from __future__ import annotations

import csv
import resource
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from ..data.loading import load_dataset
from ..metrics import benchmark_inference, per_sample_metrics, summarize_metrics
from ..models.layers import FeatureCorrection
from ..models.registry import MODEL_SPECS, build_model
from ..physics.losses import compute_loss_terms
from ..physics.losses import pde_residual
from ..utils import ensure_dir, set_seed, write_json


def _normalization(train):
    spatial_axes = tuple(range(train["x"].ndim - 1))
    x_mean = np.mean(train["x"], axis=spatial_axes, keepdims=True)
    x_std = np.std(train["x"], axis=spatial_axes, keepdims=True)
    x_std = np.where(x_std < 1.0e-8, 1.0, x_std)
    y_mean = float(np.mean(train["y"]))
    y_std = max(float(np.std(train["y"])), 1.0e-8)
    return x_mean.astype(np.float32), x_std.astype(np.float32), y_mean, y_std


def _batches(values, batch_size, rng, shuffle=True):
    indices = np.arange(len(values["x"]))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        yield values["x"][idx], values["y"][idx]


def _set_correction_warmup(model, value):
    for layer in _iter_keras_layers(model):
        if isinstance(layer, FeatureCorrection):
            layer.warmup_scale.assign(float(value))



def _iter_keras_layers(model):
    """Iterate over Keras layers for both tf.keras.Model and Functional models."""
    if hasattr(model, "_flatten_layers"):
        try:
            return list(model._flatten_layers(include_self=False, recursive=True))
        except TypeError:
            try:
                return list(model._flatten_layers(recursive=True))
            except TypeError:
                pass

    if hasattr(model, "submodules"):
        try:
            return list(model.submodules)
        except Exception:
            pass

    layers = []
    seen = set()

    def visit(obj):
        for layer in getattr(obj, "layers", []):
            if id(layer) in seen:
                continue
            seen.add(id(layer))
            layers.append(layer)
            visit(layer)

    visit(model)
    return layers

def _build_correction_probes(model):
    probes = []
    for layer in _iter_keras_layers(model):
        if isinstance(layer, FeatureCorrection):
            try:
                probes.append((layer, tf.keras.Model(model.input, layer.input)))
            except Exception:
                continue
    return probes


def _correction_diagnostics(probes, normalized_sample):
    diagnostics = {}
    for index, (layer, feature_model) in enumerate(probes, start=1):
        feature = feature_model(normalized_sample, training=False)
        delta = layer.transform(layer.norm(feature), training=False)
        feature_rms = tf.sqrt(tf.reduce_mean(tf.square(feature)) + 1.0e-12)
        delta_rms = tf.sqrt(tf.reduce_mean(tf.square(delta)) + 1.0e-12)
        spatial_mean = 1.0
        channel_mean = 1.0
        if layer.spatial_gate is not None:
            spatial = 2.0 * tf.nn.sigmoid(
                layer.spatial_gate(tf.concat([feature, delta], axis=-1))
            )
            spatial_mean = float(tf.reduce_mean(spatial).numpy())
        if layer.channel_gate is not None:
            channel = 2.0 * tf.nn.sigmoid(layer.channel_gate(feature, training=False))
            channel_mean = float(tf.reduce_mean(channel).numpy())
        if layer.gate_name == "none":
            nominal_scale = 1.0
        elif layer.gate_name == "learnable_scalar":
            nominal_scale = layer.gamma * 2.0 * float(tf.nn.sigmoid(layer.scalar_logit).numpy())
        else:
            nominal_scale = layer.gamma
        prefix = f"level_{index}"
        diagnostics[f"{prefix}_delta_feature_rms_ratio"] = float((delta_rms / feature_rms).numpy())
        diagnostics[f"{prefix}_spatial_gate_mean"] = spatial_mean
        diagnostics[f"{prefix}_channel_gate_mean"] = channel_mean
        diagnostics[f"{prefix}_effective_scale_mean"] = (
            float(layer.warmup_scale.numpy()) * nominal_scale * spatial_mean * channel_mean
        )
    return diagnostics


def _model_flops(model, input_shape):
    try:
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2,
        )
        concrete = tf.function(lambda x: model(x, training=False)).get_concrete_function(
            tf.TensorSpec([1, *input_shape], tf.float32)
        )
        frozen = convert_variables_to_constants_v2(concrete)
        options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        with frozen.graph.as_default():
            graph_info = tf.compat.v1.profiler.profile(graph=frozen.graph, options=options)
        return int(graph_info.total_float_ops)
    except Exception:
        return None


def train_experiment(dataset_path, model_id, output_dir, config, seed=0):
    set_seed(seed)
    output_dir = ensure_dir(output_dir)
    splits, metadata = load_dataset(dataset_path)
    train, val, test = splits["train"], splits["val"], splits["test"]
    x_mean, x_std, y_mean, y_std = _normalization(train)
    input_shape = tuple(train["x"].shape[1:])
    model_config = dict(config.get("model", {}))
    model_config["correction_regularization"] = config.get("loss", {}).get("correction", 0.0)
    model = build_model(model_id, input_shape, metadata["dimension"], model_config)
    _ = model(tf.zeros([1, *input_shape], tf.float32), training=False)
    gpu_peak_bytes = None
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass

    train_config = config.get("training", {})
    loss_weights = dict(config.get("loss", {}))
    if model_id == "M10":
        loss_weights.update(residual=0.0, boundary=0.0, gradient=0.0, correction=0.0)
    if model_id == "M11" and loss_weights.get("residual", 0.0) <= 0:
        loss_weights["residual"] = 1.0e-4
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=train_config.get("learning_rate", 1.0e-3),
        clipnorm=train_config.get("clipnorm", 1.0),
    )
    batch_size = int(train_config.get("batch_size", 16))
    epochs = int(train_config.get("epochs", 100))
    patience = int(train_config.get("patience", 20))
    warmup_epochs = int(round(epochs * train_config.get("correction_warmup_ratio", 0.1)))
    rng = np.random.default_rng(seed)
    correction_probes = _build_correction_probes(model)
    diagnostic_raw = val["x"][:1]
    diagnostic_normalized = tf.convert_to_tensor((diagnostic_raw - x_mean) / x_std)

    @tf.function(reduce_retracing=True)
    def step(x_raw, y_raw, training):
        x_n = (x_raw - x_mean) / x_std
        y_n = (y_raw - y_mean) / y_std
        with tf.GradientTape() as tape:
            pred_n = model(x_n, training=training)
            pred = pred_n * y_std + y_mean
            regularization = tf.add_n(model.losses) if model.losses else tf.constant(0.0)
            terms = compute_loss_terms(
                metadata["task"], x_raw, y_raw, pred, y_n, pred_n,
                metadata, loss_weights, regularization,
            )
        if training:
            grads = tape.gradient(terms["total"], model.trainable_variables)
            optimizer.apply_gradients([
                (g, v) for g, v in zip(grads, model.trainable_variables) if g is not None
            ])
        return terms

    history = []
    best_val = np.inf
    best_epoch = -1
    wait = 0
    training_start = time.perf_counter()
    weights_path = output_dir / "best.weights.h5"
    for epoch in range(epochs):
        warmup = min(1.0, (epoch + 1) / max(warmup_epochs, 1)) if warmup_epochs > 0 else 1.0
        _set_correction_warmup(model, warmup)
        accum = {key: [] for key in ("total", "data", "residual", "boundary", "gradient", "correction")}
        for xb, yb in _batches(train, batch_size, rng, shuffle=True):
            terms = step(tf.convert_to_tensor(xb), tf.convert_to_tensor(yb), True)
            for key in accum:
                accum[key].append(float(terms[key].numpy()))
        val_losses = []
        for xb, yb in _batches(val, batch_size, rng, shuffle=False):
            val_losses.append(float(step(tf.convert_to_tensor(xb), tf.convert_to_tensor(yb), False)["total"].numpy()))
        row = {"epoch": epoch + 1, "warmup": warmup,
               **{f"train_{key}": float(np.mean(values)) for key, values in accum.items()},
               "val_total": float(np.mean(val_losses))}
        row.update(_correction_diagnostics(correction_probes, diagnostic_normalized))
        history.append(row)
        print(f"epoch={epoch + 1:04d} train={row['train_total']:.6e} val={row['val_total']:.6e} warmup={warmup:.3f}")
        if row["val_total"] < best_val:
            best_val = row["val_total"]
            best_epoch = epoch + 1
            model.save_weights(weights_path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    training_seconds = time.perf_counter() - training_start
    model.load_weights(weights_path)
    _set_correction_warmup(model, 1.0)

    test_x_n = (test["x"] - x_mean) / x_std
    pred_n = model.predict(test_x_n, batch_size=batch_size, verbose=0)
    prediction = pred_n * y_std + y_mean
    per_sample = per_sample_metrics(test["y"], prediction, metadata["dimension"])
    summary = summarize_metrics(per_sample)
    residual_squares = []
    for start in range(0, len(prediction), batch_size):
        residual = pde_residual(
            metadata["task"],
            tf.convert_to_tensor(test["x"][start:start + batch_size]),
            tf.convert_to_tensor(prediction[start:start + batch_size]),
            metadata,
        )
        residual_squares.append(np.square(residual.numpy()).reshape(-1))
    test_residual_mse = float(np.mean(np.concatenate(residual_squares)))
    inference_ms = benchmark_inference(
        model, tf.convert_to_tensor(test_x_n[:min(batch_size, len(test_x_n))]), repeats=20
    )
    try:
        gpu_peak_bytes = int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
    except Exception:
        gpu_peak_bytes = None
    cpu_peak_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(output_dir / "per_sample_metrics.npz", **per_sample)
    np.savez_compressed(
        output_dir / "predictions_preview.npz",
        x=test["x"][:64], y=test["y"][:64], prediction=prediction[:64],
        group_id=test.get("group_id", np.arange(len(test["x"])))[:64],
    )
    normalization = {"x_mean": x_mean.reshape(-1).tolist(), "x_std": x_std.reshape(-1).tolist(),
                     "y_mean": y_mean, "y_std": y_std}
    write_json(output_dir / "normalization.json", normalization)
    result = {
        "dataset": str(Path(dataset_path).resolve()), "task": metadata["task"],
        "model_id": model_id, "model_label": MODEL_SPECS[model_id]["label"], "seed": seed,
        "experiment_label": config.get("experiment_label", MODEL_SPECS[model_id]["label"]),
        "result_tag": config.get("result_tag", metadata["task"]),
        "num_params": int(model.count_params()), "flops": _model_flops(model, input_shape),
        "epochs_trained": len(history), "best_epoch": best_epoch, "best_val_loss": best_val,
        "training_seconds": training_seconds, "inference_ms_per_sample": inference_ms,
        "gpu_peak_bytes": gpu_peak_bytes, "cpu_peak_kb": cpu_peak_kb,
        "metrics": summary, "config": config, "metadata": metadata,
        "test_pde_residual_mse": test_residual_mse,
    }
    write_json(output_dir / "metrics.json", result)
    write_json(output_dir / "model_summary.json", {
        "name": model.name, "input_shape": input_shape,
        "layers": [{"name": layer.name, "class": layer.__class__.__name__,
                    "params": int(layer.count_params())} for layer in model.layers],
    })
    return result
