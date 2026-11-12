import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf


def set_seed(seed: int):
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_mat(path):
    try:
        import scipy.io as sio
        data = sio.loadmat(path)
        return {k: v for k, v in data.items() if not k.startswith("__")}
    except Exception:
        import h5py
        out = {}
        with h5py.File(path, "r") as f:
            def visit(name, obj):
                if hasattr(obj, "shape"):
                    arr = np.array(obj)
                    if arr.ndim >= 2:
                        arr = arr.T
                    out[name] = arr
            f.visititems(visit)
        return out


def align_space_length(arr: np.ndarray, target_len: int, mode: str = "pad_last"):
    """
    Align spatial dimension to target_len.

    arr shape: [samples, space]
    target_len: target spatial length, e.g. u.shape[1]

    If arr has target_len - 1 points, it is usually a derivative array.
    We pad one value at the end by default.
    """
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Only 2D arrays are supported. Got shape={arr.shape}")

    n_sample, n_space = arr.shape

    if n_space == target_len:
        return arr

    if n_space == target_len - 1:
        if mode == "pad_last":
            last = arr[:, -1:]
            return np.concatenate([arr, last], axis=1)
        if mode == "pad_zero":
            zero = np.zeros((n_sample, 1), dtype=arr.dtype)
            return np.concatenate([arr, zero], axis=1)
        if mode == "crop_target":
            return arr

    if n_space > target_len:
        return arr[:, :target_len]

    if n_space < target_len:
        pad_width = target_len - n_space
        if mode == "pad_zero":
            pad = np.zeros((n_sample, pad_width), dtype=arr.dtype)
        else:
            pad = np.repeat(arr[:, -1:], pad_width, axis=1)
        return np.concatenate([arr, pad], axis=1)

    return arr


def build_residual_cnn(
    nx: int,
    in_channels: int,
    channels: int = 64,
    blocks: int = 6,
    kernel_size: int = 5,
    activation: str = "gelu",
    dropout: float = 0.0,
    lr: float = 1e-3,
):
    inp = tf.keras.Input(shape=(nx, in_channels))

    x = tf.keras.layers.Conv1D(
        channels,
        kernel_size,
        padding="same",
        activation=None,
    )(inp)

    for _ in range(blocks):
        shortcut = x

        y = tf.keras.layers.Conv1D(
            channels,
            kernel_size,
            padding="same",
            activation=None,
        )(x)

        if activation == "gelu":
            y = tf.keras.layers.Activation(tf.keras.activations.gelu)(y)
        elif activation == "swish":
            y = tf.keras.layers.Activation(tf.keras.activations.swish)(y)
        elif activation == "relu":
            y = tf.keras.layers.Activation("relu")(y)
        elif activation == "tanh":
            y = tf.keras.layers.Activation("tanh")(y)
        elif activation == "identity":
            pass
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        if dropout > 0:
            y = tf.keras.layers.Dropout(dropout)(y)

        y = tf.keras.layers.Conv1D(
            channels,
            kernel_size,
            padding="same",
            activation=None,
        )(y)

        x = tf.keras.layers.Add()([shortcut, y])

    out = tf.keras.layers.Conv1D(
        1,
        kernel_size,
        padding="same",
        activation=None,
    )(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="mse",
    )
    return model


def relative_l2(y_true, y_pred):
    num = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=(1, 2)))
    den = np.sqrt(np.sum(y_true ** 2, axis=(1, 2))) + 1e-12
    return float(np.mean(num / den))


def relative_l2_per_sample(y_true, y_pred):
    num = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=(1, 2)))
    den = np.sqrt(np.sum(y_true ** 2, axis=(1, 2))) + 1e-12
    return num / den


def plot_examples(x_raw, y_true, y_pred, input_names, outdir, n_show=4):
    n_show = min(n_show, x_raw.shape[0])
    xx = np.linspace(0.0, 1.0, x_raw.shape[1])

    fig, axs = plt.subplots(n_show, 1, figsize=(10, 3.2 * n_show), sharex=True)
    if n_show == 1:
        axs = [axs]

    for i in range(n_show):
        for c, name in enumerate(input_names):
            axs[i].plot(
                xx,
                x_raw[i, :, c],
                linewidth=1.0,
                alpha=0.75,
                label=f"Input {name}",
            )

        axs[i].plot(
            xx,
            y_true[i, :, 0],
            linewidth=1.5,
            label="True u",
        )

        axs[i].plot(
            xx,
            y_pred[i, :, 0],
            "--",
            linewidth=1.5,
            label="Predicted u",
        )

        axs[i].set_ylabel("value")
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(loc="best", fontsize=8)

    axs[-1].set_xlabel("x")
    fig.suptitle("Burgers multichannel prediction")
    fig.tight_layout()
    fig.savefig(outdir / "prediction_examples.png", dpi=300)
    plt.close(fig)


def plot_heatmap(y_true, y_pred, outdir, n_show=64):
    n_show = min(n_show, y_true.shape[0])
    true = y_true[:n_show, :, 0]
    pred = y_pred[:n_show, :, 0]
    err = np.abs(true - pred)

    vmax = max(float(np.max(np.abs(true))), float(np.max(np.abs(pred))))

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))

    im0 = axs[0].imshow(
        true,
        aspect="auto",
        cmap="jet",
        vmin=-vmax,
        vmax=vmax,
    )
    axs[0].set_title("True u")
    axs[0].set_xlabel("x index")
    axs[0].set_ylabel("test sample")

    im1 = axs[1].imshow(
        pred,
        aspect="auto",
        cmap="jet",
        vmin=-vmax,
        vmax=vmax,
    )
    axs[1].set_title("Predicted u")
    axs[1].set_xlabel("x index")

    im2 = axs[2].imshow(
        err,
        aspect="auto",
        cmap="jet",
    )
    axs[2].set_title("Absolute error")
    axs[2].set_xlabel("x index")

    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(outdir / "prediction_heatmap.png", dpi=300)
    plt.close(fig)


def plot_loss(history, outdir):
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.arange(1, len(loss) + 1), loss, label="train loss")
    ax.plot(np.arange(1, len(val_loss) + 1), val_loss, label="val loss")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "loss_curve.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Multichannel supervised Burgers experiment: [a, a_x, ...] -> u"
    )

    parser.add_argument("--mat", required=True)
    parser.add_argument("--input-vars", default="a,a_x")
    parser.add_argument("--target-var", default="u")
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--out", default="results/extensions/burgers/multichannel_R10")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument(
        "--activation",
        choices=["gelu", "swish", "relu", "tanh", "identity"],
        default="gelu",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--align-mode",
        choices=["pad_last", "pad_zero", "crop_target"],
        default="pad_last",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_mat(args.mat)

    input_vars = [v.strip() for v in args.input_vars.split(",") if v.strip()]
    target_var = args.target_var

    if target_var not in data:
        raise KeyError(f"Target variable {target_var} not found. Available keys: {list(data.keys())}")

    for v in input_vars:
        if v not in data:
            raise KeyError(f"Input variable {v} not found. Available keys: {list(data.keys())}")

    y = np.squeeze(np.asarray(data[target_var], dtype=np.float32))

    if y.ndim != 2:
        raise ValueError(f"Target variable must be 2D [samples, space]. Got shape={y.shape}")

    n_samples, target_space = y.shape

    input_arrays = []
    raw_shapes = {}

    for v in input_vars:
        arr = np.squeeze(np.asarray(data[v], dtype=np.float32))

        if arr.ndim != 2:
            raise ValueError(f"Input variable {v} must be 2D [samples, space]. Got shape={arr.shape}")

        raw_shapes[v] = list(arr.shape)

        if arr.shape[0] != n_samples:
            if arr.shape[1] == n_samples:
                arr = arr.T
            else:
                raise ValueError(
                    f"Input variable {v} sample size does not match target. "
                    f"{v}.shape={arr.shape}, target.shape={y.shape}"
                )

        arr = align_space_length(arr, target_space, mode=args.align_mode)
        input_arrays.append(arr)

    x = np.stack(input_arrays, axis=-1)
    y = y[..., None]

    x = x[:, ::args.stride, :]
    y = y[:, ::args.stride, :]

    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)

    train_n = int(0.8 * len(idx))
    train_idx = idx[:train_n]
    test_idx = idx[train_n:]

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    x_mean = np.mean(x_train, axis=(0, 1), keepdims=True)
    x_std = np.std(x_train, axis=(0, 1), keepdims=True) + 1e-8

    y_mean = np.mean(y_train)
    y_std = np.std(y_train) + 1e-8

    x_train_n = (x_train - x_mean) / x_std
    x_test_n = (x_test - x_mean) / x_std

    y_train_n = (y_train - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    nx = x_train.shape[1]
    in_channels = x_train.shape[2]

    model = build_residual_cnn(
        nx=nx,
        in_channels=in_channels,
        channels=args.channels,
        blocks=args.blocks,
        kernel_size=args.kernel_size,
        activation=args.activation,
        dropout=args.dropout,
        lr=args.lr,
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train_n,
        y_train_n,
        validation_data=(x_test_n, y_test_n),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    pred_n = model.predict(x_test_n, verbose=0)
    pred = pred_n * y_std + y_mean

    mse = float(np.mean((y_test - pred) ** 2))
    rel = relative_l2(y_test, pred)
    rel_each = relative_l2_per_sample(y_test, pred)

    linf = float(np.max(np.abs(y_test - pred)))
    mae = float(np.mean(np.abs(y_test - pred)))

    metrics = {
        "mat_file": args.mat,
        "input_vars": input_vars,
        "target_var": target_var,
        "raw_shapes": raw_shapes,
        "target_raw_shape": list(np.squeeze(data[target_var]).shape),
        "used_shape_x": list(x.shape),
        "used_shape_y": list(y.shape),
        "stride": args.stride,
        "align_mode": args.align_mode,
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "epochs_requested": args.epochs,
        "epochs_trained": int(len(history.history.get("loss", []))),
        "batch_size": args.batch_size,
        "channels": args.channels,
        "blocks": args.blocks,
        "kernel_size": args.kernel_size,
        "activation": args.activation,
        "dropout": args.dropout,
        "lr": args.lr,
        "num_params": int(model.count_params()),
        "test_mse": mse,
        "test_mae": mae,
        "test_relative_l2": rel,
        "test_relative_l2_median": float(np.median(rel_each)),
        "test_relative_l2_std": float(np.std(rel_each)),
        "test_linf": linf,
        "x_mean_per_channel": np.squeeze(x_mean).astype(float).tolist(),
        "x_std_per_channel": np.squeeze(x_std).astype(float).tolist(),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
    }

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    loss = np.array(history.history.get("loss", []), dtype=np.float64)
    val_loss = np.array(history.history.get("val_loss", []), dtype=np.float64)
    np.savetxt(
        outdir / "loss_history.csv",
        np.column_stack([loss, val_loss]),
        delimiter=",",
        header="loss,val_loss",
        comments="",
    )

    np.save(outdir / "test_input.npy", x_test)
    np.save(outdir / "test_true.npy", y_test)
    np.save(outdir / "test_pred.npy", pred)
    np.save(outdir / "test_indices.npy", test_idx)

    plot_examples(x_test, y_test, pred, input_vars, outdir)
    plot_heatmap(y_test, pred, outdir)
    plot_loss(history, outdir)

    model.save(outdir / "model.keras")

    print("完成。结果目录：", outdir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
