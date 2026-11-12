import argparse
import csv
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


class RBFKANLayer(tf.keras.layers.Layer):
    """
    A lightweight KAN-like layer based on radial basis functions.

    Input shape:
        [..., in_dim]

    Output shape:
        [..., units]

    The layer implements:
        y_o = sum_i sum_g W_{i,g,o} phi_g(x_i) + Dense(x)

    This is an efficient KAN-style feature transform suitable for bottleneck use.
    """

    def __init__(
        self,
        units: int,
        grid_size: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        base_activation: str = "identity",
        use_base: bool = True,
        clip_value: float = 5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = int(units)
        self.grid_size = int(grid_size)
        self.grid_min = float(grid_min)
        self.grid_max = float(grid_max)
        self.base_activation = base_activation
        self.use_base = bool(use_base)
        self.clip_value = float(clip_value)

        centers = np.linspace(self.grid_min, self.grid_max, self.grid_size).astype(np.float32)
        self.centers = tf.constant(centers.reshape(1, 1, self.grid_size), dtype=tf.float32)

        if self.grid_size > 1:
            self.sigma = float((self.grid_max - self.grid_min) / (self.grid_size - 1))
        else:
            self.sigma = 1.0

        self.base_dense = None

    def build(self, input_shape):
        in_dim = int(input_shape[-1])

        self.spline_weight = self.add_weight(
            name="spline_weight",
            shape=(in_dim, self.grid_size, self.units),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )

        self.spline_bias = self.add_weight(
            name="spline_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

        if self.use_base:
            self.base_dense = tf.keras.layers.Dense(
                self.units,
                activation=None,
                use_bias=False,
            )

        super().build(input_shape)

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)

        if self.clip_value > 0:
            x = tf.clip_by_value(x, -self.clip_value, self.clip_value)

        x_expanded = tf.expand_dims(x, axis=-1)
        basis = tf.exp(-tf.square(x_expanded - self.centers) / (2.0 * self.sigma * self.sigma + 1e-12))

        y = tf.einsum("...ig,igu->...u", basis, self.spline_weight)
        y = y + self.spline_bias

        if self.use_base:
            base_x = x
            if self.base_activation == "gelu":
                base_x = tf.keras.activations.gelu(base_x)
            elif self.base_activation == "swish":
                base_x = tf.keras.activations.swish(base_x)
            elif self.base_activation == "relu":
                base_x = tf.nn.relu(base_x)
            elif self.base_activation == "tanh":
                base_x = tf.nn.tanh(base_x)
            elif self.base_activation == "identity":
                pass
            else:
                raise ValueError(f"Unsupported base activation: {self.base_activation}")

            y = y + self.base_dense(base_x)

        return y

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "grid_size": self.grid_size,
            "grid_min": self.grid_min,
            "grid_max": self.grid_max,
            "base_activation": self.base_activation,
            "use_base": self.use_base,
            "clip_value": self.clip_value,
        })
        return config


class KANMiddleBlock(tf.keras.layers.Layer):
    """
    Middle KAN bottleneck block:
        x_out = x + gamma * Conv1D(KAN(LN(x)))

    It keeps the original CNN feature path and adds a controlled KAN correction.
    """

    def __init__(
        self,
        channels: int,
        kan_grid_size: int = 8,
        gamma: float = 0.1,
        activation: str = "gelu",
        use_layernorm: bool = True,
        dropout: float = 0.0,
        gate_mode: str = "fixed",
        se_reduction: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.kan_grid_size = int(kan_grid_size)
        self.gamma = float(gamma)
        self.activation = activation
        self.use_layernorm = bool(use_layernorm)
        self.dropout_rate = float(dropout)
        self.gate_mode = str(gate_mode)
        self.se_reduction = int(se_reduction)

        self.norm = tf.keras.layers.LayerNormalization(axis=-1) if self.use_layernorm else None
        self.kan = RBFKANLayer(
            units=self.channels,
            grid_size=self.kan_grid_size,
            grid_min=-2.0,
            grid_max=2.0,
            base_activation="identity",
            use_base=True,
            clip_value=5.0,
        )
        self.proj = tf.keras.layers.Conv1D(self.channels, 1, padding="same", activation=None)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None

        if self.gate_mode == "fixed":
            self.gate = None
        elif self.gate_mode == "spatial_shared":
            self.gate = tf.keras.layers.Conv1D(
                1, 1, padding="same", kernel_initializer="zeros",
                bias_initializer="zeros", name="spatial_gate_c1",
            )
        elif self.gate_mode == "spatial_channel":
            # groups=channels makes each channel's 1x1 gate independent.
            self.gate = tf.keras.layers.Conv1D(
                self.channels, 1, padding="same", groups=self.channels,
                kernel_initializer="zeros", bias_initializer="zeros",
                name="spatial_gate_per_channel",
            )
        elif self.gate_mode == "se":
            hidden = max(self.channels // self.se_reduction, 1)
            self.gate = tf.keras.Sequential([
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(hidden, activation="relu"),
                tf.keras.layers.Dense(
                    self.channels, activation=None,
                    kernel_initializer="zeros", bias_initializer="zeros",
                ),
                tf.keras.layers.Reshape((1, self.channels)),
            ], name="se_gate")
        else:
            raise ValueError(
                "gate_mode must be one of: fixed, spatial_shared, "
                "spatial_channel, se"
            )

    def call(self, x, training=False):
        z = x

        if self.norm is not None:
            z = self.norm(z)

        z = self.kan(z)

        if self.activation == "gelu":
            z = tf.keras.activations.gelu(z)
        elif self.activation == "swish":
            z = tf.keras.activations.swish(z)
        elif self.activation == "relu":
            z = tf.nn.relu(z)
        elif self.activation == "tanh":
            z = tf.nn.tanh(z)
        elif self.activation == "identity":
            pass
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        if self.dropout is not None:
            z = self.dropout(z, training=training)

        z = self.proj(z)

        if self.gate is None:
            scale = self.gamma
        else:
            scale = self.gamma * 2.0 * tf.nn.sigmoid(
                self.gate(x, training=training)
            )

        return x + scale * z

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "kan_grid_size": self.kan_grid_size,
            "gamma": self.gamma,
            "activation": self.activation,
            "use_layernorm": self.use_layernorm,
            "dropout": self.dropout_rate,
            "gate_mode": self.gate_mode,
            "se_reduction": self.se_reduction,
        })
        return config


def residual_block(x, channels, kernel_size, activation, dropout):
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

    return tf.keras.layers.Add()([shortcut, y])


def build_rescnn(
    nx,
    in_channels,
    channels=64,
    blocks=6,
    kernel_size=5,
    activation="gelu",
    dropout=0.0,
    lr=1e-3,
):
    inp = tf.keras.Input(shape=(nx, in_channels), name="input")

    x = tf.keras.layers.Conv1D(
        channels,
        kernel_size,
        padding="same",
        activation=None,
        name="stem_conv",
    )(inp)

    for i in range(blocks):
        x = residual_block(
            x,
            channels=channels,
            kernel_size=kernel_size,
            activation=activation,
            dropout=dropout,
        )

    out = tf.keras.layers.Conv1D(
        1,
        kernel_size,
        padding="same",
        activation=None,
        name="output_conv",
    )(x)

    model = tf.keras.Model(inp, out, name="ResCNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="mse",
    )
    return model


def build_rescnn_kan_middle(
    nx,
    in_channels,
    channels=64,
    blocks=6,
    kernel_size=5,
    activation="gelu",
    dropout=0.0,
    lr=1e-3,
    kan_grid_size=8,
    kan_gamma=0.1,
    kan_layernorm=True,
):
    inp = tf.keras.Input(shape=(nx, in_channels), name="input")

    x = tf.keras.layers.Conv1D(
        channels,
        kernel_size,
        padding="same",
        activation=None,
        name="stem_conv",
    )(inp)

    pre_blocks = blocks // 2
    post_blocks = blocks - pre_blocks

    for i in range(pre_blocks):
        x = residual_block(
            x,
            channels=channels,
            kernel_size=kernel_size,
            activation=activation,
            dropout=dropout,
        )

    x = KANMiddleBlock(
        channels=channels,
        kan_grid_size=kan_grid_size,
        gamma=kan_gamma,
        activation=activation,
        use_layernorm=kan_layernorm,
        dropout=dropout,
        name="kan_middle_bottleneck",
    )(x)

    for i in range(post_blocks):
        x = residual_block(
            x,
            channels=channels,
            kernel_size=kernel_size,
            activation=activation,
            dropout=dropout,
        )

    out = tf.keras.layers.Conv1D(
        1,
        kernel_size,
        padding="same",
        activation=None,
        name="output_conv",
    )(x)

    model = tf.keras.Model(inp, out, name="ResCNN_KAN_Middle")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="mse",
    )
    return model


def build_rescnn_kan_multilevel(
    nx,
    in_channels,
    channels=64,
    blocks=6,
    kernel_size=5,
    activation="gelu",
    dropout=0.0,
    lr=1e-3,
    kan_grid_size=8,
    stage_gammas=(0.02, 0.05, 0.08),
    kan_layernorm=True,
    gate_mode="fixed",
    se_reduction=8,
):
    """ResCNN with independent KAN corrections at shallow/middle/deep stages.

    ResCNN keeps a constant spatial resolution, so these are depth levels rather
    than U-Net-like resolution levels. The residual blocks are split as evenly
    as possible among the requested stages.
    """
    gammas = tuple(float(g) for g in stage_gammas)
    if len(gammas) != 3:
        raise ValueError(f"Expected three stage gammas, got: {gammas}")
    if blocks < len(gammas):
        raise ValueError("blocks must be at least the number of KAN stages")

    inp = tf.keras.Input(shape=(nx, in_channels), name="input")
    x = tf.keras.layers.Conv1D(
        channels, kernel_size, padding="same", activation=None,
        name="stem_conv",
    )(inp)

    quotient, remainder = divmod(blocks, len(gammas))
    stage_sizes = [quotient + (1 if i < remainder else 0) for i in range(len(gammas))]
    for stage_index, (stage_size, gamma) in enumerate(zip(stage_sizes, gammas)):
        for _ in range(stage_size):
            x = residual_block(
                x, channels=channels, kernel_size=kernel_size,
                activation=activation, dropout=dropout,
            )
        x = KANMiddleBlock(
            channels=channels,
            kan_grid_size=kan_grid_size,
            gamma=gamma,
            activation=activation,
            use_layernorm=kan_layernorm,
            dropout=dropout,
            gate_mode=gate_mode,
            se_reduction=se_reduction,
            name=f"kan_middle_stage_{stage_index + 1}",
        )(x)

    out = tf.keras.layers.Conv1D(
        1, kernel_size, padding="same", activation=None,
        name="output_conv",
    )(x)
    model = tf.keras.Model(inp, out, name="ResCNN_KAN_MultiLevel")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return model


def relative_l2(y_true, y_pred):
    num = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=(1, 2)))
    den = np.sqrt(np.sum(y_true ** 2, axis=(1, 2))) + 1e-12
    return float(np.mean(num / den))


def relative_l2_per_sample(y_true, y_pred):
    num = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=(1, 2)))
    den = np.sqrt(np.sum(y_true ** 2, axis=(1, 2))) + 1e-12
    return num / den


def plot_examples(x_raw, y_true, y_pred, input_names, outdir, model_name, n_show=4):
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
                alpha=0.70,
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
    fig.suptitle(f"Burgers prediction examples: {model_name}")
    fig.tight_layout()
    fig.savefig(outdir / "prediction_examples.png", dpi=300)
    plt.close(fig)


def plot_heatmap(y_true, y_pred, outdir, model_name, n_show=64):
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

    fig.suptitle(f"Prediction heatmap: {model_name}")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(outdir / "prediction_heatmap.png", dpi=300)
    plt.close(fig)


def plot_loss(history, outdir, model_name):
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.arange(1, len(loss) + 1), loss, label="train loss")
    ax.plot(np.arange(1, len(val_loss) + 1), val_loss, label="val loss")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(f"Training curve: {model_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "loss_curve.png", dpi=300)
    plt.close(fig)


def plot_compare_bar(rows, outdir):
    if not rows:
        return

    names = [r["model_name"] for r in rows]
    rel = [float(r["test_relative_l2"]) for r in rows]
    mse = [float(r["test_mse"]) for r in rows]
    linf = [float(r["test_linf"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    ax.bar(x, rel)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Relative L2")
    ax.set_title("Model comparison on Burgers")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "comparison_relative_l2.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, mse)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("MSE")
    ax.set_title("Model comparison on Burgers")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "comparison_mse.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, linf)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Linf")
    ax.set_title("Model comparison on Burgers")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "comparison_linf.png", dpi=300)
    plt.close(fig)


def prepare_data(args):
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

    meta = {
        "input_vars": input_vars,
        "target_var": target_var,
        "raw_shapes": raw_shapes,
        "target_raw_shape": list(np.squeeze(data[target_var]).shape),
        "used_shape_x": list(x.shape),
        "used_shape_y": list(y.shape),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "x_mean_per_channel": np.squeeze(x_mean).astype(float).tolist(),
        "x_std_per_channel": np.squeeze(x_std).astype(float).tolist(),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
    }

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "x_train_n": x_train_n,
        "y_train_n": y_train_n,
        "x_test_n": x_test_n,
        "y_test_n": y_test_n,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "y_mean": y_mean,
        "y_std": y_std,
        "meta": meta,
    }


def train_one_model(model_name, args, data_pack, outdir):
    model_dir = outdir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    x_train_n = data_pack["x_train_n"]
    y_train_n = data_pack["y_train_n"]
    x_test_n = data_pack["x_test_n"]
    y_test_n = data_pack["y_test_n"]

    x_test = data_pack["x_test"]
    y_test = data_pack["y_test"]
    y_mean = data_pack["y_mean"]
    y_std = data_pack["y_std"]
    input_vars = data_pack["meta"]["input_vars"]

    nx = x_train_n.shape[1]
    in_channels = x_train_n.shape[2]

    if model_name == "rescnn":
        model = build_rescnn(
            nx=nx,
            in_channels=in_channels,
            channels=args.channels,
            blocks=args.blocks,
            kernel_size=args.kernel_size,
            activation=args.activation,
            dropout=args.dropout,
            lr=args.lr,
        )
    elif model_name == "kan_middle":
        model = build_rescnn_kan_middle(
            nx=nx,
            in_channels=in_channels,
            channels=args.channels,
            blocks=args.blocks,
            kernel_size=args.kernel_size,
            activation=args.activation,
            dropout=args.dropout,
            lr=args.lr,
            kan_grid_size=args.kan_grid_size,
            kan_gamma=args.kan_gamma,
            kan_layernorm=not args.no_kan_layernorm,
        )
    elif model_name == "kan_multilevel":
        model = build_rescnn_kan_multilevel(
            nx=nx,
            in_channels=in_channels,
            channels=args.channels,
            blocks=args.blocks,
            kernel_size=args.kernel_size,
            activation=args.activation,
            dropout=args.dropout,
            lr=args.lr,
            kan_grid_size=args.kan_grid_size,
            stage_gammas=args.stage_kan_gammas,
            kan_layernorm=not args.no_kan_layernorm,
            gate_mode=args.kan_gate_mode,
            se_reduction=args.se_reduction,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=args.lr_patience,
            min_lr=args.min_lr,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print("=" * 80)
    print(f"Training model: {model_name}")
    print(model.summary())

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
    mae = float(np.mean(np.abs(y_test - pred)))
    rel = relative_l2(y_test, pred)
    rel_each = relative_l2_per_sample(y_test, pred)
    linf = float(np.max(np.abs(y_test - pred)))

    metrics = {
        "model_name": model_name,
        "mat_file": args.mat,
        "input_vars": data_pack["meta"]["input_vars"],
        "target_var": data_pack["meta"]["target_var"],
        "raw_shapes": data_pack["meta"]["raw_shapes"],
        "target_raw_shape": data_pack["meta"]["target_raw_shape"],
        "used_shape_x": data_pack["meta"]["used_shape_x"],
        "used_shape_y": data_pack["meta"]["used_shape_y"],
        "stride": args.stride,
        "align_mode": args.align_mode,
        "train_samples": data_pack["meta"]["train_samples"],
        "test_samples": data_pack["meta"]["test_samples"],
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
        "kan_grid_size": args.kan_grid_size if "kan" in model_name else None,
        "kan_gamma": args.kan_gamma if model_name == "kan_middle" else None,
        "stage_kan_gammas": list(args.stage_kan_gammas) if model_name == "kan_multilevel" else None,
        "kan_layernorm": (not args.no_kan_layernorm) if "kan" in model_name else None,
        "kan_gate_mode": args.kan_gate_mode if model_name == "kan_multilevel" else "fixed" if model_name == "kan_middle" else None,
        "test_mse": mse,
        "test_mae": mae,
        "test_relative_l2": rel,
        "test_relative_l2_median": float(np.median(rel_each)),
        "test_relative_l2_std": float(np.std(rel_each)),
        "test_linf": linf,
        "x_mean_per_channel": data_pack["meta"]["x_mean_per_channel"],
        "x_std_per_channel": data_pack["meta"]["x_std_per_channel"],
        "y_mean": data_pack["meta"]["y_mean"],
        "y_std": data_pack["meta"]["y_std"],
    }

    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    loss = np.array(history.history.get("loss", []), dtype=np.float64)
    val_loss = np.array(history.history.get("val_loss", []), dtype=np.float64)
    np.savetxt(
        model_dir / "loss_history.csv",
        np.column_stack([loss, val_loss]),
        delimiter=",",
        header="loss,val_loss",
        comments="",
    )

    np.save(model_dir / "test_input.npy", x_test)
    np.save(model_dir / "test_true.npy", y_test)
    np.save(model_dir / "test_pred.npy", pred)
    np.save(model_dir / "test_indices.npy", data_pack["test_idx"])

    plot_examples(x_test, y_test, pred, input_vars, model_dir, model_name=model_name)
    plot_heatmap(y_test, pred, model_dir, model_name=model_name)
    plot_loss(history, model_dir, model_name=model_name)

    model.save(model_dir / "model.keras")

    print("完成模型:", model_name)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    return metrics


def save_comparison(rows, outdir):
    if not rows:
        return

    with open(outdir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    keys = [
        "model_name",
        "num_params",
        "epochs_trained",
        "test_mse",
        "test_mae",
        "test_relative_l2",
        "test_relative_l2_median",
        "test_relative_l2_std",
        "test_linf",
        "channels",
        "blocks",
        "kernel_size",
        "activation",
        "kan_grid_size",
        "kan_gamma",
        "stage_kan_gammas",
        "kan_gate_mode",
        "kan_layernorm",
    ]

    with open(outdir / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, None) for k in keys})

    plot_compare_bar(rows, outdir)

    baseline = next((r for r in rows if r["model_name"] == "rescnn"), None)
    if baseline is not None:
        baseline_l2 = float(baseline["test_relative_l2"])
        changes = []
        for row in rows:
            if row["model_name"] == "rescnn":
                continue
            reduction = 100.0 * (
                baseline_l2 - float(row["test_relative_l2"])
            ) / max(baseline_l2, 1.0e-12)
            changes.append({
                "model_name": row["model_name"],
                "baseline_relative_l2": baseline_l2,
                "model_relative_l2": float(row["test_relative_l2"]),
                "relative_l2_reduction_percent": reduction,
                "exceeds_10_percent": bool(reduction > 10.0),
            })
        with open(outdir / "improvement_decision.json", "w", encoding="utf-8") as f:
            json.dump({
                "criterion": "Relative L2 reduction versus ResCNN exceeds 10%",
                "changes": changes,
            }, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare baseline ResCNN and ResCNN with middle KAN bottleneck on Burgers data."
    )

    parser.add_argument("--mat", required=True)
    parser.add_argument("--input-vars", default="a,a_x")
    parser.add_argument("--target-var", default="u")
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--out", default="results/extensions/burgers/compare_kan_middle_R10_a_ax_stride16")

    parser.add_argument(
        "--model",
        choices=["rescnn", "kan_middle", "kan_multilevel", "both", "multilevel_pair", "all"],
        default="both",
        help="Train baseline only, KAN-middle only, or both.",
    )

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
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=25)

    parser.add_argument("--kan-grid-size", type=int, default=8)
    parser.add_argument("--kan-gamma", type=float, default=0.1)
    parser.add_argument(
        "--stage-kan-gammas",
        type=lambda s: tuple(float(x.strip()) for x in s.split(",") if x.strip()),
        default=(0.02, 0.05, 0.08),
        help="Shallow-to-deep stage gamma values.",
    )
    parser.add_argument(
        "--kan-gate-mode",
        choices=["fixed", "spatial_shared", "spatial_channel", "se"],
        default="fixed",
    )
    parser.add_argument("--se-reduction", type=int, default=8)
    parser.add_argument("--no-kan-layernorm", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--align-mode",
        choices=["pad_last", "pad_zero", "crop_target"],
        default="pad_last",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.stage_kan_gammas) != 3:
        raise ValueError(
            "--stage-kan-gammas requires exactly three values; "
            f"got {args.stage_kan_gammas}"
        )

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    data_pack = prepare_data(args)

    with open(outdir / "data_meta.json", "w", encoding="utf-8") as f:
        json.dump(data_pack["meta"], f, indent=2, ensure_ascii=False)

    if args.model == "both":
        model_names = ["rescnn", "kan_middle"]
    elif args.model == "multilevel_pair":
        model_names = ["rescnn", "kan_multilevel"]
    elif args.model == "all":
        model_names = ["rescnn", "kan_middle", "kan_multilevel"]
    else:
        model_names = [args.model]

    rows = []

    for model_name in model_names:
        set_seed(args.seed)
        metrics = train_one_model(model_name, args, data_pack, outdir)
        rows.append(metrics)

    save_comparison(rows, outdir)

    print("=" * 80)
    print("全部完成。输出目录：", outdir)
    print("对比表：", outdir / "comparison.csv")
    print("对比 JSON：", outdir / "comparison.json")


if __name__ == "__main__":
    main()
