from __future__ import annotations

import numpy as np
import tensorflow as tf

from .models import ConvBlock, EncoderBlock, DecoderBlock


def _activate(x, activation: str):
    if activation == "identity":
        return x
    if activation == "relu":
        return tf.nn.relu(x)
    if activation == "tanh":
        return tf.nn.tanh(x)
    if activation == "swish":
        return tf.nn.swish(x)
    if activation == "gelu":
        return tf.keras.activations.gelu(x)
    raise ValueError(f"Unsupported activation: {activation}")


class RBFKANLayer2D(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int,
        grid_size: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        use_base: bool = True,
        clip_value: float = 5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = int(units)
        self.grid_size = int(grid_size)
        self.grid_min = float(grid_min)
        self.grid_max = float(grid_max)
        self.use_base = bool(use_base)
        self.clip_value = float(clip_value)

        centers = np.linspace(self.grid_min, self.grid_max, self.grid_size).astype(np.float32)
        self.centers = tf.constant(
            centers.reshape(1, 1, 1, 1, self.grid_size),
            dtype=tf.float32,
        )

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

        basis = tf.exp(
            -tf.square(tf.expand_dims(x, axis=-1) - self.centers)
            / (2.0 * self.sigma * self.sigma + 1.0e-12)
        )

        y = tf.einsum("...cg,cgu->...u", basis, self.spline_weight)
        y = y + self.spline_bias

        if self.use_base:
            y = y + self.base_dense(x)

        return y

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "units": self.units,
            "grid_size": self.grid_size,
            "grid_min": self.grid_min,
            "grid_max": self.grid_max,
            "use_base": self.use_base,
            "clip_value": self.clip_value,
        })
        return cfg


class KANMiddleBlock2D(tf.keras.layers.Layer):
    def __init__(
        self,
        channels: int,
        grid_size: int = 8,
        gamma: float = 0.05,
        activation: str = "identity",
        use_layernorm: bool = True,
        gate_mode: str = "fixed",
        se_reduction: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.grid_size = int(grid_size)
        self.gamma = float(gamma)
        self.activation = activation
        self.use_layernorm = bool(use_layernorm)
        self.gate_mode = str(gate_mode)
        self.se_reduction = int(se_reduction)

        self.norm = (
            tf.keras.layers.LayerNormalization(axis=-1)
            if self.use_layernorm
            else None
        )

        self.kan = RBFKANLayer2D(
            units=self.channels,
            grid_size=self.grid_size,
            grid_min=-2.0,
            grid_max=2.0,
            use_base=True,
            clip_value=5.0,
        )

        self.proj = tf.keras.layers.Conv2D(
            self.channels,
            1,
            padding="same",
            activation=None,
        )

        if self.gate_mode == "fixed":
            self.gate = None
        elif self.gate_mode == "spatial_shared":
            # One spatial gate (C=1), broadcast to every feature channel.
            self.gate = tf.keras.layers.Conv2D(
                1, 1, padding="same", kernel_initializer="zeros",
                bias_initializer="zeros", name="spatial_gate_c1",
            )
        elif self.gate_mode == "spatial_channel":
            # A separate 1x1 spatial gate for each channel (no channel mixing).
            self.gate = tf.keras.layers.DepthwiseConv2D(
                1, padding="same", depth_multiplier=1,
                depthwise_initializer="zeros", bias_initializer="zeros",
                name="spatial_gate_per_channel",
            )
        elif self.gate_mode == "se":
            hidden = max(self.channels // self.se_reduction, 1)
            self.gate = tf.keras.Sequential([
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(hidden, activation="relu"),
                tf.keras.layers.Dense(
                    self.channels, activation=None,
                    kernel_initializer="zeros", bias_initializer="zeros",
                ),
                tf.keras.layers.Reshape((1, 1, self.channels)),
            ], name="se_gate")
        else:
            raise ValueError(
                "gate_mode must be one of: fixed, spatial_shared, "
                "spatial_channel, se"
            )

    def call(self, x, training: bool = False):
        z = x

        if self.norm is not None:
            z = self.norm(z)

        z = self.kan(z)
        z = _activate(z, self.activation)
        z = self.proj(z)

        if self.gate is None:
            scale = self.gamma
        else:
            # Zero-initialized logits give scale=gamma at initialization.
            scale = self.gamma * 2.0 * tf.nn.sigmoid(
                self.gate(x, training=training)
            )

        return x + scale * z

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "channels": self.channels,
            "grid_size": self.grid_size,
            "gamma": self.gamma,
            "activation": self.activation,
            "use_layernorm": self.use_layernorm,
            "gate_mode": self.gate_mode,
            "se_reduction": self.se_reduction,
        })
        return cfg


def _validate_gammas(gammas, depth: int) -> tuple[float, ...]:
    values = tuple(float(g) for g in gammas)
    if len(values) != depth:
        raise ValueError(
            f"Expected {depth} encoder gamma values, got {len(values)}: {values}"
        )
    if any(g < 0.0 for g in values):
        raise ValueError(f"Gamma values must be non-negative: {values}")
    return values


class LinearUNetKANMiddle(tf.keras.Model):
    def __init__(
        self,
        n_classes: int = 1,
        filters: int = 32,
        depth: int = 3,
        pocket: bool = True,
        activation: str = "identity",
        kernel_size: int = 5,
        l2_weight: float = 1.0e-7,
        kan_grid_size: int = 8,
        kan_gamma: float = 0.05,
        kan_activation: str = "identity",
        kan_layernorm: bool = True,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.filters = filters
        self.depth = depth
        self.pocket = pocket
        self.activation_name = activation
        self.kernel_size = kernel_size
        self.l2_weight = l2_weight
        self.kan_grid_size = kan_grid_size
        self.kan_gamma = kan_gamma
        self.kan_activation = kan_activation
        self.kan_layernorm = kan_layernorm

        multiplier = 1 if pocket else 2

        self.encoder = []
        for i in range(depth):
            f = filters * (multiplier ** i)
            self.encoder.append(
                EncoderBlock(f, kernel_size, activation, l2_weight)
            )

        bottleneck_filters = filters * (multiplier ** depth)

        self.bottleneck = ConvBlock(
            bottleneck_filters,
            kernel_size,
            activation,
            l2_weight,
        )

        self.kan_middle = KANMiddleBlock2D(
            channels=bottleneck_filters,
            grid_size=kan_grid_size,
            gamma=kan_gamma,
            activation=kan_activation,
            use_layernorm=kan_layernorm,
            name="kan_middle_bottleneck",
        )

        self.decoder = []
        for i in range(depth - 1, -1, -1):
            f = filters * (multiplier ** i)
            self.decoder.append(
                DecoderBlock(f, kernel_size, activation, l2_weight)
            )

        self.out = tf.keras.layers.Conv2D(
            n_classes,
            1,
            padding="same",
            dtype="float32",
        )

    def call(self, x, training: bool = False):
        skips = []

        for block in self.encoder:
            skip, x = block(x, training=training)
            skips.append(skip)

        x = self.bottleneck(x, training=training)
        x = self.kan_middle(x, training=training)

        for skip, block in zip(reversed(skips), self.decoder):
            x = block(skip, x, training=training)

        return self.out(x)


def build_unet_kan_middle(
    n: int,
    filters: int = 32,
    depth: int = 3,
    activation: str = "identity",
    kernel_size: int = 5,
    l2_weight: float = 1.0e-7,
    pocket: bool = True,
    kan_grid_size: int = 8,
    kan_gamma: float = 0.05,
    kan_activation: str = "identity",
    kan_layernorm: bool = True,
) -> LinearUNetKANMiddle:
    model = LinearUNetKANMiddle(
        n_classes=1,
        filters=filters,
        depth=depth,
        pocket=pocket,
        activation=activation,
        kernel_size=kernel_size,
        l2_weight=l2_weight,
        kan_grid_size=kan_grid_size,
        kan_gamma=kan_gamma,
        kan_activation=kan_activation,
        kan_layernorm=kan_layernorm,
    )

    dummy = tf.zeros((1, n, n, 1), dtype=tf.float32)
    _ = model(dummy, training=False)

    return model


class LinearUNetKANMultiLevel(tf.keras.Model):
    """U-Net with an independent KAN-Middle branch at every encoder level.

    For depth=3, encoder_gammas are ordered from the shallow/high-resolution
    level to the deep/low-resolution level. The corrected feature is used by
    both the skip path and the subsequent downsampling path.
    """

    def __init__(
        self,
        n_classes: int = 1,
        filters: int = 32,
        depth: int = 3,
        pocket: bool = True,
        activation: str = "identity",
        kernel_size: int = 5,
        l2_weight: float = 1.0e-7,
        kan_grid_size: int = 8,
        encoder_gammas=(0.02, 0.05, 0.08),
        kan_activation: str = "identity",
        kan_layernorm: bool = True,
        gate_mode: str = "fixed",
        se_reduction: int = 8,
    ):
        super().__init__()
        self.n_classes = int(n_classes)
        self.filters = int(filters)
        self.depth = int(depth)
        self.pocket = bool(pocket)
        self.activation_name = activation
        self.kernel_size = int(kernel_size)
        self.l2_weight = float(l2_weight)
        self.kan_grid_size = int(kan_grid_size)
        self.encoder_gammas = _validate_gammas(encoder_gammas, self.depth)
        self.kan_activation = kan_activation
        self.kan_layernorm = bool(kan_layernorm)
        self.gate_mode = gate_mode
        self.se_reduction = int(se_reduction)

        multiplier = 1 if self.pocket else 2
        self.encoder = []
        self.encoder_kan = []
        for i in range(self.depth):
            channels = self.filters * (multiplier ** i)
            self.encoder.append(
                EncoderBlock(channels, self.kernel_size, activation, l2_weight)
            )
            self.encoder_kan.append(
                KANMiddleBlock2D(
                    channels=channels,
                    grid_size=self.kan_grid_size,
                    gamma=self.encoder_gammas[i],
                    activation=self.kan_activation,
                    use_layernorm=self.kan_layernorm,
                    gate_mode=self.gate_mode,
                    se_reduction=self.se_reduction,
                    name=f"kan_middle_encoder_level_{i + 1}",
                )
            )

        bottleneck_filters = self.filters * (multiplier ** self.depth)
        self.bottleneck = ConvBlock(
            bottleneck_filters, self.kernel_size, activation, l2_weight
        )
        self.decoder = []
        for i in range(self.depth - 1, -1, -1):
            channels = self.filters * (multiplier ** i)
            self.decoder.append(
                DecoderBlock(channels, self.kernel_size, activation, l2_weight)
            )
        self.out = tf.keras.layers.Conv2D(
            self.n_classes, 1, padding="same", dtype="float32"
        )

    def call(self, x, training: bool = False):
        skips = []
        for block, corrector in zip(self.encoder, self.encoder_kan):
            skip, _ = block(x, training=training)
            skip = corrector(skip, training=training)
            skips.append(skip)
            x = block.down(skip)

        x = self.bottleneck(x, training=training)
        for skip, block in zip(reversed(skips), self.decoder):
            x = block(skip, x, training=training)
        return self.out(x)


def build_unet_kan_multilevel(
    n: int,
    filters: int = 32,
    depth: int = 3,
    activation: str = "identity",
    kernel_size: int = 5,
    l2_weight: float = 1.0e-7,
    pocket: bool = True,
    kan_grid_size: int = 8,
    encoder_gammas=(0.02, 0.05, 0.08),
    kan_activation: str = "identity",
    kan_layernorm: bool = True,
    gate_mode: str = "fixed",
    se_reduction: int = 8,
) -> LinearUNetKANMultiLevel:
    model = LinearUNetKANMultiLevel(
        n_classes=1,
        filters=filters,
        depth=depth,
        pocket=pocket,
        activation=activation,
        kernel_size=kernel_size,
        l2_weight=l2_weight,
        kan_grid_size=kan_grid_size,
        encoder_gammas=encoder_gammas,
        kan_activation=kan_activation,
        kan_layernorm=kan_layernorm,
        gate_mode=gate_mode,
        se_reduction=se_reduction,
    )
    dummy = tf.zeros((1, n, n, 1), dtype=tf.float32)
    _ = model(dummy, training=False)
    return model
