from __future__ import annotations

import tensorflow as tf


def _regularizer(weight: float):
    if weight is None or weight <= 0:
        return None
    return tf.keras.regularizers.L2(weight)


def _activate(x, activation: str):
    if activation == "identity":
        return x
    if activation == "relu":
        return tf.nn.relu(x)
    if activation == "tanh":
        return tf.nn.tanh(x)
    if activation == "swish":
        return tf.nn.swish(x)
    raise ValueError(f"Unsupported activation: {activation}")


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int = 5, activation: str = "identity", l2_weight: float = 1.0e-7):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation=None,
            kernel_regularizer=_regularizer(l2_weight),
        )
        self.activation = activation

    def call(self, x, training: bool = False):
        return _activate(self.conv(x), self.activation)


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int = 5, activation: str = "identity", l2_weight: float = 1.0e-7):
        super().__init__()
        self.conv1 = ConvLayer(filters, kernel_size, activation, l2_weight)
        self.conv2 = ConvLayer(filters, kernel_size, activation, l2_weight)

    def call(self, x, training: bool = False):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int = 5, activation: str = "identity", l2_weight: float = 1.0e-7):
        super().__init__()
        self.block = ConvBlock(filters, kernel_size, activation, l2_weight)
        self.down = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

    def call(self, x, training: bool = False):
        skip = self.block(x, training=training)
        return skip, self.down(skip)


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int = 5, activation: str = "identity", l2_weight: float = 1.0e-7):
        super().__init__()
        self.up = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        self.block = ConvBlock(filters, kernel_size, activation, l2_weight)

    def call(self, skip, x, training: bool = False):
        x = self.up(x)
        x = tf.keras.layers.concatenate([skip, x], axis=-1)
        return self.block(x, training=training)


class LinearUNet(tf.keras.Model):
    """Pocket U-Net used by the paper: no normalization and identity by default."""

    def __init__(
        self,
        n_classes: int = 1,
        filters: int = 32,
        depth: int = 3,
        pocket: bool = True,
        activation: str = "identity",
        kernel_size: int = 5,
        l2_weight: float = 1.0e-7,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.filters = filters
        self.depth = depth
        self.pocket = pocket
        self.activation_name = activation
        self.kernel_size = kernel_size
        self.l2_weight = l2_weight
        multiplier = 1 if pocket else 2

        self.encoder = []
        for i in range(depth):
            f = filters * (multiplier ** i)
            self.encoder.append(EncoderBlock(f, kernel_size, activation, l2_weight))

        bottleneck_filters = filters * (multiplier ** depth)
        self.bottleneck = ConvBlock(bottleneck_filters, kernel_size, activation, l2_weight)

        self.decoder = []
        for i in range(depth - 1, -1, -1):
            f = filters * (multiplier ** i)
            self.decoder.append(DecoderBlock(f, kernel_size, activation, l2_weight))

        self.out = tf.keras.layers.Conv2D(n_classes, 1, padding="same", dtype="float32")

    @tf.function
    def call(self, x, training: bool = False):
        skips = []
        for block in self.encoder:
            skip, x = block(x, training=training)
            skips.append(skip)
        x = self.bottleneck(x, training=training)
        for skip, block in zip(reversed(skips), self.decoder):
            x = block(skip, x, training=training)
        return self.out(x)


def build_unet(
    n: int,
    filters: int = 32,
    depth: int = 3,
    activation: str = "identity",
    kernel_size: int = 5,
    l2_weight: float = 1.0e-7,
    pocket: bool = True,
) -> LinearUNet:
    model = LinearUNet(
        n_classes=1,
        filters=filters,
        depth=depth,
        pocket=pocket,
        activation=activation,
        kernel_size=kernel_size,
        l2_weight=l2_weight,
    )
    model.build((1, n, n, 1))
    return model
