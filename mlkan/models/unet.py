from __future__ import annotations

import tensorflow as tf

from .layers import BSplineKAN, FeatureCorrection, RBFKAN, conv_nd


def _pool(dimension):
    cls = tf.keras.layers.MaxPool1D if dimension == 1 else tf.keras.layers.MaxPool2D
    return cls(pool_size=2)


def _upsample(dimension):
    if dimension == 1:
        return tf.keras.layers.UpSampling1D(size=2)
    return tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")


def conv_block(x, dimension, channels, kernel_size, activation, name):
    x = conv_nd(dimension, channels, kernel_size, padding="same", activation=activation,
                name=f"{name}_conv1")(x)
    x = conv_nd(dimension, channels, kernel_size, padding="same", activation=activation,
                name=f"{name}_conv2")(x)
    return x


def build_unet(
    input_shape,
    dimension,
    base_channels=32,
    depth=3,
    kernel_size=5,
    activation="gelu",
    correction="none",
    correction_levels=(0, 1, 2),
    gammas=(0.02, 0.05, 0.08),
    kan_widths=(8, 16, 32),
    kan_grid_size=8,
    kan_basis="bspline",
    spline_order=3,
    share_kan=False,
    gate="fixed",
    correction_regularization=0.0,
):
    """Build dimension-adaptive U-Net with independent encoder corrections.

    correction: none, single, kan, conv, or mlp. The single variant places one
    KAN correction at the bottleneck; other variants use correction_levels.
    """
    if len(gammas) < depth or len(kan_widths) < depth:
        raise ValueError("gammas and kan_widths must cover every encoder level")
    inputs = tf.keras.Input(shape=input_shape, name="pde_input")
    shared_core = None
    if share_kan and correction == "kan":
        if len(set(kan_widths[:depth])) != 1:
            raise ValueError("share_kan requires equal kan_widths at all corrected levels")
        width = kan_widths[0]
        shared_core = (BSplineKAN(width, grid_size=kan_grid_size, spline_order=spline_order,
                                  name="shared_bspline_kan") if kan_basis == "bspline"
                       else RBFKAN(width, grid_size=kan_grid_size, name="shared_rbf_kan"))
    x = inputs
    skips = []
    for level in range(depth):
        channels = base_channels * (2 ** level)
        x = conv_block(x, dimension, channels, kernel_size, activation, f"encoder_{level + 1}")
        if correction in {"kan", "conv", "mlp"} and level in correction_levels:
            x = FeatureCorrection(
                dimension=dimension, channels=channels, width=kan_widths[level],
                gamma=gammas[level], transform=correction, gate=gate,
                grid_size=kan_grid_size,
                kan_basis=kan_basis, spline_order=spline_order,
                shared_core=shared_core,
                correction_regularization=correction_regularization,
                name=f"correction_level_{level + 1}",
            )(x)
        skips.append(x)
        x = _pool(dimension)(x)

    bottleneck_channels = base_channels * (2 ** depth)
    x = conv_block(x, dimension, bottleneck_channels, kernel_size, activation, "bottleneck")
    if correction == "single":
        x = FeatureCorrection(
            dimension=dimension, channels=bottleneck_channels,
            width=kan_widths[-1], gamma=gammas[-1], transform="kan", gate=gate,
            grid_size=kan_grid_size,
            kan_basis=kan_basis, spline_order=spline_order,
            correction_regularization=correction_regularization,
            name="correction_bottleneck",
        )(x)

    for level in reversed(range(depth)):
        channels = base_channels * (2 ** level)
        x = _upsample(dimension)(x)
        x = tf.keras.layers.Concatenate(axis=-1)([skips[level], x])
        x = conv_block(x, dimension, channels, kernel_size, activation, f"decoder_{level + 1}")
    outputs = conv_nd(dimension, 1, 1, padding="same", dtype="float32", name="solution")(x)
    name = f"UNet{dimension}D_{correction}_{gate}"
    return tf.keras.Model(inputs, outputs, name=name)
