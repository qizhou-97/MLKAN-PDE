from __future__ import annotations

import tensorflow as tf

from .layers import FeatureCorrection


def _residual(x, channels, kernel_size, activation, name):
    y = tf.keras.layers.Conv1D(channels, kernel_size, padding="same", activation=activation,
                               name=f"{name}_conv1")(x)
    y = tf.keras.layers.Conv1D(channels, kernel_size, padding="same", name=f"{name}_conv2")(y)
    return tf.keras.layers.Add(name=f"{name}_add")([x, y])


def build_rescnn(input_shape, channels=64, blocks=6, kernel_size=5, activation="gelu",
                 correction="none", gammas=(0.02, 0.05, 0.08), widths=(16, 16, 16),
                 grid_size=8, kan_basis="bspline", spline_order=3,
                 gate="fixed", correction_regularization=0.0):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(channels, kernel_size, padding="same", activation=activation)(inputs)
    stage_sizes = [blocks // 3 + (1 if i < blocks % 3 else 0) for i in range(3)]
    block_index = 0
    for stage, stage_size in enumerate(stage_sizes):
        for _ in range(stage_size):
            x = _residual(x, channels, kernel_size, activation, f"block_{block_index + 1}")
            block_index += 1
        if correction == "multi":
            x = FeatureCorrection(
                1, channels, widths[stage], gammas[stage], transform="kan", gate=gate,
                grid_size=grid_size, kan_basis=kan_basis, spline_order=spline_order,
                correction_regularization=correction_regularization,
                name=f"stage_correction_{stage + 1}",
            )(x)
        elif correction == "single" and stage == 1:
            x = FeatureCorrection(
                1, channels, widths[stage], gammas[stage], transform="kan", gate=gate,
                grid_size=grid_size, kan_basis=kan_basis, spline_order=spline_order,
                correction_regularization=correction_regularization,
                name="middle_correction",
            )(x)
    outputs = tf.keras.layers.Conv1D(1, kernel_size, padding="same", dtype="float32")(x)
    return tf.keras.Model(inputs, outputs, name=f"ResCNN_{correction}")
