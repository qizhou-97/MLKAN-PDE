from __future__ import annotations

import math
import numpy as np
import tensorflow as tf


def conv_nd(dimension, *args, **kwargs):
    return (tf.keras.layers.Conv1D if dimension == 1 else tf.keras.layers.Conv2D)(*args, **kwargs)


def global_pool_nd(dimension):
    return (tf.keras.layers.GlobalAveragePooling1D if dimension == 1
            else tf.keras.layers.GlobalAveragePooling2D)()


class RBFKAN(tf.keras.layers.Layer):
    """Channel-wise RBF KAN transform operating on the last tensor axis."""

    def __init__(self, units, grid_size=8, grid_min=-2.0, grid_max=2.0,
                 use_base=True, clip_value=5.0, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.grid_size = int(grid_size)
        self.grid_min = float(grid_min)
        self.grid_max = float(grid_max)
        self.use_base = bool(use_base)
        self.clip_value = float(clip_value)
        centers = np.linspace(grid_min, grid_max, grid_size).astype(np.float32)
        self.centers = tf.constant(centers, dtype=tf.float32)
        self.sigma = (grid_max - grid_min) / max(grid_size - 1, 1)
        self.base = None

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.spline_weight = self.add_weight(name="spline_weight", shape=(channels, self.grid_size, self.units),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02), trainable=True,
        )
        self.bias = self.add_weight(name="bias", shape=(self.units,), initializer="zeros")
        if self.use_base:
            self.base = tf.keras.layers.Dense(self.units, use_bias=False)
        super().build(input_shape)

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        if self.clip_value > 0:
            x = tf.clip_by_value(x, -self.clip_value, self.clip_value)
        basis = tf.exp(-tf.square(tf.expand_dims(x, -1) - self.centers) /
                       (2.0 * self.sigma * self.sigma + 1.0e-12))
        y = tf.einsum("...cg,cgu->...u", basis, self.spline_weight) + self.bias
        if self.base is not None:
            y = y + self.base(x)
        return y

    def get_config(self):
        return {**super().get_config(), "units": self.units, "grid_size": self.grid_size,
                "grid_min": self.grid_min, "grid_max": self.grid_max,
                "use_base": self.use_base, "clip_value": self.clip_value}


class BSplineKAN(tf.keras.layers.Layer):
    """KAN layer with trainable channel-wise clamped B-spline expansions."""

    def __init__(self, units, grid_size=8, spline_order=3, grid_min=-2.0,
                 grid_max=2.0, use_base=True, clip_value=5.0, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.grid_min = float(grid_min)
        self.grid_max = float(grid_max)
        self.use_base = bool(use_base)
        self.clip_value = float(clip_value)
        if self.grid_size <= self.spline_order:
            raise ValueError("grid_size must be larger than spline_order")
        n_internal = self.grid_size - self.spline_order - 1
        internal = (np.linspace(grid_min, grid_max, n_internal + 2)[1:-1]
                    if n_internal > 0 else np.asarray([], dtype=np.float32))
        knots = np.concatenate([
            np.full(self.spline_order + 1, grid_min), internal,
            np.full(self.spline_order + 1, grid_max),
        ]).astype(np.float32)
        self.knots = tf.constant(knots, dtype=tf.float32)
        self.base = None

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.spline_weight = self.add_weight(name="spline_weight", shape=(channels, self.grid_size, self.units),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02), trainable=True,
        )
        self.bias = self.add_weight(name="bias", shape=(self.units,), initializer="zeros")
        if self.use_base:
            self.base = tf.keras.layers.Dense(self.units, use_bias=False)
        super().build(input_shape)

    def _basis(self, x):
        x = tf.expand_dims(x, -1)
        left = self.knots[:-1]
        right = self.knots[1:]
        basis = tf.cast((x >= left) & (x < right), x.dtype)
        is_right_boundary = tf.cast(tf.equal(x, self.grid_max), x.dtype)
        last = tf.one_hot(tf.shape(basis)[-1] - 1, tf.shape(basis)[-1], dtype=x.dtype)
        basis = tf.maximum(basis, is_right_boundary * last)
        for degree in range(1, self.spline_order + 1):
            new_length = len(self.knots) - degree - 1
            left_den = self.knots[degree:degree + new_length] - self.knots[:new_length]
            right_den = (self.knots[degree + 1:degree + 1 + new_length]
                         - self.knots[1:1 + new_length])
            left_num = x - self.knots[:new_length]
            right_num = self.knots[degree + 1:degree + 1 + new_length] - x
            left_term = tf.where(left_den > 0, left_num / tf.where(left_den > 0, left_den, 1.0), 0.0)
            right_term = tf.where(right_den > 0, right_num / tf.where(right_den > 0, right_den, 1.0), 0.0)
            basis = left_term * basis[..., :new_length] + right_term * basis[..., 1:new_length + 1]
        return basis

    def call(self, inputs):
        x = tf.clip_by_value(tf.cast(inputs, tf.float32), self.grid_min, self.grid_max)
        basis = self._basis(x)
        y = tf.einsum("...cg,cgu->...u", basis, self.spline_weight) + self.bias
        if self.base is not None:
            y = y + self.base(tf.nn.silu(x))
        return y


class ReducedTransform(tf.keras.layers.Layer):
    """Down-project, transform, and up-project a correction feature."""

    def __init__(self, dimension, channels, width, transform="kan", grid_size=8,
                 kan_basis="bspline", spline_order=3, shared_core=None,
                 activation="gelu", **kwargs):
        super().__init__(**kwargs)
        self.dimension = int(dimension)
        self.channels = int(channels)
        self.width = int(width)
        self.transform = transform
        self.grid_size = int(grid_size)
        self.kan_basis = kan_basis
        self.spline_order = int(spline_order)
        self.activation_name = activation
        self.down = conv_nd(dimension, self.width, 1, padding="same", name="project_down")
        self.up = conv_nd(dimension, self.channels, 1, padding="same", name="project_up")
        if transform == "kan":
            if shared_core is not None:
                self.core = shared_core
            elif self.kan_basis == "bspline":
                self.core = BSplineKAN(
                    self.width, grid_size=self.grid_size, spline_order=self.spline_order,
                    name="bspline_kan",
                )
            elif self.kan_basis == "rbf":
                self.core = RBFKAN(self.width, grid_size=self.grid_size, name="rbf_kan")
            else:
                raise ValueError(f"Unknown KAN basis: {self.kan_basis}")
        elif transform == "conv":
            kernel = 3
            target = self.width * self.grid_size * self.width + self.width * self.width
            matched_width = max(2, int(round(math.sqrt(target / (kernel ** dimension)))))
            self.core = tf.keras.Sequential([
                conv_nd(dimension, matched_width, kernel, padding="same", activation=activation),
                conv_nd(dimension, self.width, 1, padding="same", activation=None),
            ], name="parameter_matched_conv")
        elif transform == "mlp":
            target = self.width * self.grid_size * self.width + self.width * self.width
            matched_width = max(2, int(round(target / max(2 * self.width, 1))))
            self.core = tf.keras.Sequential([
                tf.keras.layers.Dense(matched_width, activation=activation),
                tf.keras.layers.Dense(self.width),
            ], name="parameter_matched_mlp")
        else:
            raise ValueError(f"Unknown correction transform: {transform}")
        self.activation = tf.keras.layers.Activation(activation)

    def call(self, x, training=False):
        z = self.down(x)
        z = self.core(z, training=training)
        z = self.activation(z)
        return self.up(z)


class FeatureCorrection(tf.keras.layers.Layer):
    """Independent controlled feature correction with optional adaptive gates."""

    def __init__(self, dimension, channels, width, gamma, transform="kan",
                 gate="fixed", grid_size=8, kan_basis="bspline", spline_order=3,
                 se_reduction=8, shared_core=None,
                 correction_regularization=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dimension = int(dimension)
        self.channels = int(channels)
        self.width = int(width)
        self.gamma = float(gamma)
        self.transform_name = transform
        self.gate_name = gate
        self.grid_size = int(grid_size)
        self.kan_basis = kan_basis
        self.spline_order = int(spline_order)
        self.se_reduction = int(se_reduction)
        self.correction_regularization = float(correction_regularization)
        self.warmup_scale = self.add_weight(name="warmup_scale", shape=(), initializer="ones", trainable=False
        )
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.transform = ReducedTransform(
            dimension, channels, width, transform=transform, grid_size=grid_size,
            kan_basis=kan_basis, spline_order=spline_order,
            shared_core=shared_core,
            name=f"{transform}_transform",
        )
        if gate in {"spatial", "joint"}:
            self.spatial_gate = conv_nd(
                dimension, 1, 1, padding="same", kernel_initializer="zeros",
                bias_initializer="zeros", name="spatial_gate",
            )
        else:
            self.spatial_gate = None
        if gate in {"se", "joint"}:
            hidden = max(channels // se_reduction, 1)
            reshape = (1, channels) if dimension == 1 else (1, 1, channels)
            self.channel_gate = tf.keras.Sequential([
                global_pool_nd(dimension),
                tf.keras.layers.Dense(hidden, activation="relu"),
                tf.keras.layers.Dense(
                    channels, kernel_initializer="zeros", bias_initializer="zeros"
                ),
                tf.keras.layers.Reshape(reshape),
            ], name="se_gate")
        else:
            self.channel_gate = None
        if gate == "learnable_scalar":
            self.scalar_logit = self.add_weight(name="scalar_logit", shape=(), initializer="zeros", trainable=True
            )
        else:
            self.scalar_logit = None
        if gate not in {"none", "fixed", "learnable_scalar", "spatial", "se", "joint"}:
            raise ValueError(f"Unknown gate: {gate}")

    def call(self, x, training=False):
        delta = self.transform(self.norm(x), training=training)
        if self.gate_name == "none":
            scale = 1.0
        elif self.gate_name == "fixed":
            scale = self.gamma
        elif self.gate_name == "learnable_scalar":
            scale = self.gamma * 2.0 * tf.nn.sigmoid(self.scalar_logit)
        else:
            scale = self.gamma
            if self.spatial_gate is not None:
                spatial_input = tf.concat([x, delta], axis=-1)
                scale = scale * 2.0 * tf.nn.sigmoid(self.spatial_gate(spatial_input))
            if self.channel_gate is not None:
                scale = scale * 2.0 * tf.nn.sigmoid(self.channel_gate(x, training=training))
        if self.correction_regularization > 0:
            axes = tf.range(1, tf.rank(x))
            ratio = tf.reduce_mean(
                tf.reduce_sum(tf.square(delta), axis=axes)
                / (tf.reduce_sum(tf.square(x), axis=axes) + 1.0e-8)
            )
            self.add_loss(self.correction_regularization * ratio)
        return x + self.warmup_scale * scale * delta
