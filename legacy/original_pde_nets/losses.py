from __future__ import annotations

import numpy as np
import tensorflow as tf


def boundary_loss(u_pred, boundaries, n: int):
    left, right, bottom, top = boundaries
    left = tf.cast(left, tf.float32)
    right = tf.cast(right, tf.float32)
    bottom = tf.cast(bottom, tf.float32)
    top = tf.cast(top, tf.float32)
    parts = [
        tf.square(left - tf.reshape(u_pred[:, :, 0, :], [n])),
        tf.square(right - tf.reshape(u_pred[:, :, -1, :], [n])),
        tf.square(bottom - tf.reshape(u_pred[:, 0, :, :], [n])),
        tf.square(top - tf.reshape(u_pred[:, -1, :, :], [n])),
    ]
    return tf.reduce_mean(tf.concat(parts, axis=-1))


class EllipticFDLoss(tf.keras.losses.Loss):
    def __init__(self, n: int, h: float, f_source, boundaries, reduction="sum_over_batch_size", name="elliptic_fd_loss"):
        super().__init__(reduction=reduction, name=name)
        self.n = n
        self.h = h
        self.alpha = (h ** 2) / 4.0
        self.f = tf.cast(tf.reshape(f_source, [1, n, n, 1]), tf.float32)
        self.boundaries = tuple(tf.cast(b, tf.float32) for b in boundaries)
        kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32) / (h ** 2)
        self.lap_kernel = tf.reshape(tf.constant(kernel), [3, 3, 1, 1])

    def call(self, y_true, y_pred):
        lap = tf.nn.convolution(y_pred, self.lap_kernel, strides=1, padding="VALID")
        rhs = -lap
        interior = tf.reduce_mean(tf.square(rhs - self.f[:, 1:-1, 1:-1, :]))
        bnd = boundary_loss(y_pred, self.boundaries, self.n)
        return self.alpha * interior + (1.0 - self.alpha) * bnd


class VariableDiffusionLoss(tf.keras.losses.Loss):
    def __init__(self, n: int, h: float, f_source, boundaries, kappa, reduction="sum_over_batch_size", name="variable_diffusion_loss"):
        super().__init__(reduction=reduction, name=name)
        self.n = n
        self.h = h
        self.alpha = (h ** 2) / 4.0
        self.f = tf.cast(tf.reshape(f_source, [1, n, n, 1]), tf.float32)
        self.boundaries = tuple(tf.cast(b, tf.float32) for b in boundaries)
        k = np.asarray(kappa, dtype=np.float32)
        kc = k[1:-1, 1:-1]
        self.ke = tf.constant(2.0 / (1.0 / kc + 1.0 / k[1:-1, 2:]), dtype=tf.float32)[None, :, :, None]
        self.kw = tf.constant(2.0 / (1.0 / kc + 1.0 / k[1:-1, :-2]), dtype=tf.float32)[None, :, :, None]
        self.kn = tf.constant(2.0 / (1.0 / kc + 1.0 / k[2:, 1:-1]), dtype=tf.float32)[None, :, :, None]
        self.ks = tf.constant(2.0 / (1.0 / kc + 1.0 / k[:-2, 1:-1]), dtype=tf.float32)[None, :, :, None]

    def call(self, y_true, y_pred):
        u = y_pred
        uc = u[:, 1:-1, 1:-1, :]
        ue = u[:, 1:-1, 2:, :]
        uw = u[:, 1:-1, :-2, :]
        un = u[:, 2:, 1:-1, :]
        us = u[:, :-2, 1:-1, :]
        div = (self.ke * (ue - uc) - self.kw * (uc - uw) + self.kn * (un - uc) - self.ks * (uc - us)) / (self.h ** 2)
        residual = -div - self.f[:, 1:-1, 1:-1, :]
        interior = tf.reduce_mean(tf.square(residual))
        bnd = boundary_loss(y_pred, self.boundaries, self.n)
        return self.alpha * interior + (1.0 - self.alpha) * bnd


class ParabolicFDLoss(tf.keras.losses.Loss):
    def __init__(self, n: int, h: float, tau: float, alpha_mode: str = "notebook", reduction="sum_over_batch_size", name="parabolic_fd_loss"):
        super().__init__(reduction=reduction, name=name)
        self.n = n
        self.h = h
        self.tau = tau
        if alpha_mode == "paper":
            self.alpha = (h ** 2) / 4.0
        elif alpha_mode == "notebook":
            self.alpha = (h ** 2) * 4.0
        else:
            raise ValueError("alpha_mode must be 'paper' or 'notebook'")
        kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32) / (h ** 2)
        self.lap_kernel = tf.reshape(tf.constant(kernel), [3, 3, 1, 1])

    def call(self, inputs, y_true=None):
        u_current, u_previous, f_current, boundaries = inputs
        lap = tf.nn.convolution(u_current, self.lap_kernel, strides=1, padding="VALID")
        residual = (
            u_current[:, 1:-1, 1:-1, :]
            - u_previous[:, 1:-1, 1:-1, :]
            - self.tau * (lap + f_current[:, 1:-1, 1:-1, :])
        )
        interior = tf.reduce_mean(tf.square(residual))
        bnd = boundary_loss(u_current, boundaries, self.n)
        return self.alpha * interior + (1.0 - self.alpha) * bnd
