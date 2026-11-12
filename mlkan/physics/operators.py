from __future__ import annotations

import tensorflow as tf


def dx_periodic_1d(u, h):
    return (tf.roll(u, -1, axis=1) - tf.roll(u, 1, axis=1)) / (2.0 * h)


def dxx_periodic_1d(u, h):
    return (tf.roll(u, -1, axis=1) - 2.0 * u + tf.roll(u, 1, axis=1)) / (h * h)


def gradient_2d(u, h, periodic=False):
    if periodic:
        dx = (tf.roll(u, -1, axis=1) - tf.roll(u, 1, axis=1)) / (2.0 * h)
        dy = (tf.roll(u, -1, axis=2) - tf.roll(u, 1, axis=2)) / (2.0 * h)
        return dx, dy
    dx = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2.0 * h)
    dy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2.0 * h)
    return dx, dy


def laplacian_2d(u, h, periodic=False):
    if periodic:
        return (
            tf.roll(u, -1, axis=1) + tf.roll(u, 1, axis=1)
            + tf.roll(u, -1, axis=2) + tf.roll(u, 1, axis=2) - 4.0 * u
        ) / (h * h)
    return (
        u[:, 2:, 1:-1] + u[:, :-2, 1:-1]
        + u[:, 1:-1, 2:] + u[:, 1:-1, :-2]
        - 4.0 * u[:, 1:-1, 1:-1]
    ) / (h * h)
