from __future__ import annotations

import tensorflow as tf


class SpectralConv1D(tf.keras.layers.Layer):
    def __init__(self, out_channels, modes=16, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = int(out_channels)
        self.modes = int(modes)

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        shape = (in_channels, self.out_channels, self.modes)
        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        self.wr = self.add_weight(name="weight_real", shape=shape, initializer=init)
        self.wi = self.add_weight(name="weight_imag", shape=shape, initializer=init)

    def call(self, x):
        n = tf.shape(x)[1]
        spectrum = tf.signal.rfft(tf.transpose(x, [0, 2, 1]))
        available = tf.shape(spectrum)[-1]
        modes = tf.minimum(self.modes, available)
        weights = tf.complex(self.wr[..., :modes], self.wi[..., :modes])
        low = tf.einsum("bcm,com->bom", spectrum[..., :modes], weights)
        low = tf.pad(low, [[0, 0], [0, 0], [0, available - modes]])
        y = tf.signal.irfft(low, fft_length=[n])
        return tf.transpose(y, [0, 2, 1])


class SpectralConv2D(tf.keras.layers.Layer):
    def __init__(self, out_channels, modes=12, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = int(out_channels)
        self.modes = int(modes)

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        shape = (in_channels, self.out_channels, self.modes, self.modes)
        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        self.top_r = self.add_weight(name="top_real", shape=shape, initializer=init)
        self.top_i = self.add_weight(name="top_imag", shape=shape, initializer=init)
        self.bottom_r = self.add_weight(name="bottom_real", shape=shape, initializer=init)
        self.bottom_i = self.add_weight(name="bottom_imag", shape=shape, initializer=init)

    def call(self, x):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        spectrum = tf.signal.rfft2d(tf.transpose(x, [0, 3, 1, 2]))
        wf = tf.shape(spectrum)[-1]
        mh = tf.minimum(self.modes, h // 2)
        mw = tf.minimum(self.modes, wf)
        wt = tf.complex(self.top_r[..., :mh, :mw], self.top_i[..., :mh, :mw])
        wb = tf.complex(self.bottom_r[..., :mh, :mw], self.bottom_i[..., :mh, :mw])
        top = tf.einsum("bchw,cohw->bohw", spectrum[:, :, :mh, :mw], wt)
        bottom = tf.einsum("bchw,cohw->bohw", spectrum[:, :, -mh:, :mw], wb)
        top = tf.pad(top, [[0, 0], [0, 0], [0, 0], [0, wf - mw]])
        bottom = tf.pad(bottom, [[0, 0], [0, 0], [0, 0], [0, wf - mw]])
        middle = tf.zeros(
            [tf.shape(x)[0], self.out_channels, h - 2 * mh, wf], dtype=tf.complex64
        )
        out_spectrum = tf.concat([top, middle, bottom], axis=2)
        y = tf.signal.irfft2d(out_spectrum, fft_length=[h, w])
        return tf.transpose(y, [0, 2, 3, 1])


def build_fno(input_shape, dimension, width=48, modes=16, blocks=4):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(width)(inputs)
    spectral_cls = SpectralConv1D if dimension == 1 else SpectralConv2D
    conv_cls = tf.keras.layers.Conv1D if dimension == 1 else tf.keras.layers.Conv2D
    for index in range(blocks):
        spectral = spectral_cls(width, modes=modes, name=f"spectral_{index + 1}")(x)
        local = conv_cls(width, 1, padding="same", name=f"local_{index + 1}")(x)
        x = tf.keras.layers.Activation("gelu")(spectral + local)
    x = tf.keras.layers.Dense(128, activation="gelu")(x)
    outputs = tf.keras.layers.Dense(1, dtype="float32")(x)
    return tf.keras.Model(inputs, outputs, name=f"FNO{dimension}D")
