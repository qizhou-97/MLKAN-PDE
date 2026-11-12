from __future__ import annotations

import time
import numpy as np


def per_sample_metrics(y_true, y_pred, dimension, boundary_width=2, high_gradient_quantile=0.9):
    axes = tuple(range(1, y_true.ndim))
    error = y_pred - y_true
    l2 = np.sqrt(np.sum(error ** 2, axis=axes)) / (
        np.sqrt(np.sum(y_true ** 2, axis=axes)) + 1.0e-12
    )
    rmse = np.sqrt(np.mean(error ** 2, axis=axes))
    mae = np.mean(np.abs(error), axis=axes)
    linf = np.max(np.abs(error), axis=axes)
    if dimension == 1:
        grad_true = np.gradient(y_true[..., 0], axis=1)
        grad_pred = np.gradient(y_pred[..., 0], axis=1)
        grad_axes = (1,)
        spectrum_true = np.abs(np.fft.rfft(y_true[..., 0], axis=1))
        spectrum_pred = np.abs(np.fft.rfft(y_pred[..., 0], axis=1))
        cutoff = max(spectrum_true.shape[1] // 3, 1)
        spectral = np.linalg.norm(spectrum_pred[:, cutoff:] - spectrum_true[:, cutoff:], axis=1) / (
            np.linalg.norm(spectrum_true[:, cutoff:], axis=1) + 1.0e-12
        )
        true_position = np.argmax(y_true[..., 0], axis=1)
        pred_position = np.argmax(y_pred[..., 0], axis=1)
        peak_position_error = np.abs(pred_position - true_position) / y_true.shape[1]
        peak_amplitude_error = np.abs(
            np.max(y_pred[..., 0], axis=1) - np.max(y_true[..., 0], axis=1)
        )
    else:
        gt = np.gradient(y_true[..., 0], axis=(1, 2))
        gp = np.gradient(y_pred[..., 0], axis=(1, 2))
        grad_true = np.sqrt(gt[0] ** 2 + gt[1] ** 2)
        grad_pred = np.sqrt(gp[0] ** 2 + gp[1] ** 2)
        grad_axes = (1, 2)
        ft = np.abs(np.fft.rfft2(y_true[..., 0], axes=(1, 2)))
        fp = np.abs(np.fft.rfft2(y_pred[..., 0], axes=(1, 2)))
        kx = np.fft.fftfreq(y_true.shape[1])[:, None]
        ky = np.fft.rfftfreq(y_true.shape[2])[None, :]
        mask = np.sqrt(kx ** 2 + ky ** 2) >= 0.25
        spectral = np.linalg.norm((fp - ft)[:, mask], axis=1) / (
            np.linalg.norm(ft[:, mask], axis=1) + 1.0e-12
        )
        flat_true = y_true[..., 0].reshape(len(y_true), -1)
        flat_pred = y_pred[..., 0].reshape(len(y_pred), -1)
        true_index = np.argmax(flat_true, axis=1)
        pred_index = np.argmax(flat_pred, axis=1)
        tx, ty = np.unravel_index(true_index, y_true.shape[1:3])
        px, py = np.unravel_index(pred_index, y_true.shape[1:3])
        peak_position_error = np.sqrt((px - tx) ** 2 + (py - ty) ** 2) / max(
            y_true.shape[1], y_true.shape[2]
        )
        peak_amplitude_error = np.abs(np.max(flat_pred, axis=1) - np.max(flat_true, axis=1))
    grad_rel = np.sqrt(np.sum((grad_pred - grad_true) ** 2, axis=grad_axes)) / (
        np.sqrt(np.sum(grad_true ** 2, axis=grad_axes)) + 1.0e-12
    )
    threshold = np.quantile(np.abs(grad_true), high_gradient_quantile, axis=grad_axes, keepdims=True)
    mask = np.abs(grad_true) >= threshold
    point_error = np.abs(error[..., 0])
    high_gradient_mae = np.sum(point_error * mask, axis=grad_axes) / (np.sum(mask, axis=grad_axes) + 1.0e-12)
    if dimension == 2:
        bmask = np.zeros_like(point_error, dtype=bool)
        bmask[:, :boundary_width] = True
        bmask[:, -boundary_width:] = True
        bmask[:, :, :boundary_width] = True
        bmask[:, :, -boundary_width:] = True
        boundary_mae = np.sum(point_error * bmask, axis=(1, 2)) / np.sum(bmask, axis=(1, 2))
    else:
        boundary_mae = np.mean(
            np.concatenate([point_error[:, :boundary_width], point_error[:, -boundary_width:]], axis=1), axis=1
        )
    return {"relative_l2": l2, "rmse": rmse, "mae": mae, "linf": linf,
            "gradient_relative_l2": grad_rel, "high_frequency_error": spectral,
            "high_gradient_mae": high_gradient_mae, "boundary_mae": boundary_mae,
            "peak_position_error": peak_position_error,
            "peak_amplitude_error": peak_amplitude_error}


def summarize_metrics(per_sample):
    output = {}
    for name, values in per_sample.items():
        values = np.asarray(values, dtype=float)
        output[name] = {
            "mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)), "min": float(values.min()), "max": float(values.max()),
        }
    return output


def benchmark_inference(model, sample, warmup=5, repeats=30):
    for _ in range(warmup):
        _ = model(sample, training=False)
    start = time.perf_counter()
    for _ in range(repeats):
        _ = model(sample, training=False)
    return 1000.0 * (time.perf_counter() - start) / repeats / len(sample)
