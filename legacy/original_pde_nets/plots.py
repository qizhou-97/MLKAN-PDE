from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_loss_curve(history, out_path: str | Path, title: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(1, len(history) + 1), history)
    plt.yscale("log")
    plt.xlabel("Optimization step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_elliptic_comparison(X, Y, u_true, u_pred, out_path: str | Path, title: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err = np.abs(u_true - u_pred)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    vmin = min(float(np.min(u_true)), float(np.min(u_pred)))
    vmax = max(float(np.max(u_true)), float(np.max(u_pred)))
    for ax, data, name in zip(axes, [u_true, u_pred, err], ["True solution", "Prediction", "Absolute difference"]):
        if name == "Absolute difference":
            im = ax.imshow(data, origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()])
        else:
            im = ax.imshow(data, origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()], vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_parabolic_timeline(X, Y, times, predictions, exact_fn, out_path: str | Path, title: str, max_frames: int = 10) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    times = list(times)[:max_frames]
    predictions = list(predictions)[:max_frames]
    truths = [exact_fn(float(t)) for t in times]
    errors = [np.abs(tu - pr) for tu, pr in zip(truths, predictions)]
    rows = [("True solution", truths), ("Prediction", predictions), ("Difference", errors)]
    ncols = len(times)
    fig, axes = plt.subplots(3, ncols, figsize=(1.7 * ncols, 5.7))
    if ncols == 1:
        axes = axes[:, None]
    for r, (row_name, data_list) in enumerate(rows):
        for c, data in enumerate(data_list):
            ax = axes[r, c]
            im = ax.imshow(data, origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"t={times[c]:.1f}", fontsize=8)
            if c == 0:
                ax.set_ylabel(row_name)
            if c == ncols - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
