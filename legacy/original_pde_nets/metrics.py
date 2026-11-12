from __future__ import annotations

import numpy as np


def error_norms(u_true, u_pred, h: float) -> dict[str, float]:
    diff = np.asarray(u_true, dtype=np.float64) - np.asarray(u_pred, dtype=np.float64)
    l2_h = float(h * np.sqrt(np.sum(diff ** 2)))
    linf = float(np.max(np.abs(diff)))
    return {"l2_h": l2_h, "linf": linf}


def relative_l2(u_true, u_pred) -> float:
    u_true = np.asarray(u_true, dtype=np.float64)
    u_pred = np.asarray(u_pred, dtype=np.float64)
    return float(np.linalg.norm(u_true.ravel() - u_pred.ravel()) / (np.linalg.norm(u_true.ravel()) + 1.0e-12))
