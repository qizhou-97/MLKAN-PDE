from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def bootstrap_mean_ci(values, confidence=0.95, n_bootstrap=5000, seed=123):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def paired_statistics(baseline, candidate):
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if baseline.shape != candidate.shape:
        raise ValueError(f"Paired arrays differ: {baseline.shape} vs {candidate.shape}")
    delta = baseline - candidate
    mean_delta = float(delta.mean())
    sd = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
    dz = mean_delta / max(sd, 1.0e-12)
    t_value = mean_delta / max(sd / math.sqrt(len(delta)), 1.0e-12)
    try:
        from scipy.stats import t as student_t
        p_value = float(2.0 * student_t.sf(abs(t_value), df=max(len(delta) - 1, 1)))
    except Exception:
        p_value = float(math.erfc(abs(t_value) / math.sqrt(2.0)))
    ci_low, ci_high = bootstrap_mean_ci(delta)
    return {
        "n_pairs": len(delta), "baseline_mean": float(baseline.mean()),
        "candidate_mean": float(candidate.mean()),
        "relative_reduction_percent": 100.0 * mean_delta / max(float(baseline.mean()), 1.0e-12),
        "mean_paired_improvement": mean_delta, "ci95_low": ci_low, "ci95_high": ci_high,
        "paired_t": t_value, "p_value_two_sided": p_value, "cohen_dz": dz,
        "sample_improvement_fraction": float(np.mean(candidate < baseline)),
    }
