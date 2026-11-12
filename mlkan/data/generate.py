from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .loading import save_dataset, validate_dataset
from .solvers import (
    laplacian_dirichlet,
    solve_allen_cahn_periodic,
    solve_burgers_periodic,
    solve_heat_2d_periodic,
)
from ..utils import split_group_ids


def _smooth_periodic_ic_1d(rng, x, max_mode=6):
    u = np.zeros_like(x)
    for mode in range(1, max_mode + 1):
        scale = 1.0 / (mode ** 1.5)
        u += scale * rng.normal() * np.sin(2.0 * np.pi * mode * x)
        u += scale * rng.normal() * np.cos(2.0 * np.pi * mode * x)
    u /= max(np.max(np.abs(u)), 1.0e-8)
    return rng.uniform(0.7, 1.2) * u


def _pack_time_pairs_1d(task, trajectories, parameters, x_grid, save_dt):
    rows = []
    for group_id, states in trajectories.items():
        parameter = float(parameters[group_id])
        for step in range(len(states) - 1):
            previous = states[step]
            target = states[step + 1]
            features = np.stack([
                previous,
                np.full_like(previous, parameter),
                x_grid,
                np.full_like(previous, save_dt),
            ], axis=-1)
            rows.append((features, target[..., None], group_id, step, parameter))
    return rows


def _pack_time_pairs_2d(trajectories, parameters, X, Y, save_dt):
    rows = []
    for group_id, states in trajectories.items():
        parameter = float(parameters[group_id])
        for step in range(len(states) - 1):
            previous = states[step]
            target = states[step + 1]
            features = np.stack([
                previous,
                np.full_like(previous, parameter),
                X,
                Y,
                np.full_like(previous, save_dt),
            ], axis=-1)
            rows.append((features, target[..., None], group_id, step, parameter))
    return rows


def _rows_to_splits(rows, group_splits):
    output = {}
    for split, group_ids in group_splits.items():
        allowed = set(int(x) for x in group_ids)
        selected = [row for row in rows if int(row[2]) in allowed]
        if not selected:
            continue
        output[split] = {
            "x": np.asarray([r[0] for r in selected], dtype=np.float32),
            "y": np.asarray([r[1] for r in selected], dtype=np.float32),
            "group_id": np.asarray([r[2] for r in selected], dtype=np.int32),
            "step": np.asarray([r[3] for r in selected], dtype=np.int32),
            "parameter": np.asarray([r[4] for r in selected], dtype=np.float32),
        }
    return output


def generate_burgers(n=1024, n_groups=100, seed=42, final_time=0.1, save_dt=0.01,
                     viscosities=(1.0e-2, 1.0e-3, 1.0e-4)):
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=np.float64) / n
    trajectories, parameters = {}, {}
    for group_id in range(n_groups):
        nu = float(viscosities[group_id % len(viscosities)])
        u0 = _smooth_periodic_ic_1d(rng, x)
        _, states = solve_burgers_periodic(u0, nu, final_time, save_dt)
        trajectories[group_id] = states
        parameters[group_id] = nu
    rows = _pack_time_pairs_1d("burgers", trajectories, parameters, x, save_dt)
    splits = _rows_to_splits(rows, split_group_ids(n_groups, seed))
    metadata = {
        "task": "burgers", "equation": "u_t + u u_x = nu u_xx",
        "dimension": 1, "boundary": "periodic", "n": n, "length": 1.0,
        "h": 1.0 / n, "dt": save_dt, "final_time": final_time,
        "n_groups": n_groups, "viscosities": list(viscosities), "seed": seed,
        "input_channels": {"u_previous": 0, "viscosity": 1, "x": 2, "dt": 3},
        "target": "u_next", "solver": "dealiased Fourier pseudospectral RK4",
        "split_policy": "trajectory-level 70/15/15",
    }
    return splits, metadata


def generate_allen_cahn(n=1024, n_groups=100, seed=42, final_time=0.1, save_dt=0.01,
                        epsilons=(0.01, 0.02, 0.04)):
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=np.float64) / n
    trajectories, parameters = {}, {}
    for group_id in range(n_groups):
        epsilon = float(epsilons[group_id % len(epsilons)])
        u0 = 0.65 * _smooth_periodic_ic_1d(rng, x, max_mode=4)
        _, states = solve_allen_cahn_periodic(u0, epsilon, final_time, save_dt)
        trajectories[group_id] = states
        parameters[group_id] = epsilon
    rows = _pack_time_pairs_1d("allen_cahn", trajectories, parameters, x, save_dt)
    splits = _rows_to_splits(rows, split_group_ids(n_groups, seed))
    metadata = {
        "task": "allen_cahn", "equation": "u_t = epsilon^2 u_xx + u - u^3",
        "dimension": 1, "boundary": "periodic", "n": n, "length": 1.0,
        "h": 1.0 / n, "dt": save_dt, "final_time": final_time,
        "n_groups": n_groups, "epsilons": list(epsilons), "seed": seed,
        "input_channels": {"u_previous": 0, "epsilon": 1, "x": 2, "dt": 3},
        "target": "u_next", "solver": "semi-implicit Fourier IMEX",
        "split_policy": "trajectory-level 70/15/15",
    }
    return splits, metadata


def _boundary_arrays(u):
    mask = np.zeros_like(u)
    mask[[0, -1], :] = 1.0
    mask[:, [0, -1]] = 1.0
    values = mask * u
    return values, mask


def _elliptic_solution(task, rng, X, Y, trig_mode=4):
    if task == "trig_n4":
        amplitude = rng.uniform(0.7, 1.3)
        c1, c2 = rng.normal(0.0, 0.12, size=2)
        u = amplitude * np.sin(trig_mode * np.pi * X) * np.sin(trig_mode * np.pi * Y)
        u += c1 * np.sin(2 * np.pi * X) * np.sin(3 * np.pi * Y)
        u += c2 * np.sin(np.pi * X) * np.sin(2 * np.pi * Y)
        parameter = [amplitude, c1, c2]
    elif task == "peak":
        cx, cy = rng.uniform(0.35, 0.65, size=2)
        width = rng.uniform(35.0, 90.0)
        amplitude = rng.uniform(0.8, 1.2)
        envelope = X * (1.0 - X) * Y * (1.0 - Y)
        u = amplitude * envelope * np.exp(-width * ((X - cx) ** 2 + (Y - cy) ** 2))
        parameter = [cx, cy, width, amplitude]
    elif task == "exp_trig":
        ax, ay = rng.uniform(-1.0, 1.0, size=2)
        amplitude = rng.uniform(0.7, 1.3)
        envelope = X * (1.0 - X) * Y * (1.0 - Y)
        u = amplitude * envelope * np.exp(ax * X + ay * Y) * (
            np.sin(2 * np.pi * X) * np.sin(3 * np.pi * Y)
        )
        parameter = [ax, ay, amplitude]
    else:
        raise ValueError(task)
    return u, parameter


def generate_elliptic(task, n=64, n_groups=1000, seed=42, trig_mode=4):
    if task not in {"trig_n4", "peak", "exp_trig"}:
        raise ValueError(task)
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    h = 1.0 / (n - 1)
    rows = []
    for group_id in range(n_groups):
        u, parameter = _elliptic_solution(task, rng, X, Y, trig_mode=trig_mode)
        source = -laplacian_dirichlet(u, h)
        boundary_values, boundary_mask = _boundary_arrays(u)
        features = np.stack([source, boundary_values, boundary_mask, X, Y], axis=-1)
        rows.append((features, u[..., None], group_id, 0, parameter[0]))
    splits = _rows_to_splits(rows, split_group_ids(n_groups, seed))
    metadata = {
        "task": task, "equation": "-Delta u = f", "dimension": 2,
        "boundary": "Dirichlet", "n": n, "h": h, "n_groups": n_groups,
        "seed": seed,
        "trig_mode": trig_mode if task == "trig_n4" else None,
        "input_channels": {"source": 0, "boundary_value": 1, "boundary_mask": 2, "x": 3, "y": 4},
        "target": "u", "solver": "discrete manufactured solution using five-point Laplacian",
        "split_policy": "parameter-instance-level 70/15/15",
    }
    return splits, metadata


def generate_gaussian_heat(n=64, n_groups=100, seed=42, final_time=0.1, save_dt=0.01,
                           diffusivities=(0.005, 0.01, 0.02)):
    rng = np.random.default_rng(seed)
    axis = np.arange(n, dtype=np.float64) / n
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    trajectories, parameters = {}, {}
    for group_id in range(n_groups):
        alpha = float(diffusivities[group_id % len(diffusivities)])
        u0 = np.zeros((n, n), dtype=np.float64)
        for _ in range(rng.integers(1, 4)):
            cx, cy = rng.uniform(0.2, 0.8, size=2)
            width = rng.uniform(40.0, 100.0)
            amp = rng.uniform(0.5, 1.2)
            # Periodic distance keeps the spectral reference consistent.
            dx = np.minimum(np.abs(X - cx), 1.0 - np.abs(X - cx))
            dy = np.minimum(np.abs(Y - cy), 1.0 - np.abs(Y - cy))
            u0 += amp * np.exp(-width * (dx * dx + dy * dy))
        _, states = solve_heat_2d_periodic(u0, alpha, final_time, save_dt)
        trajectories[group_id] = states
        parameters[group_id] = alpha
    rows = _pack_time_pairs_2d(trajectories, parameters, X, Y, save_dt)
    splits = _rows_to_splits(rows, split_group_ids(n_groups, seed))
    metadata = {
        "task": "gaussian_heat", "equation": "u_t = alpha Delta u",
        "dimension": 2, "boundary": "periodic", "n": n, "h": 1.0 / n,
        "dt": save_dt, "final_time": final_time, "n_groups": n_groups,
        "diffusivities": list(diffusivities), "seed": seed,
        "input_channels": {"u_previous": 0, "diffusivity": 1, "x": 2, "y": 3, "dt": 4},
        "target": "u_next", "solver": "exact Fourier semigroup",
        "split_policy": "trajectory-level 70/15/15",
    }
    return splits, metadata


def generate_dataset(task, **kwargs):
    if task == "burgers":
        return generate_burgers(**kwargs)
    if task == "allen_cahn":
        return generate_allen_cahn(**kwargs)
    if task in {"trig_n4", "peak", "exp_trig"}:
        return generate_elliptic(task, **kwargs)
    if task == "gaussian_heat":
        return generate_gaussian_heat(**kwargs)
    raise ValueError(f"Unknown task: {task}")


def generate_and_save(task, output, **kwargs):
    splits, metadata = generate_dataset(task, **kwargs)
    errors = validate_dataset(splits, metadata)
    if errors:
        raise RuntimeError("Dataset validation failed: " + "; ".join(errors))
    save_dataset(output, splits, metadata)
    digest = hashlib.sha256(Path(output).read_bytes()).hexdigest()
    manifest = {
        **metadata,
        "file": str(Path(output).resolve()), "sha256": digest,
        "split_shapes": {
            split: {key: list(value.shape) for key, value in values.items()}
            for split, values in splits.items()
        },
        "value_ranges": {
            split: {
                "x_min": float(values["x"].min()), "x_max": float(values["x"].max()),
                "y_min": float(values["y"].min()), "y_max": float(values["y"].max()),
            } for split, values in splits.items()
        },
    }
    Path(output).with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
