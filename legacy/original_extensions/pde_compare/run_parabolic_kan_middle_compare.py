from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pde_nets.config import DEFAULT_OPTIMIZER, DEFAULT_PARABOLIC, setup_seed
from pde_nets.losses import ParabolicFDLoss
from pde_nets.metrics import error_norms
from pde_nets.models import build_unet
from pde_nets.kan_middle_unet import (
    build_unet_kan_middle,
    build_unet_kan_multilevel,
)
from pde_nets.plots import save_loss_curve, save_parabolic_timeline
from pde_nets.problems import get_parabolic_problem


def tensor2d(x: np.ndarray) -> tf.Tensor:
    return tf.convert_to_tensor(
        x.reshape(1, x.shape[0], x.shape[1], 1),
        dtype=tf.float32,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def apply_gradients_safe(optimizer, grads, variables):
    pairs = [(g, v) for g, v in zip(grads, variables) if g is not None]
    if pairs:
        optimizer.apply_gradients(pairs)


def parse_float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(x.strip()) for x in value.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated floats, got: {value}"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one gamma value is required")
    return values


def build_model(args, model_name: str, n: int):
    if model_name == "linear_unet":
        model = build_unet(
            n=n,
            filters=args.filters,
            depth=args.depth,
            activation=args.activation,
            kernel_size=args.kernel_size,
            l2_weight=args.l2,
            pocket=True,
        )

        dummy = tf.zeros((1, n, n, 1), dtype=tf.float32)
        _ = model(dummy, training=False)
        return model

    if model_name == "unet_kan_middle":
        return build_unet_kan_middle(
            n=n,
            filters=args.filters,
            depth=args.depth,
            activation=args.activation,
            kernel_size=args.kernel_size,
            l2_weight=args.l2,
            pocket=True,
            kan_grid_size=args.kan_grid_size,
            kan_gamma=args.kan_gamma,
            kan_activation=args.kan_activation,
            kan_layernorm=not args.no_kan_layernorm,
        )

    if model_name == "unet_kan_multilevel":
        return build_unet_kan_multilevel(
            n=n,
            filters=args.filters,
            depth=args.depth,
            activation=args.activation,
            kernel_size=args.kernel_size,
            l2_weight=args.l2,
            pocket=True,
            kan_grid_size=args.kan_grid_size,
            encoder_gammas=args.encoder_kan_gammas,
            kan_activation=args.kan_activation,
            kan_layernorm=not args.no_kan_layernorm,
            gate_mode=args.kan_gate_mode,
            se_reduction=args.se_reduction,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def train_one(args, problem, model_name: str, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    n_steps = int(round(args.time / args.tau))

    if abs(n_steps * args.tau - args.time) > 1.0e-8:
        raise ValueError("--time must be an integer multiple of --tau")

    model = build_model(args, model_name, problem.n)

    loss_fn = ParabolicFDLoss(
        problem.n,
        problem.h,
        args.tau,
        alpha_mode=args.alpha_mode,
    )

    optimizer = tf.keras.optimizers.Adam(
        args.lr,
        global_clipnorm=args.clipnorm,
    )

    @tf.function
    def train_step(u_previous, f_current, boundaries):
        with tf.GradientTape() as tape:
            pred = model(u_previous, training=True)

            main_loss = loss_fn.call(
                (pred, u_previous, f_current, boundaries),
                None,
            )

            reg_loss = (
                tf.add_n(model.losses)
                if model.losses
                else tf.constant(0.0, dtype=tf.float32)
            )

            total_loss = main_loss + reg_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        apply_gradients_safe(optimizer, grads, model.trainable_variables)

        return total_loss, pred

    u_previous = tensor2d(problem.u0())
    solutions = [problem.u0().astype(np.float32)]
    times = [0.0]
    history = []
    best_loss_global = np.inf
    start = time.time()

    for time_index in range(1, n_steps + 1):
        t_value = time_index * args.tau

        f_current = tensor2d(problem.source(t_value))

        b = problem.boundary(t_value)
        boundaries = tuple(
            tf.convert_to_tensor(x, dtype=tf.float32)
            for x in b
        )

        steps_this = (
            args.first_step_iters
            if time_index == 1
            else args.later_step_iters
        )

        best_loss = np.inf
        best_u = None

        for step in range(steps_this):
            loss, pred = train_step(u_previous, f_current, boundaries)
            value = float(loss.numpy())
            history.append(value)

            if np.isfinite(value) and value < best_loss:
                best_loss = value
                best_u = pred

            if np.isfinite(value) and value < best_loss_global:
                best_loss_global = value

        if best_u is None:
            best_u = pred

        u_previous = best_u

        solutions.append(
            u_previous.numpy().reshape(problem.n, problem.n).astype(np.float32)
        )
        times.append(t_value)

        print(
            f"[{problem.name}][{model_name}] "
            f"time {t_value:.2f}/{args.time:.2f} best_loss={best_loss:.6e}"
        )

    elapsed = time.time() - start

    prediction = solutions[-1]
    truth = problem.exact(args.time)
    metrics = error_norms(truth, prediction, problem.h)

    result = {
        "problem": problem.name,
        "type": "parabolic",
        "model_name": model_name,
        "n": problem.n,
        "h": problem.h,
        "final_time": args.time,
        "tau": args.tau,
        "first_step_iters": args.first_step_iters,
        "later_step_iters": args.later_step_iters,
        "depth": args.depth,
        "filters": args.filters,
        "kernel_size": args.kernel_size,
        "activation": args.activation,
        "alpha_mode": args.alpha_mode,
        "seed": args.seed,
        "num_params": int(model.count_params()),
        "best_loss": float(best_loss_global),
        "elapsed_sec": float(elapsed),
        "kan_grid_size": args.kan_grid_size if "kan" in model_name else None,
        "kan_gamma": args.kan_gamma if model_name == "unet_kan_middle" else None,
        "encoder_kan_gammas": list(args.encoder_kan_gammas) if model_name == "unet_kan_multilevel" else None,
        "kan_activation": args.kan_activation if "kan" in model_name else None,
        "kan_layernorm": (not args.no_kan_layernorm) if "kan" in model_name else None,
        "kan_gate_mode": args.kan_gate_mode if model_name == "unet_kan_multilevel" else "fixed" if model_name == "unet_kan_middle" else None,
        **metrics,
    }

    np.save(out_dir / "prediction.npy", prediction.astype(np.float32))
    np.save(out_dir / "exact.npy", truth.astype(np.float32))
    np.save(out_dir / "loss_history.npy", np.asarray(history, dtype=np.float32))
    np.save(out_dir / "times.npy", np.asarray(times, dtype=np.float32))
    np.save(out_dir / "solutions.npy", np.asarray(solutions, dtype=np.float32))
    write_json(out_dir / "metrics.json", result)

    if not args.no_plots:
        save_parabolic_timeline(
            problem.X,
            problem.Y,
            times,
            solutions,
            problem.exact,
            out_dir / "timeline.png",
            f"{problem.name}: {model_name}",
        )

        save_loss_curve(
            history,
            out_dir / "loss_curve.png",
            f"Loss: {problem.name} {model_name}",
        )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def save_comparison(rows: list[dict[str, Any]], out_root: Path) -> None:
    write_json(out_root / "comparison.json", {"results": rows})

    if not rows:
        return

    fields = [
        "problem",
        "type",
        "model_name",
        "n",
        "final_time",
        "tau",
        "first_step_iters",
        "later_step_iters",
        "depth",
        "filters",
        "activation",
        "alpha_mode",
        "seed",
        "num_params",
        "best_loss",
        "elapsed_sec",
        "kan_grid_size",
        "kan_gamma",
        "encoder_kan_gammas",
        "kan_gate_mode",
        "l2_h",
        "linf",
    ]

    with (out_root / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    baseline = next((r for r in rows if r["model_name"] == "linear_unet"), None)
    if baseline is not None:
        changes = []
        baseline_l2 = float(baseline["l2_h"])
        for row in rows:
            if row["model_name"] == "linear_unet":
                continue
            reduction = 100.0 * (baseline_l2 - float(row["l2_h"])) / max(baseline_l2, 1.0e-12)
            changes.append({
                "model_name": row["model_name"],
                "baseline_l2_h": baseline_l2,
                "model_l2_h": float(row["l2_h"]),
                "l2_reduction_percent": reduction,
                "exceeds_10_percent": bool(reduction > 10.0),
            })
        write_json(out_root / "improvement_decision.json", {
            "criterion": "L2 error reduction relative to Linear U-Net exceeds 10%",
            "changes": changes,
        })


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare Linear U-Net and U-Net + KAN-Middle on parabolic PDE cases."
    )

    p.add_argument("--case", default="Gaussian", choices=["Trig-n1", "Trig-n4", "Gaussian"])
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--time", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=DEFAULT_PARABOLIC.tau)
    p.add_argument("--first-step-iters", type=int, default=DEFAULT_PARABOLIC.first_step_iters)
    p.add_argument("--later-step-iters", type=int, default=DEFAULT_PARABOLIC.later_step_iters)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--filters", type=int, default=32)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--activation", default="identity", choices=["identity", "relu", "tanh", "swish"])
    p.add_argument("--lr", type=float, default=DEFAULT_OPTIMIZER.parabolic_lr)
    p.add_argument("--l2", type=float, default=DEFAULT_OPTIMIZER.l2_weight)
    p.add_argument("--clipnorm", type=float, default=DEFAULT_OPTIMIZER.clipnorm)
    p.add_argument("--alpha-mode", default=DEFAULT_PARABOLIC.alpha_mode, choices=["notebook", "paper"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model",
        choices=["linear_unet", "unet_kan_middle", "unet_kan_multilevel", "both", "multilevel_pair", "all"],
        default="both",
    )
    p.add_argument("--kan-grid-size", type=int, default=8)
    p.add_argument("--kan-gamma", type=float, default=0.05)
    p.add_argument(
        "--encoder-kan-gammas",
        type=parse_float_list,
        default=(0.02, 0.05, 0.08),
        help="Shallow-to-deep encoder gamma values, e.g. 0.02,0.05,0.08.",
    )
    p.add_argument(
        "--kan-gate-mode",
        choices=["fixed", "spatial_shared", "spatial_channel", "se"],
        default="fixed",
        help="Keep fixed for the first experiment; adaptive modes are ablations.",
    )
    p.add_argument("--se-reduction", type=int, default=8)
    p.add_argument("--kan-activation", default="identity", choices=["identity", "relu", "tanh", "swish", "gelu"])
    p.add_argument("--no-kan-layernorm", action="store_true")
    p.add_argument("--out", default="results/extensions/pde_kan_compare/parabolic")
    p.add_argument("--no-plots", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    if len(args.encoder_kan_gammas) != args.depth:
        raise ValueError(
            f"--encoder-kan-gammas requires {args.depth} values for depth={args.depth}; "
            f"got {args.encoder_kan_gammas}"
        )
    setup_seed(args.seed)

    problem = get_parabolic_problem(args.case, args.N)

    gamma_tag = "-".join(str(x).replace(".", "p") for x in args.encoder_kan_gammas)
    tag = (
        f"N{args.N}_T{args.time}_d{args.depth}_"
        f"gammas{gamma_tag}_gate{args.kan_gate_mode}_seed{args.seed}"
    )

    out_root = Path(args.out) / args.case / tag
    out_root.mkdir(parents=True, exist_ok=True)

    if args.model == "both":
        model_names = ["linear_unet", "unet_kan_middle"]
    elif args.model == "multilevel_pair":
        model_names = ["linear_unet", "unet_kan_multilevel"]
    elif args.model == "all":
        model_names = ["linear_unet", "unet_kan_middle", "unet_kan_multilevel"]
    else:
        model_names = [args.model]

    rows = []
    for model_name in model_names:
        setup_seed(args.seed)
        rows.append(train_one(args, problem, model_name, out_root / model_name))

    save_comparison(rows, out_root)
    print("Saved comparison to:", out_root / "comparison.csv")


if __name__ == "__main__":
    main()
