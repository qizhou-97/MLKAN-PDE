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

from pde_nets.config import DEFAULT_OPTIMIZER, setup_seed
from pde_nets.losses import EllipticFDLoss, VariableDiffusionLoss
from pde_nets.metrics import error_norms
from pde_nets.models import build_unet
from pde_nets.kan_middle_unet import build_unet_kan_middle
from pde_nets.plots import save_elliptic_comparison, save_loss_curve
from pde_nets.problems import get_elliptic_problem


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

    raise ValueError(f"Unknown model_name: {model_name}")


def train_one(args, problem, model_name: str, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args, model_name, problem.n)
    f_input = tensor2d(problem.f_source)

    if problem.kappa is None:
        loss_fn = EllipticFDLoss(
            problem.n,
            problem.h,
            problem.f_source,
            problem.boundary,
        )
    else:
        loss_fn = VariableDiffusionLoss(
            problem.n,
            problem.h,
            problem.f_source,
            problem.boundary,
            problem.kappa,
        )

    if args.cosine_lr:
        lr = tf.keras.optimizers.schedules.CosineDecay(
            args.lr,
            decay_steps=max(1, args.steps),
        )
    else:
        lr = args.lr

    optimizer = tf.keras.optimizers.Adam(
        lr,
        global_clipnorm=args.clipnorm,
    )

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            pred = model(f_input, training=True)
            main_loss = loss_fn.call(None, pred)
            reg_loss = (
                tf.add_n(model.losses)
                if model.losses
                else tf.constant(0.0, dtype=tf.float32)
            )
            total_loss = main_loss + reg_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        apply_gradients_safe(optimizer, grads, model.trainable_variables)

        return total_loss, pred

    history = []
    best_loss = np.inf
    best_pred = None
    start = time.time()

    for step in range(args.steps):
        loss, pred = train_step()
        value = float(loss.numpy())
        history.append(value)

        if np.isfinite(value) and value < best_loss:
            best_loss = value
            best_pred = pred.numpy().reshape(problem.n, problem.n)

        if (step + 1) % max(1, args.steps // 10) == 0:
            print(
                f"[{problem.name}][{model_name}] "
                f"step {step + 1}/{args.steps} loss={value:.6e}"
            )

    elapsed = time.time() - start

    if best_pred is None:
        best_pred = model(f_input, training=False).numpy().reshape(problem.n, problem.n)

    metrics = error_norms(problem.u_exact, best_pred, problem.h)

    result = {
        "problem": problem.name,
        "type": "elliptic",
        "model_name": model_name,
        "n": problem.n,
        "h": problem.h,
        "steps": args.steps,
        "depth": args.depth,
        "filters": args.filters,
        "kernel_size": args.kernel_size,
        "activation": args.activation,
        "seed": args.seed,
        "num_params": int(model.count_params()),
        "best_loss": float(best_loss),
        "elapsed_sec": float(elapsed),
        "kan_grid_size": args.kan_grid_size if model_name == "unet_kan_middle" else None,
        "kan_gamma": args.kan_gamma if model_name == "unet_kan_middle" else None,
        "kan_activation": args.kan_activation if model_name == "unet_kan_middle" else None,
        "kan_layernorm": (not args.no_kan_layernorm) if model_name == "unet_kan_middle" else None,
        **metrics,
    }

    np.save(out_dir / "prediction.npy", best_pred.astype(np.float32))
    np.save(out_dir / "exact.npy", problem.u_exact.astype(np.float32))
    np.save(out_dir / "loss_history.npy", np.asarray(history, dtype=np.float32))
    write_json(out_dir / "metrics.json", result)

    if not args.no_plots:
        save_elliptic_comparison(
            problem.X,
            problem.Y,
            problem.u_exact,
            best_pred,
            out_dir / "comparison.png",
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
        "steps",
        "depth",
        "filters",
        "activation",
        "seed",
        "num_params",
        "best_loss",
        "elapsed_sec",
        "kan_grid_size",
        "kan_gamma",
        "l2_h",
        "linf",
    ]

    with (out_root / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare Linear U-Net and U-Net + KAN-Middle on elliptic PDE cases."
    )

    p.add_argument("--case", default="Peak", choices=["Bubble", "Peak", "Exp-Trig", "Non-Const-Diff"])
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--filters", type=int, default=32)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--activation", default="identity", choices=["identity", "relu", "tanh", "swish"])
    p.add_argument("--lr", type=float, default=DEFAULT_OPTIMIZER.elliptic_lr)
    p.add_argument("--l2", type=float, default=DEFAULT_OPTIMIZER.l2_weight)
    p.add_argument("--clipnorm", type=float, default=DEFAULT_OPTIMIZER.clipnorm)
    p.add_argument("--cosine-lr", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", choices=["linear_unet", "unet_kan_middle", "both"], default="both")
    p.add_argument("--kan-grid-size", type=int, default=8)
    p.add_argument("--kan-gamma", type=float, default=0.05)
    p.add_argument("--kan-activation", default="identity", choices=["identity", "relu", "tanh", "swish", "gelu"])
    p.add_argument("--no-kan-layernorm", action="store_true")
    p.add_argument("--out", default="results/extensions/pde_kan_compare/elliptic")
    p.add_argument("--no-plots", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)

    problem = get_elliptic_problem(args.case, args.N)

    tag = (
        f"N{args.N}_M{args.steps}_d{args.depth}_"
        f"gamma{str(args.kan_gamma).replace('.', 'p')}_seed{args.seed}"
    )

    out_root = Path(args.out) / args.case / tag
    out_root.mkdir(parents=True, exist_ok=True)

    model_names = ["linear_unet", "unet_kan_middle"] if args.model == "both" else [args.model]

    rows = []
    for model_name in model_names:
        setup_seed(args.seed)
        rows.append(train_one(args, problem, model_name, out_root / model_name))

    save_comparison(rows, out_root)
    print("Saved comparison to:", out_root / "comparison.csv")


if __name__ == "__main__":
    main()
