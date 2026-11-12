from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from .losses import EllipticFDLoss, ParabolicFDLoss, VariableDiffusionLoss
from .metrics import error_norms
from .models import build_unet
from .plots import save_elliptic_comparison, save_loss_curve, save_parabolic_timeline
from .problems import EllipticProblem, ParabolicProblem


def _tensor2d(x: np.ndarray) -> tf.Tensor:
    return tf.convert_to_tensor(x.reshape(1, x.shape[0], x.shape[1], 1), dtype=tf.float32)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


class EllipticTrainer:
    def __init__(self, problem: EllipticProblem, output_dir: str | Path):
        self.problem = problem
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        steps: int = 4000,
        depth: int = 3,
        filters: int = 32,
        activation: str = "identity",
        learning_rate: float = 1.0e-3,
        l2_weight: float = 1.0e-7,
        clipnorm: float = 1.0e-2,
        save_plots: bool = True,
    ) -> dict[str, Any]:
        p = self.problem
        model = build_unet(p.n, filters=filters, depth=depth, activation=activation, l2_weight=l2_weight)
        f_input = _tensor2d(p.f_source)
        if p.kappa is None:
            loss_fn = EllipticFDLoss(p.n, p.h, p.f_source, p.boundary)
        else:
            loss_fn = VariableDiffusionLoss(p.n, p.h, p.f_source, p.boundary, p.kappa)
        lr = tf.keras.optimizers.schedules.CosineDecay(learning_rate, decay_steps=max(1, steps))
        optimizer = tf.keras.optimizers.Adam(lr, global_clipnorm=clipnorm)

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                pred = model(f_input, training=True)
                main_loss = loss_fn.call(None, pred)
                reg_loss = tf.add_n(model.losses) if model.losses else tf.constant(0.0, dtype=tf.float32)
                total_loss = main_loss + reg_loss
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return total_loss, pred

        history: list[float] = []
        best_loss = np.inf
        best_pred = None
        start = time.time()
        for step in range(steps):
            loss, pred = train_step()
            value = float(loss.numpy())
            history.append(value)
            if np.isfinite(value) and value < best_loss:
                best_loss = value
                best_pred = pred.numpy().reshape(p.n, p.n)
            if (step + 1) % max(1, steps // 10) == 0:
                print(f"[{p.name}] step {step + 1}/{steps} loss={value:.6e}")

        elapsed = time.time() - start
        if best_pred is None:
            best_pred = model(f_input, training=False).numpy().reshape(p.n, p.n)
        metrics = error_norms(p.u_exact, best_pred, p.h)
        result = {
            "problem": p.name,
            "type": "elliptic",
            "n": p.n,
            "h": p.h,
            "steps": steps,
            "depth": depth,
            "filters": filters,
            "activation": activation,
            "num_params": int(model.count_params()),
            "best_loss": float(best_loss),
            "elapsed_sec": float(elapsed),
            **metrics,
        }
        np.save(self.output_dir / "prediction.npy", best_pred.astype(np.float32))
        np.save(self.output_dir / "exact.npy", p.u_exact.astype(np.float32))
        np.save(self.output_dir / "loss_history.npy", np.asarray(history, dtype=np.float32))
        _write_json(self.output_dir / "metrics.json", result)
        if save_plots:
            save_elliptic_comparison(p.X, p.Y, p.u_exact, best_pred, self.output_dir / "comparison.png", p.name)
            save_loss_curve(history, self.output_dir / "loss_curve.png", f"Loss: {p.name}")
        return result


class ParabolicTrainer:
    def __init__(self, problem: ParabolicProblem, output_dir: str | Path):
        self.problem = problem
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        final_time: float = 0.5,
        tau: float = 0.1,
        first_step_iters: int = 1000,
        later_step_iters: int = 250,
        depth: int = 3,
        filters: int = 32,
        activation: str = "identity",
        learning_rate: float = 1.0e-4,
        l2_weight: float = 1.0e-7,
        clipnorm: float = 1.0e-2,
        alpha_mode: str = "notebook",
        save_plots: bool = True,
    ) -> dict[str, Any]:
        p = self.problem
        n_steps = int(round(final_time / tau))
        if abs(n_steps * tau - final_time) > 1.0e-8:
            raise ValueError("final_time must be an integer multiple of tau")
        model = build_unet(p.n, filters=filters, depth=depth, activation=activation, l2_weight=l2_weight)
        loss_fn = ParabolicFDLoss(p.n, p.h, tau, alpha_mode=alpha_mode)
        optimizer = tf.keras.optimizers.Adam(learning_rate, global_clipnorm=clipnorm)

        @tf.function
        def train_step(u_previous, f_current, boundaries):
            with tf.GradientTape() as tape:
                pred = model(u_previous, training=True)
                main_loss = loss_fn.call((pred, u_previous, f_current, boundaries))
                reg_loss = tf.add_n(model.losses) if model.losses else tf.constant(0.0, dtype=tf.float32)
                total_loss = main_loss + reg_loss
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return total_loss, pred

        u_previous = _tensor2d(p.u0())
        solutions = [p.u0().astype(np.float32)]
        times = [0.0]
        history: list[float] = []
        start = time.time()
        best_loss_global = np.inf
        for time_index in range(1, n_steps + 1):
            t_value = time_index * tau
            f_current = _tensor2d(p.source(t_value))
            b = p.boundary(t_value)
            boundaries = tuple(tf.convert_to_tensor(x, dtype=tf.float32) for x in b)
            steps_this = first_step_iters if time_index == 1 else later_step_iters
            best_loss = np.inf
            best_u = None
            for step in range(steps_this):
                loss, pred = train_step(u_previous, f_current, boundaries)
                value = float(loss.numpy())
                history.append(value)
                if np.isfinite(value) and value < best_loss:
                    best_loss = value
                    best_u = pred
                if value < best_loss_global:
                    best_loss_global = value
            if best_u is None:
                best_u = pred
            u_previous = best_u
            solutions.append(u_previous.numpy().reshape(p.n, p.n).astype(np.float32))
            times.append(t_value)
            print(f"[{p.name}] time {t_value:.2f}/{final_time:.2f} best_loss={best_loss:.6e}")

        elapsed = time.time() - start
        prediction = solutions[-1]
        truth = p.exact(final_time)
        metrics = error_norms(truth, prediction, p.h)
        result = {
            "problem": p.name,
            "type": "parabolic",
            "n": p.n,
            "h": p.h,
            "final_time": final_time,
            "tau": tau,
            "first_step_iters": first_step_iters,
            "later_step_iters": later_step_iters,
            "depth": depth,
            "filters": filters,
            "activation": activation,
            "alpha_mode": alpha_mode,
            "num_params": int(model.count_params()),
            "best_loss": float(best_loss_global),
            "elapsed_sec": float(elapsed),
            **metrics,
        }
        np.save(self.output_dir / "prediction.npy", prediction.astype(np.float32))
        np.save(self.output_dir / "exact.npy", truth.astype(np.float32))
        np.save(self.output_dir / "loss_history.npy", np.asarray(history, dtype=np.float32))
        np.save(self.output_dir / "times.npy", np.asarray(times, dtype=np.float32))
        np.save(self.output_dir / "solutions.npy", np.asarray(solutions, dtype=np.float32))
        _write_json(self.output_dir / "metrics.json", result)
        if save_plots:
            save_parabolic_timeline(p.X, p.Y, times, solutions, p.exact, self.output_dir / "timeline.png", p.name)
            save_loss_curve(history, self.output_dir / "loss_curve.png", f"Loss: {p.name}")
        return result
