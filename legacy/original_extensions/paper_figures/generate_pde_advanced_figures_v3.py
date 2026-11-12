from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_grid(n: int):
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y


def read_pair(tag_dir: Path):
    linear_dir = tag_dir / "linear_unet"
    kan_dir = tag_dir / "unet_kan_middle"

    if not linear_dir.exists() or not kan_dir.exists():
        return None

    required = [
        linear_dir / "metrics.json",
        linear_dir / "exact.npy",
        linear_dir / "prediction.npy",
        kan_dir / "metrics.json",
        kan_dir / "exact.npy",
        kan_dir / "prediction.npy",
    ]

    if not all(p.exists() for p in required):
        return None

    m_linear = load_json(linear_dir / "metrics.json")
    m_kan = load_json(kan_dir / "metrics.json")

    exact = np.load(linear_dir / "exact.npy")
    pred_linear = np.load(linear_dir / "prediction.npy")
    pred_kan = np.load(kan_dir / "prediction.npy")

    loss_linear = None
    loss_kan = None

    if (linear_dir / "loss_history.npy").exists():
        loss_linear = np.load(linear_dir / "loss_history.npy")
    if (kan_dir / "loss_history.npy").exists():
        loss_kan = np.load(kan_dir / "loss_history.npy")

    return {
        "tag_dir": tag_dir,
        "linear_dir": linear_dir,
        "kan_dir": kan_dir,
        "m_linear": m_linear,
        "m_kan": m_kan,
        "exact": exact,
        "pred_linear": pred_linear,
        "pred_kan": pred_kan,
        "loss_linear": loss_linear,
        "loss_kan": loss_kan,
    }


def central_zoom_box(error_map: np.ndarray, box_size_ratio: float = 0.28):
    n, m = error_map.shape

    idx = int(np.argmax(error_map))
    iy, ix = np.unravel_index(idx, error_map.shape)

    box_h = max(8, int(n * box_size_ratio))
    box_w = max(8, int(m * box_size_ratio))

    y0 = max(0, iy - box_h // 2)
    y1 = min(n, y0 + box_h)
    y0 = max(0, y1 - box_h)

    x0 = max(0, ix - box_w // 2)
    x1 = min(m, x0 + box_w)
    x0 = max(0, x1 - box_w)

    return x0, x1, y0, y1


def add_panel_label(ax, label: str):
    ax.text(
        0.02,
        0.96,
        label,
        transform=ax.transAxes,
        fontsize=13,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
    )


def save_zoomed_error_figure(exact, pred_linear, pred_kan, out_path: Path, title: str):
    err_linear = np.abs(pred_linear - exact)
    err_kan = np.abs(pred_kan - exact)
    improve = err_linear - err_kan

    combined_err = np.maximum(err_linear, err_kan)
    x0, x1, y0, y1 = central_zoom_box(combined_err, box_size_ratio=0.28)

    extent = [0, 1, 0, 1]
    n = exact.shape[0]

    rx0 = x0 / (n - 1)
    rx1 = x1 / (n - 1)
    ry0 = y0 / (n - 1)
    ry1 = y1 / (n - 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    im0 = axes[0, 0].imshow(err_linear, origin="lower", extent=extent, aspect="auto")
    axes[0, 0].add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False, linewidth=2))
    axes[0, 0].set_title("Linear error")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    add_panel_label(axes[0, 0], "(a)")

    im1 = axes[0, 1].imshow(err_kan, origin="lower", extent=extent, aspect="auto")
    axes[0, 1].add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False, linewidth=2))
    axes[0, 1].set_title("KAN error")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    add_panel_label(axes[0, 1], "(b)")

    vmax = np.max(np.abs(improve))
    im2 = axes[0, 2].imshow(improve, origin="lower", extent=extent, aspect="auto", vmin=-vmax, vmax=vmax)
    axes[0, 2].add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False, linewidth=2))
    axes[0, 2].set_title("Improvement map")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    add_panel_label(axes[0, 2], "(c)")

    im3 = axes[1, 0].imshow(err_linear[y0:y1, x0:x1], origin="lower", aspect="auto")
    axes[1, 0].set_title("Zoomed linear error")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    add_panel_label(axes[1, 0], "(d)")

    im4 = axes[1, 1].imshow(err_kan[y0:y1, x0:x1], origin="lower", aspect="auto")
    axes[1, 1].set_title("Zoomed KAN error")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    add_panel_label(axes[1, 1], "(e)")

    im5 = axes[1, 2].imshow(improve[y0:y1, x0:x1], origin="lower", aspect="auto", vmin=-vmax, vmax=vmax)
    axes[1, 2].set_title("Zoomed improvement")
    fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
    add_panel_label(axes[1, 2], "(f)")

    for ax in axes.ravel():
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(title + " | Local error magnification", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def save_final_main_comparison(exact, pred_linear, pred_kan, out_path: Path, title: str):
    err_linear = np.abs(pred_linear - exact)
    err_kan = np.abs(pred_kan - exact)
    improve = err_linear - err_kan

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    fields = [
        ("Exact", exact),
        ("Linear U-Net", pred_linear),
        ("KAN-Middle", pred_kan),
        ("KAN - Linear", pred_kan - pred_linear),
        ("Linear error", err_linear),
        ("KAN error", err_kan),
        ("Improvement map", improve),
        ("Error ratio", err_kan / (err_linear + 1e-10)),
    ]

    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    for ax, (name, data), label in zip(axes.ravel(), fields, labels):
        if name in ["Improvement map", "KAN - Linear"]:
            vmax = np.max(np.abs(data))
            im = ax.imshow(data, origin="lower", extent=[0, 1, 0, 1], aspect="auto", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(data, origin="lower", extent=[0, 1, 0, 1], aspect="auto")

        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        add_panel_label(ax, label)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title + " | Final comparison", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def save_difference_contour_with_lines(exact, pred_linear, pred_kan, out_path: Path, title: str):
    n = exact.shape[0]
    x, y, X, Y = get_grid(n)

    err_linear = np.abs(pred_linear - exact)
    err_kan = np.abs(pred_kan - exact)
    improve = err_linear - err_kan

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    levels_solution = 32
    levels_error = 32

    c0 = axes[0].contourf(X, Y, exact, levels=levels_solution)
    axes[0].contour(X, Y, exact, levels=12, linewidths=0.5)
    axes[0].set_title("Exact with contour lines")
    fig.colorbar(c0, ax=axes[0], fraction=0.046, pad=0.04)
    add_panel_label(axes[0], "(a)")

    vmax_err = max(float(np.max(err_linear)), float(np.max(err_kan)))
    c1 = axes[1].contourf(X, Y, err_linear, levels=levels_error, vmin=0.0, vmax=vmax_err)
    axes[1].contour(X, Y, err_linear, levels=10, linewidths=0.5)
    axes[1].set_title("Linear error contours")
    fig.colorbar(c1, ax=axes[1], fraction=0.046, pad=0.04)
    add_panel_label(axes[1], "(b)")

    vmax = np.max(np.abs(improve))
    c2 = axes[2].contourf(X, Y, improve, levels=levels_error, vmin=-vmax, vmax=vmax)
    axes[2].contour(X, Y, improve, levels=10, linewidths=0.5)
    axes[2].set_title("KAN improvement contours")
    fig.colorbar(c2, ax=axes[2], fraction=0.046, pad=0.04)
    add_panel_label(axes[2], "(c)")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(title + " | Contour analysis", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def save_high_res_3d_final(exact, pred_linear, pred_kan, out_path: Path, title: str):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n = exact.shape[0]
    x, y, X, Y = get_grid(n)

    err_linear = np.abs(pred_linear - exact)
    err_kan = np.abs(pred_kan - exact)
    improve = err_linear - err_kan

    fig = plt.figure(figsize=(22, 12))

    fields = [
        ("Exact", exact),
        ("Linear U-Net", pred_linear),
        ("KAN-Middle", pred_kan),
        ("Linear error", err_linear),
        ("KAN error", err_kan),
        ("Improvement surface", improve),
    ]

    for i, (name, data) in enumerate(fields, start=1):
        ax = fig.add_subplot(2, 3, i, projection="3d")

        if name == "Improvement surface":
            vmax = float(np.max(np.abs(data)))
            surf = ax.plot_surface(X, Y, data, linewidth=0, antialiased=True, vmin=-vmax, vmax=vmax)
        else:
            surf = ax.plot_surface(X, Y, data, linewidth=0, antialiased=True)

        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("value")
        ax.view_init(elev=28, azim=-135)
        fig.colorbar(surf, ax=ax, shrink=0.56, pad=0.08)

    fig.suptitle(title + " | Three-dimensional comparison", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def save_error_ratio_map(exact, pred_linear, pred_kan, out_path: Path, title: str):
    err_linear = np.abs(pred_linear - exact)
    err_kan = np.abs(pred_kan - exact)

    ratio = err_kan / (err_linear + 1e-10)
    ratio_clip = np.clip(ratio, 0.0, 2.0)

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111)

    im = ax.imshow(ratio_clip, origin="lower", extent=[0, 1, 0, 1], aspect="auto", vmin=0.0, vmax=2.0)
    ax.contour(ratio_clip, levels=[1.0], extent=[0, 1, 0, 1], linewidths=1.4)
    ax.set_title("Error ratio map: KAN error / Linear error")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def save_final_cross_section_with_error(exact, pred_linear, pred_kan, out_path: Path, title: str):
    n = exact.shape[0]
    grid = np.linspace(0.0, 1.0, n)
    mid = n // 2

    exact_x = exact[mid, :]
    linear_x = pred_linear[mid, :]
    kan_x = pred_kan[mid, :]

    exact_y = exact[:, mid]
    linear_y = pred_linear[:, mid]
    kan_y = pred_kan[:, mid]

    err_linear_x = np.abs(linear_x - exact_x)
    err_kan_x = np.abs(kan_x - exact_x)
    err_linear_y = np.abs(linear_y - exact_y)
    err_kan_y = np.abs(kan_y - exact_y)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(grid, exact_x, label="Exact")
    axes[0, 0].plot(grid, linear_x, "--", label="Linear")
    axes[0, 0].plot(grid, kan_x, "-.", label="KAN")
    axes[0, 0].set_title("Solution profile at y=0.5")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("u")
    axes[0, 0].legend()
    add_panel_label(axes[0, 0], "(a)")

    axes[0, 1].plot(grid, exact_y, label="Exact")
    axes[0, 1].plot(grid, linear_y, "--", label="Linear")
    axes[0, 1].plot(grid, kan_y, "-.", label="KAN")
    axes[0, 1].set_title("Solution profile at x=0.5")
    axes[0, 1].set_xlabel("y")
    axes[0, 1].set_ylabel("u")
    axes[0, 1].legend()
    add_panel_label(axes[0, 1], "(b)")

    axes[1, 0].plot(grid, err_linear_x, "--", label="Linear error")
    axes[1, 0].plot(grid, err_kan_x, "-.", label="KAN error")
    axes[1, 0].set_title("Error profile at y=0.5")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("absolute error")
    axes[1, 0].legend()
    add_panel_label(axes[1, 0], "(c)")

    axes[1, 1].plot(grid, err_linear_y, "--", label="Linear error")
    axes[1, 1].plot(grid, err_kan_y, "-.", label="KAN error")
    axes[1, 1].set_title("Error profile at x=0.5")
    axes[1, 1].set_xlabel("y")
    axes[1, 1].set_ylabel("absolute error")
    axes[1, 1].legend()
    add_panel_label(axes[1, 1], "(d)")

    fig.suptitle(title + " | Cross-sectional comparison", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def pair_to_output_dir(pair, figure_root: Path):
    m = pair["m_linear"]
    typ = m.get("type", "unknown")
    problem = m.get("problem", "unknown")
    seed = m.get("seed", "unknown")
    return figure_root / typ / problem / f"seed_{seed}"


def process_all_pairs(result_root: Path, figure_root: Path):
    manifest_rows = []

    for typ in ["elliptic", "parabolic"]:
        typ_root = result_root / typ
        if not typ_root.exists():
            continue

        for problem_dir in sorted(typ_root.iterdir()):
            if not problem_dir.is_dir():
                continue

            for tag_dir in sorted(problem_dir.iterdir()):
                if not tag_dir.is_dir():
                    continue

                pair = read_pair(tag_dir)
                if pair is None:
                    continue

                m = pair["m_linear"]
                exact = pair["exact"]
                pred_linear = pair["pred_linear"]
                pred_kan = pair["pred_kan"]

                problem = m.get("problem", problem_dir.name)
                seed = m.get("seed", "unknown")
                title = f"{problem} | seed={seed}"

                out_dir = pair_to_output_dir(pair, figure_root)
                ensure_dir(out_dir)

                figure_files = {
                    "final_main_comparison": out_dir / "v3_final_main_comparison.png",
                    "zoomed_error": out_dir / "v3_zoomed_error.png",
                    "difference_contour": out_dir / "v3_difference_contour.png",
                    "high_res_3d": out_dir / "v3_high_res_3d.png",
                    "error_ratio_map": out_dir / "v3_error_ratio_map.png",
                    "cross_section_error": out_dir / "v3_cross_section_error.png",
                }

                save_final_main_comparison(exact, pred_linear, pred_kan, figure_files["final_main_comparison"], title)
                save_zoomed_error_figure(exact, pred_linear, pred_kan, figure_files["zoomed_error"], title)
                save_difference_contour_with_lines(exact, pred_linear, pred_kan, figure_files["difference_contour"], title)
                save_high_res_3d_final(exact, pred_linear, pred_kan, figure_files["high_res_3d"], title)
                save_error_ratio_map(exact, pred_linear, pred_kan, figure_files["error_ratio_map"], title)
                save_final_cross_section_with_error(exact, pred_linear, pred_kan, figure_files["cross_section_error"], title)

                for name, path in figure_files.items():
                    manifest_rows.append({
                        "problem": problem,
                        "type": m.get("type", ""),
                        "seed": seed,
                        "figure_type": name,
                        "path": str(path),
                    })

    return manifest_rows


def collect_metrics(result_root: Path):
    rows = []
    for p in sorted(result_root.rglob("metrics.json")):
        try:
            d = load_json(p)
            d["source"] = str(p)
            rows.append(d)
        except Exception:
            pass

    return pd.DataFrame(rows)


def paired_dataframe(df: pd.DataFrame):
    rows = []

    for (problem, seed), sub in df.groupby(["problem", "seed"]):
        b = sub[sub["model_name"] == "linear_unet"]
        k = sub[sub["model_name"] == "unet_kan_middle"]

        if len(b) == 0 or len(k) == 0:
            continue

        b = b.iloc[0]
        k = k.iloc[0]

        b_l2 = float(b["l2_h"])
        k_l2 = float(k["l2_h"])
        b_linf = float(b["linf"])
        k_linf = float(k["linf"])

        rows.append({
            "problem": problem,
            "seed": seed,
            "baseline_l2_h": b_l2,
            "kan_l2_h": k_l2,
            "l2_change_percent": (k_l2 - b_l2) / b_l2 * 100.0,
            "baseline_linf": b_linf,
            "kan_linf": k_linf,
            "linf_change_percent": (k_linf - b_linf) / b_linf * 100.0,
            "baseline_params": float(b["num_params"]),
            "kan_params": float(k["num_params"]),
        })

    return pd.DataFrame(rows)


def save_statistical_panel(paired: pd.DataFrame, out_path: Path):
    problems = sorted(paired["problem"].unique().tolist())
    x = np.arange(len(problems))

    l2_mean = []
    l2_std = []
    linf_mean = []
    linf_std = []

    for p in problems:
        sub = paired[paired["problem"] == p]

        l2_mean.append(float(sub["l2_change_percent"].mean()))
        l2_std.append(float(sub["l2_change_percent"].std(ddof=1)) if len(sub) > 1 else 0.0)

        linf_mean.append(float(sub["linf_change_percent"].mean()))
        linf_std.append(float(sub["linf_change_percent"].std(ddof=1)) if len(sub) > 1 else 0.0)

    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    axes[0].bar(x, l2_mean, yerr=l2_std)
    axes[0].axhline(0.0, linestyle="--")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(problems, rotation=15)
    axes[0].set_ylabel("Relative change of L2_h (%)")
    axes[0].set_title("Paired L2_h change")
    add_panel_label(axes[0], "(a)")

    axes[1].bar(x, linf_mean, yerr=linf_std)
    axes[1].axhline(0.0, linestyle="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(problems, rotation=15)
    axes[1].set_ylabel("Relative change of L_inf (%)")
    axes[1].set_title("Paired L_inf change")
    add_panel_label(axes[1], "(b)")

    fig.suptitle("Paired error changes of KAN-Middle relative to Linear U-Net", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def save_model_metric_matrix(df: pd.DataFrame, out_path: Path, metric: str):
    df = df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    summary = df.groupby(["problem", "model_name"])[metric].mean().reset_index()
    problems = sorted(summary["problem"].unique().tolist())
    models = ["linear_unet", "unet_kan_middle"]

    mat = np.zeros((len(problems), len(models)), dtype=float)

    for i, p in enumerate(problems):
        for j, m in enumerate(models):
            row = summary[(summary["problem"] == p) & (summary["model_name"] == m)]
            mat[i, j] = float(row[metric].iloc[0]) if len(row) else np.nan

    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(["Linear U-Net", "KAN-Middle"], rotation=10)
    ax.set_yticks(np.arange(len(problems)))
    ax.set_yticklabels(problems)
    ax.set_title(f"Mean {metric} matrix")

    for i in range(len(problems)):
        for j in range(len(models)):
            ax.text(j, i, f"{mat[i, j]:.3e}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def save_seedwise_slope_plot(paired: pd.DataFrame, out_path: Path, metric_name: str):
    if metric_name == "l2_h":
        b_col = "baseline_l2_h"
        k_col = "kan_l2_h"
        title = "Seed-wise paired L2_h"
        ylabel = "L2_h"
    else:
        b_col = "baseline_linf"
        k_col = "kan_linf"
        title = "Seed-wise paired L_inf"
        ylabel = "L_inf"

    problems = sorted(paired["problem"].unique().tolist())
    fig, axes = plt.subplots(1, len(problems), figsize=(4.5 * len(problems), 5), squeeze=False)
    axes = axes[0]

    for ax, p in zip(axes, problems):
        sub = paired[paired["problem"] == p].sort_values("seed")

        for _, row in sub.iterrows():
            ax.plot([0, 1], [row[b_col], row[k_col]], marker="o")
            ax.text(0, row[b_col], f"s{row['seed']}", fontsize=8, ha="right")
            ax.text(1, row[k_col], f"s{row['seed']}", fontsize=8, ha="left")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Linear", "KAN"])
        ax.set_title(p)
        ax.set_ylabel(ylabel)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def write_latex_snippets(manifest_rows, out_path: Path):
    lines = []
    lines.append("% Auto-generated LaTeX figure snippets.")
    lines.append("% Copy selected blocks into your paper.")
    lines.append("")

    selected_types = [
        "final_main_comparison",
        "high_res_3d",
        "zoomed_error",
        "difference_contour",
    ]

    for row in manifest_rows:
        if row["figure_type"] not in selected_types:
            continue

        rel_path = row["path"]
        caption = (
            f"{row['problem']} 算例在 seed={row['seed']} 下的 "
            f"{row['figure_type'].replace('_', ' ')} 对比图。"
        )
        label = f"fig:{row['problem'].lower().replace('-', '_')}_seed{row['seed']}_{row['figure_type']}"

        lines.append("\\begin{figure}[htbp]")
        lines.append("  \\centering")
        lines.append(f"  \\includegraphics[width=0.98\\textwidth]{{{rel_path}}}")
        lines.append(f"  \\caption{{{caption}}}")
        lines.append(f"  \\label{{{label}}}")
        lines.append("\\end{figure}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_summary_outputs(result_root: Path, figure_root: Path):
    df = collect_metrics(result_root)

    if len(df) == 0:
        print("No metrics found.")
        return []

    for c in ["l2_h", "linf", "elapsed_sec", "num_params"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    paired = paired_dataframe(df)

    summary_dir = figure_root / "summary_v3"
    ensure_dir(summary_dir)

    df.to_csv(summary_dir / "all_metrics.csv", index=False)
    paired.to_csv(summary_dir / "paired_changes.csv", index=False)

    save_statistical_panel(paired, summary_dir / "v3_statistical_paired_changes.png")
    save_model_metric_matrix(df, summary_dir / "v3_metric_matrix_l2_h.png", "l2_h")
    save_model_metric_matrix(df, summary_dir / "v3_metric_matrix_linf.png", "linf")
    save_seedwise_slope_plot(paired, summary_dir / "v3_seedwise_slope_l2_h.png", "l2_h")
    save_seedwise_slope_plot(paired, summary_dir / "v3_seedwise_slope_linf.png", "linf")

    return paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="results/extensions/pde_kan_compare")
    parser.add_argument("--figure-root", default="results/paper_figures/pde_kan_compare_v3")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    figure_root = Path(args.figure_root)
    ensure_dir(figure_root)

    manifest_rows = process_all_pairs(result_root, figure_root)

    manifest_path = figure_root / "figure_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["problem", "type", "seed", "figure_type", "path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    write_latex_snippets(manifest_rows, figure_root / "latex_figure_snippets.tex")
    save_summary_outputs(result_root, figure_root)

    print("Saved third-version advanced figures to:", figure_root)
    print("Figure manifest:", manifest_path)
    print("LaTeX snippets:", figure_root / "latex_figure_snippets.tex")


if __name__ == "__main__":
    main()
