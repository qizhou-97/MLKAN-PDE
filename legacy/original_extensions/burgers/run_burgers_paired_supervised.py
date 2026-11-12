import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf


def load_mat(path):
    try:
        import scipy.io as sio
        data = sio.loadmat(path)
        return {k: v for k, v in data.items() if not k.startswith("__")}
    except Exception:
        import h5py
        out = {}
        with h5py.File(path, "r") as f:
            def visit(name, obj):
                if hasattr(obj, "shape"):
                    arr = np.array(obj)
                    if arr.ndim >= 2:
                        arr = arr.T
                    out[name] = arr
            f.visititems(visit)
        return out


def build_model(nx):
    inp = tf.keras.Input(shape=(nx, 1))
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="gelu")(inp)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="gelu")(x)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="gelu")(x)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="gelu")(x)
    out = tf.keras.layers.Conv1D(1, 5, padding="same")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def relative_l2(y_true, y_pred):
    num = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=(1, 2)))
    den = np.sqrt(np.sum(y_true ** 2, axis=(1, 2))) + 1e-12
    return float(np.mean(num / den))


def plot_examples(x, y, pred, outdir):
    xx = np.linspace(0, 1, x.shape[1])
    n_show = min(4, x.shape[0])
    fig, axs = plt.subplots(n_show, 1, figsize=(9, 3 * n_show), sharex=True)
    if n_show == 1:
        axs = [axs]
    for i in range(n_show):
        axs[i].plot(xx, x[i, :, 0], label="Input a(x)", linewidth=1.2)
        axs[i].plot(xx, y[i, :, 0], label="True u(x)", linewidth=1.4)
        axs[i].plot(xx, pred[i, :, 0], "--", label="Pred u(x)", linewidth=1.4)
        axs[i].grid(True, alpha=0.3)
        axs[i].legend()
    axs[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(outdir / "paired_prediction_examples.png", dpi=300)
    plt.close(fig)


def plot_heatmap(y, pred, outdir):
    n_show = min(64, y.shape[0])
    true = y[:n_show, :, 0]
    pp = pred[:n_show, :, 0]
    err = np.abs(true - pp)

    vmax = max(float(np.max(np.abs(true))), float(np.max(np.abs(pp))))

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    im0 = axs[0].imshow(true, aspect="auto", cmap="jet", vmin=-vmax, vmax=vmax)
    axs[0].set_title("True u")
    axs[0].set_xlabel("x index")
    axs[0].set_ylabel("sample")

    im1 = axs[1].imshow(pp, aspect="auto", cmap="jet", vmin=-vmax, vmax=vmax)
    axs[1].set_title("Predicted u")
    axs[1].set_xlabel("x index")

    im2 = axs[2].imshow(err, aspect="auto", cmap="jet")
    axs[2].set_title("Absolute error")
    axs[2].set_xlabel("x index")

    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outdir / "paired_prediction_heatmap.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", required=True)
    parser.add_argument("--input-var", default="a")
    parser.add_argument("--target-var", default="u")
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--out", default="results/extensions/burgers/paired_supervised_R10")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_mat(args.mat)

    if args.input_var not in data:
        raise KeyError(f"找不到输入变量 {args.input_var}，已有变量：{list(data.keys())}")
    if args.target_var not in data:
        raise KeyError(f"找不到目标变量 {args.target_var}，已有变量：{list(data.keys())}")

    a = np.squeeze(np.asarray(data[args.input_var], dtype=np.float32))
    u = np.squeeze(np.asarray(data[args.target_var], dtype=np.float32))

    if a.shape != u.shape:
        raise ValueError(f"输入和目标形状不一致：a={a.shape}, u={u.shape}")

    a = a[:, ::args.stride]
    u = u[:, ::args.stride]

    x = a[..., None]
    y = u[..., None]

    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)

    train_n = int(0.8 * len(idx))
    train_idx = idx[:train_n]
    test_idx = idx[train_n:]

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    x_mean = float(np.mean(x_train))
    x_std = float(np.std(x_train) + 1e-8)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) + 1e-8)

    x_train_n = (x_train - x_mean) / x_std
    x_test_n = (x_test - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    model = build_model(x_train.shape[1])

    hist = model.fit(
        x_train_n,
        y_train_n,
        validation_data=(x_test_n, y_test_n),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )

    pred_n = model.predict(x_test_n, verbose=0)
    pred = pred_n * y_std + y_mean

    mse = float(np.mean((y_test - pred) ** 2))
    rel = relative_l2(y_test, pred)
    linf = float(np.max(np.abs(y_test - pred)))

    metrics = {
        "mat_file": args.mat,
        "input_var": args.input_var,
        "target_var": args.target_var,
        "original_shape": list(np.squeeze(data[args.input_var]).shape),
        "used_shape": list(x.shape),
        "stride": args.stride,
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_mse": mse,
        "test_relative_l2": rel,
        "test_linf": linf,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    np.savetxt(
        outdir / "loss_history.csv",
        np.column_stack([hist.history["loss"], hist.history["val_loss"]]),
        delimiter=",",
        header="loss,val_loss",
        comments="",
    )

    plot_examples(x_test, y_test, pred, outdir)
    plot_heatmap(y_test, pred, outdir)

    model.save(outdir / "model.keras")

    print("完成。结果目录：", outdir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
