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
                    out[name] = np.array(obj)
            f.visititems(visit)
        return out


def select_numeric_array(data):
    candidates = []
    for k, v in data.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            arr = np.array(v)
            arr = np.squeeze(arr)
            if arr.ndim >= 2 and arr.size > 100:
                candidates.append((k, arr, arr.size))
    if not candidates:
        raise RuntimeError("没有找到可用的数值型 Burgers 数据数组。")
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0][0], candidates[0][1]


def to_trajectories(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        # 解释为 [T, X] 或 [X, T]
        if arr.shape[0] > arr.shape[1] and arr.shape[1] < 512:
            arr = arr.T
        return arr[None, :, :]

    if arr.ndim == 3:
        # 尝试把三维数组解释为 [S, T, X]
        best = None
        for perm in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
            a = np.transpose(arr, perm)
            S, T, X = a.shape
            if T < 2 or X < 8:
                continue
            score = 0
            if X >= T:
                score += 10
            if S >= 2:
                score += 5
            score += min(X, 2048) / 2048
            if best is None or score > best[0]:
                best = (score, a)
        if best is None:
            raise RuntimeError(f"无法解释三维数组形状：{arr.shape}")
        return best[1]

    if arr.ndim > 3:
        # 默认保留最后两个维度作为 [T, X]，其余展平为样本维
        T, X = arr.shape[-2], arr.shape[-1]
        return arr.reshape(-1, T, X).astype(np.float32)

    raise RuntimeError(f"不支持的数据维度：{arr.shape}")


def build_cnn(nx):
    inp = tf.keras.Input(shape=(nx, 1))
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="tanh")(inp)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="tanh")(x)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="tanh")(x)
    out = tf.keras.layers.Conv1D(1, 5, padding="same", activation=None)(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
    )
    return model


def rel_l2(y_true, y_pred):
    num = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=(1, 2)))
    den = np.sqrt(np.sum(y_true ** 2, axis=(1, 2))) + 1e-12
    return float(np.mean(num / den))


def plot_one_step(x, y_true, y_pred, out):
    xx = np.linspace(0, 1, x.shape[0])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xx, x[:, 0], label="Input u(t)", linewidth=1.5)
    ax.plot(xx, y_true[:, 0], label="True u(t+dt)", linewidth=1.5)
    ax.plot(xx, y_pred[:, 0], "--", label="Pred u(t+dt)", linewidth=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("Burgers one-step prediction")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_rollout(traj_true, model, out, steps=20):
    steps = min(steps, traj_true.shape[0] - 1)
    pred = [traj_true[0]]
    cur = traj_true[0][None, :, None]
    for _ in range(steps):
        nxt = model.predict(cur, verbose=0)[0, :, 0]
        pred.append(nxt)
        cur = nxt[None, :, None]
    pred = np.array(pred)
    true = traj_true[:steps + 1]

    vmax = max(np.max(np.abs(true)), np.max(np.abs(pred)))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axs[0].imshow(true, aspect="auto", origin="lower", cmap="jet", vmin=-vmax, vmax=vmax)
    axs[0].set_title("True trajectory")
    axs[0].set_xlabel("x index")
    axs[0].set_ylabel("time step")

    im1 = axs[1].imshow(pred, aspect="auto", origin="lower", cmap="jet", vmin=-vmax, vmax=vmax)
    axs[1].set_title("Predicted rollout")
    axs[1].set_xlabel("x index")

    im2 = axs[2].imshow(np.abs(true - pred), aspect="auto", origin="lower", cmap="jet")
    axs[2].set_title("Absolute error")
    axs[2].set_xlabel("x index")

    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", required=True)
    parser.add_argument("--out", default="results/extensions/burgers/supervised_quick")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_mat(args.mat)
    var_name, arr = select_numeric_array(data)
    trajs = to_trajectories(arr)

    print("使用变量：", var_name)
    print("原始形状：", arr.shape)
    print("解释为轨迹形状 [S,T,X]：", trajs.shape)

    S, T, X = trajs.shape
    if T < 2:
        raise RuntimeError("时间维度不足，无法构造 u(t)->u(t+dt) 样本。")

    xs = trajs[:, :-1, :].reshape(-1, X, 1)
    ys = trajs[:, 1:, :].reshape(-1, X, 1)

    idx = np.arange(xs.shape[0])
    np.random.shuffle(idx)
    if len(idx) > args.max_pairs:
        idx = idx[:args.max_pairs]
    xs = xs[idx]
    ys = ys[idx]

    n_train = int(0.8 * len(xs))
    x_train, y_train = xs[:n_train], ys[:n_train]
    x_test, y_test = xs[n_train:], ys[n_train:]

    model = build_cnn(X)
    hist = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        batch_size=64,
        verbose=2
    )

    y_pred = model.predict(x_test, verbose=0)
    mse = float(np.mean((y_test - y_pred) ** 2))
    r_l2 = rel_l2(y_test, y_pred)

    metrics = {
        "mat_file": args.mat,
        "selected_variable": var_name,
        "raw_shape": list(arr.shape),
        "trajectory_shape_S_T_X": list(trajs.shape),
        "num_pairs_used": int(len(xs)),
        "epochs": args.epochs,
        "test_mse": mse,
        "test_relative_l2": r_l2,
    }

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    np.savetxt(outdir / "loss_history.csv",
               np.column_stack([hist.history["loss"], hist.history["val_loss"]]),
               delimiter=",", header="loss,val_loss", comments="")

    k = 0
    plot_one_step(x_test[k], y_test[k], y_pred[k], outdir / "one_step_prediction.png")

    # 用第一条轨迹做 rollout 可视化
    plot_rollout(trajs[0], model, outdir / "rollout_prediction.png", steps=min(30, T - 1))

    model.save(outdir / "model.keras")

    print("完成。结果目录：", outdir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
