from pathlib import Path
import argparse
import numpy as np

def load_mat(path):
    try:
        import scipy.io as sio
        data = sio.loadmat(path)
        return {k: v for k, v in data.items() if not k.startswith("__")}
    except Exception as e1:
        print("scipy.io.loadmat 读取失败，尝试 h5py：", repr(e1))
        import h5py
        out = {}
        with h5py.File(path, "r") as f:
            def visit(name, obj):
                if hasattr(obj, "shape"):
                    arr = np.array(obj)
                    out[name] = arr
            f.visititems(visit)
        return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", required=True)
    args = parser.parse_args()

    data = load_mat(args.mat)
    print("文件：", args.mat)
    print("变量数量：", len(data))
    print("-" * 80)

    for k, v in data.items():
        if isinstance(v, np.ndarray):
            info = f"{k}: shape={v.shape}, dtype={v.dtype}"
            if np.issubdtype(v.dtype, np.number):
                try:
                    info += f", min={np.nanmin(v):.6g}, max={np.nanmax(v):.6g}"
                except Exception:
                    pass
            print(info)
        else:
            print(k, type(v))

if __name__ == "__main__":
    main()
