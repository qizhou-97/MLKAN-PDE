from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except ImportError:
        pass


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_group_ids(n_groups: int, seed: int, ratios=(0.70, 0.15, 0.15)):
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Split ratios must sum to one: {ratios}")
    rng = np.random.default_rng(seed)
    ids = rng.permutation(n_groups)
    n_train = int(round(ratios[0] * n_groups))
    n_val = int(round(ratios[1] * n_groups))
    n_train = min(max(n_train, 1), n_groups)
    n_val = min(max(n_val, 1), max(n_groups - n_train, 0))
    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }
