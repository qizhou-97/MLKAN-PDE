from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


def setup_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and TensorFlow seeds when TensorFlow is available."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
        tf.keras.backend.set_floatx("float32")
    except Exception:
        pass


@dataclass(frozen=True)
class OptimizerConfig:
    elliptic_lr: float = 1.0e-3
    parabolic_lr: float = 1.0e-4
    l2_weight: float = 1.0e-7
    clipnorm: float = 1.0e-2


@dataclass(frozen=True)
class NetworkConfig:
    filters: int = 32
    depth: int = 3
    activation: str = "identity"
    kernel_size: int = 5
    pocket: bool = True


@dataclass(frozen=True)
class ParabolicConfig:
    tau: float = 0.1
    first_step_iters: int = 1000
    later_step_iters: int = 250
    alpha_mode: str = "notebook"


DEFAULT_OPTIMIZER = OptimizerConfig()
DEFAULT_NETWORK = NetworkConfig()
DEFAULT_PARABOLIC = ParabolicConfig()
