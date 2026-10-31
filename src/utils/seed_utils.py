"""
src/utils/seed_utils.py
=======================
Reproducibility utilities.

Call set_global_seed(SEED) at the very start of every script that does
any random operation (training, evaluation, data splitting).
"""

import os
import random
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and TensorFlow to ensure
    reproducible results across runs.

    Parameters
    ----------
    seed : int
        The seed value. Default is 42.
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # TensorFlow (imported here to avoid mandatory dependency at module load)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass  # TensorFlow not installed yet; seed will be set when available

    # Deterministic operations in TensorFlow
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
