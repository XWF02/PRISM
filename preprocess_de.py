"""MDE construction from the official precomputed DE epoch sequences.

The manuscript defines the epoch-wise MDE exactly, and specifies 10-step,
50%-overlap windows for SEED/SEED-IV. It does not state the window-level
reduction operator. This release exposes mean pooling as an explicit default;
confirm it against the preprocessing used for the reported experiments.
"""
from __future__ import annotations

import numpy as np


def mde_sequence(de_trial: np.ndarray) -> np.ndarray:
    """Build epoch-wise MDE for one trial: [T,C,5] -> [T,C,10].

    The first-order difference at epoch t is DE[t]-DE[t-1]. The first epoch
    has zero difference because no preceding epoch exists inside that trial.
    Call this function separately for every trial; never concatenate trials
    before computing differences.
    """
    de = np.asarray(de_trial, dtype=np.float32)
    if de.ndim != 3 or de.shape[1] <= 0 or de.shape[2] != 5:
        raise ValueError("Expected one trial with shape [time, channels, 5]")
    if not np.isfinite(de).all():
        raise ValueError("DE sequence contains NaN or Inf")
    difference = np.zeros_like(de)
    if de.shape[0] > 1:
        difference[1:] = de[1:] - de[:-1]
    # Per channel: [five 0th-order bands, five 1st-order bands].
    return np.concatenate([de, difference], axis=-1)


def seed_seediv_window_mde(
    de_trial: np.ndarray,
    window: int = 10,
    stride: int = 5,
    pooling: str = "mean",
) -> np.ndarray:
    """Create one 2*C*5-dimensional MDE vector per within-trial window.

    The complete epoch-wise MDE sequence is constructed before windowing.
    This preserves the transition from the epoch immediately preceding an
    overlapping window (e.g. DE[5]-DE[4] for a window starting at epoch 5).

    `pooling='mean'` is an explicit operational choice because the manuscript
    does not specify how its 10-step window is reduced to the stated 620-D
    model input. It must be checked against the original experimental pipeline.
    """
    if window < 2 or stride < 1:
        raise ValueError("window must be >= 2 and stride must be >= 1")
    if pooling != "mean":
        raise ValueError("Only mean pooling is currently implemented")
    mde = mde_sequence(de_trial)
    if mde.shape[0] < window:
        return np.empty((0, mde.shape[1] * mde.shape[2]), dtype=np.float32)

    samples = []
    for start in range(0, mde.shape[0] - window + 1, stride):
        window_mde = mde[start:start + window]
        # Window reduction is not specified in the manuscript; mean is explicit.
        pooled = window_mde.mean(axis=0)
        samples.append(pooled.reshape(-1))
    return np.asarray(samples, dtype=np.float32)
