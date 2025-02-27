"""Inference pipeline for the H&E U-Net model.

Provides batch prediction over a test Dataset, optional morphological
post-processing (majority-vote median filter), and overlay generation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from multiplex_pipeline.hne.config import MA_KERNEL_SIZE
from scipy.ndimage import uniform_filter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def apply_morphological_filter(
    pred_mask: np.ndarray,
    kernel_size: int = MA_KERNEL_SIZE,
) -> np.ndarray:
    """Smooth a predicted label map with a majority-vote filter.

    Each output pixel is assigned the most frequent class among all pixels
    in a kernel_size × kernel_size neighborhood. This removes isolated
    mis-classified pixels and smooths class boundaries.

    Uses scipy's C-implemented uniform_filter to count neighborhood votes
    per class and then takes argmax — O(n_classes × H × W) instead of
    O(H × W) Python callbacks.

    Args:
        pred_mask: 2-D integer array of predicted class labels.
        kernel_size: Size of the square filter window.

    Returns:
        Array of the same shape and dtype as pred_mask.
    """
    n_classes = int(pred_mask.max()) + 1
    votes = np.stack(
        [
            uniform_filter(
                (pred_mask == c).astype(np.float32),
                size=kernel_size,
                mode="nearest",
            )
            for c in range(n_classes)
        ]
    )
    return votes.argmax(axis=0).astype(pred_mask.dtype)


def predict_patches(
    model: nn.Module,
    dataset: Dataset,
    device: str | torch.device = "cuda",
    output_dir: str | Path | None = None,
    apply_ma: bool = True,
    ma_kernel: int = MA_KERNEL_SIZE,
    batch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on a Dataset and collect predictions.

    Args:
        model: Trained U-Net (already on device).
        dataset: Test Dataset (augment=False).
        device: Inference device.
        output_dir: If given, prediction overlays are saved here as SVG files.
        apply_ma: Whether to apply `apply_morphological_filter` to each prediction.
        ma_kernel: Kernel size passed to the filter.
        batch_size: Samples per inference batch.

    Returns:
        Tuple of (y_true (N_px,), y_pred (N_px,), y_prob (N_px, n_classes)).
    """
    device = torch.device(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    model.eval()
    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Predicting"):
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()  # (B, C, H, W)

            for b in range(x.shape[0]):
                P = probs[b]  # (C, H, W)
                pred = np.argmax(P, axis=0).astype(np.uint8)  # (H, W)

                if apply_ma:
                    pred = apply_morphological_filter(pred, kernel_size=ma_kernel)

                gt = y[b].numpy()  # (H, W)
                all_true.append(gt.flatten())
                all_pred.append(pred.flatten())
                all_prob.append(P.transpose(1, 2, 0).reshape(-1, P.shape[0]))

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.vstack(all_prob),
    )
