"""Inference pipeline for the H&E U-Net model.

Provides batch prediction over a test Dataset, optional morphological
post-processing (majority-vote median filter), and overlay generation.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from multiplex_pipeline.hne.config import MA_KERNEL_SIZE
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def apply_morphological_filter(
    pred_mask: np.ndarray,
    kernel_size: int = MA_KERNEL_SIZE,
) -> np.ndarray:
    """Smooth a predicted label map with a majority-vote median filter.

    Each output pixel is assigned the most frequent class among all pixels
    in a kernel_size × kernel_size neighbourhood. This removes isolated
    mis-classified pixels and smooths class boundaries.

    Parameters
    ----------
    pred_mask:
        2-D integer array of predicted class labels.
    kernel_size:
        Size of the square filter window.

    Returns
    -------
    np.ndarray of the same shape and dtype as pred_mask.
    """
    return scipy.ndimage.generic_filter(
        pred_mask,
        lambda v: np.bincount(v.astype(int)).argmax(),
        size=kernel_size,
    )


def predict_patches(
    model: nn.Module,
    dataset: Dataset,
    device: str | torch.device = "cuda",
    output_dir: str | os.PathLike | None = None,
    apply_ma: bool = True,
    ma_kernel: int = MA_KERNEL_SIZE,
    batch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on a Dataset and collect predictions.

    Parameters
    ----------
    model:
        Trained U-Net (already on device).
    dataset:
        Test Dataset (augment=False).
    device:
        Inference device.
    output_dir:
        If given, prediction overlays are saved here as SVG files.
    apply_ma:
        Whether to apply :func:`apply_morphological_filter` to each prediction.
    ma_kernel:
        Kernel size passed to the filter.
    batch_size:
        Samples per inference batch.

    Returns
    -------
    y_true : np.ndarray  (N_pixels,)  – ground-truth flat labels
    y_pred : np.ndarray  (N_pixels,)  – predicted flat labels
    y_prob : np.ndarray  (N_pixels, n_classes) – per-class softmax probabilities
    """
    device = torch.device(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    model.eval()
    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(loader, desc="Predicting")):
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

                logger.debug("Patch %d predicted.", idx * batch_size + b)

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.vstack(all_prob),
    )
