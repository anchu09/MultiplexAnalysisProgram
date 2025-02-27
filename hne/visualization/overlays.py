"""Prediction overlay visualization for the H&E U-Net.

Produces side-by-side three-panel figures:
  [Original RGB] | [Predicted mask] | [Ground-truth mask]

Each figure is saved as an SVG file named overlay_<idx>.svg.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from multiplex_pipeline.hne.config import CLASS_LABELS, IGNORE_INDEX, N_CLASSES

# background=orange, invasion_front=red, stroma=green, ignore=transparent
_CLASS_RGBA: dict[int, tuple[float, float, float, float]] = {
    0: (1.0, 0.65, 0.0, 1.0),
    1: (1.0, 0.0, 0.0, 1.0),
    2: (0.0, 1.0, 0.0, 1.0),
}
_IGNORE_RGBA: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

_GT_SENTINEL = N_CLASSES + 1  # value used to represent IGNORE_INDEX pixels in GT_CMAP


def _build_unet_cmap() -> mcolors.ListedColormap:
    cmap_arr = np.zeros((256, 4))
    for k, v in _CLASS_RGBA.items():
        cmap_arr[k] = v
    return mcolors.ListedColormap(cmap_arr)


def _build_gt_cmap() -> mcolors.ListedColormap:
    # For ground-truth masks that may contain IGNORE_INDEX (−1 → stored as 255 when read as uint8)
    gt_vals = {**_CLASS_RGBA, _GT_SENTINEL: _IGNORE_RGBA}
    return mcolors.ListedColormap([gt_vals[i] for i in sorted(gt_vals)])


UNET_CMAP = _build_unet_cmap()
GT_CMAP = _build_gt_cmap()


def save_prediction_overlay(
    rgb: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    output_path: str | os.PathLike,
) -> None:
    """Save a three-panel prediction overlay figure.

    Args:
        rgb: uint8 (H, W, 3) original RGB patch.
        pred_mask: Integer (H, W) array of predicted class labels.
        gt_mask: Integer (H, W) array of ground-truth labels. Pixels with value
            IGNORE_INDEX are rendered transparently.
        output_path: Destination SVG file path. Parent directories are created if needed.
    """
    Path(output_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    gt_viz = _prepare_gt(gt_mask)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax1, ax2, ax3 = axes

    ax1.imshow(rgb)
    ax1.set_title("Original")
    ax1.axis("off")

    ax2.imshow(pred_mask, cmap=UNET_CMAP, norm=mcolors.NoNorm())
    ax2.set_title("Predicted")
    ax2.axis("off")

    ax3.imshow(gt_viz, cmap=GT_CMAP, norm=mcolors.NoNorm())
    ax3.set_title("Ground Truth")
    ax3.axis("off")

    handles = [
        mpatches.Patch(color=_CLASS_RGBA[c], label=CLASS_LABELS[c]) for c in sorted(CLASS_LABELS)
    ]
    fig.legend(
        handles,
        [CLASS_LABELS[c] for c in sorted(CLASS_LABELS)],
        loc="center right",
        bbox_to_anchor=(1.05, 0.8),
        ncol=1,
        frameon=False,
    )
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2)
    fig.savefig(str(output_path), format="svg", bbox_inches="tight")
    plt.close(fig)


def save_batch_overlays(
    rgb_paths: list[str],
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    output_dir: str | os.PathLike,
) -> None:
    """Save one overlay per patch in a list.

    Args:
        rgb_paths: List of paths to original RGB PNG files.
        pred_masks: Corresponding list of predicted label arrays.
        gt_masks: Corresponding list of ground-truth label arrays.
        output_dir: Directory where overlay_0000.svg, overlay_0001.svg, … are saved.
    """
    output_dir = str(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for idx, (rgb_path, pred, gt) in enumerate(zip(rgb_paths, pred_masks, gt_masks, strict=False)):
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        out_path = str(Path(output_dir) / f"overlay_{idx:04d}.svg")
        save_prediction_overlay(rgb, pred, gt, out_path)


def _prepare_gt(mask: np.ndarray) -> np.ndarray:
    """Map IGNORE_INDEX pixels to the sentinel value used by GT_CMAP."""
    m = mask.copy()
    m[m == IGNORE_INDEX] = _GT_SENTINEL
    return m
