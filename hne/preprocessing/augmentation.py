"""Data augmentation for H&E patch-mask pairs.

Augmentations are applied consistently to the RGB patch, the hematoxylin
channel, the eosin channel, and the segmentation mask so that spatial
correspondence is preserved.
"""

from __future__ import annotations

import random

import numpy as np

_OPS = ["rot90", "rot180", "rot270", "flip_h", "flip_v"]


def augment_patch(
    rgb: np.ndarray,
    hema: np.ndarray,
    eosin: np.ndarray,
    mask: np.ndarray,
    n_augments: int = 1,
    seed: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Generate augmented copies of a patch/mask quad.

    Each augmentation applies one randomly chosen geometric transformation
    (90°/180°/270° rotation or horizontal/vertical flip) to all four arrays.

    Parameters
    ----------
    rgb:
        uint8 (H, W, 3) RGB patch.
    hema:
        uint8 (H, W) hematoxylin channel.
    eosin:
        uint8 (H, W) eosin channel.
    mask:
        uint8 (H, W) segmentation mask.
    n_augments:
        Number of augmented copies to produce.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    list of (rgb, hema, eosin, mask) tuples, each with the same dtypes and
    shapes as the inputs.
    """
    if seed is not None:
        random.seed(seed)

    results = []
    for _ in range(n_augments):
        op = random.choice(_OPS)
        aug_rgb, aug_hema, aug_eosin, aug_mask = _apply_op(rgb, hema, eosin, mask, op)
        results.append((aug_rgb, aug_hema, aug_eosin, aug_mask))
    return results


def _apply_op(
    rgb: np.ndarray,
    hema: np.ndarray,
    eosin: np.ndarray,
    mask: np.ndarray,
    op: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if op.startswith("rot"):
        k = int(op.replace("rot", "")) // 90
        aug_rgb = np.rot90(rgb, k)
        aug_hema = np.rot90(hema, k)
        aug_eos = np.rot90(eosin, k)
        aug_mask = np.rot90(mask, k)
    elif op == "flip_h":
        aug_rgb = np.fliplr(rgb)
        aug_hema = np.fliplr(hema)
        aug_eos = np.fliplr(eosin)
        aug_mask = np.fliplr(mask)
    else:  # flip_v
        aug_rgb = np.flipud(rgb)
        aug_hema = np.flipud(hema)
        aug_eos = np.flipud(eosin)
        aug_mask = np.flipud(mask)

    return (
        np.ascontiguousarray(aug_rgb),
        np.ascontiguousarray(aug_hema),
        np.ascontiguousarray(aug_eos),
        np.ascontiguousarray(aug_mask),
    )
