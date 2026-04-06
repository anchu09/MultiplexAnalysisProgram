"""PyTorch Dataset for H&E patch loading and on-the-fly augmentation.

Each sample is a quad of files sharing the same numeric suffix:
  - patch_NNNN.png        – original RGB tile
  - hematoxilin_NNNN.png  – hematoxylin OD channel (grayscale uint8)
  - eosin_NNNN.png        – eosin OD channel (grayscale uint8)
  - mask_NNNN.png         – tri-class segmentation mask (0/1/2)

The model receives a 2-channel float32 tensor [hematoxylin, eosin] in [0, 1].
Pixels with mask values outside {0, 1, 2} are mapped to -1 (ignore index).
"""

from __future__ import annotations

import os
import random

import cv2
import numpy as np
import torch
from multiplex_pipeline.hne.config import IGNORE_INDEX, RANDOM_SEED
from torch.utils.data import Dataset


def get_valid_pairs(
    patch_dir: str | os.PathLike,
) -> list[tuple[str, str, str, str]]:
    """Return (rgb, hema, eosin, mask) path quads for all valid patches.

    A quad is valid when all four files exist and the mask contains at least
    one pixel labelled as invasion_front (1) or stroma (2).

    Parameters
    ----------
    patch_dir:
        Directory containing the four file types per patch.

    Returns
    -------
    list of (rgb_path, hema_path, eosin_path, mask_path) string tuples.
    """
    patch_dir = str(patch_dir)
    pairs: list[tuple[str, str, str, str]] = []

    for fname in os.listdir(patch_dir):
        if not (fname.startswith("patch_") and fname.endswith(".png")):
            continue
        rgb = os.path.join(patch_dir, fname)
        hema = os.path.join(patch_dir, fname.replace("patch_", "hematoxilin_"))
        eos = os.path.join(patch_dir, fname.replace("patch_", "eosin_"))
        msk = os.path.join(patch_dir, fname.replace("patch_", "mask_"))

        if not all(os.path.exists(p) for p in (rgb, hema, eos, msk)):
            continue

        mask = cv2.imread(msk, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask_clean = np.where(np.isin(mask, [0, 1, 2]), mask, IGNORE_INDEX)
        if {1, 2} & set(np.unique(mask_clean)):
            pairs.append((rgb, hema, eos, msk))

    return pairs


class PatchDataset(Dataset):
    """Dataset returning (2-channel float tensor, long mask tensor) pairs.

    Parameters
    ----------
    pairs:
        List of (rgb, hema, eosin, mask) path tuples as returned by
        :func:`get_valid_pairs`.
    augment:
        If True, apply random flips, 90° rotations, intensity jitter,
        and Gaussian noise on-the-fly during training.
    seed:
        Base random seed; per-sample randomness is not seeded further to
        preserve stochasticity across epochs.
    """

    def __init__(
        self,
        pairs: list[tuple[str, str, str, str]],
        augment: bool = False,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.pairs = pairs
        self.augment = augment
        random.seed(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_f, hema_f, eosin_f, mask_f = self.pairs[idx]

        h = cv2.imread(hema_f, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        e = cv2.imread(eosin_f, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        m = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)

        # Align mask spatial dimensions to the channel images
        if h.shape != m.shape:
            m = cv2.resize(m, (h.shape[1], h.shape[0]), interpolation=cv2.INTER_NEAREST)

        m = np.where(np.isin(m, [0, 1, 2]), m, IGNORE_INDEX)

        ch = np.stack([h, e], axis=0)  # (2, H, W)

        if self.augment:
            ch, m = self._augment(ch, m)

        ch = np.ascontiguousarray(ch, dtype=np.float32)
        m = np.ascontiguousarray(m)
        return torch.from_numpy(ch), torch.from_numpy(m).long()

    @staticmethod
    def _augment(
        ch: np.ndarray,
        m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if random.random() > 0.5:
            ch, m = ch[:, :, ::-1], m[:, ::-1]
        if random.random() > 0.5:
            ch, m = ch[:, ::-1, :], m[::-1, :]
        k = random.randint(0, 3)
        if k:
            ch = np.rot90(ch, k, axes=(1, 2))
            m = np.rot90(m, k)
        delta = np.float32(random.uniform(-0.1, 0.1))
        ch = np.clip(ch + delta, 0.0, 1.0)
        noise = np.random.normal(0, 0.02, size=ch.shape).astype(np.float32)
        ch = np.clip(ch + noise, 0.0, 1.0)
        return ch, m
