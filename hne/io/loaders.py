"""Patch-pair file discovery for the H&E pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from multiplex_pipeline.hne.config import EOSIN_PREFIX, HEMATOXYLIN_PREFIX
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_patch_pairs(
    patches_dir: str | os.PathLike,
) -> list[tuple[str, str, str, str]]:
    """Scan a directory and return valid (rgb, hema, eosin, mask) path quads.

    A quad is valid when all four files exist. Does not filter by mask content;
    use `get_valid_pairs` for content-based filtering.

    Args:
        patches_dir: Directory containing patch_*.png, hematoxylin_*.png,
            eosin_*.png, and mask_*.png files.

    Returns:
        Sorted list of (rgb_path, hema_path, eosin_path, mask_path) string tuples.
    """
    patches_dir = Path(patches_dir)
    patch_files = sorted(patches_dir.glob("patch_*.png"))

    pairs: list[tuple[str, str, str, str]] = []
    for p in tqdm(patch_files, desc=f"Scanning {patches_dir.name}"):
        hema = patches_dir / p.name.replace("patch_", HEMATOXYLIN_PREFIX + "_")
        eosin = patches_dir / p.name.replace("patch_", EOSIN_PREFIX + "_")
        mask = patches_dir / p.name.replace("patch_", "mask_")
        if hema.exists() and eosin.exists() and mask.exists():
            pairs.append((str(p), str(hema), str(eosin), str(mask)))
        else:
            logger.warning("Incomplete quad for %s, skipping.", p.name)

    logger.info("Found %d complete patch quads in %s.", len(pairs), patches_dir)
    return pairs
