"""Dataset balancing for H&E patch sets.

Two complementary steps bring class balance to an imbalanced patch corpus:

1. Oversampling – patches containing a minimum fraction of tumor pixels
   are duplicated with geometric augmentations to reach a target multiplier.
2. Purging – stroma-only patches (no tumor pixels) are removed until
   the total stroma pixel count no longer exceeds the total tumor pixel count.

Together these steps approximate equal class representation without discarding
patches that contain biologically valuable transition zones.
"""

from __future__ import annotations

import logging
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from multiplex_pipeline.hne.config import (
    MAX_CPU_WORKERS,
    MIN_TUMOR_FRACTION,
    OVERSAMPLE_FACTOR,
    RANDOM_SEED,
)
from multiplex_pipeline.hne.preprocessing.augmentation import augment_patch
from multiplex_pipeline.hne.preprocessing.color_decomposition import decompose_patch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def oversample_tumor_patches(
    src_dir: Path,
    dst_dir: Path,
    min_tumor_fraction: float = MIN_TUMOR_FRACTION,
    oversample_factor: int = OVERSAMPLE_FACTOR,
    seed: int = RANDOM_SEED,
) -> None:
    """Copy all patches to dst_dir and augment tumor-containing ones.

    For every source patch the RGB, hematoxylin, eosin, and mask files are
    written to dst_dir. Patches whose masks contain more than
    min_tumor_fraction tumor pixels are additionally duplicated
    oversample_factor - 1 times with random geometric augmentations.

    Parameters
    ----------
    src_dir:
        Directory containing patch_NNNN.png and mask_NNNN.png files.
    dst_dir:
        Destination directory (created if it does not exist).
    min_tumor_fraction:
        Minimum fraction of tumor pixels (class 1) required to oversample.
    oversample_factor:
        How many total copies (original + augmented) to generate per tumor patch.
    seed:
        Random seed used for augmentation sampling.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    patch_paths = sorted(src_dir.glob("patch_*.png"))
    mask_paths = sorted(src_dir.glob("mask_*.png"))
    if len(patch_paths) != len(mask_paths):
        raise ValueError("Number of patch files and mask files does not match.")

    pairs = list(zip(patch_paths, mask_paths, strict=False))
    logger.info("Processing %d original patches → %s", len(pairs), dst_dir)

    with ThreadPoolExecutor(max_workers=MAX_CPU_WORKERS) as pool:
        futures = [pool.submit(_process_original, p, m, dst_dir) for p, m in pairs]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Copying originals"):
            _.result()  # raise any exception from the worker thread

    tumor_pairs = [(p, m) for p, m in pairs if _tumor_fraction(m) >= min_tumor_fraction]
    logger.info("Found %d tumor-containing patches for oversampling.", len(tumor_pairs))

    existing_nums = [int(p.stem.split("_")[1]) for p in patch_paths]
    max_num = max(existing_nums) if existing_nums else 0

    n_extra = len(tumor_pairs) * (oversample_factor - 1)
    augment_tasks = [(random.choice(tumor_pairs), max_num + i + 1) for i in range(n_extra)]

    with ThreadPoolExecutor(max_workers=MAX_CPU_WORKERS) as pool:
        futures = [
            pool.submit(_process_augmented, p, m, new_num, dst_dir)
            for (p, m), new_num in augment_tasks
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Augmenting tumor patches"):
            _.result()  # raise any exception from the worker thread

    logger.info("Oversampling complete. %d patches in %s.", len(pairs) + n_extra, dst_dir)


def balance_dataset(
    src_dir: Path,
    dst_dir: Path,
    seed: int = RANDOM_SEED,
) -> None:
    """Remove pure-stroma patches until stroma pixel count ≤ tumor pixel count.

    Selected patches are copied (with sequential renaming) to dst_dir.

    Parameters
    ----------
    src_dir:
        Directory produced by :func:`oversample_tumor_patches`.
    dst_dir:
        Output directory for the balanced patch set.
    seed:
        Random seed used to shuffle the final kept-patch order.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    mask_paths = sorted(src_dir.glob("mask_*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No masks found in {src_dir}")

    id_to_counts: dict[str, dict[int, int]] = {}
    for m_path in tqdm(mask_paths, desc="Reading masks"):
        mask = cv2.imread(str(m_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            logger.warning("Could not read %s, skipping.", m_path.name)
            continue
        ident = m_path.stem[len("mask_") :]
        id_to_counts[ident] = {
            0: int((mask == 0).sum()),
            1: int((mask == 1).sum()),
            2: int((mask == 2).sum()),
        }

    totals = {cls: sum(c[cls] for c in id_to_counts.values()) for cls in (0, 1, 2)}
    threshold_stroma = totals[1]
    logger.info(
        "Pixel totals → Background: %d  Tumor: %d  Stroma: %d",
        totals[0],
        totals[1],
        totals[2],
    )

    candidates = [
        ident for ident, c in id_to_counts.items() if c[2] > c[0] and c[2] > c[1] and c[1] == 0
    ]
    candidates.sort(
        key=lambda i: id_to_counts[i][2] - max(id_to_counts[i][0], id_to_counts[i][1]),
        reverse=True,
    )

    removed: set[str] = set()
    curr_stroma = totals[2]
    for ident in candidates:
        if curr_stroma <= threshold_stroma:
            break
        removed.add(ident)
        curr_stroma -= id_to_counts[ident][2]

    logger.info("Removing %d stroma-dominant patches.", len(removed))

    selected = [i for i in id_to_counts if i not in removed]
    random.shuffle(selected)
    logger.info("Copying %d balanced patches to %s", len(selected), dst_dir)

    with ThreadPoolExecutor(max_workers=MAX_CPU_WORKERS) as pool:
        futures = [
            pool.submit(_copy_quad, ident, new_idx, src_dir, dst_dir)
            for new_idx, ident in enumerate(selected)
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Copying balanced patches"):
            _.result()  # raise any exception from the worker thread

    logger.info("Balancing complete.")


def _tumor_fraction(mask_path: Path) -> float:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return 0.0
    return float((mask == 1).sum()) / mask.size


def _process_original(patch_path: Path, mask_path: Path, dst_dir: Path) -> None:
    """Decompose a patch and write all four files to dst_dir."""
    img_bgr = cv2.imread(str(patch_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if img_bgr is None or mask is None:
        logger.warning("Could not read %s, skipping.", patch_path.name)
        return
    if np.unique(mask).size == 1:
        return  # skip uniform masks

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_u8, e_u8, _ = decompose_patch(rgb)

    num = patch_path.stem.split("_")[1]
    _write_quad(rgb, h_u8, e_u8, mask, num, dst_dir)


def _process_augmented(patch_path: Path, mask_path: Path, new_num: int, dst_dir: Path) -> None:
    """Apply one random augmentation and write the result to dst_dir."""
    img_bgr = cv2.imread(str(patch_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if img_bgr is None or mask is None:
        return

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    [(aug_rgb, _, _, aug_mask)] = augment_patch(rgb, rgb[:, :, 0], rgb[:, :, 0], mask, n_augments=1)
    h_u8, e_u8, _ = decompose_patch(aug_rgb)
    _write_quad(aug_rgb, h_u8, e_u8, aug_mask, str(new_num), dst_dir)


def _write_quad(
    rgb: np.ndarray,
    hema: np.ndarray,
    eosin: np.ndarray,
    mask: np.ndarray,
    num: str,
    dst_dir: Path,
) -> None:
    cv2.imwrite(str(dst_dir / f"patch_{num}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(dst_dir / f"hematoxilin_{num}.png"), hema)
    cv2.imwrite(str(dst_dir / f"eosin_{num}.png"), eosin)
    cv2.imwrite(str(dst_dir / f"mask_{num}.png"), mask)


def _copy_quad(ident: str, new_idx: int, src_dir: Path, dst_dir: Path) -> None:
    num_str = str(new_idx).zfill(4)
    for prefix in ("patch", "hematoxilin", "eosin", "mask"):
        src = src_dir / f"{prefix}_{ident}.png"
        dst = dst_dir / f"{prefix}_{num_str}.png"
        if src.exists():
            shutil.copyfile(str(src), str(dst))
