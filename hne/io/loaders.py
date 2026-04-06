"""I/O helpers for the H&E pipeline.

Follows the same conventions as :mod:`multiplex_pipeline.io.loaders`:
  - All functions return typed objects.
  - Errors are logged rather than silently swallowed.
  - tqdm progress bars are used for long-running loads.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from multiplex_pipeline.hne.config import BASE_FEATURES, IN_CHANNELS, N_CLASSES
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_patch_pairs(
    patches_dir: str | os.PathLike,
) -> list[tuple[str, str, str, str]]:
    """Scan a directory and return valid (rgb, hema, eosin, mask) path quads.

    A quad is valid when all four files exist.  This function does **not**
    filter by mask content; use :func:`~multiplex_pipeline.hne.data.dataset.get_valid_pairs`
    for content-based filtering.

    Parameters
    ----------
    patches_dir:
        Directory containing ``patch_*.png``, ``hematoxilin_*.png``,
        ``eosin_*.png``, and ``mask_*.png`` files.

    Returns
    -------
    Sorted list of (rgb_path, hema_path, eosin_path, mask_path) string tuples.
    """
    patches_dir = Path(patches_dir)
    patch_files = sorted(patches_dir.glob("patch_*.png"))

    pairs: list[tuple[str, str, str, str]] = []
    for p in tqdm(patch_files, desc=f"Scanning {patches_dir.name}"):
        hema = patches_dir / p.name.replace("patch_", "hematoxilin_")
        eosin = patches_dir / p.name.replace("patch_", "eosin_")
        mask = patches_dir / p.name.replace("patch_", "mask_")
        if hema.exists() and eosin.exists() and mask.exists():
            pairs.append((str(p), str(hema), str(eosin), str(mask)))
        else:
            logger.warning("Incomplete quad for %s, skipping.", p.name)

    logger.info("Found %d complete patch quads in %s.", len(pairs), patches_dir)
    return pairs


def load_model(
    checkpoint_path: str | os.PathLike,
    device: str | torch.device = "cpu",
    in_channels: int = IN_CHANNELS,
    n_classes: int = N_CLASSES,
    base_features: int = BASE_FEATURES,
) -> nn.Module:
    """Instantiate a UNet and load weights from a saved checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pth`` file saved with ``torch.save(model.state_dict(), ...)``.
    device:
        Device on which to load the model weights.
    in_channels:
        Must match the value used at training time.
    n_classes:
        Must match the value used at training time.
    base_features:
        Must match the value used at training time.

    Returns
    -------
    nn.Module in evaluation mode.

    Raises
    ------
    FileNotFoundError
        If *checkpoint_path* does not exist.
    """
    # Import here to avoid a circular dependency at module level
    from multiplex_pipeline.hne.models.unet import UNet  # noqa: PLC0415

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(device)
    model = UNet(in_channels=in_channels, n_classes=n_classes, base_features=base_features)
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Model loaded from %s on %s.", checkpoint_path, device)
    return model
