"""H&E stain-space decomposition utilities.

Converts RGB patches to separate hematoxylin and eosin optical-density
channels, giving the model explicit nuclear and cytoplasmic/stromal signals.
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2hed


def rgb_to_hed(rgb_patch: np.ndarray) -> np.ndarray:
    """Decompose an RGB H&E image into hematoxylin, eosin, and DAB channels.

    Parameters
    ----------
    rgb_patch:
        uint8 array of shape (H, W, 3) with values in [0, 255].

    Returns
    -------
    np.ndarray
        Float32 array of shape (H, W, 3) containing optical-density values
        for the H, E, and D channels respectively.
    """
    patch_float = rgb_patch.astype(np.float32) / 255.0
    hed = rgb2hed(patch_float)
    return hed.astype(np.float32)


def normalize_stain_channels(hed_patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize hematoxylin and eosin channels to uint8 [0, 255].

    Min-max normalisation is applied independently to each channel so that
    the full dynamic range is used regardless of staining variability.

    Parameters
    ----------
    hed_patch:
        Float32 array of shape (H, W, 3) as returned by :func:`rgb_to_hed`.

    Returns
    -------
    hematoxylin_uint8 : np.ndarray
        uint8 array (H, W) for the hematoxylin channel.
    eosin_uint8 : np.ndarray
        uint8 array (H, W) for the eosin channel.
    """
    h_channel = hed_patch[:, :, 0]
    e_channel = hed_patch[:, :, 1]
    return _normalize_to_uint8(h_channel), _normalize_to_uint8(e_channel)


def decompose_patch(
    rgb_patch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full decomposition pipeline: RGB → (hematoxylin_u8, eosin_u8, hed_float).

    Parameters
    ----------
    rgb_patch:
        uint8 array of shape (H, W, 3).

    Returns
    -------
    hematoxylin_uint8 : np.ndarray  (H, W) uint8
    eosin_uint8 : np.ndarray        (H, W) uint8
    hed_float : np.ndarray          (H, W, 3) float32
    """
    hed = rgb_to_hed(rgb_patch)
    h_u8, e_u8 = normalize_stain_channels(hed)
    return h_u8, e_u8, hed


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx > mn:
        norm = (arr - mn) / (mx - mn)
    else:
        norm = np.zeros_like(arr)
    return (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
