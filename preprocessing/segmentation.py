import logging
import os
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from multiplex_pipeline.utils.helpers import _in_jupyter, extract_roi_number
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects

logger = logging.getLogger(__name__)


# Structural element radius for morphological closing (pixels).
_CLOSING_DISK_RADIUS = 20


def post_process_mask(
    mask: np.ndarray, min_size: int = 0, max_hole_size: int = 10000
) -> np.ndarray:
    """Fill small holes and remove small objects from a binary mask."""
    mask_bool = mask.astype(bool)

    if max_hole_size > 0:
        mask_bool = remove_small_holes(mask_bool, max_size=max_hole_size)
        logger.debug("Holes ≤ %d px filled.", max_hole_size)

    if min_size > 0:
        mask_bool = remove_small_objects(mask_bool, max_size=min_size)
        logger.debug("Objects < %d px removed.", min_size)

    return mask_bool


def post_process_mask_closing(
    mask: np.ndarray, min_size: int = 0, max_hole_size: int = 10000
) -> np.ndarray:
    """Apply a large morphological closing to unify structures.

    min_size and max_hole_size are accepted for API compatibility but ignored —
    closing is controlled by _CLOSING_DISK_RADIUS.
    """
    mask_bool = mask.astype(bool)
    selem = disk(_CLOSING_DISK_RADIUS)
    closed = binary_closing(mask_bool, selem)
    logger.debug("Performed closing (disk radius=%d).", _CLOSING_DISK_RADIUS)
    return closed


def generate_initial_mask(
    channel_data: np.ndarray,
    score: dict[str, float] | float,
    scaling_divisor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the initial binary mask by thresholding channel data.

    Threshold = mean + (score / scaling_divisor) * 3 * std
    """
    m, s = channel_data.mean(), channel_data.std()
    thresh = m + (score / scaling_divisor) * 3 * s
    initial_mask = channel_data > thresh
    return channel_data, initial_mask


def apply_dapi_mask(
    initial_mask: np.ndarray,
    roi_num: str,
    dapi_masks: dict[str, np.ndarray],
    require_dapi: bool,
    name: str,
) -> np.ndarray | None:
    """Intersect the initial mask with the DAPI ROI mask if required.

    Returns None if the DAPI mask is missing and require_dapi is True.
    """
    if require_dapi:
        key = f"roi{roi_num}_dapi"
        dmask = dapi_masks.get(key)
        if dmask is None:
            logger.warning("Skipping %s: missing DAPI mask '%s'.", name, key)
            return None
        return initial_mask & (dmask > 0)

    return initial_mask


def save_mask(
    mask: np.ndarray,
    base_folder: str,
    name: str,
    roi_num: str,
    filename: str,
) -> None:
    """Save a binary mask to disk under '{base_folder}/{name} - ROI{roi_num}/'."""
    roi_folder = os.path.join(base_folder, f"{name} - ROI{roi_num}")
    os.makedirs(roi_folder, exist_ok=True)
    out_path = os.path.join(roi_folder, filename)
    tifffile.imwrite(out_path, mask.astype(np.uint8))
    logger.info("Saved: %s", out_path)


def display_masks(
    channel: np.ndarray,
    initial_mask: np.ndarray,
    processed_mask: np.ndarray,
    brightness_factor: float | None,
    mask_label: str,
    channel_index: int,
    name: str,
    min_size: int,
    max_hole_size: int,
) -> None:
    """Display raw channel, initial mask, and post-processed mask side by side."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    disp = channel * brightness_factor if brightness_factor else channel
    axs[0].imshow(np.clip(disp, 0, channel.max()), cmap="gray")
    axs[0].set_title(f"{name}: {mask_label} (ch {channel_index})")
    axs[0].axis("off")
    axs[1].imshow(initial_mask, cmap="gray")
    axs[1].set_title("Initial mask")
    axs[1].axis("off")
    axs[2].imshow(processed_mask, cmap="gray")
    axs[2].set_title(f"Post-processed (min={min_size}, max_hole={max_hole_size})")
    axs[2].axis("off")
    plt.tight_layout()
    if _in_jupyter():
        plt.show()
    else:
        plt.close(fig)


def create_channel_masks(
    images_dict: dict[str, np.ndarray],
    dapi_masks_dict: dict[str, np.ndarray],
    channel_index: int,
    user_scores: dict[str, float] | float,
    scaling_divisor: float,
    base_folder_path: str,
    min_size: int,
    max_hole_size: int,
    mask_label: str,
    mask_filename: str,
    post_process_funcs: list[Callable[[np.ndarray, int, int], np.ndarray]],
    brightness_factor: float | None = None,
    require_dapi: bool = False,
) -> dict[str, np.ndarray]:
    """Generate, post-process, save, and display channel masks.

    user_scores: if a dict, keys must match image names; values in [0, 3]
    where threshold = mean + (score/divisor)*3*std.

    Returns:
        Dictionary of processed masks indexed by ROI key (e.g. 'roi1').

    Raises:
        KeyError: If user_scores is a dict and the image name is not present.
    """
    out_masks: dict[str, np.ndarray] = {}

    for name, image in images_dict.items():
        roi_num = extract_roi_number(name)
        if roi_num is None:
            logger.warning("Skipping %s: cannot extract ROI number.", name)
            continue

        chan = image[channel_index].astype(float)

        if isinstance(user_scores, dict):
            if name not in user_scores:
                raise KeyError(
                    f"No threshold score defined for image '{name}' in user_scores. "
                    f"Available keys: {list(user_scores.keys())}"
                )
            score_val = user_scores[name]
        else:
            score_val = user_scores

        chan, initial_mask = generate_initial_mask(chan, score_val, scaling_divisor)

        masked = apply_dapi_mask(initial_mask, roi_num, dapi_masks_dict, require_dapi, name)
        if masked is None:
            continue

        processed = masked.copy()
        for fn in post_process_funcs:
            processed = fn(processed, min_size=min_size, max_hole_size=max_hole_size)

        out_key = f"roi{roi_num}"
        out_masks[out_key] = processed
        save_mask(processed, base_folder_path, name, roi_num, mask_filename)

        display_masks(
            chan,
            masked,
            processed,
            brightness_factor,
            mask_label,
            channel_index,
            name,
            min_size,
            max_hole_size,
        )

    return out_masks
