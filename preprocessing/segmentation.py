import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from multiplex_pipeline.config import CLOSING_DISK_RADIUS
from multiplex_pipeline.preprocessing.types import ChannelMaskSettings, PostProcessing
from multiplex_pipeline.schema import dapi_key
from multiplex_pipeline.utils.helpers import extract_roi_number, in_jupyter
from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects

logger = logging.getLogger(__name__)


def post_process_mask(
    mask: np.ndarray, min_size: int = 0, max_hole_size: int = 10000
) -> np.ndarray:
    """Fill small holes and remove small objects from a binary mask."""
    mask_bool = mask.astype(bool)

    if max_hole_size > 0:
        mask_bool = remove_small_holes(mask_bool, area_threshold=max_hole_size)

    if min_size > 0:
        # skimage remove_small_objects keeps objects >= min_size and removes smaller ones
        mask_bool = remove_small_objects(mask_bool, min_size=min_size)

    return mask_bool


def post_process_mask_closing(
    mask: np.ndarray, min_size: int = 0, max_hole_size: int = 10000
) -> np.ndarray:
    """Apply a large morphological closing to unify fragmented structures.

    min_size and max_hole_size match the signature of post_process_mask so both
    functions can be placed in the same post_process_funcs list. They have no
    effect here — closing radius is controlled by the module constant
    CLOSING_DISK_RADIUS.
    """
    mask_bool = mask.astype(bool)
    selem = disk(CLOSING_DISK_RADIUS)
    return closing(mask_bool, selem)


_POST_PROCESS_MAP = {
    PostProcessing.STANDARD: post_process_mask,
    PostProcessing.CLOSING: post_process_mask_closing,
}


def generate_initial_mask(
    channel_data: np.ndarray,
    score: float,
    scaling_divisor: float,
) -> np.ndarray:
    """Compute the initial binary mask by thresholding channel data.

    Threshold = mean + (score / scaling_divisor) * 3 * std
    """
    m, s = channel_data.mean(), channel_data.std()
    thresh = m + (score / scaling_divisor) * 3 * s
    return channel_data > thresh


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
        key = dapi_key(f"roi{roi_num}")
        dmask = dapi_masks.get(key)
        if dmask is None:
            logger.warning("Skipping %s: missing DAPI mask '%s'.", name, key)
            return None
        return initial_mask & (dmask > 0)

    return initial_mask


def save_mask(
    mask: np.ndarray,
    base_folder: str | Path,
    name: str,
    roi_num: str,
    filename: str,
) -> None:
    """Save a binary mask to disk under '{base_folder}/{name} - ROI{roi_num}/'."""
    roi_folder = Path(base_folder) / f"{name} - ROI{roi_num}"
    roi_folder.mkdir(parents=True, exist_ok=True)
    out_path = roi_folder / filename
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
    if in_jupyter():
        plt.show()
    else:
        plt.close(fig)


def create_channel_masks(
    images_dict: dict[str, np.ndarray],
    dapi_masks_dict: dict[str, np.ndarray],
    settings: ChannelMaskSettings,
    show: bool = False,
) -> dict[str, np.ndarray]:
    """Generate, post-process, and save channel masks for all images.

    Args:
        images_dict: Mapping of filename → (C, H, W) image array.
        dapi_masks_dict: DAPI labeled masks keyed by 'roi{N}_dapi'.
        settings: All channel-specific parameters (thresholds, morphology, I/O).
        show: Display QC figures per mask in Jupyter (default False).

    Returns:
        Dict of processed binary masks indexed by ROI key (e.g. 'roi1').

    Raises:
        KeyError: If settings.user_scores is a dict and an image name is absent.
    """
    out_masks: dict[str, np.ndarray] = {}

    for name, image in images_dict.items():
        roi_num = extract_roi_number(name)
        if roi_num is None:
            logger.warning("Skipping %s: cannot extract ROI number.", name)
            continue

        chan = image[settings.channel_index].astype(float)

        if isinstance(settings.user_scores, dict):
            if name not in settings.user_scores:
                raise KeyError(
                    f"No threshold score defined for image '{name}' in user_scores. "
                    f"Available keys: {list(settings.user_scores.keys())}"
                )
            score_val = settings.user_scores[name]
        else:
            score_val = settings.user_scores

        initial_mask = generate_initial_mask(chan, score_val, settings.scaling_divisor)

        masked = apply_dapi_mask(
            initial_mask, roi_num, dapi_masks_dict, settings.require_dapi, name
        )
        if masked is None:
            continue

        processed = masked.copy()
        for fn in [_POST_PROCESS_MAP[p] for p in settings.post_process_funcs]:
            processed = fn(
                processed, min_size=settings.min_size, max_hole_size=settings.max_hole_size
            )

        out_key = f"roi{roi_num}"
        out_masks[out_key] = processed
        save_mask(processed, settings.base_folder_path, name, roi_num, settings.mask_filename)

        if show:
            display_masks(
                chan,
                masked,
                processed,
                settings.brightness_factor,
                settings.mask_label,
                settings.channel_index,
                name,
                settings.min_size,
                settings.max_hole_size,
            )

    return out_masks
