import logging
import re

import numpy as np
import pandas as pd
from multiplex_pipeline.config import DAPI_CONNECTIVITY, PIXEL_AREA, ROI_PATTERN
from multiplex_pipeline.schema import (
    AREA_PIXELS,
    AREA_UM2,
    CENTROID_COL,
    CENTROID_ROW,
    DAPI_ID,
    ROI,
    dapi_key,
)
from multiplex_pipeline.utils.validation import is_binary
from skimage import measure
from tqdm import tqdm

logger = logging.getLogger(__name__)

# col_base values (post-sanitization) that require a binary mask rather than mean intensity.
_MASK_MARKERS_NGFR: frozenset[str] = frozenset({"NGFR"})
_MASK_MARKERS_CK: frozenset[str] = frozenset({"Pan_Cytokeratin_CK"})


def extract_roi_key(img_name: str) -> str | None:
    """Extract the ROI key from an image name, e.g. 'ROI1.ome.tiff' → 'roi1'."""
    m = ROI_PATTERN.search(img_name)
    return m.group(1).lower() if m else None


def label_dapi(mask: np.ndarray) -> np.ndarray:
    """Label a binary DAPI mask; return unchanged if already labeled."""
    if len(np.unique(mask)) <= 2:
        return measure.label(mask, connectivity=DAPI_CONNECTIVITY)
    return mask


def get_labels_and_counts(
    labeled_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid labels, their pixel counts, and the flattened mask."""
    flat = labeled_mask.ravel()
    counts = np.bincount(flat)
    labels = np.arange(len(counts))
    valid = (labels != 0) & (counts > 0)
    return labels[valid], counts[valid], flat


def get_centroids_map(labeled_mask: np.ndarray) -> dict[int, tuple]:
    """Return a mapping of label → centroid from regionprops."""
    return {p.label: p.centroid for p in measure.regionprops(labeled_mask)}


def compute_mean_intensities(
    mask_flat: np.ndarray,
    img_channel: np.ndarray,
    valid_labels: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Compute per-region mean intensities via bincount."""
    sums = np.bincount(mask_flat, weights=img_channel.ravel())
    return sums[valid_labels] / counts


def compute_binary_flags(
    valid_labels: np.ndarray,
    mask_flat: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Return a binary flag (0/1) indicating whether each label overlaps the mask.

    Raises:
        ValueError: If mask is not binary.
    """
    if not is_binary(mask, "mask"):
        raise ValueError(f"Expected a binary mask but got unique values: {np.unique(mask)}")
    flat = mask.ravel()
    positive = np.unique(mask_flat[flat > 0])
    return np.isin(valid_labels, positive).astype(int)


def process_roi(
    img_name: str,
    img_data: np.ndarray,
    dapi_masks: dict[str, np.ndarray],
    ck_masks: dict[str, np.ndarray],
    ngfr_masks: dict[str, np.ndarray],
    channels: list[int],
    marker_dict: dict[int, str],
    pixel_area_um2: float = PIXEL_AREA,
) -> pd.DataFrame | None:
    """Process one ROI image to extract per-cell intensities and binary flags.

    Returns None (with a warning) if a required CK/NGFR mask is absent for
    the extracted ROI key.

    Raises:
        KeyError: If the ROI key cannot be extracted from img_name, or the
            DAPI mask key is missing from dapi_masks.
        ValueError: If image and DAPI mask spatial dimensions do not match.
    """
    roi = extract_roi_key(img_name)
    if roi is None:
        raise KeyError(f"Cannot extract ROI key from image name '{img_name}'.")

    dapi_mask_key = dapi_key(roi)
    if dapi_mask_key not in dapi_masks:
        raise KeyError(f"DAPI mask '{dapi_mask_key}' not found in dapi_masks.")

    mask_dapi = dapi_masks[dapi_mask_key]
    if mask_dapi.shape != img_data.shape[1:]:
        raise ValueError(
            f"Shape mismatch for '{img_name}': "
            f"DAPI mask {mask_dapi.shape} vs image {img_data.shape[1:]}."
        )

    lbl = label_dapi(mask_dapi)
    valid_labels, counts, flat = get_labels_and_counts(lbl)
    cent_map = get_centroids_map(lbl)

    df = pd.DataFrame(
        {
            ROI: roi,
            DAPI_ID: valid_labels,
            AREA_PIXELS: counts,
            AREA_UM2: counts * pixel_area_um2,
            CENTROID_ROW: [cent_map[cell_id][0] for cell_id in valid_labels],
            CENTROID_COL: [cent_map[cell_id][1] for cell_id in valid_labels],
        }
    )

    col_bases = {
        ch: re.sub(r"[^a-zA-Z0-9]+", "_", marker_dict.get(ch, f"Ch{ch}")).strip("_")
        for ch in channels
    }

    # Pre-validate that required masks exist before processing any channel.
    for ch in channels:
        cb = col_bases[ch]
        if cb in _MASK_MARKERS_NGFR and roi not in ngfr_masks:
            logger.warning(
                "NGFR mask missing for ROI '%s'. Available ROIs: %s",
                roi,
                list(ngfr_masks.keys()),
            )
            return None
        if cb in _MASK_MARKERS_CK and roi not in ck_masks:
            logger.warning(
                "CK mask missing for ROI '%s'. Available ROIs: %s",
                roi,
                list(ck_masks.keys()),
            )
            return None

    for ch in tqdm(channels, desc=f"Channels {roi}", unit="ch"):
        col_base = col_bases[ch]
        img_ch = img_data[ch]

        if col_base in _MASK_MARKERS_NGFR:
            df[f"mean_intensity_{col_base}"] = compute_mean_intensities(
                flat, img_ch, valid_labels, counts
            )
            df[f"is_positive_{col_base}"] = compute_binary_flags(
                valid_labels, flat, ngfr_masks[roi]
            )
        elif col_base in _MASK_MARKERS_CK:
            df[f"is_positive_{col_base}"] = compute_binary_flags(valid_labels, flat, ck_masks[roi])
        else:
            df[f"mean_intensity_{col_base}"] = compute_mean_intensities(
                flat, img_ch, valid_labels, counts
            )

    return df


def intensity_to_binary(
    df: pd.DataFrame,
    thresholds: dict[str, float],
    exclude: list[str] | None = None,
) -> pd.DataFrame:
    """Convert intensity columns to binary values using per-ROI thresholds.

    Thresholds are computed independently per ROI to correct for batch effects
    and illumination differences between acquisitions. For each marker column m:

        threshold = ROI_mean(m) + thresholds[m] * ROI_std(m)

    A value of 0.0 sets the threshold at the ROI mean; positive values push it
    above mean (stricter). If m is not in thresholds, 0.0 is used with a warning.
    """
    exclude = exclude or [ROI, DAPI_ID, AREA_PIXELS, AREA_UM2, CENTROID_COL, CENTROID_ROW]
    markers = [c for c in df if c not in exclude]
    means = df.groupby(ROI)[markers].mean()
    stds = df.groupby(ROI)[markers].std()
    dfb = df.join(means, on=ROI, rsuffix="_mean").join(stds, on=ROI, rsuffix="_std")

    for m in markers:
        if m not in thresholds:
            logger.warning(
                "No threshold defined for marker '%s'. Using default 0.0 (threshold at ROI mean).",
                m,
            )
        k = thresholds.get(m, 0.0)
        dfb[f"{m}_threshold"] = dfb[f"{m}_mean"] + k * dfb[f"{m}_std"]
        dfb[f"{m}_binary"] = (dfb[m] > dfb[f"{m}_threshold"]).astype(int)

    cols = [ROI, DAPI_ID, AREA_PIXELS, AREA_UM2, CENTROID_COL, CENTROID_ROW] + [
        f"{m}_binary" for m in markers
    ]
    return dfb[cols]
