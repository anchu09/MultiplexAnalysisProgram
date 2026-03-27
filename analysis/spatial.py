import logging
import os

import numpy as np
import pandas as pd
from multiplex_pipeline.config import PIXEL_AREA, PIXEL_SIZE
from multiplex_pipeline.schema import (
    CELLS_PER_UM2_CK,
    CELLS_PER_UM2_CK_NGFR,
    CENTROID_COL,
    CENTROID_ROW,
    CK_NGFR_POSITIVE_AREA_UM2,
    CK_POSITIVE_AREA_UM2,
    DAPI_ID,
    DISTANCE_PX,
    ROI,
    SUBPOP_CELL_COUNT,
    TOTAL_AREA_ROI_UM2,
)
from scipy.spatial import cKDTree
from skimage.measure import regionprops

logger = logging.getLogger(__name__)


def compute_mask_area_summary(
    ck_masks: dict,
    ngfr_masks: dict,
    pixel_area: float = PIXEL_AREA,
) -> pd.DataFrame:
    """Compute CK and CK+NGFR area summary per ROI.

    ROIs with missing or shape-mismatched masks are skipped with a warning.
    """
    data = []
    for roi in ck_masks:
        ck_mask = ck_masks.get(roi)
        ngfr_mask = ngfr_masks.get(roi)

        if ck_mask is None or ngfr_mask is None:
            logger.warning("Skipping ROI '%s': CK or NGFR mask is None.", roi)
            continue

        if ck_mask.shape != ngfr_mask.shape:
            logger.warning(
                "Skipping ROI '%s': shape mismatch CK %s vs NGFR %s.",
                roi,
                ck_mask.shape,
                ngfr_mask.shape,
            )
            continue

        data.append(
            {
                ROI: roi,
                CK_POSITIVE_AREA_UM2: np.sum(ck_mask == 1) * pixel_area,
                CK_NGFR_POSITIVE_AREA_UM2: np.sum((ck_mask == 1) & (ngfr_mask == 1)) * pixel_area,
                TOTAL_AREA_ROI_UM2: ck_mask.size * pixel_area,
            }
        )

    return pd.DataFrame(data)


def compute_subpop_cells_per_area(
    df_binary: pd.DataFrame,
    subpop_conditions: list,
    cond_map: dict,
    mask_summary: pd.DataFrame,
    rois: list,
    out_dir: str,
    roi_col: str = ROI,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute subpopulation cell counts and densities per ROI.

    Parameters are non-obvious, so documented here:
    - subpop_conditions: e.g. ['CK_mask+', 'CD3_intensity+']
    - cond_map: maps condition names to DataFrame column names
    - mask_summary: per-ROI area DataFrame from compute_mask_area_summary

    Returns:
        Tuple of (summary_df, summary_fmt) where summary_fmt has formatted strings
        for CSV export.

    Raises:
        ValueError: If no ROIs have data after filtering.
    """
    parsed: dict = {}
    for cond in subpop_conditions:
        val = 1 if cond.strip().endswith("+") else 0
        key = cond.strip()[:-1].strip()
        col = cond_map.get(key)
        if col:
            parsed[col] = val
        else:
            logger.warning("No mapping found for condition '%s' — skipping.", cond)

    area_lookup = mask_summary.set_index(ROI)
    df_filt = df_binary[df_binary[roi_col].isin(rois)]
    missing = set(rois) - set(df_filt[roi_col].unique())
    if missing:
        logger.warning("ROIs not found in df_binary: %s", missing)

    results = []
    for roi in rois:
        grp = df_filt[df_filt[roi_col] == roi]
        if grp.empty or roi not in area_lookup.index:
            logger.warning("Skipping ROI '%s': no data or not in area lookup.", roi)
            continue

        mask_sub = pd.Series(True, index=grp.index)
        for col, val in parsed.items():
            mask_sub &= grp[col] == val

        cnt = mask_sub.sum()
        area_ck = area_lookup.at[roi, CK_POSITIVE_AREA_UM2]
        area_ckng = area_lookup.at[roi, CK_NGFR_POSITIVE_AREA_UM2]
        dens_ck = cnt / area_ck if area_ck > 0 else 0
        dens_ckng = cnt / area_ckng if area_ckng > 0 else 0

        results.append(
            {
                ROI: roi,
                SUBPOP_CELL_COUNT: cnt,
                CK_POSITIVE_AREA_UM2: area_ck,
                CK_NGFR_POSITIVE_AREA_UM2: area_ckng,
                CELLS_PER_UM2_CK: dens_ck,
                CELLS_PER_UM2_CK_NGFR: dens_ckng,
            }
        )

    if not results:
        logger.warning(
            "No valid ROI data found for conditions %s. Returning empty DataFrame.",
            subpop_conditions,
        )
        empty = pd.DataFrame(
            columns=[
                ROI,
                SUBPOP_CELL_COUNT,
                CK_POSITIVE_AREA_UM2,
                CK_NGFR_POSITIVE_AREA_UM2,
                CELLS_PER_UM2_CK,
                CELLS_PER_UM2_CK_NGFR,
            ]
        )
        return empty, empty.copy()

    summary_df = pd.DataFrame(results)
    summary_fmt = summary_df.copy()
    summary_fmt[CELLS_PER_UM2_CK] = summary_fmt[CELLS_PER_UM2_CK].map(lambda x: f"{x:.6f}")
    summary_fmt[CELLS_PER_UM2_CK_NGFR] = summary_fmt[CELLS_PER_UM2_CK_NGFR].map(
        lambda x: f"{x:.6f}"
    )
    summary_fmt[CK_POSITIVE_AREA_UM2] = summary_fmt[CK_POSITIVE_AREA_UM2].map(lambda x: f"{x:.2f}")
    summary_fmt[CK_NGFR_POSITIVE_AREA_UM2] = summary_fmt[CK_NGFR_POSITIVE_AREA_UM2].map(
        lambda x: f"{x:.2f}"
    )

    label = "_".join([c.replace("+", "Pos").replace(" ", "") for c in subpop_conditions])
    summary_fmt.rename(
        columns={
            SUBPOP_CELL_COUNT: f"Subpopulation Cell Count ({', '.join(subpop_conditions)})",
            CK_POSITIVE_AREA_UM2: "CK+ Area (um2)",
            CK_NGFR_POSITIVE_AREA_UM2: "CK+NGFR+ Area (um2)",
            CELLS_PER_UM2_CK: "Cells per um2 (CK+)",
            CELLS_PER_UM2_CK_NGFR: "Cells per um2 (CK+NGFR+)",
        },
        inplace=True,
    )

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"cell_density_area_{label}.csv")
    summary_fmt.to_csv(csv_path, index=False)
    logger.info("Saved: %s", csv_path)

    return summary_df, summary_fmt


def compute_distances(sub_df: pd.DataFrame, mask: np.ndarray, bin_col: str) -> tuple[list, list]:
    """Compute distances from each cell centroid to positive and negative mask regions.

    Cells already inside the mask (bin_col == 1) get distance 0 to the positive region;
    cells outside get distance 0 to the negative region.
    """
    pos = np.column_stack(np.where(mask == 1))
    neg = np.column_stack(np.where(mask == 0))

    tree_p = cKDTree(pos) if len(pos) else None
    tree_n = cKDTree(neg) if len(neg) else None

    d_to_pos, d_to_neg = [], []
    for _, r in sub_df.iterrows():
        cent = (r[CENTROID_ROW], r[CENTROID_COL])
        if r.get(bin_col, 0) == 1:
            d_to_pos.append(0.0)
            d_to_neg.append(tree_n.query(cent)[0] * PIXEL_SIZE if tree_n else 0.0)
        else:
            d_to_pos.append(tree_p.query(cent)[0] * PIXEL_SIZE if tree_p else 0.0)
            d_to_neg.append(0.0)

    return d_to_pos, d_to_neg


def get_centroids(dapi_mask: np.ndarray) -> pd.DataFrame:
    """Extract centroids from a labelled DAPI mask."""
    data = []
    for p in regionprops(dapi_mask):
        r, c = p.centroid
        data.append({DAPI_ID: p.label, CENTROID_ROW: int(r), CENTROID_COL: int(c)})
    return pd.DataFrame(data)


def compute_subpop_distances(subpopA_df: pd.DataFrame, subpopB_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise distances (pixels) between centroids of two subpopulations.

    Raises:
        ValueError: If either subpopulation is empty.
    """
    if subpopA_df.empty or subpopB_df.empty:
        raise ValueError(
            "Cannot compute distances: one or both subpopulations are empty. "
            f"A={len(subpopA_df)} cells, B={len(subpopB_df)} cells."
        )

    A = subpopA_df.assign(key=1)
    B = subpopB_df.assign(key=1)
    df = pd.merge(B, A, on="key", suffixes=("_b", "_a")).drop("key", axis=1)
    df[DISTANCE_PX] = np.sqrt(
        (df["centroid_row_b"] - df["centroid_row_a"]) ** 2
        + (df["centroid_col_b"] - df["centroid_col_a"]) ** 2
    )

    return df.rename(
        columns={
            "DAPI_ID_b": "B_cell_id",
            "centroid_row_b": "B_row",
            "centroid_col_b": "B_col",
            "DAPI_ID_a": "A_cell_id",
            "centroid_row_a": "A_row",
            "centroid_col_a": "A_col",
        }
    )
