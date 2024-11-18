import logging
from pathlib import Path

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
from scipy.spatial import KDTree
from skimage.measure import regionprops

logger = logging.getLogger(__name__)


def compute_mask_area_summary(
    ck_masks: dict[str, np.ndarray],
    ngfr_masks: dict[str, np.ndarray],
    pixel_area_um2: float = PIXEL_AREA,
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
                CK_POSITIVE_AREA_UM2: np.sum(ck_mask == 1) * pixel_area_um2,
                CK_NGFR_POSITIVE_AREA_UM2: np.sum((ck_mask == 1) & (ngfr_mask == 1))
                * pixel_area_um2,
                TOTAL_AREA_ROI_UM2: ck_mask.size * pixel_area_um2,
            }
        )

    return pd.DataFrame(data)


def _parse_subpop_conditions(
    subpop_conditions: list[str], cond_map: dict[str, str]
) -> dict[str, int]:
    """Convert condition strings to {column_name: value} pairs, warning on unknown keys."""
    parsed: dict[str, int] = {}
    for cond in subpop_conditions:
        val = 1 if cond.strip().endswith("+") else 0
        key = cond.strip()[:-1].strip()
        col = cond_map.get(key)
        if col:
            parsed[col] = val
        else:
            logger.warning("No mapping found for condition '%s' — skipping.", cond)
    return parsed


def _format_and_save_summary(
    summary_df: pd.DataFrame,
    subpop_conditions: list[str],
    out_dir: str | None,
) -> pd.DataFrame:
    """Format numeric columns and optionally write the summary CSV."""
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
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        csv_path = str(Path(out_dir) / f"cell_density_area_{label}.csv")
        summary_fmt.to_csv(csv_path, index=False)
        logger.info("Saved: %s", csv_path)
    return summary_fmt


def compute_subpop_cells_per_area(
    df_binary: pd.DataFrame,
    subpop_conditions: list[str],
    cond_map: dict[str, str],
    mask_summary: pd.DataFrame,
    rois: list[str],
    out_dir: str | None = None,
    roi_col: str = ROI,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute subpopulation cell counts and densities per ROI.

    subpop_conditions: e.g. ['CK_mask+', 'CD3_intensity+']
    cond_map: maps condition shorthand names to DataFrame column names
    mask_summary: per-ROI area DataFrame from compute_mask_area_summary

    Returns:
        Tuple of (summary_df, summary_fmt) where summary_fmt has formatted
        strings suitable for CSV export.
    """
    parsed = _parse_subpop_conditions(subpop_conditions, cond_map)
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
        results.append(
            {
                ROI: roi,
                SUBPOP_CELL_COUNT: cnt,
                CK_POSITIVE_AREA_UM2: area_ck,
                CK_NGFR_POSITIVE_AREA_UM2: area_ckng,
                CELLS_PER_UM2_CK: cnt / area_ck if area_ck > 0 else 0,
                CELLS_PER_UM2_CK_NGFR: cnt / area_ckng if area_ckng > 0 else 0,
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
    return summary_df, _format_and_save_summary(summary_df, subpop_conditions, out_dir)


def compute_distances(
    sub_df: pd.DataFrame, mask: np.ndarray, bin_col: str
) -> tuple[list[float], list[float]]:
    """Compute distances from each cell centroid to positive and negative mask regions.

    Cells inside the mask (bin_col == 1) get distance 0 to the positive region;
    cells outside get distance 0 to the negative region. Uses batch KDTree queries
    for efficiency — avoids iterating row-by-row.

    Raises:
        KeyError: If bin_col is not a column in sub_df.
    """
    if bin_col not in sub_df.columns:
        raise KeyError(
            f"Column '{bin_col}' not found in sub_df. Available columns: {list(sub_df.columns)}"
        )

    pos = np.column_stack(np.where(mask == 1))
    neg = np.column_stack(np.where(mask == 0))

    tree_p = KDTree(pos) if len(pos) else None
    tree_n = KDTree(neg) if len(neg) else None

    centroids = sub_df[[CENTROID_ROW, CENTROID_COL]].to_numpy()
    flags = sub_df[bin_col].to_numpy()

    d_to_pos = np.zeros(len(sub_df))
    d_to_neg = np.zeros(len(sub_df))

    inside = flags == 1
    if inside.any() and tree_n is not None:
        d_to_neg[inside] = tree_n.query(centroids[inside])[0] * PIXEL_SIZE
    if (~inside).any() and tree_p is not None:
        d_to_pos[~inside] = tree_p.query(centroids[~inside])[0] * PIXEL_SIZE

    return d_to_pos.tolist(), d_to_neg.tolist()


def get_centroids(dapi_mask: np.ndarray) -> pd.DataFrame:
    """Extract centroids from a labeled DAPI mask."""
    data = []
    for p in regionprops(dapi_mask):
        r, c = p.centroid
        data.append({DAPI_ID: p.label, CENTROID_ROW: int(r), CENTROID_COL: int(c)})
    return pd.DataFrame(data)


def compute_subpop_distances(subpop_a: pd.DataFrame, subpop_b: pd.DataFrame) -> pd.DataFrame:
    """Compute nearest-neighbor distances (pixels) between centroids of two subpopulations.

    For each cell in A finds its nearest cell in B, and for each cell in B finds
    its nearest cell in A. Returns the union of both directions (duplicates removed).

    Uses KDTree for O((N+M) log N) complexity instead of the O(N×M) Cartesian join.

    Raises:
        ValueError: If either subpopulation is empty.
    """
    if subpop_a.empty or subpop_b.empty:
        raise ValueError(
            "Cannot compute distances: one or both subpopulations are empty. "
            f"A={len(subpop_a)} cells, B={len(subpop_b)} cells."
        )

    coords_a = subpop_a[[CENTROID_ROW, CENTROID_COL]].to_numpy(dtype=float)
    coords_b = subpop_b[[CENTROID_ROW, CENTROID_COL]].to_numpy(dtype=float)
    ids_a = subpop_a[DAPI_ID].to_numpy()
    ids_b = subpop_b[DAPI_ID].to_numpy()

    dists_a2b, idx_b = KDTree(coords_b).query(coords_a)
    df_a2b = pd.DataFrame(
        {
            "A_cell_id": ids_a,
            "A_row": coords_a[:, 0],
            "A_col": coords_a[:, 1],
            "B_cell_id": ids_b[idx_b],
            "B_row": coords_b[idx_b, 0],
            "B_col": coords_b[idx_b, 1],
            DISTANCE_PX: dists_a2b,
        }
    )

    dists_b2a, idx_a = KDTree(coords_a).query(coords_b)
    df_b2a = pd.DataFrame(
        {
            "A_cell_id": ids_a[idx_a],
            "A_row": coords_a[idx_a, 0],
            "A_col": coords_a[idx_a, 1],
            "B_cell_id": ids_b,
            "B_row": coords_b[:, 0],
            "B_col": coords_b[:, 1],
            DISTANCE_PX: dists_b2a,
        }
    )

    return pd.concat([df_a2b, df_b2a], ignore_index=True).drop_duplicates(
        subset=["A_cell_id", "B_cell_id"]
    )
