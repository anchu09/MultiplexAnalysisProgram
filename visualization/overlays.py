"""Orchestration layer for multiplex cell overlay visualizations.

Coordinates data loading, subpopulation filtering, distance computation, and
plotting per ROI. Pure helpers live in data_prep.py and plotting.py.

All public names from those modules are re-exported here for backward
compatibility with existing notebook code.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from multiplex_pipeline.analysis.spatial import (
    compute_distances,
    compute_subpop_distances,
    get_centroids,
)
from multiplex_pipeline.config import DISTANCES_POPULATIONS_DIR, PIXEL_SIZE
from multiplex_pipeline.schema import (
    CENTROID_COL,
    CENTROID_ROW,
    DAPI_ID,
    DISTANCE_CK_NEGATIVE,
    DISTANCE_CK_POSITIVE,
    DISTANCE_NGFR_NEGATIVE,
    DISTANCE_NGFR_POSITIVE,
    DISTANCE_PX,
    DISTANCE_UM,
    IS_POSITIVE_CK,
    IS_POSITIVE_NGFR,
    ROI,
)
from multiplex_pipeline.visualization.data_prep import (  # re-exported
    filter_cells_by_combination,
    load_distance_matrices_for_plot,
    parse_conditions,
    parse_distance_matrix_filename,
    select_subpopulation,
)
from multiplex_pipeline.visualization.plotting import (  # re-exported
    _show_figure,
    create_color_composite,
    create_marker_plot,
    plot_subpopulations_and_distances,
    shade_selected_masks,
    voronoi_finite_polygons_2d,
)
from skimage.measure import regionprops

logger = logging.getLogger(__name__)


def _build_centroid_dataframe(
    dapi_mask: np.ndarray,
    df_bin: pd.DataFrame,
    roi: str,
) -> pd.DataFrame:
    """Merge DAPI centroids with per-cell binary intensity data.

    Left-joins centroid DataFrame with df_bin on cell ID, then fills NaN values
    in the CK/NGFR positivity columns with 0 (cells present in the mask but
    absent from intensity data).

    Raises:
        KeyError: If IS_POSITIVE_CK or IS_POSITIVE_NGFR are absent from the
            merged result — indicates a mismatch between the DAPI mask
            and the intensity extraction step.
    """
    props = regionprops(dapi_mask)
    centroids = [
        {
            "cell_id": p.label,
            CENTROID_ROW: int(p.centroid[0]),
            CENTROID_COL: int(p.centroid[1]),
        }
        for p in props
    ]
    df_bin_no_centroids = df_bin.drop(columns=[CENTROID_ROW, CENTROID_COL], errors="ignore")
    merged = pd.merge(
        pd.DataFrame(centroids),
        df_bin_no_centroids,
        left_on="cell_id",
        right_on=DAPI_ID,
        how="left",
    )
    for col in (IS_POSITIVE_CK, IS_POSITIVE_NGFR):
        if col not in merged.columns:
            raise KeyError(
                f"Column '{col}' missing from merged DataFrame for ROI '{roi}'. "
                "Ensure intensity extraction ran successfully for this ROI."
            )
        n_nans = int(merged[col].isna().sum())
        if n_nans > 0:
            logger.warning(
                "ROI %s: %d NaN values in '%s' filled with 0 — some cells may lack intensity data.",
                roi,
                n_nans,
                col,
            )
        merged[col] = merged[col].fillna(0).astype(int)
    return merged


__all__ = [
    # data_prep
    "parse_conditions",
    "select_subpopulation",
    "parse_distance_matrix_filename",
    "load_distance_matrices_for_plot",
    "filter_cells_by_combination",
    # plotting
    "create_color_composite",
    "voronoi_finite_polygons_2d",
    "shade_selected_masks",
    "create_marker_plot",
    "plot_subpopulations_and_distances",
    # orchestration (defined here)
    "plot_conditional_cells_channels",
    "plot_roi_split_markers",
    "compute_and_save",
    "compute_and_plot_subpop_distances_for_all_rois",
]


def plot_conditional_cells_channels(
    rois: list[str],
    conditions: list[str],
    dapi_masks_dict: dict[str, np.ndarray],
    images_dict: dict[str, np.ndarray],
    df_binary: pd.DataFrame,
    marker_dict: dict[int, str],
    ck_masks_dict: dict[str, dict[str, np.ndarray]],
    ngfr_masks_dict: dict[str, dict[str, np.ndarray]],
    condition_column_map: dict[str, str],
    brightness_factor: float = 1.0,
) -> None:
    """Plot cells and channels for the specified conditions in each ROI.

    Renders a two-row figure per ROI:
    - Top row: multi-channel composite + individual marker channels/masks + mean image.
    - Bottom row: same channels at 2× brightness with positive cells overlaid.
    """
    import matplotlib.pyplot as plt

    def _is_mask_marker(name: str) -> bool:
        return name.lower().endswith("_mask")

    def _is_intensity_marker(name: str) -> bool:
        return name.lower().endswith("_intensity")

    def _get_channel_for_marker(marker_short: str) -> int | None:
        if _is_mask_marker(marker_short):
            return None
        base_name = marker_short.replace("_intensity", "").strip()
        for ch, full_name in marker_dict.items():
            if base_name.lower() == full_name.lower():
                logger.debug("Marker '%s' matched channel %d.", base_name, ch)
                return ch
        logger.warning("Channel not found for marker '%s'.", marker_short)
        return None

    parsed_conditions = parse_conditions(conditions, condition_column_map)
    mask_dicts_combined = {"CK_mask": ck_masks_dict, "NGFR_mask": ngfr_masks_dict}

    markers_to_plot = []
    for cond in conditions:
        shorthand = cond[:-1]
        ch = _get_channel_for_marker(shorthand)
        markers_to_plot.append((shorthand, ch))

    for roi in rois:
        logger.info("Processing %s...", roi)
        roi_lower = roi.lower()
        roi_dapi_key = f"{roi_lower}_dapi"

        if roi_dapi_key not in dapi_masks_dict:
            logger.warning("%s not found in dapi_masks_dict. Skipping.", roi_dapi_key)
            continue

        roi_image_key = next((k for k in images_dict if roi_lower in k.lower()), None)
        if roi_image_key is None:
            logger.warning("No image found for %s in images_dict.", roi)
            continue

        img_data = images_dict[roi_image_key]
        cell_mask = dapi_masks_dict[roi_dapi_key]
        df_roi = df_binary[df_binary[ROI] == roi]

        condition_series = pd.Series([True] * len(df_roi), index=df_roi.index)
        for marker_col, val in parsed_conditions.items():
            condition_series &= df_roi[marker_col] == val

        selected_cells = df_roi[condition_series][DAPI_ID].tolist()
        logger.info("ROI %s: %d cells meet conditions.", roi, len(selected_cells))

        mask_selected = (
            np.isin(cell_mask, selected_cells).astype(int)
            if selected_cells
            else np.zeros_like(cell_mask, dtype=int)
        )

        n_markers = len(markers_to_plot)
        n_cols = n_markers + 2

        plt.figure(figsize=(6 * n_cols, 12))

        cell_labels = np.unique(cell_mask)
        cell_labels = cell_labels[cell_labels != 0]
        num_labels = len(cell_labels) + 1
        np.random.seed(42)
        rand_colors = np.random.rand(num_labels, 3)
        rand_colors[0] = [0, 0, 0]
        cmap_cells = ListedColormap(rand_colors)

        top_images = [None] * n_cols
        bottom_images = [None] * n_cols

        ax_top_left = plt.subplot(2, n_cols, 1)
        composite = create_color_composite(
            img_data, mask_dicts_combined, markers_to_plot, roi_lower, brightness_factor
        )
        ax_top_left.imshow(composite, interpolation="nearest")
        ax_top_left.set_title("Multicolor Composite\n(Intensity + Masks)")
        ax_top_left.axis("off")
        top_images[0] = composite

        for i, (marker_name, ch) in enumerate(markers_to_plot, start=2):
            ax_top = plt.subplot(2, n_cols, i)
            if _is_mask_marker(marker_name):
                if (
                    marker_name in mask_dicts_combined
                    and roi_lower in mask_dicts_combined[marker_name]
                ):
                    mask_img = mask_dicts_combined[marker_name][roi_lower]
                    ax_top.imshow(mask_img, cmap="gray")
                    top_images[i - 1] = mask_img.astype(float)
                else:
                    zeros_img = np.zeros_like(cell_mask, dtype=float)
                    ax_top.imshow(zeros_img, cmap="gray")
                    top_images[i - 1] = zeros_img
                ax_top.set_title(f"Mask: {marker_name}")

            elif _is_intensity_marker(marker_name):
                if ch is not None:
                    channel_data = img_data[ch, :, :] * brightness_factor
                    cmin, cmax = channel_data.min(), channel_data.max()
                    if cmax == cmin:
                        channel_norm = np.zeros_like(channel_data)
                    else:
                        channel_norm = (channel_data - cmin) / (cmax - cmin)
                    channel_corrected = channel_norm**0.5
                    ax_top.imshow(channel_corrected, cmap="gray")
                    top_images[i - 1] = channel_corrected
                else:
                    zeros_img = np.zeros_like(cell_mask, dtype=float)
                    ax_top.imshow(zeros_img, cmap="gray")
                    top_images[i - 1] = zeros_img
                ax_top.set_title(f"Intensity: {marker_name}")
            else:
                zeros_img = np.zeros_like(cell_mask, dtype=float)
                ax_top.imshow(zeros_img, cmap="gray")
                top_images[i - 1] = zeros_img
                ax_top.set_title(f"?? {marker_name}")
            ax_top.axis("off")

        ax_top_right = plt.subplot(2, n_cols, n_markers + 2)
        avg_top = np.mean([top_images[idx] for idx in range(1, n_markers + 1)], axis=0)
        ax_top_right.imshow(avg_top, cmap="gray")
        ax_top_right.set_title("Mean (Top Row)")
        ax_top_right.axis("off")
        top_images[n_markers + 1] = avg_top

        ax_bottom_left = plt.subplot(2, n_cols, n_cols + 1)
        composite_2x = np.clip(top_images[0] * 2, 0, 1)
        ax_bottom_left.imshow(composite_2x, interpolation="nearest")
        ax_bottom_left.imshow(cell_mask, cmap=cmap_cells, alpha=0.5, interpolation="nearest")
        ax_bottom_left.set_title("All Cells (over Composite)")
        ax_bottom_left.axis("off")
        bottom_images[0] = composite_2x

        for j, (marker_name, _) in enumerate(markers_to_plot, start=2):
            ax_bottom = plt.subplot(2, n_cols, n_cols + j)
            background = top_images[j - 1]
            background_2x = np.clip(background * 2, 0, 1)
            bottom_images[j - 1] = background_2x
            ax_bottom.imshow(background_2x, cmap="gray")

            col_bin = condition_column_map.get(marker_name)
            if col_bin is not None:
                val = parsed_conditions.get(col_bin)
                selected_for_marker = df_roi[df_roi[col_bin] == (val if val is not None else 1)][
                    DAPI_ID
                ].tolist()
            else:
                selected_for_marker = []

            mask_positive = np.isin(cell_mask, selected_for_marker).astype(int)
            ax_bottom.imshow(
                mask_positive * cell_mask, cmap=cmap_cells, interpolation="nearest", alpha=0.5
            )
            ax_bottom.set_title(f"Pos Cells\n{marker_name}")
            ax_bottom.axis("off")

        ax_bottom_right = plt.subplot(2, n_cols, 2 * n_cols)
        avg_bottom = np.mean([bottom_images[idx] for idx in range(1, n_markers + 1)], axis=0)
        ax_bottom_right.imshow(avg_bottom, cmap="gray")

        if selected_cells:
            cmap_selected = ListedColormap(
                ["black"] + [np.random.rand(3) for _ in range(len(selected_cells))]
            )
            ax_bottom_right.imshow(
                mask_selected * cell_mask, cmap=cmap_selected, interpolation="nearest", alpha=0.7
            )
            ax_bottom_right.set_title(f"Final Filter:\n{conditions}")
        else:
            ax_bottom_right.imshow(mask_selected, cmap="gray", alpha=0.7, interpolation="nearest")
            ax_bottom_right.set_title(f"Final Filter (0)\n{conditions}")

        ax_bottom_right.axis("off")
        bottom_images[n_markers + 1] = avg_bottom

        plt.tight_layout()
        plt.show()
        logger.info("Plot generated for %s.", roi)


def plot_roi_split_markers(
    roi: str,
    dapi_masks: dict,
    ck_masks: dict,
    ngfr_masks: dict,
    df_bin: pd.DataFrame,
    col_map: dict,
    subpop_name: str,
    subpop_conditions: list,
    max_cells: int | None = None,
):
    """Plot CK and NGFR marker distance figures for a single ROI.

    Returns:
        Tuple (fig_ck, fig_ngfr), either of which may be None if skipped.
    """
    dapi = dapi_masks.get(f"{roi}_dapi")
    if dapi is None:
        logger.warning("No DAPI mask for %s.", roi)
        return None, None

    ck = ck_masks.get(roi, np.zeros_like(dapi, dtype=np.uint8))
    ngfr = ngfr_masks.get(roi, np.zeros_like(dapi, dtype=np.uint8))

    if not regionprops(dapi):
        logger.warning("No cells in DAPI mask for %s.", roi)
        return None, None

    merged = _build_centroid_dataframe(dapi, df_bin, roi)

    fig_ck = create_marker_plot(
        roi, ck, "CK", "red", dapi, merged, col_map, subpop_name, subpop_conditions, max_cells
    )
    fig_ngfr = create_marker_plot(
        roi, ngfr, "NGFR", "blue", dapi, merged, col_map, subpop_name, subpop_conditions, max_cells
    )
    return fig_ck, fig_ngfr


def compute_and_save(
    roi: str,
    subpop_name: str,
    subpop_conditions: list,
    path_save: str,
    dapi_masks: dict,
    ck_masks: dict,
    ngfr_masks: dict,
    df_bin: pd.DataFrame,
    col_map: dict,
    max_cells: int | None = None,
) -> None:
    """Generate marker plots and distance CSV for one ROI and subpopulation."""
    out_dir = os.path.join(path_save, subpop_name)
    os.makedirs(out_dir, exist_ok=True)

    fig_ck, fig_ngfr = plot_roi_split_markers(
        roi,
        dapi_masks,
        ck_masks,
        ngfr_masks,
        df_bin,
        col_map,
        subpop_name,
        subpop_conditions,
        max_cells,
    )

    if fig_ck is None and fig_ngfr is None:
        logger.info("%s: %s — no figures (distance 0 everywhere).", roi, subpop_name)
        return

    if fig_ck:
        fig_ck.savefig(os.path.join(out_dir, f"roi_{roi}_ck.svg"))
        _show_figure(fig_ck)
    if fig_ngfr:
        fig_ngfr.savefig(os.path.join(out_dir, f"roi_{roi}_ngfr.svg"))
        _show_figure(fig_ngfr)

    dapi = dapi_masks.get(f"{roi}_dapi")
    ck = ck_masks.get(roi, np.zeros_like(dapi, dtype=np.uint8))
    ngfr = ngfr_masks.get(roi, np.zeros_like(dapi, dtype=np.uint8))

    merged = _build_centroid_dataframe(dapi, df_bin, roi)

    parsed = parse_conditions(subpop_conditions, col_map)
    sub_df = select_subpopulation(merged, parsed)

    ck_pos, ck_neg = compute_distances(sub_df, ck, IS_POSITIVE_CK)
    ng_pos, ng_neg = compute_distances(sub_df, ngfr, IS_POSITIVE_NGFR)

    sub_df = sub_df.copy()
    sub_df[DISTANCE_CK_POSITIVE] = ck_pos
    sub_df[DISTANCE_CK_NEGATIVE] = ck_neg
    sub_df[DISTANCE_NGFR_POSITIVE] = ng_pos
    sub_df[DISTANCE_NGFR_NEGATIVE] = ng_neg

    dist_tbl = sub_df[
        [
            "cell_id",
            CENTROID_ROW,
            CENTROID_COL,
            DISTANCE_CK_POSITIVE,
            DISTANCE_CK_NEGATIVE,
            DISTANCE_NGFR_POSITIVE,
            DISTANCE_NGFR_NEGATIVE,
        ]
    ].copy()
    dist_tbl.insert(0, ROI, roi)

    keep = (
        (dist_tbl[DISTANCE_CK_POSITIVE] > 0)
        | (dist_tbl[DISTANCE_CK_NEGATIVE] > 0)
        | (dist_tbl[DISTANCE_NGFR_POSITIVE] > 0)
        | (dist_tbl[DISTANCE_NGFR_NEGATIVE] > 0)
    )
    dist_tbl = dist_tbl[keep]

    if dist_tbl.empty:
        logger.info("%s: %s — CSV empty (all distances are 0).", roi, subpop_name)
        return

    csv_path = os.path.join(out_dir, f"roi_{roi}_distance_table.csv")
    dist_tbl.to_csv(csv_path, index=False)
    logger.info("CSV saved → %s", csv_path)


def compute_and_plot_subpop_distances_for_all_rois(
    rois: list,
    subpop_conditions_A: list,
    subpop_conditions_B: list,
    df_binary: pd.DataFrame,
    dapi_masks_dict: dict,
    condition_column_map: dict,
    pixel_size: float = PIXEL_SIZE,
    max_pairs: int | None = None,
    masks_to_shade: list | None = None,
    shading_dict: dict | None = None,
    save_matrix_as_csv: bool = False,
    path_save: str = str(DISTANCES_POPULATIONS_DIR),
    print_pivot_head: bool = False,
    plot_type: str = "line",
    subpopB_label: str | None = None,
) -> dict:
    """Compute and plot subpopulation distances for all ROIs.

    For each ROI:
    1. Parse subpopulation A and B conditions.
    2. Select the corresponding cells.
    3. Compute all pairwise distances (A × B).
    4. Build and optionally save a distance pivot table.
    5. Plot using lines or Voronoi diagram.

    subpopB_label: Optional display label override for subpopulation B.

    Returns:
        Dict mapping ROI → distances DataFrame.
    """
    subpopA_name = " & ".join(subpop_conditions_A)
    subpopB_name = subpopB_label if subpopB_label is not None else " & ".join(subpop_conditions_B)
    parsedA = parse_conditions(subpop_conditions_A, condition_column_map)
    parsedB = parse_conditions(subpop_conditions_B, condition_column_map)

    all_distances: dict = {}

    for roi in rois:
        logger.info("=== ROI %s ===", roi)
        dapi_key = roi.lower() + "_dapi"
        if dapi_key not in dapi_masks_dict:
            logger.warning("Missing DAPI mask for %s. Skipping.", dapi_key)
            continue

        dapi_mask = dapi_masks_dict[dapi_key]
        df_roi = df_binary[df_binary[ROI] == roi].copy()
        df_roi = df_roi.drop(columns=[CENTROID_ROW, CENTROID_COL], errors="ignore")
        centroids_df = get_centroids(dapi_mask)
        df_roi = pd.merge(df_roi, centroids_df, on=DAPI_ID, how="left")

        subpopA = select_subpopulation(df_roi, parsedA)
        subpopB = select_subpopulation(df_roi, parsedB)

        try:
            dist_df = compute_subpop_distances(subpopA, subpopB)
        except ValueError as exc:
            logger.warning("Skipping ROI %s: %s", roi, exc)
            all_distances[roi] = pd.DataFrame()
            continue

        dist_df[ROI] = roi
        dist_df[DISTANCE_UM] = dist_df[DISTANCE_PX] * pixel_size
        all_distances[roi] = dist_df

        dist_df["A_roi_cell"] = dist_df[ROI].astype(str) + "_" + dist_df["A_cell_id"].astype(str)
        dist_df["B_roi_cell"] = dist_df[ROI].astype(str) + "_" + dist_df["B_cell_id"].astype(str)
        distance_matrix = dist_df.pivot(
            index="A_roi_cell", columns="B_roi_cell", values=DISTANCE_UM
        )

        logger.info("rows: %s", subpopA_name)
        logger.info("columns: %s", subpopB_name)
        if print_pivot_head:
            logger.info("\n%s", distance_matrix.head(10).to_string())

        if save_matrix_as_csv:
            csv_filename = f"distance_matrix_{roi}_{subpopA_name}_vs_{subpopB_name}.csv"
            os.makedirs(path_save, exist_ok=True)
            csv_filepath = os.path.join(path_save, csv_filename)
            distance_matrix.to_csv(csv_filepath, index=True)
            logger.info("Matrix saved: %s", csv_filepath)

        if len(subpopA) > 0 and len(subpopB) > 0:
            plot_filename = os.path.join(
                path_save, f"plot_{roi}_{subpopA_name}_vs_{subpopB_name}.svg"
            )
            plot_subpopulations_and_distances(
                roi=roi,
                dapi_mask=dapi_mask,
                subpopA_df=subpopA,
                subpopB_df=subpopB,
                dist_df=dist_df,
                subpopA_name=subpopA_name,
                subpopB_name=subpopB_name,
                masks_to_shade=masks_to_shade,
                shading_dict=shading_dict,
                pixel_size=pixel_size,
                max_pairs=max_pairs,
                plot_type=plot_type,
                save_plot=True,
                plot_filename=plot_filename,
            )
        else:
            logger.info("One or both subpopulations empty in %s. No plot.", roi)

    return all_distances
