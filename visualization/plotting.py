"""Pure plotting functions for multiplex microscopy visualization.

No data loading or DataFrame transformations here — see data_prep.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Polygon
from multiplex_pipeline.config import MASK_ALPHA, PIXEL_SIZE, RANDOM_STATE
from multiplex_pipeline.schema import (
    CENTROID_COL,
    CENTROID_ROW,
    DISTANCE_CK_MASK,
    DISTANCE_NGFR_MASK,
    DISTANCE_PX,
    IS_POSITIVE_CK,
    IS_POSITIVE_NGFR,
    NEAREST_CK_COL,
    NEAREST_CK_ROW,
    NEAREST_NGFR_COL,
    NEAREST_NGFR_ROW,
)
from multiplex_pipeline.utils.helpers import in_jupyter
from multiplex_pipeline.visualization.data_prep import parse_conditions, select_subpopulation
from scipy.spatial import KDTree, QhullError, Voronoi
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from skimage.color import label2rgb
from skimage.measure import find_contours

logger = logging.getLogger(__name__)

_DISTANCE_CMAP = "hot"  # colormap for distance-coded lines and scatter plots

# Fixed color cycle for multi-channel composite overlays (RGB primaries + secondaries)
_CHANNEL_COLORS = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
    ],
    dtype=np.float32,
)


def show_figure(fig: Figure) -> None:
    """Display a figure: IPython.display in Jupyter kernels, plt.show() in scripts."""
    if in_jupyter():
        from IPython.display import display

        display(fig)
    else:
        plt.show()


def create_color_composite(
    img_data: np.ndarray,
    mask_dicts: dict[str, dict[str, np.ndarray]],
    markers_to_plot: list[tuple[str, int | None]],
    roi_lower: str,
    brightness_factor: float = 1.0,
) -> np.ndarray:
    """Create an additive multi-channel RGB composite image.

    Returns RGB composite array of shape (H, W, 3) clipped to [0, 1].
    """
    height, width = img_data.shape[1], img_data.shape[2]
    composite = np.zeros((height, width, 3), dtype=np.float32)

    for i, (marker_name, ch) in enumerate(markers_to_plot):
        color = _CHANNEL_COLORS[i % len(_CHANNEL_COLORS)]
        if ch is not None:
            channel_data = img_data[ch, :, :] * brightness_factor
            cmin, cmax = channel_data.min(), channel_data.max()
            if cmax == cmin:
                channel_norm = np.zeros_like(channel_data, dtype=np.float32)
            else:
                channel_norm = (channel_data - cmin) / (cmax - cmin)
            channel_gamma = channel_norm**0.5
            composite[:, :, 0] += channel_gamma * color[0]
            composite[:, :, 1] += channel_gamma * color[1]
            composite[:, :, 2] += channel_gamma * color[2]
        else:
            mask_data = None
            if marker_name in mask_dicts and roi_lower in mask_dicts[marker_name]:
                mask_data = mask_dicts[marker_name][roi_lower]
            if mask_data is not None:
                mask_binary = (mask_data > 0).astype(float)
                composite[:, :, 0] += mask_binary * color[0]
                composite[:, :, 1] += mask_binary * color[1]
                composite[:, :, 2] += mask_binary * color[2]

    return np.clip(composite, 0, 1)


def voronoi_finite_polygons_2d(
    vor: Voronoi, radius: float | None = None
) -> tuple[list, np.ndarray]:
    """Reconstruct finite Voronoi polygons in 2D.

    radius: Extension radius for infinite regions. Defaults to 2× the bounding-box diagonal.

    Raises:
        ValueError: If the Voronoi input is not 2D.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("voronoi_finite_polygons_2d requires 2D input.")
    if radius is not None and radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}.")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points).max() * 2

    all_ridges: dict[int, list] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices, strict=False):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        vs = np.array([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)].tolist()
        new_regions.append(new_region)

    return new_regions, np.array(new_vertices)


def _reorder_masks(masks_to_shade: list[str]) -> list[str]:
    """Sort mask names so CK_mask is drawn first."""

    def _priority(name: str) -> int:
        return 0 if name == "CK_mask" else 1

    return sorted(masks_to_shade, key=_priority)


def shade_selected_masks(
    ax: Axes,
    roi: str,
    masks_to_shade: list[str],
    shading_dict: dict[str, tuple],
    alpha: float = 0.25,
) -> list[Patch]:
    """Overlay multiple masks as translucent colored areas on ax."""
    legend_handles = []
    for mask_name in _reorder_masks(masks_to_shade):
        if mask_name not in shading_dict:
            continue
        mask_dict, color = shading_dict[mask_name]
        if roi in mask_dict:
            mask_data = mask_dict[roi]
            custom_cmap = ListedColormap([(0, 0, 0, 0), color + (1.0,)])
            ax.imshow(mask_data, cmap=custom_cmap, alpha=alpha)
            legend_handles.append(Patch(facecolor=color, alpha=alpha, label=mask_name))
    return legend_handles


def _compute_mask_distances(
    sub_df: pd.DataFrame,
    marker_mask: np.ndarray,
    bin_col: str,
    dist_col: str,
    row_col: str,
    col_col: str,
) -> pd.DataFrame:
    """Add distance-to-mask columns to sub_df and filter to cells with distance > 0."""
    pos_pts = np.column_stack(np.where(marker_mask == 1))
    neg_pts = np.column_stack(np.where(marker_mask == 0))
    tree_pos = KDTree(pos_pts) if len(pos_pts) else None
    tree_neg = KDTree(neg_pts) if len(neg_pts) else None

    centroids = sub_df[[CENTROID_ROW, CENTROID_COL]].to_numpy()
    flags = sub_df[bin_col].to_numpy().astype(bool)
    dists = np.zeros(len(sub_df))
    near_row_arr = centroids[:, 0].astype(int).copy()
    near_col_arr = centroids[:, 1].astype(int).copy()

    out_idx = np.where(~flags)[0]
    if len(out_idx) and tree_pos is not None:
        d, qi = tree_pos.query(centroids[out_idx])
        dists[out_idx] = d * PIXEL_SIZE
        near_row_arr[out_idx] = tree_pos.data[qi, 0].astype(int)
        near_col_arr[out_idx] = tree_pos.data[qi, 1].astype(int)

    in_idx = np.where(flags)[0]
    if len(in_idx) and tree_neg is not None:
        d, qi = tree_neg.query(centroids[in_idx])
        dists[in_idx] = d * PIXEL_SIZE
        near_row_arr[in_idx] = tree_neg.data[qi, 0].astype(int)
        near_col_arr[in_idx] = tree_neg.data[qi, 1].astype(int)

    result = sub_df.copy()
    result[dist_col] = dists
    result[row_col] = near_row_arr
    result[col_col] = near_col_arr
    return result[result[dist_col] > 0]


def _draw_marker_ax(
    ax: Axes,
    dapi_mask: np.ndarray,
    marker_mask: np.ndarray,
    marker_name: str,
    marker_color: str,
    plot_df: pd.DataFrame,
    subpop_name: str,
    bin_col: str,
    col_col: str,
    row_col: str,
) -> None:
    """Render the marker overlay and distance lines onto ax."""
    dapi_rgb = label2rgb(dapi_mask, bg_label=0, alpha=MASK_ALPHA, colors=["lightgray"])
    ax.imshow(dapi_rgb)
    cmap_name = "Reds" if marker_name == "CK" else "Blues"
    ax.imshow(marker_mask, cmap=cmap_name, alpha=MASK_ALPHA)
    for contour in find_contours(marker_mask, 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color=marker_color, lw=1)

    ax.scatter(
        plot_df[CENTROID_COL],
        plot_df[CENTROID_ROW],
        c="yellow",
        s=30,
        edgecolors="black",
        label=subpop_name,
    )
    for _, r in plot_df.iterrows():
        line_color = marker_color if r[bin_col] == 0 else "green"
        ax.plot(
            [r[CENTROID_COL], r[col_col]],
            [r[CENTROID_ROW], r[row_col]],
            color=line_color,
            ls="--",
            lw=1.5,
        )

    ax.legend(
        handles=[
            Patch(color=marker_color, alpha=MASK_ALPHA, label=f"{marker_name} Mask"),
            mlines.Line2D([], [], color=marker_color, lw=2, label=f"{marker_name} Boundary"),
            mlines.Line2D(
                [],
                [],
                color="yellow",
                marker="o",
                markeredgecolor="black",
                markersize=6,
                lw=0,
                label=subpop_name,
            ),
            mlines.Line2D(
                [], [], color=marker_color, ls="--", lw=2, label=f"Marker- ({marker_color}) → Mask"
            ),
            mlines.Line2D([], [], color="green", ls="--", lw=2, label="Marker+ (green) → Stroma"),
        ],
        loc="upper right",
    )
    ax.set_title(f"ROI {marker_name} | {subpop_name}")
    ax.axis("off")


def create_marker_plot(
    roi: str,
    marker_mask: np.ndarray,
    marker_name: str,
    marker_color: str,
    dapi_mask: np.ndarray,
    merged_df: pd.DataFrame,
    col_map: dict[str, str],
    subpop_name: str,
    subpop_conditions: list[str],
    max_cells: int | None = None,
) -> Figure | None:
    """Plot one marker mask with distances from a subpopulation.

    Returns a matplotlib Figure, or None if the subpopulation has 0 cells or
    all distances are 0.
    """
    parsed = parse_conditions(subpop_conditions, col_map)
    sub_df = select_subpopulation(merged_df, parsed)
    if sub_df.empty:
        logger.info("%s / %s: 0 cells — skipping.", roi, subpop_name)
        return None

    if marker_name == "CK":
        bin_col = IS_POSITIVE_CK
        dist_col, row_col, col_col = DISTANCE_CK_MASK, NEAREST_CK_ROW, NEAREST_CK_COL
    else:
        bin_col = IS_POSITIVE_NGFR
        dist_col, row_col, col_col = DISTANCE_NGFR_MASK, NEAREST_NGFR_ROW, NEAREST_NGFR_COL

    sub_df = _compute_mask_distances(sub_df, marker_mask, bin_col, dist_col, row_col, col_col)
    if sub_df.empty:
        logger.info("%s / %s: all distances are 0 — nothing to show.", roi, subpop_name)
        return None

    if max_cells is not None and max_cells <= 0:
        raise ValueError(f"max_cells must be a positive integer, got {max_cells}.")
    plot_df = (
        sub_df.sample(max_cells, random_state=RANDOM_STATE)
        if max_cells is not None and max_cells < len(sub_df)
        else sub_df
    )

    fig, ax = plt.subplots(figsize=(12, 12))
    _draw_marker_ax(
        ax,
        dapi_mask,
        marker_mask,
        marker_name,
        marker_color,
        plot_df,
        subpop_name,
        bin_col,
        col_col,
        row_col,
    )
    fig.tight_layout()
    return fig


@dataclass
class SubpopPlotOptions:
    """Optional parameters for plot_subpopulations_and_distances."""

    masks_to_shade: list[str] | None = None
    shading_dict: dict[str, tuple] | None = None
    pixel_size: float = PIXEL_SIZE
    max_pairs: int | None = None
    plot_type: Literal["line", "voronoi"] = "line"
    save_plot: bool = False
    plot_filename: str | None = None


def _draw_line_mode(
    ax: Axes,
    fig: Figure,
    dist_df: pd.DataFrame,
    subpop_a_df: pd.DataFrame,
    subpop_b_df: pd.DataFrame,
    subpop_a_name: str,
    subpop_b_name: str,
    opts: SubpopPlotOptions,
    shading_handles: list[Patch],
) -> None:
    """Render nearest-neighbor distance lines between two subpopulations onto ax."""
    min_distA2B = dist_df.loc[dist_df.groupby("A_cell_id")[DISTANCE_PX].idxmin()]
    min_distB2A = dist_df.loc[dist_df.groupby("B_cell_id")[DISTANCE_PX].idxmin()]

    cmap = colormaps[_DISTANCE_CMAP]
    sm: ScalarMappable | None = None
    if not min_distA2B.empty:
        distances = min_distA2B[DISTANCE_PX]
        norm = Normalize(vmin=distances.min(), vmax=distances.max())
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

    ax.scatter(
        subpop_a_df[CENTROID_COL],
        subpop_a_df[CENTROID_ROW],
        s=20,
        c="darkblue",
        label=subpop_a_name,
        zorder=1,
        alpha=0.5,
    )

    sample_A2B = (
        min_distA2B.sample(min(len(min_distA2B), opts.max_pairs), random_state=RANDOM_STATE)
        if opts.max_pairs is not None
        else min_distA2B
    )
    if sm is not None:
        for _, row in sample_A2B.iterrows():
            ax.plot(
                [row["A_col"], row["B_col"]],
                [row["A_row"], row["B_row"]],
                color=cmap(norm(row[DISTANCE_PX])),
                linewidth=1,
                alpha=0.8,
                zorder=2,
            )

    ax.scatter(
        subpop_b_df[CENTROID_COL],
        subpop_b_df[CENTROID_ROW],
        s=20,
        c="darkgreen",
        label=subpop_b_name,
        zorder=3,
    )

    sample_B2A = (
        min_distB2A.sample(min(len(min_distB2A), opts.max_pairs), random_state=RANDOM_STATE)
        if opts.max_pairs is not None
        else min_distB2A
    )
    for _, row in sample_B2A.iterrows():
        ax.plot(
            [row["B_col"], row["A_col"]],
            [row["B_row"], row["A_row"]],
            c="lightgreen",
            linewidth=2,
            alpha=0.8,
            zorder=4,
        )

    ax.set_title(f"Distances from '{subpop_b_name}' to '{subpop_a_name}'")
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    ax.legend(
        existing_handles + shading_handles,
        existing_labels + [h.get_label() for h in shading_handles],
        loc="best",
    )
    if sm is not None:
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04).set_label("Distance (px)")


def _draw_voronoi_mode(
    ax: Axes,
    fig: Figure,
    subpop_a_df: pd.DataFrame,
    subpop_b_df: pd.DataFrame,
    subpop_a_name: str,
    subpop_b_name: str,
    dapi_mask: np.ndarray,
    opts: SubpopPlotOptions,
    shading_handles: list[Patch],
) -> bool:
    """Render Voronoi tessellation onto ax. Returns False if not enough seed points."""
    if len(subpop_a_df) <= len(subpop_b_df):
        seeds_df, points_df = subpop_a_df.copy(), subpop_b_df.copy()
        seeds_name, points_name = subpop_a_name, subpop_b_name
    else:
        seeds_df, points_df = subpop_b_df.copy(), subpop_a_df.copy()
        seeds_name, points_name = subpop_b_name, subpop_a_name

    seed_points = seeds_df[[CENTROID_COL, CENTROID_ROW]].values
    if len(seed_points) < 4:
        logger.warning("Not enough seed points for Voronoi (%d). Skipping.", len(seed_points))
        return False

    point_coords = points_df[[CENTROID_COL, CENTROID_ROW]].values
    distances, indices = KDTree(seed_points).query(point_coords, k=1)
    points_df = points_df.copy()
    points_df[DISTANCE_PX] = distances.flatten()

    norm = Normalize(vmin=points_df[DISTANCE_PX].min(), vmax=points_df[DISTANCE_PX].max())
    cmap = colormaps[_DISTANCE_CMAP]
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    try:
        vor = Voronoi(seed_points)
        regions, vertices = voronoi_finite_polygons_2d(vor)
        bounding_polygon = ShapelyPolygon(
            [
                (0, 0),
                (0, dapi_mask.shape[0]),
                (dapi_mask.shape[1], dapi_mask.shape[0]),
                (dapi_mask.shape[1], 0),
            ]
        )
        num_regions = len(regions)
        vor_colors = colormaps["tab20"](np.linspace(0, 1, num_regions))
        poly_patches, colors_voronoi = [], []
        for region_idx, region in enumerate(regions):
            shapely_poly = ShapelyPolygon(vertices[region]).intersection(bounding_polygon)
            if shapely_poly.is_empty:
                continue
            polys = shapely_poly.geoms if isinstance(shapely_poly, MultiPolygon) else [shapely_poly]
            for poly in polys:
                poly_patches.append(Polygon(np.array(poly.exterior.coords)))
                colors_voronoi.append(vor_colors[region_idx % num_regions])
        ax.add_collection(
            PatchCollection(
                poly_patches, facecolor=colors_voronoi, edgecolor="k", alpha=0.3, zorder=1
            )
        )
    except (QhullError, ValueError) as exc:
        logger.warning("Skipping Voronoi tessellation: %s", exc)

    ax.scatter(
        points_df[CENTROID_COL],
        points_df[CENTROID_ROW],
        c=points_df[DISTANCE_PX],
        cmap=cmap,
        s=20,
        label=f"{points_name} (by distance)",
        zorder=3,
        alpha=0.4,
    )
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04).set_label("Distance to Centroid (px)")
    ax.scatter(
        seeds_df[CENTROID_COL],
        seeds_df[CENTROID_ROW],
        s=40,
        c="green",
        marker="o",
        label=seeds_name,
        zorder=4,
    )
    ax.set_title(f"Voronoi between '{seeds_name}' and '{points_name}'")
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    ax.legend(
        existing_handles + shading_handles,
        existing_labels + [h.get_label() for h in shading_handles],
        loc="best",
    )
    return True


def plot_subpopulations_and_distances(
    roi: str,
    dapi_mask: np.ndarray,
    subpop_a_df: pd.DataFrame,
    subpop_b_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    *,
    subpop_a_name: str,
    subpop_b_name: str,
    options: SubpopPlotOptions | None = None,
) -> None:
    """Plot subpopulations and their pairwise distances within a ROI.

    plot_type: 'line' draws nearest-neighbor lines; 'voronoi' draws Voronoi regions.

    Raises:
        ValueError: If plot_type is not 'line' or 'voronoi'.
    """
    opts = options or SubpopPlotOptions()
    if opts.plot_type not in ("line", "voronoi"):
        raise ValueError(
            f"Unknown plot_type '{opts.plot_type}'. Supported values: 'line', 'voronoi'."
        )

    unique_labels = np.unique(dapi_mask)
    unique_labels = unique_labels[unique_labels != 0]
    rng = np.random.default_rng(RANDOM_STATE)
    colors = rng.random((len(unique_labels) + 1, 3))
    colors[0] = [0, 0, 0]
    cmap_cells = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(dapi_mask, cmap=cmap_cells, alpha=0.4, zorder=0)

    shading_handles: list[Patch] = []
    if opts.masks_to_shade and opts.shading_dict:
        shading_handles = shade_selected_masks(
            ax, roi, opts.masks_to_shade, opts.shading_dict, alpha=0.25
        )

    if opts.plot_type == "line":
        _draw_line_mode(
            ax,
            fig,
            dist_df,
            subpop_a_df,
            subpop_b_df,
            subpop_a_name,
            subpop_b_name,
            opts,
            shading_handles,
        )
    else:
        ok = _draw_voronoi_mode(
            ax,
            fig,
            subpop_a_df,
            subpop_b_df,
            subpop_a_name,
            subpop_b_name,
            dapi_mask,
            opts,
            shading_handles,
        )
        if not ok:
            plt.close(fig)
            return

    ax.set_axis_off()

    if opts.save_plot and opts.plot_filename:
        Path(opts.plot_filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(opts.plot_filename, format="svg", bbox_inches="tight")
        logger.info("Plot saved: %s", opts.plot_filename)

    show_figure(fig)
