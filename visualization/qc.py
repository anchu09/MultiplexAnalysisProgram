import logging
import re
from collections.abc import Callable
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from multiplex_pipeline.config import BOXPLOTS_DISTANCES_HEATMAPS_DIR, RANDOM_STATE
from multiplex_pipeline.schema import ROI
from multiplex_pipeline.visualization.data_prep import (
    SIGN_MINUS,
    SIGN_PLUS,
    parse_distance_matrix_filename,
)
from multiplex_pipeline.visualization.plotting import show_figure

logger = logging.getLogger(__name__)


def plot_masks(dapi_masks_dict: dict[str, np.ndarray]) -> None:
    """Plot segmented DAPI masks with random per-label colors."""
    for extracted_name, mask in dapi_masks_dict.items():
        unique_labels = np.unique(mask)
        num_labels = len(unique_labels)
        rng = np.random.default_rng(RANDOM_STATE)
        colors = rng.random((num_labels, 3))
        colors[0] = [0, 0, 0]
        cmap = ListedColormap(colors)

        fig_mask, ax_mask = plt.subplots(figsize=(12, 12))
        im = ax_mask.imshow(mask, cmap=cmap, interpolation="nearest")
        ax_mask.set_title(f"Segmented Mask - {extracted_name}")
        fig_mask.colorbar(
            im,
            ax=ax_mask,
            boundaries=np.arange(-0.5, num_labels, 1),
            ticks=np.linspace(0, num_labels - 1, min(20, num_labels), dtype=int),
        )
        ax_mask.set_xlabel("X (px)")
        ax_mask.set_ylabel("Y (px)")
        show_figure(fig_mask)
        plt.close(fig_mask)

        logger.info(
            "Mask '%s': %d unique labels (excluding background).",
            extracted_name,
            num_labels - 1,
        )


def _pad_for_violin(data: list) -> list:
    """Return data unchanged, or duplicated if only 1 point (violin minimum).

    A duplicated single value satisfies matplotlib's minimum but the resulting
    violin shape carries no distributional information.
    """
    if len(data) == 1:
        logger.warning(
            "Only 1 data point available for violin plot — duplicating to satisfy matplotlib "
            "minimum. The resulting shape is not statistically meaningful."
        )
        return data * 2
    return data


def _render_paired_violin_box(
    ax: plt.Axes,
    subdict: dict[str, pd.DataFrame],
    positive_col: str,
    negative_col: str,
    label: str,
    key1: str,
) -> tuple[str, list]:
    """Render paired positive/negative violin+box onto ax. Returns (title, all_vals)."""
    names2 = sorted(subdict.keys())
    pos_data: list[list] = []
    neg_data: list[list] = []
    for roi in names2:
        df = subdict[roi]
        pos_data.append(
            _pad_for_violin(df[positive_col][df[positive_col] != 0].tolist() or [np.nan])
        )
        neg_data.append(
            _pad_for_violin(df[negative_col][df[negative_col] != 0].tolist() or [np.nan])
        )

    positions = np.arange(1, len(names2) + 1)
    off = 0.2
    for data, color, offset in [
        (pos_data, "lightcoral", -off),
        (neg_data, "lightblue", +off),
    ]:
        vp = ax.violinplot(
            data,
            positions=positions + offset,
            widths=0.35,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for b in vp["bodies"]:
            b.set_facecolor(color)
            b.set_edgecolor("black")
            b.set_alpha(0.5)
        ax.boxplot(
            data,
            positions=positions + offset,
            widths=0.15,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="white", edgecolor="black"),
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(names2, rotation=45, ha="right")
    ax.legend(
        handles=[
            mpatches.Patch(color="lightcoral", label=f"{label}+"),
            mpatches.Patch(color="lightblue", label=f"{label}-"),
        ],
        loc="best",
    )
    all_vals = sum(pos_data + neg_data, [])
    return f"{key1} \u27f6 {label}+ vs {label}-", all_vals


def _render_single_violin_box(
    ax: plt.Axes,
    subdict: dict[str, pd.DataFrame],
    key1: str,
    label: str,
) -> tuple[str, list]:
    """Render single-series violin+box onto ax. Returns (title, all_vals)."""
    vals: list = []
    for df in subdict.values():
        nums = df.select_dtypes(include=[np.number]).values.flatten()
        nums = nums[nums != 0]
        vals.extend((nums if nums.size > 0 else np.array([np.nan])).tolist())
    vals = _pad_for_violin(vals)

    vp = ax.violinplot(
        [vals], positions=[1], widths=0.6, showmeans=False, showmedians=False, showextrema=False
    )
    for b in vp["bodies"]:
        b.set_facecolor("skyblue")
        b.set_edgecolor("black")
        b.set_alpha(0.5)
    ax.boxplot(
        [vals],
        positions=[1],
        widths=0.2,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="white", edgecolor="black"),
    )
    ax.set_xticks([1])
    ax.set_xticklabels([key1], rotation=45, ha="right")
    ax.legend(handles=[mpatches.Patch(color="skyblue", label=label)], loc="best")
    return f"{key1} \u27f6 {label} distances", vals


def generate_boxplots_nested(
    nested_data: dict[str, dict[str, pd.DataFrame]],
    positive_col: str,
    negative_col: str,
    label: str,
    output_dir: str,
    prefix: str,
) -> None:
    """Generate violin + box plots for nested data grouped by positive/negative columns."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for key1, subdict in nested_data.items():
        if not subdict:
            logger.warning("Skipping '%s': empty subdict.", key1)
            continue
        df0 = next(iter(subdict.values()))
        has_paired = positive_col in df0.columns and negative_col in df0.columns

        if has_paired:
            fig, ax = plt.subplots(figsize=(10, 6))
            title, all_vals = _render_paired_violin_box(
                ax, subdict, positive_col, negative_col, label, key1
            )
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            title, all_vals = _render_single_violin_box(ax, subdict, key1, label)

        all_vals = [v for v in all_vals if np.isfinite(v)]
        if any(v > 0 for v in all_vals):
            ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}"))
        ax.set_ylabel("Distance (\u00b5m)")
        ax.set_title(title, pad=15)
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        f = str(Path(output_dir) / f"{prefix}_{key1}.svg")
        fig.savefig(f)
        logger.info("Saved: %s", f)
        show_figure(fig)
        plt.close(fig)


def _add_violin_boxplot_pair(
    ax: plt.Axes,
    data_plus: list,
    data_minus: list,
    pos_plus: list[int],
    pos_minus: list[int],
    c_plus: str,
    c_minus: str,
) -> None:
    """Draw paired boxplots and violins for NGFR+/- groups on ax."""
    bp1 = ax.boxplot(data_plus, positions=pos_plus, patch_artist=True, widths=0.6, showfliers=False)
    bp2 = ax.boxplot(
        data_minus, positions=pos_minus, patch_artist=True, widths=0.6, showfliers=False
    )
    for b in bp1["boxes"]:
        b.set(facecolor=c_plus, alpha=0.5)
    for b in bp2["boxes"]:
        b.set(facecolor=c_minus, alpha=0.5)
    vp1 = ax.violinplot(data_plus, positions=pos_plus, widths=0.6, showextrema=False)
    vp2 = ax.violinplot(data_minus, positions=pos_minus, widths=0.6, showextrema=False)
    for v in vp1["bodies"]:
        v.set_facecolor(c_plus)
        v.set_edgecolor("black")
        v.set_alpha(0.5)
    for v in vp2["bodies"]:
        v.set_facecolor(c_minus)
        v.set_edgecolor("black")
        v.set_alpha(0.5)


def generate_combined_boxplots(
    dic_distances: dict[str, dict[str, pd.DataFrame]],
    save_path: str = str(BOXPLOTS_DISTANCES_HEATMAPS_DIR),
) -> None:
    """Generate combined violin + box plots for NGFR+/- subpopulation distances."""
    Path(save_path).mkdir(parents=True, exist_ok=True)

    pop_dict: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for _, files in dic_distances.items():
        for fname, df in files.items():
            result = parse_distance_matrix_filename(fname)
            if result is None:
                continue
            roi, sign, pop = result
            arr = pd.to_numeric(df.values.flatten(), errors="coerce")
            arr = arr[~np.isnan(arr)]
            pop_dict.setdefault(pop, {}).setdefault(roi, {})[
                SIGN_PLUS if sign == "+" else SIGN_MINUS
            ] = arr

    c_plus, c_minus = "lightgreen", "lightcoral"

    for pop, rois in pop_dict.items():
        roi_digits = []
        for x in rois:
            found = re.findall(r"\d+", x)
            if not found:
                logger.warning("Cannot extract ROI number from '%s'. Skipping sort.", x)
                sorted_rois = list(rois.keys())
                break
            roi_digits.append((int(found[0]), x))
        else:
            sorted_rois = [x for _, x in sorted(roi_digits)]

        fig, ax = plt.subplots(figsize=(8, 5))
        pos_plus, data_plus = [], []
        pos_minus, data_minus = [], []

        for i, roi in enumerate(sorted_rois):
            d = rois[roi]
            if SIGN_PLUS in d and SIGN_MINUS in d:
                pos_plus.append(3 * i + 1)
                data_plus.append(d[SIGN_PLUS])
                pos_minus.append(3 * i + 2)
                data_minus.append(d[SIGN_MINUS])

        if not (data_plus and data_minus):
            logger.warning("No valid data for subpopulation '%s'.", pop)
            plt.close(fig)
            continue

        _add_violin_boxplot_pair(ax, data_plus, data_minus, pos_plus, pos_minus, c_plus, c_minus)

        centers = [
            (3 * i + 1 + 3 * i + 2) / 2
            for i, roi in enumerate(sorted_rois)
            if SIGN_PLUS in rois[roi] and SIGN_MINUS in rois[roi]
        ]
        valid_labels = [
            roi for roi in sorted_rois if SIGN_PLUS in rois[roi] and SIGN_MINUS in rois[roi]
        ]
        ax.set_xticks(centers)
        ax.set_xticklabels(valid_labels)
        ax.set_ylabel("Distance [\u00b5m]")

        pop_match = re.search(r"vs_(.*)", pop)
        pop_name = pop_match.group(1).replace("_", " ") if pop_match else pop
        ax.set_title(f"Distances tumor vs {pop_name}")
        ax.legend(
            [mpatches.Patch(facecolor=c_plus), mpatches.Patch(facecolor=c_minus)],
            ["NGFR+", "NGFR\u2212"],
            loc="upper right",
        )
        plt.tight_layout()
        safe = re.sub(r"[^\w\-_\. ]", "_", pop_name)
        plt.savefig(str(Path(save_path) / f"{safe}.svg"), format="svg")
        show_figure(fig)
        plt.close(fig)

    # Per-ROI plots
    roi_dict: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for pop, rois in pop_dict.items():
        for roi, d in rois.items():
            roi_dict.setdefault(roi, {})[pop] = d

    for roi, pops in roi_dict.items():
        sorted_pops = sorted(pops)
        fig, ax = plt.subplots(figsize=(10, 6))
        pos_plus, data_plus = [], []
        pos_minus, data_minus = [], []

        for i, pop in enumerate(sorted_pops):
            d = pops[pop]
            if SIGN_PLUS in d and SIGN_MINUS in d:
                pos_plus.append(3 * i + 1)
                pos_minus.append(3 * i + 2)
                data_plus.append(d[SIGN_PLUS])
                data_minus.append(d[SIGN_MINUS])

        if not (data_plus and data_minus):
            logger.warning("No valid data for ROI '%s'.", roi)
            plt.close(fig)
            continue

        _add_violin_boxplot_pair(ax, data_plus, data_minus, pos_plus, pos_minus, c_plus, c_minus)

        centers = [(3 * i + 1 + 3 * i + 2) / 2 for i in range(len(sorted_pops))]
        ax.set_xticks(centers)
        ax.set_xticklabels([p.replace("_", " ") for p in sorted_pops], rotation=90, ha="center")
        ax.set_ylabel("Distance [\u00b5m]")
        ax.set_title(f"Distances for ROI {roi}")
        ax.legend(
            [mpatches.Patch(facecolor=c_plus), mpatches.Patch(facecolor=c_minus)],
            ["NGFR+", "NGFR\u2212"],
            loc="upper right",
        )
        plt.tight_layout()
        safe = re.sub(r"[^\w\-_\. ]", "_", roi)
        plt.savefig(str(Path(save_path) / f"{safe}.svg"), format="svg")
        show_figure(fig)
        plt.close(fig)


def plot_combination_counts(
    df: pd.DataFrame,
    rois: list[str],
    combinations: dict[str, Callable[[pd.DataFrame], pd.Series]],
    output_dir: str | None = None,
    base_filename: str = "combination_counts",
    plot_title: str | None = None,
    figsize: tuple[int, int] = (20, 10),
    rotation: int = 45,
    font_scale: float = 1.0,
) -> pd.DataFrame:
    """Count and plot cells matching each combination condition per ROI.

    If output_dir is given, saves a CSV and an SVG barplot there.
    """
    filtered = df[df[ROI].isin(rois)]
    counts_df = pd.DataFrame()
    grouped = filtered.groupby(ROI)

    for name, cond in combinations.items():
        counts_df[name] = grouped.apply(lambda x, c=cond: c(x).sum())
    counts_df = counts_df.reset_index()
    counts_df["Total Cells"] = grouped.size().values

    counts_melted = counts_df.melt(
        id_vars=[ROI, "Total Cells"], var_name="Combination", value_name="Count"
    )

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        csv_path = str(Path(output_dir) / f"{base_filename}.csv")
        counts_df.to_csv(csv_path, index=False)
        logger.info("Table saved: %s", csv_path)

    fig, ax = plt.subplots(figsize=figsize)
    sns.set_context("talk", font_scale=font_scale)
    sns.barplot(x=ROI, y="Count", hue="Combination", data=counts_melted, ax=ax)

    for p in ax.patches:
        h = p.get_height()
        if pd.notnull(h) and h > 0:
            ax.text(
                p.get_x() + p.get_width() / 2,
                h + 0.1,
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=10,
                rotation=90,
            )

    new_lbls = [
        f"{roi}\n(Total: {tot})"
        for roi, tot in zip(counts_df[ROI], counts_df["Total Cells"], strict=False)
    ]
    ax.set_xticklabels(new_lbls, rotation=rotation)
    ax.set_yscale("log")
    ax.set_title(plot_title or base_filename, fontsize=16)
    ax.set_xlabel(ROI, fontsize=14)
    ax.set_ylabel("Number of Positive Cells", fontsize=14)
    ax.legend(title="Combination", fontsize=12, title_fontsize=14)
    fig.tight_layout()

    if output_dir is not None:
        svg_path = str(Path(output_dir) / f"{base_filename}.svg")
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        logger.info("Plot saved: %s", svg_path)

    show_figure(fig)
    plt.close(fig)
    return counts_df
