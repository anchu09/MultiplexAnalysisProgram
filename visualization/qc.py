import logging
import os
import re
from collections.abc import Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from multiplex_pipeline.config import BOXPLOTS_DISTANCES_HEATMAPS_DIR
from multiplex_pipeline.schema import ROI

logger = logging.getLogger(__name__)


def plot_masks(dapi_masks_dict: dict[str, np.ndarray]) -> None:
    """Plot segmented DAPI masks with random per-label colours."""
    for extracted_name, mask in dapi_masks_dict.items():
        unique_labels = np.unique(mask)
        num_labels = len(unique_labels)
        colors = np.random.rand(num_labels, 3)
        colors[0] = [0, 0, 0]
        cmap = ListedColormap(colors)

        plt.figure(figsize=(12, 12))
        plt.imshow(mask, cmap=cmap, interpolation="nearest")
        plt.title(f"Segmented Mask - {extracted_name}")
        plt.colorbar(
            boundaries=np.arange(-0.5, num_labels, 1),
            ticks=np.linspace(0, num_labels - 1, min(20, num_labels), dtype=int),
        )
        plt.xlabel("X (px)")
        plt.ylabel("Y (px)")
        plt.show()

        logger.info(
            "Mask '%s': %d unique labels (excluding background).",
            extracted_name,
            num_labels - 1,
        )


def generate_boxplots_nested(
    nested_data: dict[str, dict[str, pd.DataFrame]],
    positive_col: str,
    negative_col: str,
    label: str,
    output_dir: str,
    prefix: str,
) -> None:
    """Generate violin + box plots for nested data grouped by positive/negative columns."""
    os.makedirs(output_dir, exist_ok=True)

    for key1, subdict in nested_data.items():
        df0 = next(iter(subdict.values()))

        if positive_col in df0.columns and negative_col in df0.columns:
            names2 = sorted(subdict.keys())
            pos_data, neg_data = [], []
            for roi in names2:
                df = subdict[roi]
                p = df[positive_col][df[positive_col] != 0].tolist() or [np.nan]
                n = df[negative_col][df[negative_col] != 0].tolist() or [np.nan]
                if len(p) == 1:
                    p *= 2
                if len(n) == 1:
                    n *= 2
                pos_data.append(p)
                neg_data.append(n)

            fig, ax = plt.subplots(figsize=(10, 6))
            positions = np.arange(1, len(names2) + 1)
            off = 0.2
            vp1 = ax.violinplot(
                pos_data,
                positions=positions - off,
                widths=0.35,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for b in vp1["bodies"]:
                b.set_facecolor("lightcoral")
                b.set_edgecolor("black")
                b.set_alpha(0.5)
            ax.boxplot(
                pos_data,
                positions=positions - off,
                widths=0.15,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor="white", edgecolor="black"),
            )
            vp2 = ax.violinplot(
                neg_data,
                positions=positions + off,
                widths=0.35,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for b in vp2["bodies"]:
                b.set_facecolor("lightblue")
                b.set_edgecolor("black")
                b.set_alpha(0.5)
            ax.boxplot(
                neg_data,
                positions=positions + off,
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
            title = f"{key1} \u27f6 {label}+ vs {label}-"

        else:
            vals = []
            for df in subdict.values():
                nums = df.select_dtypes(include=[np.number]).values.flatten()
                nums = nums[nums != 0]
                if nums.size == 0:
                    nums = np.array([np.nan])
                vals.extend(nums.tolist())
            if len(vals) == 1:
                vals *= 2

            fig, ax = plt.subplots(figsize=(6, 6))
            vp = ax.violinplot(
                [vals],
                positions=[1],
                widths=0.6,
                showmeans=False,
                showmedians=False,
                showextrema=False,
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
            title = f"{key1} \u27f6 {label} distances"
            pos_data = neg_data = None

        all_vals = sum(pos_data + neg_data, []) if pos_data is not None else vals
        all_vals = [v for v in all_vals if np.isfinite(v)]
        if any(v > 0 for v in all_vals):
            ax.set_yscale("log")

        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}"))
        ax.set_ylabel("Distance (\u00b5m)")
        ax.set_title(title, pad=15)
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        f = os.path.join(output_dir, f"{prefix}_{key1}.svg")
        fig.savefig(f)
        logger.info("Saved: %s", f)

        plt.show()
        plt.close(fig)


def generate_combined_boxplots(
    dic_distances: dict, save_path: str = str(BOXPLOTS_DISTANCES_HEATMAPS_DIR)
) -> None:
    """Generate combined violin + box plots for NGFR+/- subpopulation distances."""
    os.makedirs(save_path, exist_ok=True)

    pop_dict: dict = {}
    for _, files in dic_distances.items():
        for fname, df in files.items():
            m = re.match(r"(distance_matrix_(roi\d+)_.*NGFR_intensity)([\+\-])(_vs.*)\.csv", fname)
            if not m:
                continue
            roi, sign = m.group(2), m.group(3)
            pop = m.group(1).replace(f"distance_matrix_{roi}_", "") + m.group(4)
            arr = pd.to_numeric(df.values.flatten(), errors="coerce")
            arr = arr[~np.isnan(arr)]
            pop_dict.setdefault(pop, {}).setdefault(roi, {})["plus" if sign == "+" else "minus"] = (
                arr
            )

    c_plus, c_minus = "lightgreen", "lightcoral"

    for pop, rois in pop_dict.items():
        roi_digits = []
        for x in rois:
            found = re.findall(r"\d+", x)
            if not found:
                logger.warning("Cannot extract ROI number from '%s'. Skipping sort.", x)
                roi_digits = list(rois.keys())
                break
            roi_digits.append((int(found[0]), x))
        else:
            roi_digits_sorted = [x for _, x in sorted(roi_digits)]
            sorted_rois = roi_digits_sorted

        fig, ax = plt.subplots(figsize=(8, 5))
        pos_plus, data_plus = [], []
        pos_minus, data_minus = [], []

        for i, roi in enumerate(sorted_rois):
            d = rois[roi]
            if "plus" in d and "minus" in d:
                pos_plus.append(3 * i + 1)
                data_plus.append(d["plus"])
                pos_minus.append(3 * i + 2)
                data_minus.append(d["minus"])

        if not (data_plus and data_minus):
            logger.warning("No valid data for subpopulation '%s'.", pop)
            plt.close(fig)
            continue

        bp1 = ax.boxplot(
            data_plus, positions=pos_plus, patch_artist=True, widths=0.6, showfliers=False
        )
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

        centers = [
            (3 * i + 1 + 3 * i + 2) / 2
            for i, roi in enumerate(sorted_rois)
            if "plus" in rois[roi] and "minus" in rois[roi]
        ]
        valid_labels = [roi for roi in sorted_rois if "plus" in rois[roi] and "minus" in rois[roi]]
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
        plt.savefig(os.path.join(save_path, f"{safe}.svg"), format="svg")
        plt.show()
        plt.close(fig)

    # Per-ROI plots
    roi_dict: dict = {}
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
            if "plus" in d and "minus" in d:
                pos_plus.append(3 * i + 1)
                pos_minus.append(3 * i + 2)
                data_plus.append(d["plus"])
                data_minus.append(d["minus"])

        if not (data_plus and data_minus):
            logger.warning("No valid data for ROI '%s'.", roi)
            plt.close(fig)
            continue

        bp1 = ax.boxplot(
            data_plus, positions=pos_plus, patch_artist=True, widths=0.6, showfliers=False
        )
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
        plt.savefig(os.path.join(save_path, f"{safe}.svg"), format="svg")
        plt.show()
        plt.close(fig)


def plot_combination_counts(
    df: pd.DataFrame,
    rois: list[str],
    combinations: dict[str, Callable[[pd.DataFrame], pd.Series]],
    output_dir: str,
    base_filename: str,
    plot_title: str,
    figsize: tuple[int, int] = (20, 10),
    rotation: int = 45,
    font_scale: float = 1.0,
) -> pd.DataFrame:
    """Count and plot cells matching each combination condition per ROI.

    Saves a CSV and an SVG barplot to output_dir.
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

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    counts_df.to_csv(csv_path, index=False)
    logger.info("Table saved: %s", csv_path)

    plt.figure(figsize=figsize)
    sns.set_context("talk", font_scale=font_scale)
    ax = sns.barplot(x=ROI, y="Count", hue="Combination", data=counts_melted)

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
    ax.set_title(plot_title, fontsize=16)
    ax.set_xlabel(ROI, fontsize=14)
    ax.set_ylabel("Number of Positive Cells", fontsize=14)
    ax.legend(title="Combination", fontsize=12, title_fontsize=14)
    plt.tight_layout()

    svg_path = os.path.join(output_dir, f"{base_filename}.svg")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    logger.info("Plot saved: %s", svg_path)

    plt.show()
    return counts_df
