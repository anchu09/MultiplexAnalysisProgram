"""Exploratory data-analysis plots for the balanced H&E patch dataset.

Each function accepts a pandas DataFrame (one row per patch) and an output
path, generates a matplotlib figure, saves it as SVG, and closes the figure
to avoid memory leaks.

Expected DataFrame columns match multiplex_pipeline.hne.schema.
"""

from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiplex_pipeline.hne.schema import (
    DOMINANT_CLASS,
    MEAN_EOSIN,
    MEAN_GRAY,
    MEAN_HEMATOXYLIN,
    PROP_BACKGROUND,
    PROP_FRONT,
    PROP_STROMA,
    STROMA_QUARTILE,
)
from pandas.plotting import scatter_matrix

_CLASS_COLORS: dict[str, str] = {
    PROP_BACKGROUND: "lightgray",
    PROP_FRONT: "salmon",
    PROP_STROMA: "lightgreen",
}


def plot_class_proportions(df: pd.DataFrame, output_path: str | os.PathLike) -> None:
    """Histograms of background, invasion-front, and stroma proportions.

    Parameters
    ----------
    df:
        DataFrame with columns PROP_BACKGROUND, PROP_FRONT, PROP_STROMA.
    output_path:
        Path to the output SVG file.
    """
    cols = [PROP_BACKGROUND, PROP_FRONT, PROP_STROMA]
    colors = ["lightgray", "salmon", "lightgreen"]
    titles = ["Prop Background", "Prop Front Of Invasion", "Prop Stroma"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, color, title in zip(axes, cols, colors, titles, strict=False):
        df[col].hist(bins=30, ax=ax, color=color, edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel("Proportion")
        ax.set_ylabel("Number of Patches")
    fig.tight_layout()
    _save(fig, output_path)


def plot_intensity_distributions(df: pd.DataFrame, output_path: str | os.PathLike) -> None:
    """Histograms of mean grayscale brightness, hematoxylin, and eosin intensity.

    Parameters
    ----------
    df:
        DataFrame with columns MEAN_GRAY, MEAN_HEMATOXYLIN, MEAN_EOSIN.
    output_path:
        Path to the output SVG file.
    """
    cols = [MEAN_GRAY, MEAN_HEMATOXYLIN, MEAN_EOSIN]
    colors = ["silver", "steelblue", "orchid"]
    titles = ["Mean Brightness (Grayscale)", "Mean Hematoxylin Intensity", "Mean Eosin Intensity"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, color, title in zip(axes, cols, colors, titles, strict=False):
        df[col].hist(bins=30, ax=ax, color=color, edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel("Normalized Value")
        ax.set_ylabel("Number of Patches")
    fig.tight_layout()
    _save(fig, output_path)


def plot_correlation_matrix(df: pd.DataFrame, output_path: str | os.PathLike) -> None:
    """Annotated Pearson correlation heatmap between class proportions and intensities.

    Parameters
    ----------
    df:
        DataFrame with proportion and intensity columns.
    output_path:
        Path to the output SVG file.
    """
    features = [PROP_BACKGROUND, PROP_FRONT, PROP_STROMA, MEAN_GRAY, MEAN_HEMATOXYLIN, MEAN_EOSIN]
    corr = df[features].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ticks = range(len(features))
    ax.set_xticks(list(ticks))
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_yticks(list(ticks))
    ax.set_yticklabels(features)
    for i, j in np.ndindex(corr.shape):
        ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    _save(fig, output_path)


def plot_scatter_matrix(df: pd.DataFrame, output_path: str | os.PathLike) -> None:
    """Pairwise scatter matrix with KDE distributions on the diagonal, coloured by dominant class.

    Parameters
    ----------
    df:
        DataFrame with proportion and intensity columns.
    output_path:
        Path to the output SVG file.
    """
    features = [PROP_BACKGROUND, PROP_FRONT, PROP_STROMA, MEAN_GRAY, MEAN_HEMATOXYLIN, MEAN_EOSIN]

    df = df.copy()
    if DOMINANT_CLASS not in df.columns:
        df[DOMINANT_CLASS] = df[[PROP_BACKGROUND, PROP_FRONT, PROP_STROMA]].idxmax(axis=1)

    color_series = df[DOMINANT_CLASS].map(_CLASS_COLORS).fillna("lightgray")

    sm = scatter_matrix(
        df[features],
        alpha=0.5,
        figsize=(10, 10),
        diagonal=None,
        marker="o",
        color=color_series,
    )

    groups = {cls: df[df[DOMINANT_CLASS] == cls] for cls in _CLASS_COLORS}
    for i, feat in enumerate(features):
        ax = sm[i, i]
        ax.clear()
        for cls, data in groups.items():
            data[feat].plot(
                kind="kde",
                ax=ax,
                color=_CLASS_COLORS[cls],
                label=cls.replace("prop_", "").replace("_", " ").title(),
            )
        ax.set_xlabel(feat)
        if i == 0:
            ax.legend(title="Dominant Class", loc="best", fontsize=7)

    handles = [
        mpatches.Patch(
            color=_CLASS_COLORS[k], label=k.replace("prop_", "").replace("_", " ").title()
        )
        for k in _CLASS_COLORS
    ]
    plt.gcf().legend(handles=handles, loc="upper right", title="Dominant Class")
    plt.suptitle("Pairwise Scatter Matrix with Colored Diagonal", y=1.02)
    plt.tight_layout()
    _save(plt.gcf(), output_path)


def plot_eosin_by_stroma_quartile(df: pd.DataFrame, output_path: str | os.PathLike) -> None:
    """Box-plots of mean eosin intensity stratified by stroma proportion quartile.

    Parameters
    ----------
    df:
        DataFrame with PROP_STROMA and MEAN_EOSIN.
    output_path:
        Path to the output SVG file.
    """
    df = df.copy()
    df[STROMA_QUARTILE] = pd.qcut(df[PROP_STROMA], 4, labels=["Q1", "Q2", "Q3", "Q4"])

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [df[df[STROMA_QUARTILE] == q][MEAN_EOSIN].dropna() for q in ["Q1", "Q2", "Q3", "Q4"]]
    ax.boxplot(
        data,
        tick_labels=["Q1", "Q2", "Q3", "Q4"],
        patch_artist=True,
        boxprops=dict(facecolor="mistyrose", edgecolor="black"),
        medianprops=dict(color="black"),
    )
    ax.set_xlabel("Stroma Quartile")
    ax.set_ylabel("Mean Eosin Intensity")
    ax.set_title("Eosin Intensity by Stroma Quartile")
    fig.tight_layout()
    _save(fig, output_path)


def _save(fig: plt.Figure, path: str | os.PathLike) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(str(path), format="svg", bbox_inches="tight")
    plt.close(fig)
