"""Data preparation utilities for visualization — no matplotlib calls here."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_conditions(cond_list: list[str], col_map: dict[str, str]) -> dict[str, int]:
    """Convert condition strings to {column_name: expected_value} pairs.

    Example::

        parse_conditions(['CK_mask+', 'NGFR_mask-'], col_map)
        # -> {'is_positive_Pan_Cytokeratin_CK_binary': 1,
        #     'is_positive_NGFR_binary': 0}
    """
    parsed: dict[str, int] = {}
    for cond in (c.strip() for c in cond_list):
        if cond.endswith("+"):
            base, val = cond[:-1].strip(), 1
        elif cond.endswith("-"):
            base, val = cond[:-1].strip(), 0
        else:
            logger.warning("Ignoring invalid condition '%s' (must end with '+' or '-').", cond)
            continue
        if base in col_map:
            parsed[col_map[base]] = val
        else:
            logger.warning("No column mapping found for condition base '%s'.", base)
    return parsed


def select_subpopulation(df: pd.DataFrame, parsed: dict[str, int]) -> pd.DataFrame:
    """Filter a DataFrame by parsed conditions.

    Returns an empty DataFrame if parsed is empty.
    """
    if not parsed:
        return df.iloc[0:0]
    mask = pd.Series(True, index=df.index)
    for col, val in parsed.items():
        if col not in df.columns:
            logger.warning(
                "Column '%s' not found in DataFrame — subpopulation filter will return 0 cells.",
                col,
            )
            mask &= False
        else:
            mask &= df[col] == val
    return df[mask]


def parse_distance_matrix_filename(
    fname: str,
) -> tuple[str, str, str] | None:
    """Parse a distance-matrix CSV filename into (roi, sign, population_name).

    Expected pattern: distance_matrix_{roi}_...NGFR_intensity{sign}_vs{suffix}.csv

    Returns None if the filename does not match.
    """
    m = re.match(
        r"(distance_matrix_(roi\d+)_.*NGFR_intensity)([\+\-])(_vs.*)\.csv",
        fname,
    )
    if not m:
        return None
    roi = m.group(2)
    sign = m.group(3)
    pop = m.group(1).replace(f"distance_matrix_{roi}_", "") + m.group(4)
    return roi, sign, pop


def load_distance_matrices_for_plot(
    base_path: Path,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Load and organise distance matrix CSVs for combined boxplot generation.

    Groups files by population name and ROI, keeping 'plus' (NGFR+) and
    'minus' (NGFR-) arrays.

    Returns:
        Dict of shape {population: {roi: {'plus': array, 'minus': array}}}.
    """
    pop_dict: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    base_path = Path(base_path)

    for sub in base_path.iterdir():
        if not sub.is_dir():
            continue
        for csv_file in sub.glob("*.csv"):
            result = parse_distance_matrix_filename(csv_file.name)
            if result is None:
                continue
            roi, sign, pop = result
            df = pd.read_csv(csv_file)
            arr = pd.to_numeric(df.values.flatten(), errors="coerce")
            arr = arr[~np.isnan(arr)]
            key = "plus" if sign == "+" else "minus"
            pop_dict.setdefault(pop, {}).setdefault(roi, {})[key] = arr

    return pop_dict


def filter_cells_by_combination(
    df: pd.DataFrame,
    combination_func,
) -> pd.DataFrame:
    """Apply a combination filter function to a DataFrame.

    combination_func: callable that receives df and returns a boolean Series
    (e.g. a lambda from config.CHARACTERIZATION_COMBINATIONS).
    """
    mask = combination_func(df)
    return df[mask]
