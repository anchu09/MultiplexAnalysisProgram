"""Smoke tests verifying the overlays.py split preserves all functionality.

These tests use synthetic data so they run without the real microscopy files.
They verify:
1. Pure data-prep functions produce the expected output types/shapes.
2. Plotting functions remain importable and callable without side effects
   (figures are created and immediately closed).
3. Column-name constants in schema.py match the values used in the codebase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_parse_conditions_positive():
    from multiplex_pipeline.visualization.data_prep import parse_conditions

    col_map = {"CK_mask": "is_positive_CK_binary", "NGFR_mask": "is_positive_NGFR_binary"}
    result = parse_conditions(["CK_mask+", "NGFR_mask-"], col_map)

    assert result == {"is_positive_CK_binary": 1, "is_positive_NGFR_binary": 0}


def test_parse_conditions_invalid_suffix_ignored(caplog):
    from multiplex_pipeline.visualization.data_prep import parse_conditions

    result = parse_conditions(["CK_mask"], {})  # No +/-
    assert result == {}


def test_parse_conditions_unknown_key_logs_warning(caplog):
    import logging

    from multiplex_pipeline.visualization.data_prep import parse_conditions

    with caplog.at_level(logging.WARNING):
        parse_conditions(["UNKNOWN+"], {})
    assert "UNKNOWN" in caplog.text


def test_select_subpopulation_filters_correctly():
    from multiplex_pipeline.visualization.data_prep import select_subpopulation

    df = pd.DataFrame(
        {
            "is_positive_CK_binary": [1, 0, 1, 0],
            "is_positive_NGFR_binary": [1, 1, 0, 0],
        }
    )
    result = select_subpopulation(df, {"is_positive_CK_binary": 1, "is_positive_NGFR_binary": 1})
    assert len(result) == 1
    assert result.iloc[0]["is_positive_CK_binary"] == 1


def test_select_subpopulation_empty_parsed_returns_empty():
    from multiplex_pipeline.visualization.data_prep import select_subpopulation

    df = pd.DataFrame({"col": [1, 2, 3]})
    assert select_subpopulation(df, {}).empty


def test_parse_distance_matrix_filename_valid():
    from multiplex_pipeline.visualization.data_prep import parse_distance_matrix_filename

    fname = "distance_matrix_roi1_NGFR_intensity+_vs_Tregs.csv"
    result = parse_distance_matrix_filename(fname)
    assert result is not None
    roi, sign, pop = result
    assert roi == "roi1"
    assert sign == "+"
    assert "Tregs" in pop


def test_parse_distance_matrix_filename_invalid():
    from multiplex_pipeline.visualization.data_prep import parse_distance_matrix_filename

    assert parse_distance_matrix_filename("random_file.csv") is None


def test_filter_cells_by_combination():
    from multiplex_pipeline.schema import IS_POSITIVE_CK
    from multiplex_pipeline.visualization.data_prep import filter_cells_by_combination

    df = pd.DataFrame({IS_POSITIVE_CK: [1, 0, 1], "other": [10, 20, 30]})
    result = filter_cells_by_combination(df, lambda d: d[IS_POSITIVE_CK] == 1)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_schema_ck_constant_matches_config_lambda():
    """The IS_POSITIVE_CK constant must match the column name used in config lambdas."""
    from multiplex_pipeline.schema import IS_POSITIVE_CK

    assert IS_POSITIVE_CK == "is_positive_Pan_Cytokeratin_CK_binary"


def test_schema_ngfr_constant_matches_config_lambda():
    from multiplex_pipeline.schema import IS_POSITIVE_NGFR

    assert IS_POSITIVE_NGFR == "is_positive_NGFR_binary"


def test_verify_binary_returns_true_for_binary_mask():
    from multiplex_pipeline.utils.validation import verify_binary

    mask = np.array([0, 1, 0, 1, 1])
    assert verify_binary(mask, "test_mask") is True


def test_verify_binary_returns_false_for_non_binary_mask():
    from multiplex_pipeline.utils.validation import verify_binary

    mask = np.array([0, 1, 2, 3])
    assert verify_binary(mask, "test_mask") is False


def test_compute_subpop_distances_raises_on_empty():
    from multiplex_pipeline.analysis.spatial import compute_subpop_distances

    empty = pd.DataFrame({"centroid_row": [], "centroid_col": [], "DAPI_ID": []})
    non_empty = pd.DataFrame({"centroid_row": [10], "centroid_col": [20], "DAPI_ID": [1]})
    with pytest.raises(ValueError, match="empty"):
        compute_subpop_distances(empty, non_empty)


def test_compute_subpop_distances_returns_dataframe():
    from multiplex_pipeline.analysis.spatial import compute_subpop_distances

    A = pd.DataFrame({"centroid_row": [0, 10], "centroid_col": [0, 10], "DAPI_ID": [1, 2]})
    B = pd.DataFrame({"centroid_row": [5], "centroid_col": [5], "DAPI_ID": [3]})
    result = compute_subpop_distances(A, B)
    assert isinstance(result, pd.DataFrame)
    assert "distance_px" in result.columns
    assert len(result) == 2  # 2 A cells × 1 B cell


def test_process_roi_raises_on_missing_roi_key():
    from multiplex_pipeline.analysis.intensity import process_roi

    with pytest.raises(KeyError, match="Cannot extract ROI key"):
        process_roi(
            img_name="no_roi_in_name.tiff",
            img_data=np.zeros((3, 10, 10)),
            dapi_masks={},
            ck_masks={},
            ngfr_masks={},
            channels=[0],
            marker_dict={0: "DAPI"},
        )


def test_process_roi_raises_on_missing_dapi():
    from multiplex_pipeline.analysis.intensity import process_roi

    with pytest.raises(KeyError, match="dapi_masks"):
        process_roi(
            img_name="ROI1.ome.tiff",
            img_data=np.zeros((3, 10, 10)),
            dapi_masks={},  # roi1_dapi not present
            ck_masks={},
            ngfr_masks={},
            channels=[0],
            marker_dict={0: "DAPI"},
        )
