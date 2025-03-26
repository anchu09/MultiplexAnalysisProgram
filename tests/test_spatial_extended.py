"""Extended tests for analysis/spatial.py — covers compute_subpop_cells_per_area."""

from __future__ import annotations

import pandas as pd
import pytest
from multiplex_pipeline.analysis.spatial import compute_subpop_cells_per_area
from multiplex_pipeline.schema import (
    CELLS_PER_UM2_CK,
    CK_NGFR_POSITIVE_AREA_UM2,
    CK_POSITIVE_AREA_UM2,
    IS_POSITIVE_CK,
    ROI,
    SUBPOP_CELL_COUNT,
    TOTAL_AREA_ROI_UM2,
)


@pytest.fixture()
def binary_df():
    return pd.DataFrame(
        {
            ROI: ["roi1"] * 6,
            IS_POSITIVE_CK: [1, 1, 1, 0, 0, 0],
            "mean_intensity_CD3_binary": [1, 0, 1, 1, 0, 0],
        }
    )


@pytest.fixture()
def area_df():
    return pd.DataFrame(
        {
            ROI: ["roi1"],
            CK_POSITIVE_AREA_UM2: [100.0],
            CK_NGFR_POSITIVE_AREA_UM2: [50.0],
            TOTAL_AREA_ROI_UM2: [200.0],
        }
    )


@pytest.fixture()
def cond_map():
    return {
        "CK_mask": IS_POSITIVE_CK,
        "CD3_intensity": "mean_intensity_CD3_binary",
    }


class TestComputeSubpopCellsPerArea:
    def test_basic_count(self, binary_df, area_df, cond_map):
        summary, _ = compute_subpop_cells_per_area(
            df_binary=binary_df,
            subpop_conditions=["CK_mask+"],
            cond_map=cond_map,
            mask_summary=area_df,
            rois=["roi1"],
        )
        assert len(summary) == 1
        assert summary.iloc[0][SUBPOP_CELL_COUNT] == 3  # 3 CK+ cells

    def test_intersection_condition(self, binary_df, area_df, cond_map):
        summary, _ = compute_subpop_cells_per_area(
            df_binary=binary_df,
            subpop_conditions=["CK_mask+", "CD3_intensity+"],
            cond_map=cond_map,
            mask_summary=area_df,
            rois=["roi1"],
        )
        assert summary.iloc[0][SUBPOP_CELL_COUNT] == 2  # CK+ AND CD3+

    def test_density_calculation(self, binary_df, area_df, cond_map):
        summary, _ = compute_subpop_cells_per_area(
            df_binary=binary_df,
            subpop_conditions=["CK_mask+"],
            cond_map=cond_map,
            mask_summary=area_df,
            rois=["roi1"],
        )
        expected_density = 3 / 100.0
        assert summary.iloc[0][CELLS_PER_UM2_CK] == pytest.approx(expected_density)

    def test_returns_empty_for_missing_roi(self, binary_df, area_df, cond_map):
        summary, _ = compute_subpop_cells_per_area(
            df_binary=binary_df,
            subpop_conditions=["CK_mask+"],
            cond_map=cond_map,
            mask_summary=area_df,
            rois=["roi99"],  # non-existent
        )
        assert summary.empty

    def test_no_file_written_when_out_dir_none(self, binary_df, area_df, cond_map, tmp_path):
        compute_subpop_cells_per_area(
            df_binary=binary_df,
            subpop_conditions=["CK_mask+"],
            cond_map=cond_map,
            mask_summary=area_df,
            rois=["roi1"],
            out_dir=None,
        )
        assert list(tmp_path.iterdir()) == []  # nothing written

    def test_file_written_when_out_dir_given(self, binary_df, area_df, cond_map, tmp_path):
        compute_subpop_cells_per_area(
            df_binary=binary_df,
            subpop_conditions=["CK_mask+"],
            cond_map=cond_map,
            mask_summary=area_df,
            rois=["roi1"],
            out_dir=str(tmp_path),
        )
        csvs = list(tmp_path.glob("*.csv"))
        assert len(csvs) == 1

    def test_unknown_condition_key_is_skipped(self, binary_df, area_df, cond_map, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            compute_subpop_cells_per_area(
                df_binary=binary_df,
                subpop_conditions=["UNKNOWN_marker+"],
                cond_map=cond_map,
                mask_summary=area_df,
                rois=["roi1"],
            )
        assert "UNKNOWN_marker" in caplog.text
