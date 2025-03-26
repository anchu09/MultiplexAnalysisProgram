"""Extended tests for visualization/data_prep.py."""

from __future__ import annotations

import logging

import pandas as pd
import pytest
from multiplex_pipeline.visualization.data_prep import (
    load_distance_matrices_for_plot,
    parse_distance_matrix_filename,
    select_subpopulation,
)


class TestParseDistanceMatrixFilename:
    def test_valid_plus(self):
        result = parse_distance_matrix_filename("distance_matrix_roi3_NGFR_intensity+_vs_Tregs.csv")
        assert result is not None
        roi, sign, pop = result
        assert roi == "roi3"
        assert sign == "+"

    def test_valid_minus(self):
        result = parse_distance_matrix_filename("distance_matrix_roi1_NGFR_intensity-_vs_CD8.csv")
        assert result is not None
        _, sign, _ = result
        assert sign == "-"

    def test_invalid_returns_none(self):
        assert parse_distance_matrix_filename("roi1_distances.csv") is None
        assert parse_distance_matrix_filename("random_file.txt") is None

    def test_population_name_extracted(self):
        result = parse_distance_matrix_filename(
            "distance_matrix_roi5_NGFR_intensity+_vs_Macrophages.csv"
        )
        assert result is not None
        _, _, pop = result
        assert "Macrophages" in pop


class TestSelectSubpopulation:
    @pytest.fixture()
    def df(self):
        return pd.DataFrame(
            {
                "col_a": [1, 0, 1, 0],
                "col_b": [1, 1, 0, 0],
            }
        )

    def test_single_condition(self, df):
        result = select_subpopulation(df, {"col_a": 1})
        assert len(result) == 2
        assert (result["col_a"] == 1).all()

    def test_combined_conditions(self, df):
        result = select_subpopulation(df, {"col_a": 1, "col_b": 1})
        assert len(result) == 1

    def test_empty_parsed_returns_empty(self, df):
        assert select_subpopulation(df, {}).empty

    def test_missing_column_returns_empty(self, df, caplog):
        with caplog.at_level(logging.WARNING):
            result = select_subpopulation(df, {"nonexistent_col": 1})
        assert result.empty
        assert "nonexistent_col" in caplog.text


class TestLoadDistanceMatricesForPlot:
    def test_loads_valid_csvs(self, tmp_path):
        subdir = tmp_path / "Tregs"
        subdir.mkdir()
        csv = subdir / "distance_matrix_roi1_NGFR_intensity+_vs_Tregs.csv"
        df = pd.DataFrame({"distance": [1.0, 2.0, 3.0]})
        df.to_csv(csv, index=False)

        result = load_distance_matrices_for_plot(tmp_path)
        assert len(result) > 0

    def test_skips_non_matching_files(self, tmp_path):
        subdir = tmp_path / "random"
        subdir.mkdir()
        (subdir / "something_else.csv").write_text("x,y\n1,2\n")

        result = load_distance_matrices_for_plot(tmp_path)
        assert result == {}

    def test_empty_directory(self, tmp_path):
        result = load_distance_matrices_for_plot(tmp_path)
        assert result == {}
