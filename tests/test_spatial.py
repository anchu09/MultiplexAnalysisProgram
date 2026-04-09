"""Unit tests for analysis/spatial.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from multiplex_pipeline.analysis.spatial import (
    compute_distances,
    compute_mask_area_summary,
    get_centroids,
)
from multiplex_pipeline.schema import (
    CENTROID_COL,
    CENTROID_ROW,
    CK_NGFR_POSITIVE_AREA_UM2,
    CK_POSITIVE_AREA_UM2,
    DAPI_ID,
    ROI,
    TOTAL_AREA_ROI_UM2,
)


@pytest.fixture()
def square_masks():
    """3×3 binary masks: CK positive in top row, NGFR positive in left column."""
    ck = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
    ngfr = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.uint8)
    return ck, ngfr


@pytest.fixture()
def labelled_dapi():
    """3×3 labelled DAPI mask with two single-pixel regions."""
    m = np.zeros((3, 3), dtype=int)
    m[0, 0] = 1
    m[2, 2] = 2
    return m


@pytest.fixture()
def centroid_df():
    return pd.DataFrame(
        {
            DAPI_ID: [1, 2, 3],
            CENTROID_ROW: [0.0, 5.0, 10.0],
            CENTROID_COL: [0.0, 0.0, 0.0],
        }
    )


class TestComputeMaskAreaSummary:
    def test_returns_dataframe_with_expected_columns(self, square_masks):
        ck, ngfr = square_masks
        result = compute_mask_area_summary({"roi1": ck}, {"roi1": ngfr}, pixel_area=1.0)
        assert isinstance(result, pd.DataFrame)
        for col in [ROI, CK_POSITIVE_AREA_UM2, CK_NGFR_POSITIVE_AREA_UM2, TOTAL_AREA_ROI_UM2]:
            assert col in result.columns

    def test_ck_area_counts_positive_pixels(self, square_masks):
        ck, ngfr = square_masks
        result = compute_mask_area_summary({"roi1": ck}, {"roi1": ngfr}, pixel_area=1.0)
        assert result.iloc[0][CK_POSITIVE_AREA_UM2] == pytest.approx(3.0)

    def test_ck_ngfr_overlap_correct(self, square_masks):
        ck, ngfr = square_masks
        # Overlap: pixel (0,0) where both are 1
        result = compute_mask_area_summary({"roi1": ck}, {"roi1": ngfr}, pixel_area=1.0)
        assert result.iloc[0][CK_NGFR_POSITIVE_AREA_UM2] == pytest.approx(1.0)

    def test_total_area_is_full_mask(self, square_masks):
        ck, ngfr = square_masks
        result = compute_mask_area_summary({"roi1": ck}, {"roi1": ngfr}, pixel_area=2.0)
        assert result.iloc[0][TOTAL_AREA_ROI_UM2] == pytest.approx(9 * 2.0)

    def test_pixel_area_scales_output(self, square_masks):
        ck, ngfr = square_masks
        r1 = compute_mask_area_summary({"roi1": ck}, {"roi1": ngfr}, pixel_area=1.0)
        r2 = compute_mask_area_summary({"roi1": ck}, {"roi1": ngfr}, pixel_area=0.5)
        assert r1.iloc[0][CK_POSITIVE_AREA_UM2] == pytest.approx(
            r2.iloc[0][CK_POSITIVE_AREA_UM2] * 2
        )

    def test_missing_ngfr_roi_skipped(self, square_masks, caplog):
        ck, _ = square_masks
        result = compute_mask_area_summary({"roi1": ck}, {}, pixel_area=1.0)
        assert result.empty

    def test_shape_mismatch_skipped(self, square_masks, caplog):
        ck, _ = square_masks
        ngfr_wrong = np.ones((5, 5), dtype=np.uint8)
        result = compute_mask_area_summary({"roi1": ck}, {"roi1": ngfr_wrong}, pixel_area=1.0)
        assert result.empty

    def test_multiple_rois(self, square_masks):
        ck, ngfr = square_masks
        result = compute_mask_area_summary(
            {"roi1": ck, "roi2": ck},
            {"roi1": ngfr, "roi2": ngfr},
            pixel_area=1.0,
        )
        assert len(result) == 2


class TestGetCentroids:
    def test_returns_dataframe(self, labelled_dapi):
        result = get_centroids(labelled_dapi)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, labelled_dapi):
        result = get_centroids(labelled_dapi)
        assert set(result.columns) == {DAPI_ID, CENTROID_ROW, CENTROID_COL}

    def test_one_row_per_region(self, labelled_dapi):
        result = get_centroids(labelled_dapi)
        assert len(result) == 2

    def test_centroid_position_correct(self):
        mask = np.zeros((5, 5), dtype=int)
        mask[2, 3] = 1  # single pixel at (2, 3)
        result = get_centroids(mask)
        assert result.iloc[0][CENTROID_ROW] == pytest.approx(2)
        assert result.iloc[0][CENTROID_COL] == pytest.approx(3)

    def test_empty_mask_returns_empty_df(self):
        result = get_centroids(np.zeros((5, 5), dtype=int))
        assert result.empty


class TestComputeDistances:
    def test_returns_two_lists(self, centroid_df):
        mask = np.zeros((15, 5), dtype=np.uint8)
        mask[0, 0] = 1
        d_pos, d_neg = compute_distances(centroid_df, mask, DAPI_ID)
        assert isinstance(d_pos, list)
        assert isinstance(d_neg, list)

    def test_output_lengths_match_input(self, centroid_df):
        mask = np.zeros((15, 5), dtype=np.uint8)
        mask[0, 0] = 1
        d_pos, d_neg = compute_distances(centroid_df, mask, DAPI_ID)
        assert len(d_pos) == len(centroid_df)
        assert len(d_neg) == len(centroid_df)

    def test_cell_on_positive_region_has_zero_pos_distance(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[0, 0] = 1
        bin_col = "is_ck"
        df = pd.DataFrame({CENTROID_ROW: [0.0], CENTROID_COL: [0.0], bin_col: [1]})
        d_pos, d_neg = compute_distances(df, mask, bin_col)
        assert d_pos[0] == pytest.approx(0.0)

    def test_cell_on_negative_region_has_zero_neg_distance(self):
        mask = np.ones((5, 5), dtype=np.uint8)
        mask[0, 0] = 0  # only one negative pixel
        bin_col = "is_ck"
        df = pd.DataFrame({CENTROID_ROW: [0.0], CENTROID_COL: [0.0], bin_col: [0]})
        d_pos, d_neg = compute_distances(df, mask, bin_col)
        assert d_neg[0] == pytest.approx(0.0)
