"""Unit tests for analysis/intensity.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from multiplex_pipeline.analysis.intensity import (
    compute_binary_flags,
    compute_mean_intensities,
    get_labels_and_counts,
    intensity_to_binary,
    label_dapi,
    process_roi,
)
from multiplex_pipeline.schema import (
    AREA_PIXELS,
    AREA_UM2,
    CENTROID_COL,
    CENTROID_ROW,
    DAPI_ID,
    ROI,
)


@pytest.fixture()
def simple_dapi_mask():
    """5×5 labelled mask with two regions (labels 1 and 2)."""
    m = np.zeros((5, 5), dtype=int)
    m[0:2, 0:2] = 1  # 4 pixels, label 1
    m[3:5, 3:5] = 2  # 4 pixels, label 2
    return m


@pytest.fixture()
def simple_image(simple_dapi_mask):
    """3-channel 5×5 image; channel 1 = CK, channel 2 = NGFR."""
    img = np.zeros((3, 5, 5), dtype=float)
    img[1] = simple_dapi_mask.astype(float)  # CK channel: brighter in region 1
    img[2] = (simple_dapi_mask == 2).astype(float)  # NGFR channel: brighter in region 2
    return img


@pytest.fixture()
def roi_data(simple_dapi_mask, simple_image):
    roi = "ROI1"
    dapi_masks = {f"{roi.lower()}_dapi": simple_dapi_mask}
    ck_mask = (simple_dapi_mask == 1).astype(np.uint8)
    ngfr_mask = (simple_dapi_mask == 2).astype(np.uint8)
    return {
        "img_name": f"sample_{roi}.tiff",
        "img_data": simple_image,
        "dapi_masks": dapi_masks,
        "ck_masks": {"roi1": ck_mask},
        "ngfr_masks": {"roi1": ngfr_mask},
        "channels": [1, 2],
        "marker_dict": {1: "Pan_Cytokeratin_CK", 2: "NGFR"},
    }


class TestLabelDapi:
    def test_binary_mask_gets_labelled(self):
        binary = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=int)
        result = label_dapi(binary)
        assert result.max() >= 1

    def test_already_labelled_mask_returned_unchanged(self, simple_dapi_mask):
        result = label_dapi(simple_dapi_mask)
        np.testing.assert_array_equal(result, simple_dapi_mask)


class TestGetLabelsAndCounts:
    def test_returns_non_zero_labels(self, simple_dapi_mask):
        labels, counts, flat = get_labels_and_counts(simple_dapi_mask)
        assert 0 not in labels
        assert set(labels) == {1, 2}

    def test_counts_match_pixel_sizes(self, simple_dapi_mask):
        labels, counts, flat = get_labels_and_counts(simple_dapi_mask)
        for lbl, cnt in zip(labels, counts, strict=False):
            assert cnt == (simple_dapi_mask == lbl).sum()

    def test_flat_is_ravelled_mask(self, simple_dapi_mask):
        _, _, flat = get_labels_and_counts(simple_dapi_mask)
        np.testing.assert_array_equal(flat, simple_dapi_mask.ravel())


class TestComputeMeanIntensities:
    def test_uniform_channel_returns_channel_value(self, simple_dapi_mask):
        labels, counts, flat = get_labels_and_counts(simple_dapi_mask)
        channel = np.full(simple_dapi_mask.shape, 5.0)
        means = compute_mean_intensities(flat, channel, labels, counts)
        np.testing.assert_allclose(means, 5.0)

    def test_different_means_per_region(self, simple_dapi_mask):
        labels, counts, flat = get_labels_and_counts(simple_dapi_mask)
        channel = (simple_dapi_mask == 1).astype(float) * 10
        means = compute_mean_intensities(flat, channel, labels, counts)
        label1_idx = list(labels).index(1)
        label2_idx = list(labels).index(2)
        assert means[label1_idx] == pytest.approx(10.0)
        assert means[label2_idx] == pytest.approx(0.0)


class TestComputeBinaryFlags:
    def test_label_in_mask_is_positive(self, simple_dapi_mask):
        labels, _, flat = get_labels_and_counts(simple_dapi_mask)
        mask = (simple_dapi_mask == 1).astype(np.uint8)
        flags = compute_binary_flags(labels, flat, mask)
        label1_idx = list(labels).index(1)
        label2_idx = list(labels).index(2)
        assert flags[label1_idx] == 1
        assert flags[label2_idx] == 0

    def test_raises_on_non_binary_mask(self, simple_dapi_mask):
        labels, _, flat = get_labels_and_counts(simple_dapi_mask)
        with pytest.raises(ValueError, match="binary"):
            compute_binary_flags(labels, flat, simple_dapi_mask)


class TestProcessRoi:
    def test_returns_dataframe(self, roi_data):
        result = process_roi(**roi_data)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_present(self, roi_data):
        result = process_roi(**roi_data)
        for col in [ROI, DAPI_ID, AREA_PIXELS, AREA_UM2, CENTROID_ROW, CENTROID_COL]:
            assert col in result.columns

    def test_roi_value_set_correctly(self, roi_data):
        result = process_roi(**roi_data)
        assert (result[ROI] == "roi1").all()

    def test_raises_on_shape_mismatch(self, roi_data):
        roi_data["dapi_masks"]["roi1_dapi"] = np.zeros((3, 3), dtype=int)
        with pytest.raises(ValueError, match="Shape mismatch"):
            process_roi(**roi_data)


class TestIntensityToBinary:
    @pytest.fixture()
    def intensity_df(self):
        return pd.DataFrame(
            {
                ROI: ["roi1"] * 5,
                DAPI_ID: range(1, 6),
                AREA_PIXELS: [10] * 5,
                AREA_UM2: [0.3] * 5,
                CENTROID_ROW: range(5),
                CENTROID_COL: range(5),
                "mean_intensity_CD3": [1.0, 2.0, 3.0, 4.0, 10.0],
            }
        )

    def test_binary_column_created(self, intensity_df):
        result = intensity_to_binary(intensity_df, {"mean_intensity_CD3": 1.0})
        assert "mean_intensity_CD3_binary" in result.columns

    def test_binary_values_are_zero_or_one(self, intensity_df):
        result = intensity_to_binary(intensity_df, {"mean_intensity_CD3": 1.0})
        assert set(result["mean_intensity_CD3_binary"].unique()).issubset({0, 1})

    def test_high_threshold_suppresses_positives(self, intensity_df):
        result = intensity_to_binary(intensity_df, {"mean_intensity_CD3": 100.0})
        assert result["mean_intensity_CD3_binary"].sum() == 0

    def test_negative_threshold_makes_all_positive(self, intensity_df):
        result = intensity_to_binary(intensity_df, {"mean_intensity_CD3": -100.0})
        assert result["mean_intensity_CD3_binary"].sum() == len(intensity_df)

    def test_metadata_columns_preserved(self, intensity_df):
        result = intensity_to_binary(intensity_df, {})
        assert ROI in result.columns
        assert DAPI_ID in result.columns
