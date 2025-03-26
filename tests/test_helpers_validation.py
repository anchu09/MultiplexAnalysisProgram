"""Tests for utils/helpers.py and utils/validation.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from multiplex_pipeline.utils.helpers import extract_roi_number, invert_dict
from multiplex_pipeline.utils.validation import is_binary


class TestExtractRoiNumber:
    def test_standard_ome_tiff(self):
        assert extract_roi_number("ROI1.ome.tiff") == "1"

    def test_multidigit(self):
        assert extract_roi_number("ROI13.ome.tiff") == "13"

    def test_lowercase(self):
        assert extract_roi_number("roi7.tif") == "7"

    def test_no_roi_returns_none(self):
        assert extract_roi_number("image.tiff") is None

    def test_embedded_roi(self):
        assert extract_roi_number("sample_ROI3_scan.tiff") == "3"


class TestInvertDict:
    def test_basic_inversion(self):
        data = {
            "Tregs": {"roi1": pd.DataFrame({"x": [1]}), "roi2": pd.DataFrame({"x": [2]})},
            "CD8": {"roi1": pd.DataFrame({"x": [3]})},
        }
        result = invert_dict(data)
        assert "roi1" in result
        assert "roi2" in result
        assert "Tregs" in result["roi1"]
        assert "CD8" in result["roi1"]
        assert "Tregs" in result["roi2"]
        assert "CD8" not in result["roi2"]

    def test_raises_on_non_dict_value(self):
        with pytest.raises(TypeError, match="Expected dict"):
            invert_dict({"subpop": "not_a_dict"})

    def test_empty_input(self):
        assert invert_dict({}) == {}


class TestVerifyBinary:
    def test_binary_mask_returns_true(self):
        assert is_binary(np.array([0, 1, 1, 0]), "mask") is True

    def test_non_binary_returns_false(self):
        assert is_binary(np.array([0, 1, 2, 3]), "mask") is False

    def test_all_zeros_is_binary(self):
        assert is_binary(np.zeros((5, 5), dtype=np.uint8), "mask") is True

    def test_all_ones_is_binary(self):
        assert is_binary(np.ones((5, 5), dtype=np.uint8), "mask") is True
