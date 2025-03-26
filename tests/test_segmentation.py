"""Tests for preprocessing/segmentation.py."""

from __future__ import annotations

import numpy as np
from multiplex_pipeline.preprocessing.segmentation import (
    generate_initial_mask,
    post_process_mask,
    post_process_mask_closing,
)


class TestPostProcessMask:
    def test_fills_small_holes(self):
        mask = np.ones((10, 10), dtype=bool)
        mask[4, 4] = False  # single-pixel hole
        result = post_process_mask(mask, min_size=0, max_hole_size=5)
        assert result[4, 4]

    def test_removes_small_objects(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[4, 4] = True  # isolated single pixel
        result = post_process_mask(mask, min_size=5, max_hole_size=0)
        assert not result[4, 4]

    def test_returns_bool_array(self):
        mask = np.ones((5, 5), dtype=np.uint8)
        result = post_process_mask(mask)
        assert result.dtype == bool

    def test_no_change_when_thresholds_zero(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:3, 1:3] = True
        result = post_process_mask(mask, min_size=0, max_hole_size=0)
        np.testing.assert_array_equal(result, mask)


class TestPostProcessMaskClosing:
    def test_returns_bool_array(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[10:20, 10:20] = True
        result = post_process_mask_closing(mask)
        assert result.dtype == bool

    def test_ignored_params_do_not_affect_output(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[10:20, 10:20] = True
        r1 = post_process_mask_closing(mask, min_size=0, max_hole_size=0)
        r2 = post_process_mask_closing(mask, min_size=999, max_hole_size=999)
        np.testing.assert_array_equal(r1, r2)


class TestGenerateInitialMask:
    def test_threshold_at_mean(self):
        channel = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = generate_initial_mask(channel, score=0.0, scaling_divisor=1.0)
        # threshold = mean + 0*std = 2.5
        assert mask[1, 1]  # 4.0 > 2.5
        assert not mask[0, 0]  # 1.0 < 2.5

    def test_score_zero_threshold_at_mean(self):
        # With score=0, threshold=mean — exactly half the pixels exceed it
        channel = np.array([[1.0, 2.0, 3.0, 4.0]])
        mask = generate_initial_mask(channel, score=0.0, scaling_divisor=1.0)
        # mean=2.5; pixels 3.0 and 4.0 exceed it
        assert mask[0, 2]  # 3.0 > 2.5
        assert not mask[0, 0]  # 1.0 < 2.5

    def test_returns_bool_array(self):
        channel = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = generate_initial_mask(channel, score=0.0, scaling_divisor=1.0)
        assert mask.dtype == bool
