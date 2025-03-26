"""Unit tests for hne/preprocessing/color_decomposition.py and augmentation.py."""

import pytest

pytest.importorskip("cv2", reason="requires uv sync --group ml")

import numpy as np
import pytest
from multiplex_pipeline.hne.preprocessing.augmentation import augment_patch
from multiplex_pipeline.hne.preprocessing.color_decomposition import (
    decompose_patch,
    normalize_stain_channels,
    rgb_to_hed,
)


@pytest.fixture
def synthetic_rgb() -> np.ndarray:
    """Random uint8 (64, 64, 3) H&E-like patch."""
    rng = np.random.default_rng(42)
    patch = rng.integers(100, 230, size=(64, 64, 3), dtype=np.uint8)
    return patch


@pytest.fixture
def synthetic_mask() -> np.ndarray:
    """Tri-class (0/1/2) mask aligned with synthetic_rgb."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 1  # invasion front in the middle
    mask[0:16, :] = 2  # stroma at the top
    return mask


class TestRgbToHed:
    def test_output_shape(self, synthetic_rgb):
        hed = rgb_to_hed(synthetic_rgb)
        assert hed.shape == (64, 64, 3)

    def test_output_dtype(self, synthetic_rgb):
        hed = rgb_to_hed(synthetic_rgb)
        assert hed.dtype == np.float32

    def test_uniform_white_patch(self):
        """A white patch should produce near-zero optical density."""
        white = np.full((32, 32, 3), 255, dtype=np.uint8)
        hed = rgb_to_hed(white)
        assert np.allclose(hed, 0.0, atol=0.1)


class TestNormalizeStainChannels:
    def test_output_shapes_and_dtypes(self, synthetic_rgb):
        hed = rgb_to_hed(synthetic_rgb)
        h_u8, e_u8 = normalize_stain_channels(hed)
        assert h_u8.shape == (64, 64)
        assert e_u8.shape == (64, 64)
        assert h_u8.dtype == np.uint8
        assert e_u8.dtype == np.uint8

    def test_values_in_range(self, synthetic_rgb):
        hed = rgb_to_hed(synthetic_rgb)
        h_u8, e_u8 = normalize_stain_channels(hed)
        assert h_u8.min() == 0
        assert h_u8.max() == 255
        assert e_u8.min() == 0
        assert e_u8.max() == 255

    def test_constant_channel_returns_zeros(self):
        """When a stain channel is spatially constant, output should be all zeros."""
        hed = np.zeros((16, 16, 3), dtype=np.float32)
        h_u8, _ = normalize_stain_channels(hed)
        assert (h_u8 == 0).all()


class TestDecomposePatch:
    def test_returns_three_items(self, synthetic_rgb):
        result = decompose_patch(synthetic_rgb)
        assert len(result) == 3

    def test_shapes(self, synthetic_rgb):
        h_u8, e_u8, hed = decompose_patch(synthetic_rgb)
        assert h_u8.shape == (64, 64)
        assert e_u8.shape == (64, 64)
        assert hed.shape == (64, 64, 3)


class TestAugmentPatch:
    def test_returns_n_augments(self, synthetic_rgb, synthetic_mask):
        h = np.zeros((64, 64), dtype=np.uint8)
        e = np.zeros((64, 64), dtype=np.uint8)
        results = augment_patch(synthetic_rgb, h, e, synthetic_mask, n_augments=3)
        assert len(results) == 3

    def test_output_shapes_match_input(self, synthetic_rgb, synthetic_mask):
        h = np.zeros((64, 64), dtype=np.uint8)
        e = np.zeros((64, 64), dtype=np.uint8)
        for aug_rgb, _aug_h, _aug_e, aug_mask in augment_patch(
            synthetic_rgb, h, e, synthetic_mask, n_augments=5
        ):
            assert aug_rgb.shape == synthetic_rgb.shape
            assert aug_mask.shape == synthetic_mask.shape

    def test_mask_labels_preserved(self, synthetic_rgb, synthetic_mask):
        """Augmentation must not introduce new class labels."""
        h = np.zeros((64, 64), dtype=np.uint8)
        e = np.zeros((64, 64), dtype=np.uint8)
        original_labels = set(np.unique(synthetic_mask))
        for _, _, _, aug_mask in augment_patch(synthetic_rgb, h, e, synthetic_mask, n_augments=6):
            assert set(np.unique(aug_mask)).issubset(original_labels)

    def test_deterministic_with_seed(self, synthetic_rgb, synthetic_mask):
        h = np.zeros((64, 64), dtype=np.uint8)
        e = np.zeros((64, 64), dtype=np.uint8)
        r1 = augment_patch(synthetic_rgb, h, e, synthetic_mask, n_augments=2, seed=0)
        r2 = augment_patch(synthetic_rgb, h, e, synthetic_mask, n_augments=2, seed=0)
        for (rgb1, *_), (rgb2, *_) in zip(r1, r2, strict=False):
            assert np.array_equal(rgb1, rgb2)
