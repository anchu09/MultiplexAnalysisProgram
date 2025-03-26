"""Unit tests for hne/models/unet.py."""

import pytest

pytest.importorskip("torch", reason="requires uv sync --group ml")

import pytest
import torch
from multiplex_pipeline.hne.config import IN_CHANNELS, N_CLASSES
from multiplex_pipeline.hne.models.unet import DoubleConv, SEBlock, UNet


class TestSEBlock:
    def test_output_shape_unchanged(self):
        block = SEBlock(in_ch=32)
        x = torch.randn(2, 32, 16, 16)
        assert block(x).shape == x.shape

    def test_values_scaled_between_0_and_1(self):
        block = SEBlock(in_ch=8)
        x = torch.ones(1, 8, 4, 4)
        out = block(x)
        # SE gates are sigmoid ∈ (0,1), so output ≤ input for positive inputs
        assert out.max().item() <= x.max().item() + 1e-5


class TestDoubleConv:
    def test_output_channels(self):
        block = DoubleConv(3, 16)
        x = torch.randn(1, 3, 32, 32)
        assert block(x).shape == (1, 16, 32, 32)

    def test_spatial_dims_preserved(self):
        block = DoubleConv(8, 8)
        x = torch.randn(2, 8, 64, 64)
        assert block(x).shape[2:] == (64, 64)

    def test_dilated_conv_spatial_dims(self):
        block = DoubleConv(8, 8, dilation=2)
        x = torch.randn(1, 8, 32, 32)
        assert block(x).shape == (1, 8, 32, 32)


class TestUNet:
    @pytest.fixture
    def model(self):
        # Use small base_features to keep the test fast
        return UNet(in_channels=IN_CHANNELS, n_classes=N_CLASSES, base_features=8)

    def test_forward_output_shape(self, model):
        """Output spatial dimensions must match input."""
        x = torch.randn(1, IN_CHANNELS, 256, 256)
        out = model(x)
        assert out.shape == (1, N_CLASSES, 256, 256)

    def test_forward_non_square(self, model):
        x = torch.randn(1, IN_CHANNELS, 128, 192)
        out = model(x)
        assert out.shape == (1, N_CLASSES, 128, 192)

    def test_batch_size_two(self, model):
        x = torch.randn(2, IN_CHANNELS, 128, 128)
        out = model(x)
        assert out.shape[0] == 2

    def test_default_config(self):
        """Default constructor uses IN_CHANNELS / N_CLASSES / BASE_FEATURES from config."""
        model = UNet()
        x = torch.randn(1, IN_CHANNELS, 64, 64)
        out = model(x)
        assert out.shape == (1, N_CLASSES, 64, 64)

    def test_gradients_flow(self, model):
        x = torch.randn(1, IN_CHANNELS, 64, 64)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
