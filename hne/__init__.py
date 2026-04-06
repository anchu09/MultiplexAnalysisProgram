"""H&E invasion-front detection subpackage.

Provides a modular pipeline for:
  - Preprocessing H&E whole-slide images (patch extraction, stain decomposition,
    augmentation, class balancing)
  - Training and running inference with a U-Net model
  - Evaluating segmentation quality
  - Visualizing results and exploratory data

Quickstart
----------
>>> from multiplex_pipeline.hne.models.unet import UNet
>>> from multiplex_pipeline.hne.data.dataset import PatchDataset, get_valid_pairs
>>> from multiplex_pipeline.hne.training.trainer import train_model
>>> from multiplex_pipeline.hne.inference.predictor import predict_patches
"""

from multiplex_pipeline.hne.data.dataset import PatchDataset, get_valid_pairs
from multiplex_pipeline.hne.inference.predictor import predict_patches
from multiplex_pipeline.hne.models.unet import UNet
from multiplex_pipeline.hne.training.trainer import train_model

__all__ = [
    "UNet",
    "PatchDataset",
    "get_valid_pairs",
    "train_model",
    "predict_patches",
]
