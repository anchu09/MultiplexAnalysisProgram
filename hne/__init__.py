"""H&E invasion-front detection: patch extraction, U-Net training, inference, and metrics.

See notebooks/tumor_invasion_front_detection.ipynb for a full walkthrough.
"""

from multiplex_pipeline.hne.data.dataset import PatchDataset, get_valid_pairs
from multiplex_pipeline.hne.inference.predictor import predict_patches
from multiplex_pipeline.hne.models.unet import UNet, load_unet_checkpoint
from multiplex_pipeline.hne.training.trainer import train_model

__all__ = [
    "UNet",
    "load_unet_checkpoint",
    "PatchDataset",
    "get_valid_pairs",
    "train_model",
    "predict_patches",
]
