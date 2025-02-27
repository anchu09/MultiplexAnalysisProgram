"""U-Net with Squeeze-and-Excitation attention for H&E tissue segmentation.

Architecture summary
--------------------
* Input: 2 channels (normalized hematoxylin + eosin optical-density maps).
* Encoder: 4 downsampling stages followed by a bottleneck.
  - Stages 1–2 use standard convolutions.
  - Stages 3–4 and the bottleneck use dilated convolutions (dilation 2 and 4)
    to increase the receptive field without additional pooling.
* Decoder: 4 upsampling stages via transposed convolutions with skip connections.
* Output: 1×1 convolution to n_classes logits (default 3: background,
  invasion front, stroma).
* Every convolutional block ends with a Squeeze-and-Excitation (SE) module
  that re-calibrates channel responses with a learned attention vector.

References
----------
Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image
Segmentation", MICCAI 2015.
Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from multiplex_pipeline.hne.config import BASE_FEATURES, IN_CHANNELS, N_CLASSES, SE_REDUCTION

logger = logging.getLogger(__name__)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel-attention block.

    Args:
        in_ch: Number of input channels.
        reduction: Bottleneck ratio for the fully-connected layers.
    """

    def __init__(self, in_ch: int, reduction: int = SE_REDUCTION) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_ch, in_ch // reduction)
        self.fc2 = nn.Linear(in_ch // reduction, in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale each channel of x by a learned attention weight in (0, 1)."""
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class DoubleConv(nn.Module):
    """Two sequential Conv→BN→ReLU layers followed by an SE block.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        kernel: Convolution kernel size (default 3).
        dilation: Dilation factor; padding is adjusted automatically to
            maintain spatial dimensions.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        pad = ((kernel - 1) * dilation) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two Conv→BN→ReLU steps then SE channel attention."""
        return self.se(self.block(x))


class UNet(nn.Module):
    """U-Net with SE attention and dilated convolutions.

    Args:
        in_channels: Number of input channels (default 2: hematoxylin + eosin).
        n_classes: Number of output segmentation classes (default 3).
        base_features: Channel width at the first encoder level; doubles at each stage.
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        n_classes: int = N_CLASSES,
        base_features: int = BASE_FEATURES,
    ) -> None:
        super().__init__()
        c1 = base_features
        c2 = base_features * 2
        c3 = base_features * 4
        c4 = base_features * 8
        c5 = base_features * 16

        # Encoder
        self.enc1 = DoubleConv(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(c2, c3, dilation=2)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(c3, c4, dilation=4)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(c4, c5, dilation=4)

        # Decoder
        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c4 * 2, c4, dilation=2)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 * 2, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 * 2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 * 2, c1)

        # Output
        self.final_conv = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, n_classes, H, W) logits for input x of shape (B, in_channels, H, W)."""
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        bn = self.bottleneck(self.pool4(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final_conv(d1)


def load_unet_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    in_channels: int = IN_CHANNELS,
    n_classes: int = N_CLASSES,
    base_features: int = BASE_FEATURES,
) -> UNet:
    """Instantiate a UNet and load weights from a saved checkpoint.

    Args:
        checkpoint_path: Path to a .pth file saved by trainer.py.
        device: Device on which to load the model weights.
        in_channels: Input channels — must match the training architecture.
        n_classes: Output classes — must match the training architecture.
        base_features: Encoder channel width — must match the training architecture.

    Returns:
        UNet in evaluation mode.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(device)
    model = UNet(in_channels=in_channels, n_classes=n_classes, base_features=base_features)
    # weights_only=False because checkpoints include non-tensor metadata
    # (epoch, val_loss) saved by trainer.py via torch.save({...}).
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Model loaded from %s on %s.", checkpoint_path, device)
    return model
