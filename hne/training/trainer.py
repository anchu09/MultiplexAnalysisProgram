"""Training loop for the H&E U-Net segmentation model.

Uses mixed-precision (AMP) and gradient accumulation to fit the large
2048×2048 patches on consumer-grade GPUs with limited VRAM.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from multiplex_pipeline.hne.config import (
    BATCH_SIZE,
    DEFAULT_EPOCHS,
    GRAD_ACCUMULATION,
    IGNORE_INDEX,
    LEARNING_RATE,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: str | torch.device = "cuda",
    epochs: int = DEFAULT_EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    grad_accumulation: int = GRAD_ACCUMULATION,
    criterion: nn.Module | None = None,
    checkpoint_path: str | os.PathLike | None = None,
) -> dict[str, list[float]]:
    """Train model on train_dataset and evaluate on val_dataset.

    Training uses:
    - Adam optimiser
    - Cross-entropy loss with ignore_index=-1 by default (unannotated pixels)
    - Mixed-precision (float16 forward pass, float32 parameter updates)
    - Gradient accumulation over grad_accumulation micro-batches
    - Best-model checkpointing by validation loss

    Parameters
    ----------
    model:
        U-Net instance (already moved to device).
    train_dataset:
        Training split (typically PatchDataset(..., augment=True)).
    val_dataset:
        Validation split (augment=False).
    device:
        PyTorch device string or instance.
    epochs:
        Total training epochs.
    lr:
        Adam learning rate.
    batch_size:
        Samples per micro-batch.
    grad_accumulation:
        Number of micro-batches to accumulate before an optimiser step.
    criterion:
        Loss function. Defaults to CrossEntropyLoss(ignore_index=-1).
        Pass any nn.Module that accepts (logits, targets) to use a custom loss.
    checkpoint_path:
        If given, the best model weights (lowest validation loss) are saved
        to this path via torch.save.

    Returns
    -------
    dict with keys "train_loss" and "val_loss", each a list of per-epoch floats.
    """
    device = torch.device(device)
    criterion = (
        criterion if criterion is not None else nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else Path("best_model.pt")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        running_loss = 0.0

        for step, (x, y) in enumerate(tqdm(train_loader, desc=f"Train epoch {epoch}/{epochs}")):
            x, y = x.to(device), y.to(device)
            with autocast():
                logits = model(x)
                if logits.shape[2:] != y.shape[1:]:
                    y = (
                        F.interpolate(
                            y.unsqueeze(1).float(),
                            size=logits.shape[2:],
                            mode="nearest",
                        )
                        .squeeze(1)
                        .long()
                    )
                loss = criterion(logits, y) / grad_accumulation

            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accumulation

            if (step + 1) % grad_accumulation == 0 or (step + 1) == len(train_loader):
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)

        avg_train = running_loss / len(train_loader)
        history["train_loss"].append(avg_train)
        logger.info("[Epoch %d] Train loss: %.4f", epoch, avg_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"  Val  epoch {epoch}/{epochs}"):
                x, y = x.to(device), y.to(device)
                with autocast():
                    logits = model(x)
                    if logits.shape[2:] != y.shape[1:]:
                        y = (
                            F.interpolate(
                                y.unsqueeze(1).float(),
                                size=logits.shape[2:],
                                mode="nearest",
                            )
                            .squeeze(1)
                            .long()
                        )
                    val_loss += criterion(logits, y).item()

        avg_val = val_loss / len(val_loader)
        history["val_loss"].append(avg_val)
        logger.info("[Epoch %d] Val   loss: %.4f", epoch, avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": avg_val},
                ckpt_path,
            )
            logger.info(
                "[Epoch %d] New best val loss %.4f — checkpoint saved to %s",
                epoch,
                avg_val,
                ckpt_path,
            )

    return history
