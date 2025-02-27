"""Central configuration for the H&E invasion-front detection pipeline."""

from __future__ import annotations

import os
from pathlib import Path

PATCH_SIZE: int = 2048
STRIDE: int = PATCH_SIZE  # no overlap by default
PNG_COMPRESSION: int = 1  # lossless, fast

TISSUE_THRESHOLD: float = 0.05  # min fraction of dark pixels to keep a patch
HOLE_THRESHOLD: int = 190  # mean RGB > this → background glass

# Keys must match the annotation class names as defined in the QuPath project.
CLASS_MAP: dict[str, int] = {
    "ignore-fondo": 0,  # QuPath class: background / glass regions
    "tumor": 1,  # QuPath class: invasion-front tumor cells
    "Stroma": 2,  # QuPath class: stromal tissue (capitalized in QuPath)
}
CLASS_LABELS: dict[int, str] = {
    0: "background",
    1: "invasion_front",
    2: "stroma",
}
N_CLASSES: int = len(CLASS_LABELS)
IGNORE_INDEX: int = -1  # pixels masked out from loss computation

MIN_TUMOR_FRACTION: float = 0.10  # min fraction of tumor pixels to oversample
OVERSAMPLE_FACTOR: int = 6  # target multiplier for tumor-containing patches

HEMATOXYLIN_PREFIX: str = "hematoxylin"
EOSIN_PREFIX: str = "eosin"

IN_CHANNELS: int = 2  # hematoxylin + eosin
BASE_FEATURES: int = 64  # channel width at first encoder level
SE_REDUCTION: int = 16  # squeeze-and-excitation bottleneck ratio

LEARNING_RATE: float = 1e-4
BATCH_SIZE: int = 1
GRAD_ACCUMULATION: int = 4
DEFAULT_EPOCHS: int = 10
TRAIN_SPLIT: float = 0.8  # fraction used for training (rest → test)
RANDOM_SEED: int = 42

MA_KERNEL_SIZE: int = 5  # morphological median filter window

RESULTS_BASE_DIR: Path = Path(os.environ.get("HNE_RESULTS_DIR", "results_hne"))
MODELS_DIR: Path = RESULTS_BASE_DIR / "models"
OVERLAYS_DIR: Path = RESULTS_BASE_DIR / "overlays"
METRICS_DIR: Path = RESULTS_BASE_DIR / "metrics"
EDA_DIR: Path = RESULTS_BASE_DIR / "eda"

MAX_CPU_WORKERS: int = min(os.cpu_count() or 1, 8)
MAX_IO_WORKERS: int = 2
