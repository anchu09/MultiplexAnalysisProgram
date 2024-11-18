"""Configuration types for the IF preprocessing module."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class PostProcessing(str, Enum):
    """Post-processing operations available for channel masks."""

    STANDARD = "standard"  # fill holes + remove small objects
    CLOSING = "closing"  # morphological closing to unify structures


@dataclass
class ChannelMaskSettings:
    """Configuration for a single channel mask generation run."""

    channel_index: int
    user_scores: dict[str, float] | float
    scaling_divisor: float
    base_folder_path: str | Path
    min_size: int
    max_hole_size: int
    mask_label: str
    mask_filename: str
    post_process_funcs: list[PostProcessing]
    brightness_factor: float | None = None
    require_dapi: bool = False
