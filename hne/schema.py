"""Central registry of DataFrame column names for the H&E pipeline."""

from __future__ import annotations

PATCH_ID = "patch_id"
MASK_ID = "mask_id"

DOMINANT_CLASS = "dominant_class"  # name of the majority class in a patch

PROP_BACKGROUND = "prop_background"
PROP_FRONT = "prop_front_of_invasion"
PROP_STROMA = "prop_stroma"

COUNT_BACKGROUND = "background"
COUNT_FRONT = "front_of_invasion"
COUNT_STROMA = "stroma"

MEAN_GRAY = "mean_gray"
STD_GRAY = "std_gray"
MEAN_HEMATOXYLIN = "mean_hematoxylin"
MEAN_EOSIN = "mean_eosin"

STROMA_QUARTILE = "stroma_q"
