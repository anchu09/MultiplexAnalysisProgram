"""Technical configuration for the multiplex IF pipeline.

Data paths, file patterns, imaging constants, results directories,
and pipeline settings (CK/NGFR mask generation).
Biological domain data (markers, phenotypes, subpopulations) lives in domain.py.

Data paths are controlled by environment variables for portability:

    export MULTIPLEX_DATA_DIR=/path/to/images
    export MULTIPLEX_DAPI_DIR=/path/to/dapi_masks
    export MULTIPLEX_MASKS_DIR=/path/to/exported_masks
    export MULTIPLEX_RESULTS_DIR=/path/to/results
"""

import os
import re
from pathlib import Path

from multiplex_pipeline.preprocessing.types import ChannelMaskSettings, PostProcessing

# ---------------------------------------------------------------------------
# Data paths — override via environment variables
# ---------------------------------------------------------------------------
DATA_FOLDER = Path(os.environ.get("MULTIPLEX_DATA_DIR", "data"))
EXPORT_DAPI_FOLDER = Path(os.environ.get("MULTIPLEX_DAPI_DIR", "data/export_DAPI"))
EXPORTED_MASKS_FOLDER = Path(os.environ.get("MULTIPLEX_MASKS_DIR", "data/exportedmasks"))

# File patterns
IMAGE_EXTENSIONS = [".ome.tiff", ".tiff", ".tif"]
DAPI_PATTERN = re.compile(r"roi(\d+)_dapi", re.IGNORECASE)
CSV_EXTENSION = ".csv"
ROI_PATTERN = re.compile(r"(ROI\d+)", re.IGNORECASE)
DAPI_CONNECTIVITY = 1

# Results directories
RESULTS_BASE_DIR = Path(os.environ.get("MULTIPLEX_RESULTS_DIR", "results_spatial_analysis"))
CELL_COUNT_OUTPUT_DIR = RESULTS_BASE_DIR / "cell_counting"
CELL_DENSITY_OUTPUT_DIR = RESULTS_BASE_DIR / "cell_density_area"
BOXPLOTS_DISTANCES_DIR = RESULTS_BASE_DIR / "boxplots_distances_to_mask"
BOXPLOTS_DISTANCES_HEATMAPS_DIR = RESULTS_BASE_DIR / "boxplots_distances_between_populations"
DISTANCES_SUBPOP_DIR = RESULTS_BASE_DIR / "distances_between_mask_and_subpopulation"
DISTANCES_POPULATIONS_DIR = RESULTS_BASE_DIR / "distances_between_populations"

# Imaging constants
PIXEL_SIZE = 0.17  # micrometers per pixel
RANDOM_STATE = 42  # global random seed for reproducible subsampling across the IF pipeline

# Morphological closing radius for NGFR/CK mask post-processing (pixels).
# Value of 20 px ≈ 3.4 µm at 0.17 µm/px — empirically chosen to bridge gaps
# between fragmented tumor-cell clusters without merging adjacent structures.
CLOSING_DISK_RADIUS = 20
PIXEL_AREA = PIXEL_SIZE**2  # µm²
MASK_ALPHA = 0.5
DEFAULT_BRIGHTNESS = 10

# Visualization — default overlay shading colors per mask type (RGB tuples in [0, 1])
SHADING_COLORS: dict[str, tuple[float, float, float]] = {
    "CK_mask": (1.0, 0.0, 0.0),  # red
    "NGFR_mask": (1.0, 0.0, 1.0),  # magenta
}

# Mask creation settings
CK_SETTINGS = ChannelMaskSettings(
    channel_index=10,
    user_scores=-1,
    scaling_divisor=10,
    base_folder_path=EXPORTED_MASKS_FOLDER,
    min_size=2000,
    max_hole_size=10000,
    mask_label="CK",
    mask_filename="CK_mask_binary.tif",
    post_process_funcs=[PostProcessing.STANDARD],
    brightness_factor=None,
    require_dapi=False,
)
NGFR_SETTINGS = ChannelMaskSettings(
    channel_index=13,
    user_scores={
        "ROI1.ome.tiff": 2.7,
        "ROI2.ome.tiff": 2.7,
        "ROI3.ome.tiff": 2.7,
        "ROI4.ome.tiff": 1.8,
        "ROI5.ome.tiff": 1.8,
        "ROI6.ome.tiff": 1.8,
        "ROI7.ome.tiff": 1.5,
        "ROI8.ome.tiff": 1.2,
        "ROI9.ome.tiff": 1.2,
        "ROI10.ome.tiff": 2.7,
        "ROI11.ome.tiff": 2.4,
        "ROI12.ome.tiff": 2.7,
        "ROI13.ome.tiff": 2.1,
    },
    scaling_divisor=3,
    base_folder_path=EXPORTED_MASKS_FOLDER,
    min_size=200,
    max_hole_size=75000,
    mask_label="NGFR",
    mask_filename="NGFR_mask_binary.tif",
    post_process_funcs=[PostProcessing.CLOSING, PostProcessing.STANDARD],
    brightness_factor=25,
    require_dapi=True,
)
# ---------------------------------------------------------------------------
# ROIs to analyze — override via MULTIPLEX_ROIS env var (comma-separated)
# Example: export MULTIPLEX_ROIS="roi1,roi2,roi7,roi11"
# ---------------------------------------------------------------------------
_rois_env = os.environ.get("MULTIPLEX_ROIS", "")
ROIS_TO_ANALYZE: list[str] = [r.strip() for r in _rois_env.split(",") if r.strip()] or [
    "roi1",
    "roi2",
    "roi3",
    "roi4",
    "roi5",
    "roi6",
    "roi7",
    "roi8",
    "roi9",
    "roi10",
    "roi11",
    "roi12",
    "roi13",
]
