"""Entry point for ``python -m multiplex_pipeline``.

Prints a concise usage summary. For full pipeline execution see the notebooks
or call the public API directly from Python scripts.
"""

from __future__ import annotations


def main() -> None:
    """Print a concise usage summary for the multiplex_pipeline package."""
    print(
        "multiplex_pipeline — Spatial Biology Image Analysis Pipeline\n"
        "\n"
        "Usage:\n"
        "  python -m multiplex_pipeline          # this help\n"
        "  python -m multiplex_pipeline if       # IF pipeline quick-start\n"
        "  python -m multiplex_pipeline hne      # H&E pipeline quick-start\n"
        "\n"
        "Environment variables (IF pipeline):\n"
        "  MULTIPLEX_DATA_DIR    OME-TIFF image folder\n"
        "  MULTIPLEX_DAPI_DIR    DAPI mask export folder\n"
        "  MULTIPLEX_MASKS_DIR   Tissue mask output folder\n"
        "  MULTIPLEX_RESULTS_DIR Analysis results root\n"
        "  MULTIPLEX_ROIS        Comma-separated ROIs  (e.g. roi1,roi2,roi3)\n"
        "\n"
        "Environment variables (H&E pipeline):\n"
        "  HNE_RESULTS_DIR       H&E pipeline results root\n"
        "\n"
        "Example — run IF pipeline from Python:\n"
        "\n"
        "  from multiplex_pipeline.io.loaders import load_ome_tif_images, load_dapi_masks\n"
        "  from multiplex_pipeline.preprocessing.segmentation import create_channel_masks\n"
        "  from multiplex_pipeline.analysis.intensity import process_roi, intensity_to_binary\n"
        "  from multiplex_pipeline.config import CK_SETTINGS, NGFR_SETTINGS, ROIS_TO_ANALYZE\n"
        "\n"
        "  images    = load_ome_tif_images()\n"
        "  dapi      = load_dapi_masks()\n"
        "  ck_masks  = create_channel_masks(images, dapi, CK_SETTINGS)\n"
        "  ngfr_masks= create_channel_masks(images, dapi, NGFR_SETTINGS)\n"
        "\n"
        "See notebooks/IF_analysis.ipynb for the complete walkthrough.\n"
        "See notebooks/tumor_invasion_front_detection.ipynb for the H&E pipeline.\n"
        "See README.md for full documentation.\n"
    )


if __name__ == "__main__":
    import sys

    sys.exit(main())
