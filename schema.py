"""DataFrame column-name constants for the multiplex IF pipeline.

Column casing reflects origin: ``ROI``/``DAPI_ID`` come from QuPath,
``Area_pixels`` from regionprops, computed coordinates use ``lower_snake``,
and binary outputs use ``is_positive_<marker>_binary``. These strings appear
in exported CSVs — changing them is a breaking change.
"""

from __future__ import annotations

ROI = "ROI"
DAPI_ID = "DAPI_ID"

CENTROID_ROW = "centroid_row"  # regionprops centroid[0] — first image axis
CENTROID_COL = "centroid_col"  # regionprops centroid[1] — second image axis

AREA_PIXELS = "Area_pixels"
AREA_UM2 = "Area_um2"

IS_POSITIVE_CK = "is_positive_Pan_Cytokeratin_CK_binary"
IS_POSITIVE_NGFR = "is_positive_NGFR_binary"

DISTANCE_PX = "distance_px"
DISTANCE_UM = "distance_um"
DISTANCE_CK_POSITIVE = "distance_ck_positive"
DISTANCE_CK_NEGATIVE = "distance_ck_negative"
DISTANCE_NGFR_POSITIVE = "distance_ngfr_positive"
DISTANCE_NGFR_NEGATIVE = "distance_ngfr_negative"

# cell-to-mask distances used in create_marker_plot
DISTANCE_CK_MASK = "distance_ck_mask"
NEAREST_CK_ROW = "nearest_ck_row"
NEAREST_CK_COL = "nearest_ck_col"
DISTANCE_NGFR_MASK = "distance_ngfr_mask"
NEAREST_NGFR_ROW = "nearest_ngfr_row"
NEAREST_NGFR_COL = "nearest_ngfr_col"


def intensity_binary_col(marker: str) -> str:
    """Return the binary positivity column name for a marker.

    Example: intensity_binary_col("CD3") → "mean_intensity_CD3_binary".
    """
    return f"mean_intensity_{marker}_binary"


SUBPOP_CELL_COUNT = "Subpopulation_Cell_Count"
CK_POSITIVE_AREA_UM2 = "CK_Positive_Area_um2"
CK_NGFR_POSITIVE_AREA_UM2 = "CK_NGFR_Positive_Area_um2"
TOTAL_AREA_ROI_UM2 = "total_area_roi_um2"
CELLS_PER_UM2_CK = "Cells_per_um2_CK_Positive"
CELLS_PER_UM2_CK_NGFR = "Cells_per_um2_CK_NGFR_Positive"


def dapi_key(roi: str) -> str:
    """Return the dict key used to store the DAPI mask for a given ROI.

    Example: dapi_key("roi1") → "roi1_dapi".
    """
    return f"{roi}_dapi"
