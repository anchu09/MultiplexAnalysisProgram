"""End-to-end integration test for the full multiplex IF pipeline.

Runs all analysis stages on a synthetic 10×10 ROI with two cells whose
spatial relationships can be verified analytically:

    Cell 1  label=1  centroid (1.5, 1.5)  CK+  NGFR-
    Cell 2  label=2  centroid (7.5, 7.5)  CK-  NGFR+

Masks:  CK = top half (rows 0-4),  NGFR = bottom half (rows 5-9)
"""

from __future__ import annotations

import numpy as np
import pytest
from multiplex_pipeline.analysis.intensity import intensity_to_binary, process_roi
from multiplex_pipeline.analysis.spatial import (
    compute_distances,
    compute_mask_area_summary,
    compute_subpop_distances,
    get_centroids,
)
from multiplex_pipeline.schema import (
    CENTROID_COL,
    CENTROID_ROW,
    CK_NGFR_POSITIVE_AREA_UM2,
    CK_POSITIVE_AREA_UM2,
    DAPI_ID,
    DISTANCE_PX,
    IS_POSITIVE_CK,
    IS_POSITIVE_NGFR,
    TOTAL_AREA_ROI_UM2,
)

_PIXEL_AREA = 1.0  # 1 px² — keeps area assertions simple integers
_MARKER_DICT = {1: "Pan_Cytokeratin_CK", 2: "NGFR"}
# Raw mask-flag column produced by process_roi (before intensity_to_binary adds the _binary suffix)
_IS_POSITIVE_CK_RAW = "is_positive_Pan_Cytokeratin_CK"
_IS_POSITIVE_NGFR_RAW = "is_positive_NGFR"


@pytest.fixture(scope="module")
def roi_arrays():
    size = 10

    dapi_mask = np.zeros((size, size), dtype=int)
    dapi_mask[1:3, 1:3] = 1  # cell 1 — 4 pixels, centroid (1.5, 1.5)
    dapi_mask[7:9, 7:9] = 2  # cell 2 — 4 pixels, centroid (7.5, 7.5)

    img = np.zeros((3, size, size), dtype=float)
    img[2, 5:, :] = 1.0  # channel 2 (NGFR) bright only in bottom half

    ck_mask = np.zeros((size, size), dtype=np.uint8)
    ck_mask[:5, :] = 1  # top half → cell 1 is CK+, cell 2 is CK-

    ngfr_mask = np.zeros((size, size), dtype=np.uint8)
    ngfr_mask[5:, :] = 1  # bottom half → cell 1 is NGFR-, cell 2 is NGFR+

    return {
        "dapi_mask": dapi_mask,
        "img": img,
        "ck_mask": ck_mask,
        "ngfr_mask": ngfr_mask,
    }


@pytest.fixture(scope="module")
def pipeline(roi_arrays):
    """Run every pipeline stage once; return all intermediate results."""
    dm = roi_arrays["dapi_mask"]
    ck = roi_arrays["ck_mask"]
    ng = roi_arrays["ngfr_mask"]

    df = process_roi(
        img_name="sample_ROI1.ome.tiff",
        img_data=roi_arrays["img"],
        dapi_masks={"roi1_dapi": dm},
        ck_masks={"roi1": ck},
        ngfr_masks={"roi1": ng},
        channels=[1, 2],
        marker_dict=_MARKER_DICT,
        pixel_area_um2=_PIXEL_AREA,
    )
    assert df is not None, "process_roi returned None on synthetic data"

    df_bin = intensity_to_binary(df, thresholds={})

    area_df = compute_mask_area_summary({"roi1": ck}, {"roi1": ng}, pixel_area_um2=_PIXEL_AREA)

    centroids_df = get_centroids(dm)

    d_ck_pos, d_ck_neg = compute_distances(df_bin, ck, IS_POSITIVE_CK)

    ck_cells = df_bin[df_bin[IS_POSITIVE_CK] == 1][[DAPI_ID, CENTROID_ROW, CENTROID_COL]]
    ngfr_cells = df_bin[df_bin[IS_POSITIVE_NGFR] == 1][[DAPI_ID, CENTROID_ROW, CENTROID_COL]]
    subpop_dist = compute_subpop_distances(ck_cells, ngfr_cells)

    return {
        "df": df,
        "df_bin": df_bin,
        "area_df": area_df,
        "centroids_df": centroids_df,
        "d_ck_pos": d_ck_pos,
        "d_ck_neg": d_ck_neg,
        "subpop_dist": subpop_dist,
    }


class TestProcessRoiStage:
    def test_two_cells_detected(self, pipeline):
        assert len(pipeline["df"]) == 2

    def test_ck_flag_assigned_by_mask(self, pipeline):
        # rows sorted by centroid_row: cell 1 first (CK+), cell 2 second (CK-)
        flags = pipeline["df"].sort_values(CENTROID_ROW)[_IS_POSITIVE_CK_RAW].tolist()
        assert flags == [1, 0]

    def test_ngfr_flag_assigned_by_mask(self, pipeline):
        flags = pipeline["df"].sort_values(CENTROID_ROW)[_IS_POSITIVE_NGFR_RAW].tolist()
        assert flags == [0, 1]

    def test_ngfr_intensity_higher_for_ngfr_positive_cell(self, pipeline):
        cells = pipeline["df"].sort_values(CENTROID_ROW)
        assert cells.iloc[1]["mean_intensity_NGFR"] > cells.iloc[0]["mean_intensity_NGFR"]


class TestIntensityToBinaryStage:
    def test_is_positive_ck_column_present(self, pipeline):
        assert IS_POSITIVE_CK in pipeline["df_bin"].columns

    def test_is_positive_ngfr_column_present(self, pipeline):
        assert IS_POSITIVE_NGFR in pipeline["df_bin"].columns

    def test_ck_binary_matches_mask_assignment(self, pipeline):
        cells = pipeline["df_bin"].sort_values(CENTROID_ROW)
        assert cells.iloc[0][IS_POSITIVE_CK] == 1  # cell 1 — CK+
        assert cells.iloc[1][IS_POSITIVE_CK] == 0  # cell 2 — CK-

    def test_ngfr_binary_matches_mask_assignment(self, pipeline):
        cells = pipeline["df_bin"].sort_values(CENTROID_ROW)
        assert cells.iloc[0][IS_POSITIVE_NGFR] == 0  # cell 1 — NGFR-
        assert cells.iloc[1][IS_POSITIVE_NGFR] == 1  # cell 2 — NGFR+


class TestAreaSummaryStage:
    def test_single_roi_row(self, pipeline):
        assert len(pipeline["area_df"]) == 1

    def test_ck_positive_area(self, pipeline):
        # top 5 rows × 10 cols × 1.0 px² = 50.0
        assert pipeline["area_df"].iloc[0][CK_POSITIVE_AREA_UM2] == pytest.approx(50.0)

    def test_ck_ngfr_overlap_is_zero(self, pipeline):
        # CK = top half, NGFR = bottom half — no shared pixels
        assert pipeline["area_df"].iloc[0][CK_NGFR_POSITIVE_AREA_UM2] == pytest.approx(0.0)

    def test_total_area(self, pipeline):
        assert pipeline["area_df"].iloc[0][TOTAL_AREA_ROI_UM2] == pytest.approx(100.0)


class TestCentroidsStage:
    def test_two_centroids_returned(self, pipeline):
        assert len(pipeline["centroids_df"]) == 2

    def test_centroid_positions_correct(self, pipeline):
        # get_centroids stores int(centroid) — 2×2 block at rows 1-2 → int(1.5) = 1
        c = pipeline["centroids_df"].sort_values(CENTROID_ROW)
        assert c.iloc[0][CENTROID_ROW] == 1
        assert c.iloc[0][CENTROID_COL] == 1
        assert c.iloc[1][CENTROID_ROW] == 7
        assert c.iloc[1][CENTROID_COL] == 7


class TestDistancesStage:
    def test_output_length_matches_cell_count(self, pipeline):
        assert len(pipeline["d_ck_pos"]) == 2
        assert len(pipeline["d_ck_neg"]) == 2

    def test_ck_positive_cell_has_zero_distance_to_ck(self, pipeline):
        # Cell 1 is CK+ → already inside the CK region → d_ck_pos = 0
        assert pipeline["d_ck_pos"][0] == pytest.approx(0.0)

    def test_ck_negative_cell_has_zero_distance_to_background(self, pipeline):
        # Cell 2 is CK- → already inside the CK=0 region → d_ck_neg = 0
        assert pipeline["d_ck_neg"][1] == pytest.approx(0.0)

    def test_distances_are_non_negative(self, pipeline):
        assert all(d >= 0 for d in pipeline["d_ck_pos"])
        assert all(d >= 0 for d in pipeline["d_ck_neg"])


class TestSubpopDistancesStage:
    def test_one_pair_returned(self, pipeline):
        # 1 CK+ cell × 1 NGFR+ cell = 1 row
        assert len(pipeline["subpop_dist"]) == 1

    def test_distance_column_present(self, pipeline):
        assert DISTANCE_PX in pipeline["subpop_dist"].columns

    def test_distance_is_geometrically_correct(self, pipeline):
        # cell 1 at (1.5, 1.5), cell 2 at (7.5, 7.5)
        # dist = sqrt((7.5-1.5)² + (7.5-1.5)²) = sqrt(72) ≈ 8.485
        dist = pipeline["subpop_dist"].iloc[0][DISTANCE_PX]
        assert dist == pytest.approx(np.sqrt(72), rel=1e-3)
