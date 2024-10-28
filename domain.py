"""Biological domain data for the HNSCC multiplex IF analysis.

This module contains the 22-marker panel definition, per-marker intensity thresholds,
cell-phenotype combinations, subpopulation definitions, and distance-analysis
groupings — all specific to the head-and-neck squamous cell carcinoma (HNSCC)
dataset processed at CNIO.

These are data, not configuration: they describe the biology, not the system.
"""

from multiplex_pipeline.schema import IS_POSITIVE_CK, IS_POSITIVE_NGFR, intensity_binary_col

# ---------------------------------------------------------------------------
# 22-channel biomarker panel (MACSima™ cycIF)
# ---------------------------------------------------------------------------
CHANNELS_OF_INTEREST = [10, 13, 7, 17, 1, 0, 15, 5, 20, 21, 9, 19, 3]

MARKER_LABELS: dict[int, str] = {
    0: "FOXP3",
    1: "IFN-gamma",
    2: "CD20 - cyto",
    3: "HLA-DR",
    4: "CD279",
    5: "CD4",
    6: "A_Podoplan",
    7: "Ki67",
    8: "CD163",
    9: "CD11b",
    10: "Pan-Cytokeratin CK",
    11: "A_Actin",
    12: "CD31",
    13: "NGFR",
    14: "DAPI",
    15: "CD3",
    16: "CD56",
    17: "CD274",
    18: "CD45",
    19: "CD11c",
    20: "CD8a",
    21: "CD68",
}

# ---------------------------------------------------------------------------
# Per-marker adaptive thresholds: T = μ_ROI + k · σ_ROI
# k = 0 → threshold at ROI mean; k > 0 → stricter.
#
# Keys are the raw column names produced by process_roi, which mixes two
# column types:
#   - "mean_intensity_<marker>"  → float intensity columns (threshold applied)
#   - "is_positive_<marker>"     → binary mask-flag columns from process_roi
#                                  (also pass through intensity_to_binary with
#                                   k=0 so they become is_positive_<marker>_binary)
# ---------------------------------------------------------------------------
INTENSITY_THRESHOLDS: dict[str, float] = {
    "is_positive_Pan_Cytokeratin_CK": 0,
    "mean_intensity_NGFR": 0.25,
    "is_positive_NGFR": 0,
    "mean_intensity_Ki67": 0,
    "mean_intensity_CD274": 2.5,
    "mean_intensity_IFN_gamma": 1.5,
    "mean_intensity_FOXP3": 2.25,
    "mean_intensity_CD3": 1.25,
    "mean_intensity_CD4": 1,
    "mean_intensity_CD8a": 2.5,
    "mean_intensity_CD68": 2,
    "mean_intensity_CD11b": 2,
    "mean_intensity_CD11c": 2,
    "mean_intensity_HLA_DR": 1.5,
}

# ---------------------------------------------------------------------------
# Tumour characterization phenotypes
# ---------------------------------------------------------------------------
CHARACTERIZATION_COMBINATIONS = {
    "Tumor": lambda df: df[IS_POSITIVE_CK] == 1,
    "Tumor NGFR+": lambda df: (df[IS_POSITIVE_CK] == 1) & (df[intensity_binary_col("NGFR")] == 1),
    "Tumor NGFR-": lambda df: (df[IS_POSITIVE_CK] == 1) & (df[intensity_binary_col("NGFR")] == 0),
    "Tumor NGFR+ Ki67+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[intensity_binary_col("NGFR")] == 1)
        & (df[intensity_binary_col("Ki67")] == 1)
    ),
    "Tumor NGFR- Ki67+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[intensity_binary_col("NGFR")] == 0)
        & (df[intensity_binary_col("Ki67")] == 1)
    ),
    "Tumor NGFR+ CD274+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[intensity_binary_col("NGFR")] == 1)
        & (df[intensity_binary_col("CD274")] == 1)
    ),
    "Tumor NGFR- CD274+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[intensity_binary_col("NGFR")] == 0)
        & (df[intensity_binary_col("CD274")] == 1)
    ),
    "Tumor NGFR+ IFN+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[intensity_binary_col("NGFR")] == 1)
        & (df[intensity_binary_col("IFN_gamma")] == 1)
    ),
    "Tumor NGFR- IFN+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[intensity_binary_col("NGFR")] == 0)
        & (df[intensity_binary_col("IFN_gamma")] == 1)
    ),
    "Tumor NGFR+ HLA_DR+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[intensity_binary_col("NGFR")] == 1)
        & (df[intensity_binary_col("HLA_DR")] == 1)
    ),
    "Tumor NGFR- HLA_DR+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[intensity_binary_col("NGFR")] == 0)
        & (df[intensity_binary_col("HLA_DR")] == 1)
    ),
}

# ---------------------------------------------------------------------------
# Immune infiltration phenotypes
# ---------------------------------------------------------------------------
INFILTRATION_COMBINATIONS = {
    "Tumor CD3+ CD4+ FOXP3+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[intensity_binary_col("CD3")] == 1)
        & (df[intensity_binary_col("CD4")] == 1)
        & (df[intensity_binary_col("FOXP3")] == 1)
    ),
    "Stroma CD3+ CD4+ FOXP3+": lambda df: (
        (df[IS_POSITIVE_CK] == 0)
        & (df[intensity_binary_col("CD3")] == 1)
        & (df[intensity_binary_col("CD4")] == 1)
        & (df[intensity_binary_col("FOXP3")] == 1)
    ),
    "Tumor CD3+ CD8+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[intensity_binary_col("CD3")] == 1)
        & (df[intensity_binary_col("CD8a")] == 1)
    ),
    "Stroma CD3+ CD8+": lambda df: (
        (df[IS_POSITIVE_CK] == 0)
        & (df[intensity_binary_col("CD3")] == 1)
        & (df[intensity_binary_col("CD8a")] == 1)
    ),
    "Tumor CD3+ CD8+ IFN-gamma+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[intensity_binary_col("CD3")] == 1)
        & (df[intensity_binary_col("CD8a")] == 1)
        & (df[intensity_binary_col("IFN_gamma")] == 1)
    ),
    "Stroma CD3+ CD8+ IFN-gamma+": lambda df: (
        (df[IS_POSITIVE_CK] == 0)
        & (df[intensity_binary_col("CD3")] == 1)
        & (df[intensity_binary_col("CD8a")] == 1)
        & (df[intensity_binary_col("IFN_gamma")] == 1)
    ),
    "Tumor CD68+": lambda df: (df[IS_POSITIVE_CK] == 1) & (df[intensity_binary_col("CD68")] == 1),
    "Stroma CD68+": lambda df: (df[IS_POSITIVE_CK] == 0) & (df[intensity_binary_col("CD68")] == 1),
    "Tumor CD11b+": lambda df: (df[IS_POSITIVE_CK] == 1) & (df[intensity_binary_col("CD11b")] == 1),
    "Stroma CD11b+": lambda df: (
        (df[IS_POSITIVE_CK] == 0) & (df[intensity_binary_col("CD11b")] == 1)
    ),
    "Tumor CD11c+": lambda df: (df[IS_POSITIVE_CK] == 1) & (df[intensity_binary_col("CD11c")] == 1),
    "Stroma CD11c+": lambda df: (
        (df[IS_POSITIVE_CK] == 0) & (df[intensity_binary_col("CD11c")] == 1)
    ),
    "Tumor MHCII+": lambda df: (
        (df[IS_POSITIVE_CK] == 1) & (df[intensity_binary_col("HLA_DR")] == 1)
    ),
    "Stroma MHCII+": lambda df: (
        (df[IS_POSITIVE_CK] == 0) & (df[intensity_binary_col("HLA_DR")] == 1)
    ),
}

NGFR_INFILTRATION_COMBINATIONS = {
    "Tumor NGFR+": lambda df: (df[IS_POSITIVE_CK] == 1) & (df[IS_POSITIVE_NGFR] == 1),
    "Tumor NGFR-": lambda df: (df[IS_POSITIVE_CK] == 1) & (df[IS_POSITIVE_NGFR] == 0),
    "Tumor NGFR+ CD3+ CD4+ FOXP3+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[IS_POSITIVE_NGFR] == 1)
        & (df[intensity_binary_col("CD3")] == 1)
        & (df[intensity_binary_col("CD4")] == 1)
        & (df[intensity_binary_col("FOXP3")] == 1)
    ),
    "Tumor NGFR- CD3+ CD4+ FOXP3+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[IS_POSITIVE_NGFR] == 0)
        & (df[intensity_binary_col("CD3")] == 1)
        & (df[intensity_binary_col("CD4")] == 1)
        & (df[intensity_binary_col("FOXP3")] == 1)
    ),
    "Tumor NGFR+ CD3+ CD8+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[IS_POSITIVE_NGFR] == 1)
        & (df[intensity_binary_col("CD3")] == 1)
        & (df[intensity_binary_col("CD8a")] == 1)
    ),
    "Tumor NGFR- CD3+ CD8+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[IS_POSITIVE_NGFR] == 0)
        & (df[intensity_binary_col("CD3")] == 1)
        & (df[intensity_binary_col("CD8a")] == 1)
    ),
    "Tumor NGFR+ CD3+ CD8+ IFN+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[IS_POSITIVE_NGFR] == 1)
        & (df[intensity_binary_col("CD3")] == 1)
        & (df[intensity_binary_col("CD8a")] == 1)
        & (df[intensity_binary_col("IFN_gamma")] == 1)
    ),
    "Tumor NGFR- CD3+ CD8+ IFN+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[IS_POSITIVE_NGFR] == 0)
        & (df[intensity_binary_col("CD3")] == 1)
        & (df[intensity_binary_col("CD8a")] == 1)
        & (df[intensity_binary_col("IFN_gamma")] == 1)
    ),
    "Tumor NGFR+ CD68+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[IS_POSITIVE_NGFR] == 1)
        & (df[intensity_binary_col("CD68")] == 1)
    ),
    "Tumor NGFR- CD68+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[IS_POSITIVE_NGFR] == 0)
        & (df[intensity_binary_col("CD68")] == 1)
    ),
    "Tumor NGFR+ CD11b+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[IS_POSITIVE_NGFR] == 1)
        & (df[intensity_binary_col("CD11b")] == 1)
    ),
    "Tumor NGFR- CD11b+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[IS_POSITIVE_NGFR] == 0)
        & (df[intensity_binary_col("CD11b")] == 1)
    ),
    "Tumor NGFR+ CD11c+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[IS_POSITIVE_NGFR] == 1)
        & (df[intensity_binary_col("CD11c")] == 1)
    ),
    "Tumor NGFR- CD11c+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[IS_POSITIVE_NGFR] == 0)
        & (df[intensity_binary_col("CD11c")] == 1)
    ),
    "Tumor NGFR+ MHCII+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[IS_POSITIVE_NGFR] == 1)
        & (df[intensity_binary_col("HLA_DR")] == 1)
    ),
    "Tumor NGFR- MHCII+": lambda df: (
        (df[IS_POSITIVE_CK] == 1)
        & (df[IS_POSITIVE_NGFR] == 0)
        & (df[intensity_binary_col("HLA_DR")] == 1)
    ),
}

# ---------------------------------------------------------------------------
# Condition shorthand → DataFrame column mapping
# ---------------------------------------------------------------------------
CONDITION_COLUMN_MAP: dict[str, str] = {
    "FOXP3_intensity": intensity_binary_col("FOXP3"),
    "IFN-gamma_intensity": intensity_binary_col("IFN_gamma"),
    "HLA-DR_intensity": intensity_binary_col("HLA_DR"),
    "CD4_intensity": intensity_binary_col("CD4"),
    "Ki67_intensity": intensity_binary_col("Ki67"),
    "CD163_intensity": intensity_binary_col("CD163"),
    "CD11b_intensity": intensity_binary_col("CD11b"),
    "CD3_intensity": intensity_binary_col("CD3"),
    "CD274_intensity": intensity_binary_col("CD274"),
    "CD11c_intensity": intensity_binary_col("CD11c"),
    "CD8a_intensity": intensity_binary_col("CD8a"),
    "CD68_intensity": intensity_binary_col("CD68"),
    "CK_mask": IS_POSITIVE_CK,
    "NGFR_mask": IS_POSITIVE_NGFR,
    "NGFR_intensity": intensity_binary_col("NGFR"),
}

# ---------------------------------------------------------------------------
# Subpopulation definitions for distance analysis
# ---------------------------------------------------------------------------
SUBPOPULATIONS: dict[str, list[str]] = {
    "Tregs": ["CD3_intensity+", "CD4_intensity+", "FOXP3_intensity+"],
    "T CD8+": ["CD3_intensity+", "CD8a_intensity+"],
    "T CD8+ Activator": ["CD3_intensity+", "CD8a_intensity+", "IFN-gamma_intensity+"],
    "Macrophages CD68+": ["CD68_intensity+"],
    "DC CD11b+": ["CD11b_intensity+"],
    "DC CD11c+": ["CD11c_intensity+"],
    "DC HLA-DR+": ["HLA-DR_intensity+"],
}

# Reference populations for A×B distance analysis
SUBPOPULATION_A_POSITIVE: list[str] = ["CK_mask+", "NGFR_intensity+"]
SUBPOPULATION_A_NEGATIVE: list[str] = ["CK_mask+", "NGFR_intensity-"]
