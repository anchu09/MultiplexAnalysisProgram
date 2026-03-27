import logging
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from multiplex_pipeline.config import (
    CSV_EXTENSION,
    DAPI_PATTERN,
    DATA_FOLDER,
    EXPORT_DAPI_FOLDER,
    IMAGE_EXTENSIONS,
)
from tifffile import TiffFileError, imread
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_ome_tif_images(
    data_folder: Path = DATA_FOLDER,
    show_progress: bool = True,
) -> dict[str, np.ndarray]:
    """Load all OME-TIFF images from the specified folder.

    Returns dict mapping filenames to image arrays of shape (C, H, W).

    Raises:
        FileNotFoundError: If data_folder does not exist.
    """
    base = Path(data_folder)
    if not base.is_dir():
        raise FileNotFoundError(f"Image folder not found: {base}")

    exts = {e.lower() for e in IMAGE_EXTENSIONS}
    candidates = [p for p in base.iterdir() if p.is_file() and "".join(p.suffixes).lower() in exts]

    iterator = (
        tqdm(candidates, desc="Loading OME-TIFF images", unit="img")
        if show_progress
        else candidates
    )

    images: dict[str, np.ndarray] = {}
    for path in iterator:
        try:
            images[path.name] = imread(path)
        except (OSError, TiffFileError) as exc:
            logger.error("Failed to load image '%s': %s", path.name, exc)
            raise

    return images


def load_dapi_masks(
    export_folder: Path = EXPORT_DAPI_FOLDER,
    show_progress: bool = True,
) -> dict[str, np.ndarray]:
    """Load DAPI mask files matching DAPI_PATTERN from the export folder.

    Returns dict mapping 'roi{N}_dapi' keys to 2-D mask arrays.

    Raises:
        FileNotFoundError: If export_folder does not exist.
        OSError: If a matching mask file cannot be read.
    """
    base = Path(export_folder)
    if not base.is_dir():
        raise FileNotFoundError(f"DAPI export folder not found: {base}")

    tif_paths = [p for p in base.iterdir() if p.suffix.lower() == ".tif"]
    iterator = (
        tqdm(tif_paths, desc="Loading DAPI masks", unit="mask") if show_progress else tif_paths
    )

    masks: dict[str, np.ndarray] = {}
    for path in iterator:
        m = DAPI_PATTERN.search(path.stem)
        if not m:
            logger.debug("Skipping '%s': does not match DAPI pattern.", path.name)
            continue
        key = f"roi{m.group(1)}_dapi"
        try:
            masks[key] = imageio.imread(path)
        except OSError as exc:
            logger.error("Failed to load DAPI mask '%s': %s", path.name, exc)
            raise

    return masks


def load_csv_data(
    base_path: Path,
    show_progress: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Recursively load all CSV files under base_path into a nested dict.

    The returned structure is {subfolder_name: {filename: DataFrame}}.

    Raises:
        FileNotFoundError: If base_path does not exist.
        pd.errors.ParserError: If a CSV file cannot be parsed.
    """
    base = Path(base_path)
    if not base.is_dir():
        raise FileNotFoundError(f"CSV base path not found: {base}")

    subdirs = [d for d in base.iterdir() if d.is_dir()]
    dir_iter = (
        tqdm(subdirs, desc="Loading CSV subfolders", unit="dir") if show_progress else subdirs
    )

    result: dict[str, dict[str, pd.DataFrame]] = {}
    for subdir in dir_iter:
        files_dict: dict[str, pd.DataFrame] = {}
        csv_files = list(subdir.glob(f"*{CSV_EXTENSION}"))
        file_iter = (
            tqdm(csv_files, desc=f"Reading CSVs in {subdir.name}", unit="file")
            if show_progress
            else csv_files
        )
        for file in file_iter:
            try:
                files_dict[file.name] = pd.read_csv(file)
            except (OSError, pd.errors.ParserError) as exc:
                logger.error("Failed to read CSV '%s': %s", file, exc)
                raise
        result[subdir.name] = files_dict

    return result


def load_distance_matrices(base_path: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """Load distance matrices from subfolders under base_path.

    Expected layout::

        base_path/
            {group}/           e.g. "A_neg", "A_pos"
                {subpop}/
                    *.csv

    Returns dict mapping group -> {subpop/filename: DataFrame}.

    Raises:
        FileNotFoundError: If base_path does not exist.
        pd.errors.ParserError: If a CSV file cannot be parsed.
    """
    base_path = Path(base_path)
    if not base_path.is_dir():
        raise FileNotFoundError(f"Distance matrices path not found: {base_path}")

    result: dict[str, dict[str, pd.DataFrame]] = {}

    for group in os.listdir(base_path):
        path_group = base_path / group
        if not path_group.is_dir():
            continue

        files_dict: dict[str, pd.DataFrame] = {}
        for subpop in os.listdir(path_group):
            path_subpop = path_group / subpop
            if not path_subpop.is_dir():
                continue
            for fname in os.listdir(path_subpop):
                if fname.lower().endswith(".csv"):
                    full = path_subpop / fname
                    try:
                        files_dict[f"{subpop}/{fname}"] = pd.read_csv(full)
                    except (OSError, pd.errors.ParserError) as exc:
                        logger.error("Failed to read distance matrix '%s': %s", full, exc)
                        raise

        result[group] = files_dict

    return result
