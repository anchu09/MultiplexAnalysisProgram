"""Whole-slide image patch extractor for H&E FFPE tissue sections.

Reads an OME-TIFF whole-slide image together with QuPath GeoJSON annotations
and extracts non-overlapping tiles, rasterizing polygon annotations into
per-tile tri-class segmentation masks (background=0, invasion_front=1, stroma=2).
Unannotated pixels inside an annotated tile are classified by brightness:
mean RGB > HOLE_THRESHOLD → background, otherwise → stroma.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import tifffile
from multiplex_pipeline.hne.config import (
    CLASS_MAP,
    HOLE_THRESHOLD,
    MAX_CPU_WORKERS,
    MAX_IO_WORKERS,
    PATCH_SIZE,
    PNG_COMPRESSION,
    STRIDE,
    TISSUE_THRESHOLD,
)
from shapely.affinity import translate
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

logger = logging.getLogger(__name__)

_UNINITIALIZED_PIXEL: int = 255  # sentinel for unannotated pixels in rasterized masks
_CLASS_BACKGROUND = "ignore-fondo"  # QuPath class name for background/glass regions
_CLASS_STROMA = "Stroma"  # QuPath class name for stromal tissue (note capital S)


class RobustPatchExtractor:
    """Extract patches and aligned segmentation masks from a whole-slide H&E image.

    Args:
        img_path: Directory containing the OME-TIFF file.
        ome_filename: Name of the OME-TIFF file inside img_path.
        ann_path: Directory containing QuPath GeoJSON annotation files (searched
            recursively).
        output_dir: Where to write patch_NNNN.png and mask_NNNN.png files.
        patch_size: Tile edge length in pixels.
        stride: Step between tile origins. Equal to patch_size means no overlap.
        tissue_threshold: Minimum fraction of foreground (dark) pixels to keep a tile.
        class_map: Mapping from annotation class name to integer label.
        hole_threshold: Mean RGB above which unannotated pixels are background (0);
            below → stroma (2).
    """

    def __init__(
        self,
        img_path: Path,
        ome_filename: str,
        ann_path: Path | str,
        output_dir: Path,
        patch_size: int = PATCH_SIZE,
        stride: int = STRIDE,
        tissue_threshold: float = TISSUE_THRESHOLD,
        class_map: dict[str, int] | None = None,
        hole_threshold: int = HOLE_THRESHOLD,
    ) -> None:
        self.ome_file = Path(img_path) / ome_filename
        self.ann_dir = Path(ann_path)
        self.output_dir = Path(output_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.tissue_threshold = tissue_threshold
        self.class_map = class_map if class_map is not None else CLASS_MAP
        self.hole_threshold = hole_threshold

        _required_keys = {_CLASS_BACKGROUND, _CLASS_STROMA}
        _missing = _required_keys - self.class_map.keys()
        if _missing:
            raise ValueError(
                f"class_map is missing required keys: {_missing}. "
                f"Expected keys: '{_CLASS_BACKGROUND}' (background glass) and '{_CLASS_STROMA}'."
            )

        self.annotations: list[tuple[int, BaseGeometry]] = []
        self.img: np.ndarray | None = None
        self.H = self.W = 0

    def extract(self) -> None:
        """Run the full extraction pipeline and write output files to disk."""
        self.load_scene()
        self.load_annotations()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        n_x = int(np.ceil((self.W - self.patch_size) / self.stride)) + 1
        n_y = int(np.ceil((self.H - self.patch_size) / self.stride)) + 1
        tasks = [
            (
                i,
                j,
                min(j * self.stride, self.W - self.patch_size),
                min(i * self.stride, self.H - self.patch_size),
            )
            for i in range(n_y)
            for j in range(n_x)
        ]

        results = []
        # ThreadPoolExecutor is used here because cv2/numpy release the GIL,
        # so thread-based parallelism is effective despite Python's GIL.
        # Switch to ProcessPoolExecutor if profiling shows GIL contention.
        with ThreadPoolExecutor(max_workers=MAX_CPU_WORKERS) as pool:
            futures = {pool.submit(self._process_task, t): t for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(tasks), desc="Extracting patches"):
                res = fut.result()
                if res is not None:
                    results.append(res)

        with ThreadPoolExecutor(max_workers=MAX_IO_WORKERS) as io_pool:
            for i, j, patch, mask in results:
                pid = i * n_x + j
                io_pool.submit(self._save_patch, pid, patch, mask)

        logger.info(
            "Patch extraction complete. %d patches saved to %s", len(results), self.output_dir
        )

    def load_scene(self) -> None:
        """Load the OME-TIFF and convert to (H, W, 3) uint8."""
        logger.info("Loading %s", self.ome_file)
        arr = tifffile.imread(self.ome_file)
        arr = np.squeeze(arr)

        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3:
            c = arr.shape[2]
            if c not in (1, 2, 3, 4):
                arr = arr.transpose(1, 2, 0)
            if arr.shape[2] > 3:
                logger.warning("Image has %d channels; using first 3.", arr.shape[2])
                arr = arr[:, :, :3]
            elif arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
        else:
            for _ in range(arr.ndim - 3):
                arr = arr.max(axis=0)
            arr = np.squeeze(arr)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)

        self.img = arr.astype(np.uint8)
        self.H, self.W = self.img.shape[:2]
        logger.info("Image loaded: %s", self.img.shape)

    def load_annotations(self) -> None:
        """Parse all GeoJSON files in ann_dir and store polygon/class pairs."""
        logger.info("Loading annotations from %s", self.ann_dir)
        for fn in self.ann_dir.rglob("*.geojson"):
            data = json.loads(fn.read_text(encoding="utf-8"))
            for feat in data.get("features", []):
                name = feat["properties"]["classification"]["name"]
                if name not in self.class_map:
                    continue
                cls_idx = self.class_map[name]
                geom = shape(feat["geometry"])
                if geom.geom_type == "Polygon":
                    self.annotations.append((cls_idx, geom))
                else:
                    for poly in geom.geoms:
                        self.annotations.append((cls_idx, poly))
        logger.info("Loaded %d polygons.", len(self.annotations))

    def is_tissue(self, patch: np.ndarray) -> bool:
        """Return True if the patch contains enough tissue (non-background)."""
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bool((thr == 0).mean() > self.tissue_threshold)

    def rasterize_mask(self, x0: int, y0: int) -> np.ndarray:
        """Rasterize annotation polygons into a tile-sized label mask.

        Unannotated pixels are initialized to _UNINITIALIZED_PIXEL (sentinel for holes).
        Interior rings (polygon holes) are also reset to _UNINITIALIZED_PIXEL.
        """
        mask = np.full((self.patch_size, self.patch_size), _UNINITIALIZED_PIXEL, dtype=np.uint8)
        for cls_idx, poly in self.annotations:
            minx, miny, maxx, maxy = poly.bounds
            if maxx < x0 or minx > x0 + self.patch_size or maxy < y0 or miny > y0 + self.patch_size:
                continue
            shp = translate(poly, xoff=-x0, yoff=-y0)
            ext = np.array(shp.exterior.coords).round().astype(np.int32)
            cv2.fillPoly(mask, [ext], int(cls_idx))
            for interior in shp.interiors:
                hole = np.array(interior.coords).round().astype(np.int32)
                cv2.fillPoly(mask, [hole], _UNINITIALIZED_PIXEL)
        return mask

    def _process_task(
        self, task: tuple[int, int, int, int]
    ) -> tuple[int, int, np.ndarray, np.ndarray] | None:
        i, j, x0, y0 = task
        if self.img is None:
            raise RuntimeError("load_scene() must be called before extract().")
        patch = self.img[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        patch = self._pad_patch(patch)
        if not self.is_tissue(patch):
            return None

        mask = self.rasterize_mask(x0, y0)

        if np.any(mask != _UNINITIALIZED_PIXEL):
            hole_mask = mask == _UNINITIALIZED_PIXEL
            if hole_mask.any():
                brightness = patch.mean(axis=2)
                mask[hole_mask & (brightness > self.hole_threshold)] = self.class_map[
                    _CLASS_BACKGROUND
                ]
                mask[hole_mask & (brightness <= self.hole_threshold)] = self.class_map[
                    _CLASS_STROMA
                ]

        return (i, j, patch, mask)

    def _pad_patch(self, patch: np.ndarray) -> np.ndarray:
        h, w = patch.shape[:2]
        dh = self.patch_size - h
        dw = self.patch_size - w
        if dh > 0 or dw > 0:
            patch = np.pad(
                patch,
                ((0, max(dh, 0)), (0, max(dw, 0)), (0, 0)),
                mode="constant",
                constant_values=_UNINITIALIZED_PIXEL,
            )
        return patch

    def _save_patch(self, patch_id: int, patch: np.ndarray, mask: np.ndarray) -> None:
        cv2.imwrite(
            str(self.output_dir / f"patch_{patch_id:04d}.png"),
            cv2.cvtColor(patch, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
        )
        cv2.imwrite(
            str(self.output_dir / f"mask_{patch_id:04d}.png"),
            mask,
            [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
        )
