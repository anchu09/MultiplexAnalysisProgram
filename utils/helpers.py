import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def in_jupyter() -> bool:
    """Return True when running inside a Jupyter kernel."""
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def extract_roi_number(filename: str) -> str | None:
    """Extract the ROI number from an image filename, e.g. 'ROI1.ome.tiff' → '1'."""
    match = re.search(r"ROI(\d+)", filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def invert_dict(subpop_data: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Invert a nested dict from {subpopulation: {roi: df}} to {roi: {subpopulation: df}}.

    Raises:
        TypeError: If subpop_data values are not dicts.
    """
    roi_dict: dict[str, dict[str, Any]] = {}
    for subpop, roi_data in subpop_data.items():
        if not isinstance(roi_data, dict):
            raise TypeError(
                f"Expected dict for subpopulation '{subpop}', got {type(roi_data).__name__}."
            )
        for roi, df in roi_data.items():
            if roi not in roi_dict:
                roi_dict[roi] = {}
            roi_dict[roi][subpop] = df
    return roi_dict
