import logging

import numpy as np

logger = logging.getLogger(__name__)


def is_binary(mask: np.ndarray, mask_name: str = "mask") -> bool:
    """Return True if the mask contains only values 0 and 1."""
    unique_values = np.unique(mask)
    result = set(unique_values).issubset({0, 1})
    if not result:
        logger.warning("%s is NOT binary. Unique values found: %s", mask_name, unique_values)
    return result
