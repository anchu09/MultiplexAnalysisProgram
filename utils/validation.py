import logging

import numpy as np

logger = logging.getLogger(__name__)


def verify_binary(mask: np.ndarray, mask_name: str = "mask") -> bool:
    """Return True if the mask contains only values 0 and 1."""
    unique_values = np.unique(mask)
    is_binary = set(unique_values).issubset({0, 1})
    if not is_binary:
        logger.warning("%s is NOT binary. Unique values found: %s", mask_name, unique_values)
    return is_binary
