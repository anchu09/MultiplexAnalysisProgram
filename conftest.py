"""pytest configuration: make 'multiplex_pipeline' importable during tests.

The repository directory is named MultiplexAnalysisProgram but the package
import path is multiplex_pipeline. Running ``uv sync`` (or ``pip install -e .``)
registers the editable install and makes the import work automatically.

This file provides a fallback for running pytest directly without a prior
install, by inserting the parent directory into sys.path so that Python
resolves ``import multiplex_pipeline`` via the symlink that the editable
install creates in the parent directory.

If neither the editable install nor the symlink exists, run ``uv sync`` first.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).parent  # .../MultiplexAnalysisProgram/
_projects = _repo_root.parent  # .../Projects/

try:
    import multiplex_pipeline  # noqa: F401 — already installed, nothing to do
except ModuleNotFoundError:
    # Editable install not active: insert the parent directory so Python can
    # find the package via the symlink created by `uv sync`.
    if not (_projects / "multiplex_pipeline").exists():
        raise ModuleNotFoundError(
            "Package 'multiplex_pipeline' is not installed and no editable-install symlink "
            f"was found in {_projects}. Run `uv sync` (or `pip install -e .`) first."
        ) from None
    if str(_projects) not in sys.path:
        sys.path.insert(0, str(_projects))
