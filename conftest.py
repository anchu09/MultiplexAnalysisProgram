"""pytest configuration: make 'multiplex_pipeline' importable during tests.

The repository directory is named MultiplexAnalysisProgram but the package
import path is multiplex_pipeline. Running ``pip install -e .`` (or
``uv sync``) registers the editable install and makes the import work
automatically.

This file provides a fallback for running pytest directly without a prior
install, by inserting the parent directory into sys.path so that Python
resolves ``import multiplex_pipeline`` via the ``multiplex_pipeline``
symlink that uv/pip creates in the parent directory on editable install.
If the symlink does not exist yet, we create it once here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).parent  # .../MultiplexAnalysisProgram/
_projects = _repo_root.parent  # .../Projects/
_symlink = _projects / "multiplex_pipeline"

# Only create the symlink if the package is not already importable (i.e. the
# editable install has not been run yet).
try:
    import multiplex_pipeline  # noqa: F401 — already installed, nothing to do
except ModuleNotFoundError:
    if not _symlink.exists():
        _symlink.symlink_to(_repo_root)
    if str(_projects) not in sys.path:
        sys.path.insert(0, str(_projects))
