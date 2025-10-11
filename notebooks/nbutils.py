"""Utility functions for notebooks."""

import sys
from pathlib import Path


def setup_path():
    """Add the project root to Python path for importing local modules."""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
