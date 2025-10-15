"""Utility functions for notebooks."""

import sys
from pathlib import Path

import polars as pl
from polars import DataFrame


def setup_path():
    """Add the project root to Python path for importing local modules."""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def display_polars(df: DataFrame, lim=50, page_idx: int = 1) -> None:
    """
    Display a polars dataframe with pagination & adjusted view limits
    for notebooks.
    """
    pl.Config.set_tbl_rows(lim)  # Show 100 rowpl.Config.set_fmt_str_lengths(1000)
    pl.Config.set_fmt_table_cell_list_len(10)
    pl.Config.set_fmt_str_lengths(1000)

    print(df[0 + (page_idx - 1) * lim : lim * page_idx])
