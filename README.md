# Music Tagger

A Python utility for programmatically managing tags in Rekordbox 6 databases.

## Features

- Add tags to tracks in your Rekordbox library
- Query and export content data with Polars
- Utilities for bulk tag operations

## Requirements

- Python 3.10+
- Rekordbox 6

## Installation

```bash
uv sync
```

## Dependencies

- `polars` - Fast dataframe operations
- `pyrekordbox` - Rekordbox database interface

## Usage

See [utils.py](utils.py) for available functions:

- `add_tag()` - Add a tag to a track
- `get_db_content()` - Export Rekordbox content to Polars DataFrame

## Project Structure

```text
music_tagger/
├── utils.py      # Rekordbox utility functions
├── notebooks/    # Analysis and exploration
├── data/         # Data files
└── docs/         # Documentation
```
