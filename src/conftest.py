"""Shared pytest configuration.

Most tests here run offline on synthetic smiles and need no data at all. The
tests that read pulled data are skipped, with a message saying what to run,
until the pull has happened. A skipped test earns no points, so this changes
the error message and nothing about how the assignment is scored.
"""

from pathlib import Path

import pytest

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

# Which pulled or computed files each data-backed test needs, and the command
# that produces them.
REQUIRED_DATA = {
    "test_single_name_bounds.py::test_index_probabilities_computed": (
        [OUTPUT_DIR / "index_crash_probabilities.parquet"],
        "doit single_names",
    ),
    "test_svix.py::test_svix_matches_martin": (
        [
            DATA_DIR / "optionm_vol_surface.parquet",
            DATA_DIR / "optionm_security_prices.parquet",
            DATA_DIR / "optionm_zero_curve.parquet",
        ],
        "doit single_names",
    ),
    "test_single_name_bounds.py::test_spx_crash_probability_matches_martin": (
        [OUTPUT_DIR / "index_crash_probabilities.parquet"],
        "doit single_names",
    ),
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        key = f"{Path(str(item.fspath)).name}::{item.name}"
        if key not in REQUIRED_DATA:
            continue
        paths, command = REQUIRED_DATA[key]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            item.add_marker(
                pytest.mark.skip(reason=f"needs {', '.join(missing)}: run `{command}` first (it needs WRDS)")
            )
