"""Tests for the ChartBook manifest, chartbook.toml.

The manifest is what `chartbook build` reads to generate the site: it lists
the notebooks to publish and documents the dataframes the pipeline produces.
"""

import tomllib
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "chartbook.toml"


def test_chartbook_toml():
    with open(MANIFEST_PATH, "rb") as f:
        manifest = tomllib.load(f)

    notebooks = manifest.get("notebooks", {})
    for notebook in [
        "01_es_options_data_ipynb",
        "02_svix_and_crash_probability_ipynb",
        "03_daily_monitor_ipynb",
        "04_single_name_crash_bounds_ipynb",
    ]:
        assert notebook in notebooks, f"chartbook.toml should have a [notebooks.{notebook}] entry"
        entry = notebooks[notebook]
        for field in ["name", "description"]:
            assert entry.get(field), f"[notebooks.{notebook}] should have a {field}"
        assert entry.get("path", "").endswith(f"{notebook}.ipynb"), (
            f"[notebooks.{notebook}] should point at the executed notebook in _output/"
        )

    dataframes = manifest.get("dataframes", {})
    assert "svix_monitor_history" in dataframes, (
        "chartbook.toml should document the monitor's history as [dataframes.svix_monitor_history]"
    )
    history = dataframes["svix_monitor_history"]
    for field in ["name", "description", "path", "date_col", "docs"]:
        assert history.get(field), f"[dataframes.svix_monitor_history] should have a {field}"
    assert history["path"].endswith("svix_monitor_history.csv")
    assert history["date_col"] == "date"

    charts = manifest.get("charts", {})
    assert charts.get("svix_monitor", {}).get("dataframe") == "svix_monitor_history", (
        "The chart refers to the history dataframe by name"
    )
