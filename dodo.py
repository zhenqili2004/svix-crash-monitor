"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based.

Run everything with `uv run doit`. The default tasks are the index pipeline:
pull the E-mini option settlements and the FRED series over the window in
the settings (START_DATE to END_DATE, the trailing week by default), add
every session in it that the history does not have, draw the charts, build
the monitor's page, execute the notebooks, build the ChartBook site, run the
tests. So `uv run doit` is also the backfill: set START_DATE earlier, in
`.env` or the environment, and the next run pulls the longer window and
fills the history. Two task groups are not run by default:

- `uv run doit latest`: the daily update. Pull the most recent session only,
  add its row to the history if it is not there yet, and rebuild the
  monitor's page. This is what the GitHub Actions workflow runs every
  morning, before it publishes the page.
- `uv run doit single_names`: the single-name part, which pulls the
  OptionMetrics surface from WRDS and computes the Martin-Shi bounds.
"""

#######################################
## Configuration and Helpers for PyDoit
#######################################
## Make sure the src folder is in the path
import sys

sys.path.insert(1, "./src/")

import shutil
from os import environ
from pathlib import Path

from doit.tools import config_changed
from settings import config

DOIT_CONFIG = {
    "backend": "sqlite3",
    "dep_file": "./.doit-db.sqlite",
    "default_tasks": [
        "config",
        "pull",
        "compute_history",
        "monitor_chart",
        "monitor_page",
        "run_notebooks",
        "build_chartbook_site",
        "run_pytest",
    ],
}

BASE_DIR = config("BASE_DIR")
DATA_DIR = config("DATA_DIR")
MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
OS_TYPE = config("OS_TYPE")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")

## Helpers for handling Jupyter Notebook tasks
environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
def jupyter_execute_notebook(notebook_path):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
def jupyter_to_html(notebook_path, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir={output_dir} {notebook_path}"
def jupyter_to_md(notebook_path, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --output-dir={output_dir} {notebook_path}"
def jupyter_clear_output(notebook_path):
    """Clear the output of a notebook"""
    return f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
# fmt: on


def mv(from_path, to_path):
    """Move a file to a folder"""
    from_path = Path(from_path)
    to_path = Path(to_path)
    to_path.mkdir(parents=True, exist_ok=True)
    if OS_TYPE == "nix":
        command = f"mv {from_path} {to_path}"
    else:
        command = f"move {from_path} {to_path}"
    return command


##################################
## Begin rest of PyDoit tasks here
##################################


def task_config():
    """Create the data and output directories and record the pull window.

    `python ./src/settings.py` also writes `_data/pull_window.txt`, the
    window the pulls cover (START_DATE to END_DATE). The pull tasks depend
    on that file, so they run again when the window changes; this task
    reruns, and rewrites the file, whenever the window does.
    """
    return {
        "actions": ["python ./src/settings.py"],
        "targets": [DATA_DIR, OUTPUT_DIR, DATA_DIR / "pull_window.txt"],
        "file_dep": ["./src/settings.py"],
        "uptodate": [config_changed({"start_date": START_DATE, "end_date": END_DATE})],
        "clean": [],
    }


def task_pull():
    """Pull data from external sources.

    The Databento pull is free under the course subscription; it verifies
    this with the free cost-estimate endpoint and refuses to download
    anything that is not $0.00. The pull runs again whenever the window in
    the settings changes (START_DATE or END_DATE; END_DATE is today by
    default, so the first `doit` of a day pulls again) and is otherwise
    skipped. Set START_DATE earlier and `doit` backfills. `doit latest`
    pulls the most recent session only.
    """
    yield {
        "name": "es_options",
        "doc": "Pull E-mini S&P 500 option settlements and definitions from Databento",
        "actions": [
            "python ./src/settings.py",
            "python ./src/pull_es_options.py",
        ],
        "targets": [
            DATA_DIR / "es_options_settlements.parquet",
            DATA_DIR / "es_futures_settlements.parquet",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/pull_es_options.py",
            "./src/source_status.py",
            # The window (START_DATE to END_DATE): pull again when it changes.
            DATA_DIR / "pull_window.txt",
        ],
        "verbosity": 2,  # show the cost estimates
        "clean": [],
    }
    yield {
        "name": "fred",
        "doc": "Pull the VIX and the one-month Treasury yield from FRED (no API key required)",
        "actions": [
            "python ./src/settings.py",
            "python ./src/pull_fred.py",
        ],
        "targets": [DATA_DIR / "fred_vix_rates.parquet"],
        "file_dep": [
            "./src/settings.py",
            "./src/pull_fred.py",
            "./src/source_status.py",
            # FRED posts each close the next morning: refresh with the window.
            DATA_DIR / "pull_window.txt",
        ],
        "verbosity": 2,
        "clean": [],
    }


ENGINE = [
    "./src/source_status.py",
    "./src/black_scholes.py",
    "./src/smile.py",
    "./src/rnd.py",
    "./src/svix.py",
    "./src/crash_prob.py",
    "./src/utility_correction.py",
]


def task_compute_history():
    """Append the sessions in the pull to the monitor's history, once per pull.

    Same computation as `latest`, but as a normal build step over the whole
    pull window: it runs when the pull or the engine changes and is
    otherwise up to date. The history is an output like any
    other, in `_output/`; the deploy workflow keeps a copy of it on the
    `monitor-history` branch between runs.
    """
    return {
        "actions": ["python ./src/monitor.py"],
        "file_dep": [
            "./src/monitor.py",
            "./src/pull_es_options.py",
            "./src/pull_fred.py",
            *ENGINE,
            DATA_DIR / "es_options_settlements.parquet",
            DATA_DIR / "fred_vix_rates.parquet",
        ],
        "targets": [
            OUTPUT_DIR / "svix_monitor_history.csv",
            OUTPUT_DIR / "svix_monitor_latest.csv",
        ],
        "verbosity": 2,
        "clean": True,
    }


def task_latest():
    """The daily update: pull the latest session, add its row, rebuild the page.

    Always runs when invoked (`doit latest`): a daily update's job is to
    check again. It pulls the most recent session's settlements and the FRED
    series, computes the rows for the sessions the history does not have
    (normally one), appends them, and rebuilds the monitor's page from the
    history. When the history is already current, nothing changes but the
    page's timestamp. The backfill is the default `doit`, over the window in
    the settings. Not in the default task list.
    """
    return {
        "actions": [
            "python ./src/settings.py",
            "python ./src/pull_es_options.py --latest",
            "python ./src/pull_fred.py",
            "python ./src/monitor.py",
            "python ./src/build_page.py",
        ],
        "file_dep": [
            "./src/pull_es_options.py",
            "./src/pull_fred.py",
            "./src/monitor.py",
            "./src/build_page.py",
            "./src/monitor_chart.py",
            *ENGINE,
        ],
        "uptodate": [False],
        "verbosity": 2,
    }


def task_monitor_chart():
    """Draw the monitor's charts (PNG and HTML) from the history"""
    return {
        "actions": ["python ./src/monitor_chart.py"],
        "file_dep": [
            "./src/monitor_chart.py",
            "./src/monitor.py",
            OUTPUT_DIR / "svix_monitor_history.csv",
        ],
        "targets": [
            OUTPUT_DIR / "svix_monitor.png",
            OUTPUT_DIR / "svix_monitor.html",
        ],
        "task_dep": ["compute_history"],
        "clean": True,
    }


def task_monitor_page():
    """Build the monitor's page, one HTML file, from the history

    `_output/monitor_page/index.html` is what the deploy workflow publishes
    on `gh-pages`: the latest reading, the chart, and the last ten sessions.
    """
    return {
        "actions": ["python ./src/build_page.py"],
        "file_dep": [
            "./src/build_page.py",
            "./src/monitor_chart.py",
            "./src/monitor.py",
            OUTPUT_DIR / "svix_monitor_history.csv",
        ],
        "targets": [OUTPUT_DIR / "monitor_page" / "index.html"],
        "task_dep": ["compute_history"],
        "clean": True,
    }


def task_single_names():
    """The single-name part: the OptionMetrics surface from WRDS and the Martin-Shi bounds.

    Not a default task: it needs a WRDS account, and the GitHub Actions
    deployment of the monitor has none. Run `doit single_names` once; the
    results feed notebook 04.
    """
    yield {
        "name": "pull_optionmetrics",
        "doc": "Pull the volatility surface, prices, and zero curve for SPX, Apple, and AIG",
        "actions": [
            "python ./src/settings.py",
            "python ./src/pull_optionmetrics.py",
        ],
        "targets": [
            DATA_DIR / "optionm_vol_surface.parquet",
            DATA_DIR / "optionm_security_prices.parquet",
            DATA_DIR / "optionm_zero_curve.parquet",
        ],
        "file_dep": ["./src/settings.py", "./src/pull_optionmetrics.py"],
        "verbosity": 2,
        "clean": [],
    }
    yield {
        "name": "bounds",
        "doc": "Clean the surface and compute the crash bounds and the index probabilities",
        "actions": ["python ./src/single_name_bounds.py"],
        "file_dep": [
            "./src/single_name_bounds.py",
            "./src/bounds.py",
            *ENGINE,
            DATA_DIR / "optionm_vol_surface.parquet",
            DATA_DIR / "optionm_security_prices.parquet",
            DATA_DIR / "optionm_zero_curve.parquet",
        ],
        "targets": [
            OUTPUT_DIR / "single_name_bounds.parquet",
            OUTPUT_DIR / "index_crash_probabilities.parquet",
        ],
        "verbosity": 2,
        "clean": True,
    }
    yield {
        "name": "svix_vs_martin",
        "doc": "SVIX from the OptionMetrics surface, against Martin's published series",
        "actions": ["python ./src/calc_svix_vs_martin.py"],
        "file_dep": [
            "./src/calc_svix_vs_martin.py",
            "./src/single_name_bounds.py",
            *ENGINE,
            MANUAL_DATA_DIR / "martin_2017/svix2.csv",
            DATA_DIR / "optionm_vol_surface.parquet",
            DATA_DIR / "optionm_security_prices.parquet",
            DATA_DIR / "optionm_zero_curve.parquet",
        ],
        "targets": [OUTPUT_DIR / "svix_vs_martin.parquet"],
        "verbosity": 2,
        "clean": True,
    }


# Notebook 04 reads the single-name results when they exist and says so when
# they do not, so the site builds with or without a WRDS account. Its file
# dependencies include the results only if they are present: doit refuses to
# run a task whose dependency is missing, and the public monitor has no WRDS.
_single_name_results = [
    p
    for p in (
        OUTPUT_DIR / "single_name_bounds.parquet",
        OUTPUT_DIR / "index_crash_probabilities.parquet",
        OUTPUT_DIR / "svix_vs_martin.parquet",
    )
    if p.exists()
]

notebook_tasks = {
    "01_es_options_data_ipynb": {
        "path": "./src/01_es_options_data_ipynb.py",
        "file_dep": [
            "./src/figures.py",
            "./src/monitor.py",
            OUTPUT_DIR / "svix_monitor_history.csv",
            "./src/pull_es_options.py",
            "./src/pull_fred.py",
            *ENGINE,
            DATA_DIR / "es_options_settlements.parquet",
            DATA_DIR / "es_futures_settlements.parquet",
            DATA_DIR / "fred_vix_rates.parquet",
        ],
        "targets": [],
    },
    "02_svix_and_crash_probability_ipynb": {
        "path": "./src/02_svix_and_crash_probability_ipynb.py",
        "file_dep": [
            "./src/figures.py",
            "./src/monitor.py",
            OUTPUT_DIR / "svix_monitor_history.csv",
            "./src/pull_es_options.py",
            "./src/pull_fred.py",
            *ENGINE,
            DATA_DIR / "es_options_settlements.parquet",
            DATA_DIR / "fred_vix_rates.parquet",
        ],
        "targets": [],
    },
    "03_daily_monitor_ipynb": {
        "path": "./src/03_daily_monitor_ipynb.py",
        "file_dep": [
            "./src/figures.py",
            *ENGINE,
            "./src/monitor.py",
            "./src/monitor_chart.py",
            OUTPUT_DIR / "svix_monitor_history.csv",
            OUTPUT_DIR / "svix_monitor.png",
        ],
        "targets": [],
    },
    "04_single_name_crash_bounds_ipynb": {
        "path": "./src/04_single_name_crash_bounds_ipynb.py",
        "file_dep": [
            "./src/figures.py",
            *ENGINE,
            "./src/single_name_bounds.py",
            "./src/bounds.py",
            "./src/calc_svix_vs_martin.py",
            MANUAL_DATA_DIR / "martin_2017/crashprob.csv",
            MANUAL_DATA_DIR / "martin_2017/svix2.csv",
            *_single_name_results,
        ],
        "targets": [],
    },
}


# fmt: off
def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks if the script version of it has been changed.
    """
    for notebook in notebook_tasks.keys():
        pyfile_path = Path(notebook_tasks[notebook]["path"])
        notebook_path = pyfile_path.with_suffix(".ipynb")
        notebook_name = notebook_path.stem
        yield {
            "name": notebook,
            "actions": [
                """python -c "import sys; from datetime import datetime; print(f'Start """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
                f"jupytext --to notebook --output {notebook_path} {pyfile_path}",
                jupyter_execute_notebook(notebook_path),
                jupyter_to_html(notebook_path),
                mv(notebook_path, OUTPUT_DIR),
                """python -c "import sys; from datetime import datetime; print(f'End """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
            ],
            "file_dep": [
                pyfile_path,
                *notebook_tasks[notebook]["file_dep"],
            ],
            "targets": [
                OUTPUT_DIR / f"{notebook_name}.html",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
        }
# fmt: on

sphinx_targets = [
    "./docs/index.html",
]


def task_build_chartbook_site():
    """Compile Sphinx Docs"""
    notebook_scripts = [
        Path(notebook_tasks[notebook]["path"])
        for notebook in notebook_tasks.keys()
    ]
    file_dep = [
        "./README.md",
        "./chartbook.toml",
        *notebook_scripts,
    ]

    return {
        "actions": [
            "chartbook build -f",
        ],  # Use docs as build destination
        "targets": sphinx_targets,
        "file_dep": file_dep,
        "task_dep": [
            "run_notebooks",
            "monitor_chart",
        ],
        "clean": True,
    }


def task_run_pytest():
    """Run pytest and save results to OUTPUT_DIR"""
    src_py_files = list(Path("./src").glob("*.py"))
    test_output = OUTPUT_DIR / "pytest_results.xml"

    def run_pytest():
        import subprocess

        result = subprocess.run(
            ["pytest", f"--junitxml={test_output}"],
        )
        if result.returncode != 0:
            # Remove the XML so doit won't consider the target up-to-date
            Path(test_output).unlink(missing_ok=True)
            raise RuntimeError(f"pytest failed with exit code {result.returncode}")

    return {
        "actions": [run_pytest],
        "targets": [test_output],
        "file_dep": src_py_files,
        "clean": True,
        "verbosity": 2,
    }
