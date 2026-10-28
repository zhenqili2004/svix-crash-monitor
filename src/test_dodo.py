"""Tests for the PyDoit task definitions in dodo.py.

These tests check that the task-creator functions in dodo.py return
well-formed task dictionaries: the actions run the right scripts, the file
dependencies point at the code and data the task depends on, and the targets
list the files the task produces. See https://pydoit.org/tasks.html for the
anatomy of a doit task.
"""

import sys
from pathlib import Path

# dodo.py lives in the repository root, one level above this test file
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dodo  # noqa: E402


def _as_strings(values):
    return [str(value) for value in values]


def test_task_latest():
    """The daily update: pull the latest session, append, rebuild the page, and never be up to date."""
    task = dodo.task_latest()
    assert isinstance(task, dict)

    actions = " ".join(_as_strings(task.get("actions", [])))
    for script in ["pull_es_options.py", "pull_fred.py", "monitor.py", "build_page.py"]:
        assert script in actions, f"The latest task should run src/{script}"
    assert "pull_es_options.py --latest" in actions, (
        "The daily update pulls the most recent session only: python ./src/pull_es_options.py --latest"
    )
    assert actions.index("pull_es_options.py") < actions.index("monitor.py") < actions.index("build_page.py"), (
        "Pull first, then compute the history, then build the page from it"
    )

    file_dep = " ".join(_as_strings(task.get("file_dep", [])))
    for module in ["monitor.py", "smile.py", "rnd.py", "svix.py", "crash_prob.py"]:
        assert module in file_dep, f"The latest task should depend on {module}"

    # A daily update's job is to check again: the task must run every time it
    # is invoked, which doit expresses as an `uptodate` that is always False.
    assert task.get("uptodate") == [False], (
        "The latest task must always run: set 'uptodate': [False]"
    )


def test_task_run_notebooks():
    subtasks = {subtask["name"]: subtask for subtask in dodo.task_run_notebooks()}
    for notebook in [
        "01_es_options_data_ipynb",
        "02_svix_and_crash_probability_ipynb",
        "03_daily_monitor_ipynb",
        "04_single_name_crash_bounds_ipynb",
    ]:
        assert notebook in subtasks, f"notebook_tasks should have an entry named {notebook}"
        subtask = subtasks[notebook]
        actions = " ".join(_as_strings(subtask["actions"]))
        assert f"{notebook}.py" in actions
        file_dep = " ".join(_as_strings(subtask["file_dep"]))
        assert f"{notebook}.py" in file_dep
        targets = " ".join(_as_strings(subtask["targets"]))
        assert f"{notebook}.html" in targets

    file_dep = " ".join(_as_strings(subtasks["03_daily_monitor_ipynb"]["file_dep"]))
    for expected in ["monitor.py", "svix_monitor_history.csv", "svix_monitor.png"]:
        assert expected in file_dep, f"The monitor notebook should depend on {expected}"
