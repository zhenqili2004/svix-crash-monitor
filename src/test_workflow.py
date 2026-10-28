"""Tests for the daily monitor's workflow, .github/workflows/deploy_pages.yml.

A workflow is configuration, so the tests read the YAML and check the five
things the monitor needs: a schedule, the secret, the permission to push,
the step that runs the daily update, and the branch the page is deployed to.
"""

import re
from pathlib import Path

WORKFLOW_PATH = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "deploy_pages.yml"


def _workflow():
    text = WORKFLOW_PATH.read_text()
    # Drop comments so that the examples in them do not count as answers.
    uncommented = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    return uncommented


def _parse():
    import yaml  # PyYAML, installed with the notebook tooling

    return yaml.safe_load(WORKFLOW_PATH.read_text())


def test_workflow_has_a_daily_schedule():
    wf = _parse()
    on = wf.get("on") or wf.get(True)  # PyYAML reads a bare `on:` key as True
    assert on and "schedule" in on, "The workflow needs an `on: schedule:` trigger"
    crons = [entry.get("cron") for entry in on["schedule"]]
    assert crons and all(crons), "Each schedule entry needs a `cron:` line"
    fields = crons[0].split()
    assert len(fields) == 5, "A cron line has five fields: minute hour day-of-month month day-of-week"
    minute, hour, dom, month, dow = fields
    assert (dom, month, dow) == ("*", "*", "*"), "Every day: the last three fields are *"
    assert (minute, hour) == ("30", "14"), (
        "Run at 14:30 UTC, after CME's final settlements of the previous evening have reached Databento"
    )


def test_workflow_reads_the_key_from_a_secret():
    text = _workflow()
    assert re.search(r"DATABENTO_API_KEY:\s*\$\{\{\s*secrets\.DATABENTO_API_KEY\s*\}\}", text), (
        "The build step's env should set DATABENTO_API_KEY from ${{ secrets.DATABENTO_API_KEY }}"
    )
    assert not re.search(r"db-[A-Za-z0-9]{20,}", text), "Never write a key into the workflow file"


def test_workflow_can_push():
    wf = _parse()
    permissions = wf.get("permissions", {})
    assert permissions.get("contents") == "write", (
        "The job pushes the history branch and force-pushes gh-pages, so it needs `permissions: contents: write`"
    )


def test_workflow_runs_the_pipeline_with_uv():
    text = _workflow()
    assert re.search(r"run:\s*uv run doit latest\b", text), (
        "A step should run the daily update, which also rebuilds the page: `uv run doit latest`"
    )
    assert "astral-sh/setup-uv" in text, "Set up uv with the astral-sh/setup-uv action before using it"


def test_workflow_deploys_to_gh_pages():
    text = _workflow()
    assert re.search(r"git checkout --orphan gh-pages", text), "Deploy from an orphan gh-pages branch"
    assert re.search(r"git push --force origin gh-pages", text), "Force-push the orphan branch to gh-pages"
