"""Score a homework submission from a pytest JUnit XML report.

GitHub Classroom used to do this with the `education/autograding` action.
Classroom was retired in August 2026, so the same job is done here: read the
point map in `.github/autograding.json`, look up each test's outcome in the
JUnit XML that pytest wrote, and total the points.

Usage:
    pytest --junitxml=_output/test-results.xml   # allowed to fail
    python .github/scripts/autograde.py _output/test-results.xml

The script prints a colored report to the log, in the style of GitHub
Classroom's old autograder, writes a per-part table to the GitHub Actions job
summary, and saves a machine-readable `_output/score.json`.
It exits non-zero unless every point is earned, so that an incomplete
submission shows a red X on the commit.
"""

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
POINT_MAP = REPO_ROOT / ".github" / "autograding.json"


def parse_results(xml_path):
    """Map "src/test_file.py::test_name" -> ("passed"|"failed"|"error"|"skipped", message).

    pytest records the module as a dotted classname ("src.test_dodo"), so it
    is converted back into the node id form used by the point map.
    """
    results = {}
    if not xml_path.exists():
        return results

    for case in ET.parse(xml_path).getroot().iter("testcase"):
        classname = case.get("classname") or ""
        name = case.get("name") or ""
        if not classname or not name:
            continue
        nodeid = f"{classname.replace('.', '/')}.py::{name}"

        status, message = "passed", ""
        for tag, label in (("failure", "failed"), ("error", "error"), ("skipped", "skipped")):
            found = case.find(tag)
            if found is not None:
                status = label
                message = (found.get("message") or "").strip().splitlines()
                message = message[0] if message else ""
                break
        results[nodeid] = (status, message)
    return results


MARKS = {"passed": "✅", "failed": "❌", "error": "💥", "skipped": "⏭️", "missing": "❓"}

# The log printout copies the one GitHub Classroom's autograder used to print:
# each test in green or red, a row of sparkles when everything passes, and the
# points on a bold line at the end. GitHub's log viewer renders these ANSI
# codes; set NO_COLOR to turn them off.
USE_COLOR = not os.environ.get("NO_COLOR") and (
    os.environ.get("GITHUB_ACTIONS") == "true" or sys.stdout.isatty()
)
STYLES = {
    "bold": "1",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "cyan": "36",
    "bold_green": "1;32",
    "bold_red": "1;31",
}
STATUS_STYLE = {
    "passed": "green",
    "failed": "red",
    "error": "red",
    "skipped": "yellow",
    "missing": "red",
}
SPARKLES = "✨🌟💖💎🦄💎💖🌟✨🌟💖💎🦄💎💖🌟✨"


def paint(text, style):
    if not USE_COLOR:
        return text
    return f"\x1b[{STYLES[style]}m{text}\x1b[0m"


def print_console_report(parts, earned_total, possible_total):
    """Print the colored, Classroom-style report to the job log."""
    print()
    for name, earned, possible, rows in parts:
        print(paint(f"📝 {name}", "cyan"))
        for nodeid, status, got, points, message in rows:
            test_name = nodeid.split("::")[-1]
            line = f"{MARKS[status]} {test_name}  ({got}/{points})"
            print(paint(line, STATUS_STYLE[status]))
            if message and status != "passed":
                print(f"      {message[:160]}")
        part_style = "green" if earned == possible else "red"
        print(paint(f"   {name}: {earned}/{possible}", part_style))
        print()

    points = f"Points {earned_total}/{possible_total}"
    if earned_total == possible_total:
        print(paint("🎉 All tests passed! 🎉", "bold_green"))
        print()
        print(SPARKLES)
        print()
        print(paint(f"🏆 {points}", "bold_green"))
        # Shows the score as a notice on the run's summary page as well.
        if os.environ.get("GITHUB_ACTIONS") == "true":
            print(f"::notice title=Score::{points}. All tests passed.")
    else:
        failing = sum(
            1 for _, _, _, rows in parts for row in rows if row[1] != "passed"
        )
        noun = "test is" if failing == 1 else "tests are"
        print(paint(f"❌ {points}", "bold_red"))
        print(paint(f"{failing} {noun} not passing yet. See the list above.", "red"))
        if os.environ.get("GITHUB_ACTIONS") == "true":
            print(f"::error title=Score::{points}. {failing} {noun} not passing yet.")
    print()


def main():
    xml_path = Path(sys.argv[1] if len(sys.argv) > 1 else "_output/test-results.xml")
    if not xml_path.is_absolute():
        xml_path = REPO_ROOT / xml_path

    point_map = json.loads(POINT_MAP.read_text())
    results = parse_results(xml_path)

    lines = [f"# {point_map['assignment']}", ""]
    earned_total = possible_total = 0
    detail = []
    parts = []

    for part in point_map["parts"]:
        earned = possible = 0
        rows = []
        for test in part["tests"]:
            nodeid = test["nodeid"]
            points = test["points"]
            status, message = results.get(nodeid, ("missing", "test did not run"))
            got = points if status == "passed" else 0

            earned += got
            possible += points
            rows.append((nodeid, status, got, points, message))
            detail.append(
                {
                    "part": part["name"],
                    "nodeid": nodeid,
                    "status": status,
                    "earned": got,
                    "possible": points,
                    "message": message,
                }
            )

        earned_total += earned
        possible_total += possible
        parts.append((part["name"], earned, possible, rows))

        lines += [f"## {part['name']} — {earned}/{possible}", ""]
        lines += ["| | Test | Points |", "|---|---|---|"]
        for nodeid, status, got, points, message in rows:
            test_name = nodeid.split("::")[-1]
            note = f" <br><sub>{message[:160]}</sub>" if message else ""
            lines.append(f"| {MARKS[status]} | `{test_name}`{note} | {got}/{points} |")
        lines.append("")

    if not results:
        lines += [
            "> **No test results were found.** pytest did not produce a report, "
            "which usually means the whole suite failed to start. Check the "
            "test step's log above.",
            "",
        ]

    lines.insert(1, f"**Score: {earned_total} / {possible_total}**")
    lines.insert(2, "")
    summary = "\n".join(lines)

    print_console_report(parts, earned_total, possible_total)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write(summary + "\n")

    score_path = REPO_ROOT / "_output" / "score.json"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(
        json.dumps(
            {
                "assignment": point_map["assignment"],
                "earned": earned_total,
                "possible": possible_total,
                "tests": detail,
            },
            indent=2,
        )
    )

    if github_output := os.environ.get("GITHUB_OUTPUT"):
        with open(github_output, "a", encoding="utf-8") as fh:
            fh.write(f"score={earned_total}\n")
            fh.write(f"possible={possible_total}\n")

    return 0 if earned_total == possible_total else 1


if __name__ == "__main__":
    sys.exit(main())
