"""Tests for the retry policy and the outage marker, offline."""

import pytest

import source_status


@pytest.fixture
def marker(tmp_path, monkeypatch):
    path = tmp_path / "SOURCE_UNAVAILABLE"
    monkeypatch.setattr(source_status, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(source_status, "MARKER", path)
    monkeypatch.setattr(source_status.time, "sleep", lambda seconds: None)
    return path


class Outage(Exception):
    pass


def test_retry_recovers_from_a_transient_error(marker):
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise Outage("HTTP 504")
        return "data"

    assert source_status.retry(flaky, source="Test", transient=(Outage,)) == "data"
    assert len(calls) == 3
    assert not marker.exists()


def test_retry_marks_an_outage_that_outlasts_every_try(marker):
    def down():
        raise Outage("HTTP 500")

    with pytest.raises(Outage):
        source_status.retry(down, source="Test", transient=(Outage,), max_tries=4)
    assert marker.exists() and "Test was unavailable" in marker.read_text()


def test_retry_does_not_retry_a_bug(marker):
    calls = []

    def bug():
        calls.append(1)
        raise KeyError("not a transient error")

    with pytest.raises(KeyError):
        source_status.retry(bug, source="Test", transient=(Outage,))
    assert len(calls) == 1
    assert not marker.exists()


def test_retry_accepts_a_predicate(marker):
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("stream ended")
        return "ok"

    assert source_status.retry(flaky, source="Test", transient=lambda e: "stream" in str(e)) == "ok"
    with pytest.raises(RuntimeError):
        source_status.retry(lambda: (_ for _ in ()).throw(RuntimeError("bug")), source="Test",
                            transient=lambda e: "stream" in str(e))
