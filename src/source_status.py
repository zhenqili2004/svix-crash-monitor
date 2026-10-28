"""Record that an external data source was down, so the workflow can tell an
outage from a bug.

The pull scripts retry server and network errors with backoff. If a source
is still down after the last retry, the script calls ``mark_unavailable``
and then fails. The deploy workflow reads the marker: an outage ends the run
with a warning and leaves yesterday's site up (the next morning's pull, which
covers the trailing week, fills the gap); any other failure, a bug, fails the
run as usual.
"""

import time
from pathlib import Path

from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))
MARKER = OUTPUT_DIR / "SOURCE_UNAVAILABLE"

MAX_TRIES = 6  # waits of 15, 30, 60, 120, and 240 seconds between them


def clear():
    """Remove a marker left by an earlier run."""
    MARKER.unlink(missing_ok=True)


def mark_unavailable(source, error):
    """Record that ``source`` was still down after the last retry."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MARKER, "a") as f:
        f.write(f"{source} was unavailable after {MAX_TRIES} tries: {str(error)[:200]}\n")


def is_transient(error, transient):
    """Whether ``error`` is worth retrying.

    ``transient`` is an exception class, a tuple of them, or a function that
    takes the error and returns True or False.
    """
    if isinstance(transient, (type, tuple)):
        return isinstance(error, transient)
    return bool(transient(error))


def retry(call, *args, source, transient, max_tries=MAX_TRIES, **kwargs):
    """Call ``call(*args, **kwargs)``, retrying the errors ``transient`` names.

    Waits 15 seconds before the second try and doubles each time. If the last
    try fails too, records the outage with ``mark_unavailable`` and re-raises.
    Any other exception, an authentication error or a bug, is raised at once.
    """
    for attempt in range(max_tries):
        try:
            return call(*args, **kwargs)
        except Exception as error:
            if not is_transient(error, transient):
                raise
            if attempt == max_tries - 1:
                mark_unavailable(source, error)
                raise
            wait = 15 * 2**attempt
            print(f"  {source}: retrying in {wait}s after {type(error).__name__}: {str(error)[:80]}")
            time.sleep(wait)


class ServerError(Exception):
    """A web source answered with a 5xx status: a server problem, worth retrying."""


def http_get(url, *, source, params=None, timeout=60):
    """``requests.get`` with the retry policy above.

    Server errors (5xx), dropped connections, and timeouts are retried. A
    4xx answer, a bad URL or a bad request, raises at once.
    """
    import requests

    def get():
        response = requests.get(url, params=params, timeout=timeout)
        if response.status_code >= 500:
            raise ServerError(f"HTTP {response.status_code} from {source}")
        response.raise_for_status()
        return response

    transient = (ServerError, requests.exceptions.ConnectionError, requests.exceptions.Timeout)
    return retry(get, source=source, transient=transient)
