from published_site import (
    I_HAVE_DEPLOYED_MY_MONITOR,
    URL_TO_MY_MONITOR_SITE,
    URL_TO_MY_PUBLIC_REPO,
)


def test_monitor_deployed():
    assert I_HAVE_DEPLOYED_MY_MONITOR
    assert URL_TO_MY_MONITOR_SITE.startswith("https://")
    assert URL_TO_MY_MONITOR_SITE not in ("", "YOUR_SITE_URL")
    assert URL_TO_MY_PUBLIC_REPO.startswith("https://github.com/")
    assert URL_TO_MY_PUBLIC_REPO not in ("", "YOUR_REPO_URL")
