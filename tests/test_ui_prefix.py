"""The Control Room under a Fuzzball path endpoint: ui/server.py's PathPrefix.

    python -m pytest tests/test_ui_prefix.py -v

A path endpoint passes the whole path (/endpoints/accounts/.../ui/control-room/...) to the service, as Jupyter's
catalog entry shows (it sets base_url to $FB_ENDPOINT_PATH_JUPYTER). The page must answer there and at /.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ui"))
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

PREFIX = "/endpoints/accounts/acct/workflows/wf/ui/control-room/"


@pytest.fixture()
def app():
    import server as S
    ops = S.Ops("http://127.0.0.1:9", {"box-a": "http://127.0.0.1:9", "box-b": "http://127.0.0.1:9"})
    return S, S.create_app(ops, S.Agent(ops, None, "x", None), telemetry=False)


def test_page_static_and_api_answer_under_the_prefix_and_at_root(app):
    S, inner = app
    with TestClient(S.PathPrefix(inner, PREFIX)) as c:
        for base in (PREFIX, "/"):
            page = c.get(base)
            assert page.status_code == 200 and 'src="static/app.js"' in page.text, base
            assert c.get(base + "static/app.js").status_code == 200, base
            assert c.get(base + "api/events").status_code == 200, base


def test_the_bare_prefix_redirects_to_the_trailing_slash(app):
    S, inner = app
    with TestClient(S.PathPrefix(inner, PREFIX)) as c:
        r = c.get(PREFIX.rstrip("/"), follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == PREFIX


def test_no_prefix_or_a_subdomain_endpoint_changes_nothing(app):
    S, inner = app
    for prefix in ("", "/"):
        with TestClient(S.PathPrefix(inner, prefix)) as c:
            assert c.get("/").status_code == 200
            assert c.get("/endpoints/x/").status_code == 404


def test_the_page_and_its_scripts_are_checked_again_by_the_browser(app):
    """After a deploy a browser must not run an old app.js against the new server: "no-cache" on the page and
    on its static files, at the root and under the prefix. (The header had been lost for the scripts: the
    path was read after the static mount had changed it.)"""
    S, inner = app
    with TestClient(S.PathPrefix(inner, PREFIX)) as c:
        for base in (PREFIX, "/"):
            for path in ("", "static/app.js", "static/style.css"):
                assert c.get(base + path).headers.get("cache-control") == "no-cache", base + path
            assert "cache-control" not in c.get(base + "api/events").headers, base

