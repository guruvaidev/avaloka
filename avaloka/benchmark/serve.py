"""Serve the benchmark scorecard on localhost, so a human can look at it.

:mod:`avaloka.benchmark.report` writes a file; this opens it in a browser. The
reason it exists rather than leaving people to double-click the HTML is the
refresh loop: pointed at a ``scorecard.json``, this re-renders on every request,
so the way to see the result of a re-run is to reload the tab. Pointed at an
already-rendered ``.html`` it re-reads the file per request, which does the same
for a report somebody else generates.

Two deliberate restrictions.

*Loopback only.* The default host is ``127.0.0.1`` and a non-loopback address is
refused outright, including ``0.0.0.0``. A scorecard names internal datasets,
column names, artifact paths and filesystem layout, and this server has no
authentication of any kind; binding it to a routable interface publishes all of
that to anything that can reach the host. Serving it off a developer laptop or a
CI runner on a shared network is exactly the case that has to not happen by
accident, so the refusal is in code rather than in a comment.

*Nothing is served off the filesystem.* The handler answers a fixed set of
routes and 404s everything else. There is no
:class:`~http.server.SimpleHTTPRequestHandler`, no directory index, and --
importantly -- no path translation at all: the request path is compared against
a literal allowlist and is never joined to a directory or resolved against the
disk. Path traversal is not filtered here, it is structurally absent, because
there is no code path from a URL to a filename. The one file this process will
read is the one named in ``serve(path=...)``.
"""

from __future__ import annotations

import errno
import http.server
import ipaddress
import json
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from avaloka.benchmark.report import render_html

__all__ = ["serve"]

#: The only paths that answer. Everything else is a 404; see the module
#: docstring on why this is a literal allowlist and not a lookup.
_PAGE_ROUTES = ("/", "/index.html")
_JSON_ROUTE = "/scorecard.json"


def _is_loopback(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # A hostname we cannot resolve to a literal. Refuse it: we cannot prove
        # it is loopback, and guessing wrong is the failure this guards against.
        return False


class _ScorecardHandler(http.server.BaseHTTPRequestHandler):
    """Answers the scorecard routes and nothing else."""

    server_version = "avaloka-benchmark"
    sys_version = ""

    # Set by serve() on the server instance.
    source_path: Path
    source_is_json: bool
    page_title: str

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # The point of this server is that a reload shows the latest run.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _render_page(self) -> tuple[int, bytes, str]:
        """The page, or a 5xx explaining why not. Never raises."""
        try:
            raw = self.source_path.read_text(encoding="utf-8")
        except OSError as exc:
            return (503, self._plain(f"Cannot read {self.source_path}: {exc}"),
                    "text/plain; charset=utf-8")
        if not self.source_is_json:
            return (200, raw.encode("utf-8"), "text/html; charset=utf-8")
        try:
            card = json.loads(raw)
        except json.JSONDecodeError as exc:
            # A run that is midway through writing the file lands here. Say so
            # and keep serving; the next reload will succeed.
            return (503, self._plain(
                f"{self.source_path} is not valid JSON right now ({exc}). "
                "If a benchmark run is writing it, reload in a moment."),
                "text/plain; charset=utf-8")
        try:
            page = render_html(card, title=self.page_title)
        except Exception as exc:  # noqa: BLE001 - a bad card must not kill the server
            return (500, self._plain(f"Could not render the scorecard: {exc!r}"),
                    "text/plain; charset=utf-8")
        return (200, page.encode("utf-8"), "text/html; charset=utf-8")

    @staticmethod
    def _plain(message: str) -> bytes:
        return (message + "\n").encode("utf-8")

    def _handle(self) -> None:
        route = urlsplit(self.path).path
        if route in _PAGE_ROUTES:
            self._send(*self._render_page())
            return
        if route == _JSON_ROUTE and self.source_is_json:
            try:
                body = self.source_path.read_bytes()
            except OSError as exc:
                self._send(503, self._plain(f"Cannot read {self.source_path}: {exc}"),
                           "text/plain; charset=utf-8")
                return
            self._send(200, body, "application/json; charset=utf-8")
            return
        self._send(404, self._plain(
            "Not found. This server answers / only (and /scorecard.json when it "
            "was given a JSON scorecard); it does not serve files from disk."),
            "text/plain; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming
        self._handle()

    def do_HEAD(self) -> None:  # noqa: N802 - http.server's naming
        self._handle()

    def log_message(self, fmt: str, *args: Any) -> None:
        # One compact line per request instead of http.server's default, which
        # includes a timestamp wide enough to wrap in a narrow terminal.
        print(f"  {self.command} {self.path} -> {fmt % args}".rstrip())


def serve(path: Path, *, port: int = 8765, host: str = "127.0.0.1",
          open_browser: bool = False) -> None:
    """Serve the scorecard at ``http://host:port/`` until interrupted.

    ``path`` is either a ``scorecard.json`` -- re-rendered through
    :func:`~avaloka.benchmark.report.render_html` on every request, so a reload
    picks up a re-run -- or an already-rendered ``.html``, which is re-read and
    returned as-is. The choice is made on the suffix: ``.json`` means render,
    anything else means serve verbatim.

    **Loopback only.** ``host`` must be a loopback address; ``0.0.0.0`` and any
    routable address raise :class:`ValueError`. The scorecard carries dataset
    names, column names and artifact paths, and nothing here authenticates
    anyone, so exposing it beyond the local machine is not a configuration this
    function offers. See the module docstring.

    Raises :class:`SystemExit` with a readable message -- not a traceback -- when
    ``path`` is missing or the port is already taken.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"No scorecard at {path}. Run the benchmark with --json first.")
    if not _is_loopback(host):
        raise ValueError(
            f"serve() binds loopback only; {host!r} is not a loopback address. "
            "The scorecard is unauthenticated and names internal datasets and "
            "paths, so it is not served to the network. Use 127.0.0.1 and an SSH "
            "tunnel if you need it from another machine."
        )

    source_is_json = path.suffix.lower() == ".json"
    handler = type("_BoundScorecardHandler", (_ScorecardHandler,), {
        "source_path": path.resolve(),
        "source_is_json": source_is_json,
        "page_title": "Avaloka benchmark scorecard",
    })

    try:
        httpd = http.server.ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        if exc.errno in (errno.EADDRINUSE, errno.EACCES):
            reason = ("is already in use" if exc.errno == errno.EADDRINUSE
                      else "is not available to this user")
            raise SystemExit(
                f"Port {port} on {host} {reason}. Pass a different port, or stop "
                f"whatever is listening on {port}."
            ) from None
        raise SystemExit(f"Could not bind {host}:{port}: {exc}") from None

    httpd.daemon_threads = True
    mode = ("re-rendering on each request" if source_is_json
            else "serving the rendered file as-is")
    print(f"Avaloka benchmark scorecard — {path} ({mode})")
    print(f"  http://{host}:{port}/")
    print("  loopback only; Ctrl-C to stop")

    if open_browser:
        try:
            webbrowser.open(f"http://{host}:{port}/")
        except Exception as exc:  # noqa: BLE001 - a headless box has no browser
            print(f"  (could not open a browser: {exc})")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.shutdown()
        httpd.server_close()
