"""``avaloka catalog list|show|use|forget`` (design §C.6, 16g).

Two homes for the catalog, one command set:

- **local** (the default): the SQLite file under ``~/.avaloka`` that
  ``avaloka analyze`` registers into (``avaloka/catalog/local.py``);
- **a server** (``--server URL --token TOKEN``, or ``AVALOKA_API_URL`` and
  ``AVALOKA_API_TOKEN``): the hosted catalog through ``/catalog/datasets``,
  over plain ``urllib`` -- the CLI gains no dependency for it.

``use`` resolves a reference to one (dataset, version) and prints the command
that runs on it; it keeps no hidden state, so what you see is what runs.
``forget`` asks before it deletes.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

console = Console()
catalog_app = typer.Typer(
    help="Your dataset catalog: every dataset your analyses touched, with its schema history. "
         "Local by default (~/.avaloka/catalog.db); --server reaches a hosted catalog.",
    no_args_is_help=True,
)

_SERVER_HELP = "An Avaloka API base URL (env AVALOKA_API_URL). Without it, the local catalog."
_TOKEN_HELP = "Bearer token for --server (env AVALOKA_API_TOKEN); never printed."


def _die(problem: str, *, fix: Optional[str] = None, code: int = 1) -> typer.Exit:
    # escape(): a message may carry "[sources]" or a dataset name in brackets,
    # which rich would otherwise read as markup and swallow.
    console.print(f"[red]{escape(problem)}[/red]")
    if fix:
        console.print(f"[dim]{escape(fix)}[/dim]")
    return typer.Exit(code=code)


class _Remote:
    """The hosted catalog over HTTP. Only the paths the API router serves."""

    def __init__(self, base: str, token: Optional[str]) -> None:
        self.base = base.rstrip("/")
        self.token = token

    def _call(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        req = urllib.request.Request(url, method=method, headers={"Accept": "application/json"})
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310  (the user named the server)
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else None
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("detail", "")
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f"the server answered {exc.code}: {detail or exc.reason}") from None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"could not reach {self.base}: {exc.reason}") from None

    def list(self, query: Optional[str], kind: Optional[str]) -> List[Dict[str, Any]]:
        return (self._call("GET", "/catalog/datasets", {"q": query, "kind": kind}) or {}).get("datasets", [])

    def get(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self._call("GET", f"/catalog/datasets/{urllib.parse.quote(dataset_id, safe='')}")
        except RuntimeError as exc:
            if "404" in str(exc):
                return None
            raise

    def resolve(self, ref: str) -> Dict[str, Any]:
        return self._call("GET", "/catalog/resolve", {"ref": ref})

    def forget(self, dataset_id: str) -> Dict[str, Any]:
        return self._call("DELETE", f"/catalog/datasets/{urllib.parse.quote(dataset_id, safe='')}")


class _Local:
    """The local catalog, through the store itself."""

    def __init__(self) -> None:
        from avaloka.catalog import local
        self.local = local
        self.store = local.store()

    def list(self, query: Optional[str], kind: Optional[str]) -> List[Dict[str, Any]]:
        return self.store.list(query=query, kind=kind)

    def get(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        return self.store.get(dataset_id)

    def resolve(self, ref: str) -> Dict[str, Any]:
        binding = self.store.resolve(ref)
        out = binding.as_dict()
        out["path"] = binding.locator.get("path") or binding.locator.get("output_location")
        return out

    def forget(self, dataset_id: str) -> Dict[str, Any]:
        return self.store.forget(dataset_id)


def _backend(server: Optional[str], token: Optional[str]):
    server = server or os.environ.get("AVALOKA_API_URL", "").strip() or None
    token = token or os.environ.get("AVALOKA_API_TOKEN", "").strip() or None
    if server:
        return _Remote(server, token), f"server {server}"
    try:
        local = _Local()
    except Exception as exc:  # noqa: BLE001  (the store said which extra)
        raise _die(f"The local catalog isn't available: {exc}",
                   fix="Install the sources extra: pip install 'avaloka[sources]'")
    return local, f"local catalog ({local.local.catalog_url()})"


def _server_opt():
    return typer.Option(None, "--server", help=_SERVER_HELP)


def _token_opt():
    return typer.Option(None, "--token", help=_TOKEN_HELP, show_default=False)


@catalog_app.command("list")
def list_datasets(
    query: Optional[str] = typer.Argument(None, help="Words to search for: a name, a tag, a term, an analysis id."),
    kind: Optional[str] = typer.Option(None, "--kind", help="file|object|db_table|warehouse_table|lake_table|kafka_window|derived"),
    as_json: bool = typer.Option(False, "--json", help="Machine output."),
    server: Optional[str] = _server_opt(),
    token: Optional[str] = _token_opt(),
):
    """List your datasets, newest first, with the `my.<name>` to use each by."""
    backend, where = _backend(server, token)
    try:
        items = backend.list(query, kind)
    except Exception as exc:  # noqa: BLE001
        raise _die(f"I couldn't list your catalog: {exc}")
    if as_json:
        console.print_json(json.dumps(items, default=str))
        return
    if not items:
        console.print(f"[dim]No datasets{' matching ' + repr(query) if query else ' yet'} in your {where}. "
                      "Run `avaloka analyze <file> --goal ...` and the file is registered.[/dim]")
        return
    table = Table(title=f"Your datasets ({where})")
    for col in ("Use as", "Kind", "Version", "Columns", "Rows", "Captured", "Id"):
        table.add_column(col)
    for it in items:
        rows = it.get("row_count")
        table.add_row(it.get("name") or "—", it.get("kind", ""), str(it.get("version_no", "")),
                      str(it.get("columns", "")), f"{rows:,}" if isinstance(rows, int) else "—",
                      (it.get("captured_at") or "")[:10], it.get("dataset_id", ""))
    console.print(table)
    console.print("[dim]avaloka analyze my.<name> --goal ... runs on one; my.<name>@2 picks a version; "
                  "analysis:<id> the version a past run read.[/dim]")


@catalog_app.command("show")
def show_dataset(
    reference: str = typer.Argument(..., help="A dataset id, my.<name>, my.<name>@<version> or analysis:<id>."),
    as_json: bool = typer.Option(False, "--json", help="Machine output."),
    server: Optional[str] = _server_opt(),
    token: Optional[str] = _token_opt(),
):
    """Show a dataset: every version with its columns, the latest schema change, the analyses that used it."""
    backend, _where = _backend(server, token)
    try:
        dataset_id = reference
        if reference.startswith(("my.", "analysis:")):
            dataset_id = backend.resolve(reference)["dataset_id"]
        detail = backend.get(dataset_id)
    except Exception as exc:  # noqa: BLE001
        raise _die(f"I couldn't show that dataset: {exc}")
    if detail is None:
        raise _die(f"No dataset of yours is called {reference!r}.", fix="avaloka catalog list")
    if as_json:
        console.print_json(json.dumps(detail, default=str))
        return
    if detail.get("tombstone"):
        console.print(f"[dim]{detail['dataset_id']} was forgotten on {detail.get('deleted_at', '')[:10]}.[/dim]")
        return
    console.print(f"[bold]{detail.get('display_name') or detail['dataset_id']}[/bold]  "
                  f"({detail['kind']}, use as [cyan]{detail.get('name') or detail['dataset_id']}[/cyan])")
    console.print(f"[dim]{detail.get('summary', '')}[/dim]")
    table = Table(title="Versions")
    for col in ("Version", "Captured", "Rows", "Columns", "Used by"):
        table.add_column(col)
    for v in detail.get("versions", []):
        cols = ", ".join(f"{c[0]}:{c[1]}" for c in v.get("columns", []))
        rows = v.get("row_count")
        table.add_row(str(v.get("version_no")), (v.get("captured_at") or "")[:19],
                      f"{rows:,}" if isinstance(rows, int) else "—", cols, ", ".join(v.get("used_by", [])) or "—")
    console.print(table)
    change = detail.get("schema_change")
    if change and any(change.values()):
        console.print(f"[dim]latest schema change: added {change['added'] or '—'}, removed {change['removed'] or '—'}, "
                      f"retyped {change['retyped'] or '—'}[/dim]")
    if detail.get("terms"):
        console.print(f"[dim]terms: {', '.join(detail['terms'])}[/dim]")


@catalog_app.command("use")
def use_dataset(
    reference: str = typer.Argument(..., help="my.<name>, my.<name>@<version> or analysis:<id>."),
    as_json: bool = typer.Option(False, "--json", help="Machine output."),
    server: Optional[str] = _server_opt(),
    token: Optional[str] = _token_opt(),
):
    """Resolve a reference to one (dataset, version) and print the command that runs on it."""
    backend, _where = _backend(server, token)
    try:
        binding = backend.resolve(reference)
    except Exception as exc:  # noqa: BLE001
        raise _die(f"I couldn't resolve {reference!r}: {exc}", fix="avaloka catalog list")
    if as_json:
        console.print_json(json.dumps(binding, default=str))
        return
    pinned = f", pinned by analysis {binding['pinned_by']}" if binding.get("pinned_by") else ""
    cols = ", ".join(c[0] for c in binding.get("columns", []))
    console.print(f"[bold]{binding.get('display_name') or binding['dataset_id']}[/bold] version "
                  f"{binding['version_no']}{pinned}: {cols or 'columns unknown'}")
    path = binding.get("path")
    if path:
        console.print(f"[dim]file: {path}[/dim]")
        console.print(f"Run: [cyan]avaloka analyze {reference} --goal \"...\"[/cyan]")
    else:
        console.print("[dim]This catalog holds the schema and history; the data itself is read where it lives "
                      "(upload it or pass its path to analyze).[/dim]")


@catalog_app.command("forget")
def forget_dataset(
    dataset_id: str = typer.Argument(..., help="The dataset id (from `avaloka catalog list`)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask."),
    as_json: bool = typer.Option(False, "--json", help="Machine output."),
    server: Optional[str] = _server_opt(),
    token: Optional[str] = _token_opt(),
):
    """Delete a dataset from your catalog: its versions, terms and lineage. Leaves a nameless tombstone."""
    backend, where = _backend(server, token)
    if not yes and not typer.confirm(f"Forget {dataset_id} from your {where}? The source data is untouched."):
        raise typer.Exit(code=0)
    try:
        report = backend.forget(dataset_id)
    except Exception as exc:  # noqa: BLE001
        raise _die(f"I couldn't forget {dataset_id!r}: {exc}")
    if as_json:
        console.print_json(json.dumps(report, default=str))
        return
    if report.get("already_deleted"):
        console.print(f"[dim]{dataset_id} was already forgotten.[/dim]")
        return
    removed = report.get("removed", {})
    console.print(f"Forgot {dataset_id}: {removed.get('versions', 0)} version(s), {removed.get('terms', 0)} term(s). "
                  f"The source data is untouched.")
    if report.get("derived_kept"):
        console.print(f"[dim]Kept its derived datasets, which you may forget separately: "
                      f"{', '.join(report['derived_kept'])}[/dim]")


__all__ = ["catalog_app"]
