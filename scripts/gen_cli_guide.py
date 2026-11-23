#!/usr/bin/env python3
"""Generate docs/cli-user-guide.html from the CLI's own help.

The guide is a single self-contained HTML file (no build step, renders at
phone width) that documents every ``avaloka`` command and option. It is not
written by hand: the command tree, the option names, their defaults and their
help text are read from the Typer app itself, so a flag added to
``avaloka/cli.py`` appears here on the next run, and ``tests/avaloka/
test_cli_guide.py`` fails when the committed guide and the CLI disagree.

    PYTHONPATH=. python scripts/gen_cli_guide.py            # rewrite the guide
    PYTHONPATH=. python scripts/gen_cli_guide.py --stdout   # print it instead

The worked examples are the one hand-written part. Every ``--option`` they
mention is checked against the command they are filed under, so an example
cannot name a flag the CLI does not have. The ``--debug`` transcript is not
pasted in: the script runs a real two-clone analysis on
``avaloka/examples/churn.csv`` (no model, no network — the CLI's own fireflies)
and narrates its trace with :mod:`avaloka.swarm_narration`, so the transcript
in the guide is what the CLI prints. The version comes from
:mod:`avaloka.version`, never typed here.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "cli-user-guide.html"

_FLAG = re.compile(r"(?<![\w-])(--?[a-zA-Z][\w-]*)")


# --------------------------------------------------------------------------- #
# The command tree, from Typer
# --------------------------------------------------------------------------- #

#: Click's own names for the parameter types the CLI uses. Typer 0.12-0.19 hands
#: us click parameters, whose ``type.name`` is one of these already; typer 0.20+
#: keeps the Python annotation and builds the click type at call time, so the
#: name is derived from the annotation here, to the same vocabulary. Either way
#: the page reads "text", "integer", "float", "boolean", "path".
_CLICK_TYPE_NAMES: dict[Any, str] = {str: "text", int: "integer", float: "float", bool: "boolean"}
#: typer._click (typer >= 0.20) names the same types after Python; map them back.
_VENDORED_CLICK_NAMES: dict[str, str] = {"str": "text", "int": "integer", "bool": "boolean", "float": "float"}


def _unwrap_optional(annotation: Any) -> Any:
    """``Optional[X]`` / ``X | None`` -> ``X``; anything else unchanged."""
    import types
    import typing

    origin = typing.get_origin(annotation)
    if origin is typing.Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _type_name(p: Any) -> str:
    """The parameter's type as click names it, on every typer/click pairing.

    ``p.type`` is click's public attribute and is used when present. Newer
    typer parameters carry the annotation instead (``get_runtime_param()``);
    its bare type is mapped to click's vocabulary, and anything unmapped falls
    back to typer's own ``bare_type()`` so a new type is visible, not hidden.
    """
    import enum
    import pathlib

    click_type = getattr(p, "type", None)
    if click_type is not None:
        name = getattr(click_type, "name", str(click_type))
        # typer 0.20+ vendors its own click (typer._click), whose parameter
        # types are named after the Python type ("str", "int"); public click
        # says "text", "integer". One vocabulary on the page, whichever is loaded.
        return _VENDORED_CLICK_NAMES.get(name, name)
    annotation = None
    getter = getattr(p, "get_runtime_param", None)
    if callable(getter):
        try:
            annotation = getattr(getter(), "annotation", None)
        except Exception:  # noqa: BLE001  (a parameter typer cannot resolve is still listed)
            annotation = None
    bare = _unwrap_optional(annotation) if annotation is not None else None
    if bare in _CLICK_TYPE_NAMES:
        return _CLICK_TYPE_NAMES[bare]
    if isinstance(bare, type):
        if issubclass(bare, pathlib.PurePath):
            return "path"
        if issubclass(bare, enum.Enum):
            return "choice"
    bare_type = getattr(p, "bare_type", None)
    if callable(bare_type):
        try:
            return str(bare_type())
        except Exception:  # noqa: BLE001
            pass
    return str(bare) if bare is not None else "text"


def command_tree() -> list[dict[str, Any]]:
    """Every command as ``{path, help, arguments, options}``, depth first.

    A group that runs without a subcommand (``benchmark``) is listed as a
    command of its own and its subcommands follow it (``benchmark serve``).
    """
    import typer

    from avaloka.cli import app

    try:
        import click
    except ImportError:  # typer >= 0.20 vendors click and no longer depends on the package
        from typer._click import core as click  # type: ignore[no-redef]

    root = typer.main.get_command(app)
    out: list[dict[str, Any]] = []

    def param(p: click.Parameter) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": p.name,
            "kind": "argument" if p.param_type_name == "argument" else "option",
            "opts": list(p.opts) + list(p.secondary_opts),
            "type": _type_name(p),
            "required": bool(p.required),
            "default": None if p.default in (None, ...) or callable(p.default) else p.default,
            "help": (getattr(p, "help", None) or "").strip(),
            "is_flag": bool(getattr(p, "is_flag", False)),
        }
        return d

    def walk(cmd: click.Command, path: list[str]) -> None:
        # Duck-typed: Typer's group class is not a subclass of this click's
        # ``Group`` in every version pairing, but every group lists commands.
        is_group = hasattr(cmd, "list_commands") and hasattr(cmd, "get_command")
        params = [param(p) for p in cmd.params if "--help" not in getattr(p, "opts", ())]
        out.append({
            "path": " ".join(path) or "avaloka",
            "help": (cmd.help or "").strip(),
            "short": (cmd.get_short_help_str(limit=200) if cmd.help else "").strip(),
            "arguments": [p for p in params if p["kind"] == "argument"],
            "options": [p for p in params if p["kind"] == "option"],
            "is_group": is_group,
        })
        if is_group:
            for name in cmd.list_commands(click.Context(cmd)):
                walk(cmd.get_command(click.Context(cmd), name), path + [name])

    walk(root, [])
    return out


def option_names(tree: list[dict[str, Any]]) -> dict[str, set[str]]:
    """``command path -> every --flag and -f it accepts``."""
    return {c["path"]: {o for p in c["options"] for o in p["opts"]} for c in tree}


# --------------------------------------------------------------------------- #
# The hand-written part: worked examples, checked against the tree
# --------------------------------------------------------------------------- #

EXAMPLES: dict[str, list[tuple[str, str, str]]] = {
    "analyze": [
        ("A dataset and a question, in; a defensible bundle out",
         'avaloka analyze app/sample_data/salaries.csv --goal "what drives salary here?"',
         "Sizes the file, profiles it, runs the goal-specific calculation, validates the result and "
         "leaves a bundle (executive_report.html, README.md, analysis.py) you can rerun."),
        ("Cap the spend and choose where the bundle goes",
         'avaloka analyze sales.parquet --goal "why did revenue drop in Q3?" --budget 2 --output ./q3',
         "The budget is a hard ceiling in USD. Local execution is free; the ceiling matters for hosted models."),
    ],
    "train": [
        ("Predict a column, and find out whether to trust the model",
         "avaloka train avaloka/examples/churn.csv --target churned --metric roc_auc",
         "Trains a baseline and a candidate panel, selects by the metric, and reports the safe deployment level."),
        ("Make it deployable",
         "avaloka train avaloka/examples/churn.csv --target churned --deployable",
         "Adds the ML engineer: a container, an API server, schemas and a Kubernetes manifest in the bundle."),
    ],
    "chat": [
        ("Work it out conversationally",
         "avaloka chat avaloka/examples/churn.csv",
         "An interactive session that starts from the dataset; each turn is a small mission."),
    ],
    "batch": [
        ("The full analysis as a Ray swarm",
         'avaloka batch big.parquet --goal "profile the whole book of business"',
         "Where analyze samples a large file, batch runs the exact answer over every row on Ray."),
    ],
    "coordinate": [
        ("Two Avaloka clones over one dataset",
         'avaloka coordinate app/sample_data/healthcare_dataset.csv --goal "what drives billing amount?" --swarm 2',
         "Avaloka reads the file once, cuts it into two shards, runs a full sub-Avaloka on each and merges "
         "one answer. Two is the open-source and Free cap; ask for more and one line tells you the cap."),
        ("Watch the team work",
         'avaloka coordinate app/sample_data/healthcare_dataset.csv --goal "what drives billing amount?" --swarm 2 --debug',
         "Every sentence is produced from a trace record as it is written, so what is printed is what ran. "
         "See the transcript below. -v is the lighter level: the team's story without every step."),
        ("A fleet trained on a database table",
         "avaloka coordinate 'sqlite:///./app.db#customers' --goal \"who churns\" --kind train --target churned",
         "The source may be any URI from the Sources section; it is resolved to a local file first."),
    ],
    "infer": [
        ("A REST API on Kubernetes from the trained model",
         "avaloka infer ./avaloka-train/model --target kubernetes --apply",
         "MLflow model → Docker image → Kubernetes Deployment and Service."),
    ],
    "deploy": [
        ("Check a model against the deployment gates",
         "avaloka deploy ./avaloka-train/model --target kubernetes --approve",
         "Prints the plan and the gates it must pass; nothing is promoted without --approve."),
    ],
    "benchmark": [
        ("The regression suite, against this CLI",
         "avaloka benchmark --json scorecard.json --html scorecard.html",
         "Runs the data-engineering and data-science tasks and writes a scorecard."),
    ],
    "benchmark serve": [
        ("Read the scorecard in a browser",
         "avaloka benchmark serve --scorecard scorecard.json --port 8765",
         "Binds loopback only and serves nothing but the scorecard."),
    ],
    "catalog": [
        ("Your datasets, with their schema history",
         "avaloka catalog list",
         "Every dataset your analyses have read, newest first, each with the my.<name> to use it by. "
         "Local by default (~/.avaloka/catalog.db); --server reaches a hosted catalog."),
    ],
    "catalog list": [
        ("Find a dataset you used before",
         "avaloka catalog list orders --kind file",
         "Matches names, tags, terms and the ids of analyses that used a dataset. --json for machines."),
    ],
    "catalog show": [
        ("Every version, its columns, and who used it",
         "avaloka catalog show my.orders_2026",
         "The versions table shows what changed; a version is written only when the schema or the file changed."),
    ],
    "catalog use": [
        ("Pin the version a past analysis read",
         "avaloka catalog use analysis:an_7f3c",
         "Prints the (dataset, version) and the analyze command that runs on it. my.<name>@2 picks a version by number."),
    ],
    "catalog forget": [
        ("Delete a dataset from your catalog",
         "avaloka catalog forget ds:file:9a1b --yes",
         "Removes its versions, terms and lineage and leaves a nameless tombstone; the source data is untouched."),
    ],
    "version": [
        ("Which Avaloka is this?", "avaloka version", "Also: avaloka --version."),
    ],
}

#: Format rows for the guide. The suffixes are checked against the loader's
#: own ``SUPPORTED_SUFFIXES`` when that constant exists (feat/2.0-upload-
#: formats-statistics adds it), so a format the loader gains or loses is a
#: generator error here, not a stale page.
FORMATS = [
    ("CSV, TSV, TXT", ".csv .tsv .txt", "delimiter sniffed for .csv/.txt; header found below a title block"),
    ("Parquet", ".parquet .pq", "read with pyarrow"),
    ("JSON", ".json", "records, or nested objects flattened with a separator"),
    ("JSON Lines", ".jsonl .ndjson", "one record per line"),
    ("XML", ".xml", "repeated elements become rows"),
    ("ORC", ".orc", "read with pyarrow"),
    ("Avro", ".avro", "needs the avro extra"),
    ("Excel", ".xlsx .xlsm .xls", "first sheet, header found below a title block"),
]

#: Directories of part files (one Spark/Daft output, many files) are read as
#: one dataset when every part has the same suffix from this set; the loader's
#: ``_PART_SUFFIXES`` is used when present.
PART_SUFFIXES_FALLBACK = ".parquet .pq .csv .tsv .json .jsonl .ndjson .xml .orc .avro"


def check_formats() -> str:
    """The suffix list, verified against the loader when it publishes one."""
    listed = {suffix for _, suffixes, _ in FORMATS for suffix in suffixes.split()}
    try:
        from avaloka.io import loader
    except ImportError:  # pragma: no cover - the CLI always has its loader
        return PART_SUFFIXES_FALLBACK
    supported = getattr(loader, "SUPPORTED_SUFFIXES", None)
    if supported is not None and set(supported) != listed:
        sys.exit(f"gen_cli_guide: FORMATS lists {sorted(listed)} but the loader supports "
                 f"{sorted(supported)}; update FORMATS")
    parts = getattr(loader, "_PART_SUFFIXES", None)
    return " ".join(sorted(parts)) if parts else PART_SUFFIXES_FALLBACK

SOURCES = [
    ("/path/data.csv, file:///path/data.csv", "a local file"),
    ("sqlite:///abs/app.db#table", "a SQLite table, or #query=SELECT …"),
    ("postgresql://user:pw@host/db#table", "any SQLAlchemy-supported database"),
    ("s3://bucket/key.parquet", "object storage via fsspec (needs s3fs)"),
    ("gs://bucket/key.csv", "Google Cloud Storage (needs gcsfs)"),
    ("az://container/key.csv", "Azure Blob (needs adlfs)"),
]


def check_examples(tree: list[dict[str, Any]]) -> None:
    names = option_names(tree)
    for path, examples in EXAMPLES.items():
        if path not in names:
            sys.exit(f"gen_cli_guide: example filed under unknown command {path!r}")
        for _, shell, _ in examples:
            body = shell.split(path, 1)[1] if path in shell else shell
            for flag in _FLAG.findall(body):
                if flag not in names[path] and flag != "--version":
                    sys.exit(f"gen_cli_guide: example for {path!r} uses {flag!r}, which it does not accept")
    missing = [c["path"] for c in tree if c["path"] != "avaloka" and c["path"] not in EXAMPLES]
    if missing:
        sys.exit(f"gen_cli_guide: no worked example for {missing}")


# --------------------------------------------------------------------------- #
# The transcript: a real run, narrated from its records
# --------------------------------------------------------------------------- #

def debug_transcript() -> tuple[list[str], str]:
    """Lines ``coordinate --swarm 2 --debug`` prints for churn.csv, and the cap sentence."""
    from avaloka.coordinator import MainAvaloka
    from avaloka.swarm_limits import grant, limits_for
    from avaloka.swarm_narration import DEBUG, SwarmNarrative
    from avaloka.swarm_trace import SwarmTraceRecorder

    oss = limits_for("oss", has_commercial_build=False)
    story = SwarmNarrative(DEBUG)
    lines: list[str] = []
    recorder = SwarmTraceRecorder(sink=lambda rec: lines.extend("  › " + l for l in story.feed(rec)))
    with tempfile.TemporaryDirectory(prefix="avaloka-guide-") as tmp:
        MainAvaloka(swarm_limits=oss).coordinate(
            str(ROOT / "avaloka" / "examples" / "churn.csv"), goal="what drives churn",
            output_dir=Path(tmp) / "fleet", workers=2, recorder=recorder)
    capped = grant(5, None, oss).message
    return lines, capped


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

CSS = """
:root{--fg:#1a1a1a;--muted:#5a6270;--line:#e3e6ea;--accent:#2d5b8e;--bg:#fff;--card:#fafbfc;--ok:#2f7d4f}
@media (prefers-color-scheme:dark){:root{--fg:#e6e8ea;--muted:#9aa4b2;--line:#2a2f37;--accent:#7fb0e8;--bg:#14171a;--card:#1b1f24;--ok:#7fd49a}}
*{box-sizing:border-box}
body{margin:0;padding:0 1rem 4rem;background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:52rem;margin:0 auto}
header{padding:2.5rem 0 1.25rem;border-bottom:1px solid var(--line)}
h1{margin:0 0 .3rem;font-size:1.9rem;letter-spacing:-.02em}
h2{margin:2.5rem 0 .5rem;font-size:1.3rem;letter-spacing:-.01em}
h3{margin:1.5rem 0 .4rem;font-size:1.05rem}
h2 code,h3 code{font-size:.95em}
.tag,.lead{color:var(--muted);margin:.15rem 0 .5rem}
.meta{color:var(--muted);font-size:.85rem}
nav.toc{margin:1rem 0 0;padding:.75rem 1rem;border:1px solid var(--line);border-radius:8px;background:var(--card)}
nav.toc ol{margin:.25rem 0 0;padding-left:1.2rem;columns:2;column-gap:2rem}
@media (max-width:600px){nav.toc ol{columns:1}}
nav.toc a{color:var(--accent);text-decoration:none}
a{color:var(--accent)}
code{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:.05em .3em;font-size:.88em;word-break:break-word}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.85rem 1rem;overflow-x:auto;font-size:.84em;line-height:1.5;white-space:pre-wrap;word-break:break-word}
pre code{background:none;border:0;padding:0;font-size:1em}
.table{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{text-align:left;vertical-align:top;padding:.5rem .7rem;border-bottom:1px solid var(--line)}
th{background:var(--card);color:var(--muted);font-weight:600;font-size:.8rem;text-transform:uppercase;letter-spacing:.03em}
tr:last-child td{border-bottom:0}
.req{color:var(--ok);font-weight:600;font-size:.8rem}
.example{margin:1rem 0;padding:.9rem 1rem;border:1px solid var(--line);border-radius:8px;background:var(--card)}
.example b{display:block;margin-bottom:.35rem}
.example pre{margin:.4rem 0;background:var(--bg)}
.example p{margin:.3rem 0 0;color:var(--muted);font-size:.92rem}
.levels{display:grid;gap:.6rem;grid-template-columns:repeat(auto-fit,minmax(14rem,1fr))}
.levels div{padding:.75rem .9rem;border:1px solid var(--line);border-radius:8px;background:var(--card)}
.levels b{display:block;color:var(--accent)}
.cap{border-left:4px solid var(--accent);padding:.5rem .9rem;background:var(--card);border-radius:0 8px 8px 0;margin:.75rem 0}
footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);color:var(--muted);font-size:.85rem}
"""


def esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def anchor(path: str) -> str:
    return "cmd-" + path.replace(" ", "-")


def render_params(cmd: dict[str, Any]) -> str:
    parts = []
    if cmd["arguments"]:
        rows = "".join(
            f"<tr><td><code>{esc(a['name'])}</code></td><td>{esc(a['type'])}</td>"
            f"<td>{'<span class=req>required</span>' if a['required'] else ''}</td>"
            f"<td>{esc(a['help'])}</td></tr>" for a in cmd["arguments"])
        parts.append(f"<div class=table><table><thead><tr><th>Argument</th><th>Type</th><th></th><th>Meaning</th></tr></thead>"
                     f"<tbody>{rows}</tbody></table></div>")
    if cmd["options"]:
        rows = []
        for o in cmd["options"]:
            flags = " ".join(f"<code>{esc(f)}</code>" for f in o["opts"])
            typ = "" if o["is_flag"] else esc(o["type"])
            default = ("" if o["default"] is None or o["is_flag"] and o["default"] is False
                       else f"<br><span class=meta>default: <code>{esc(o['default'])}</code></span>")
            req = "<span class=req>required</span>" if o["required"] else ""
            rows.append(f"<tr><td>{flags}</td><td>{typ}{default}</td><td>{req}</td><td>{esc(o['help'])}</td></tr>")
        parts.append(f"<div class=table><table><thead><tr><th>Option</th><th>Type</th><th></th><th>Meaning</th></tr></thead>"
                     f"<tbody>{''.join(rows)}</tbody></table></div>")
    return "\n".join(parts)


def render_examples(path: str) -> str:
    out = []
    for title, shell, note in EXAMPLES.get(path, []):
        out.append(f"<div class=example><b>{esc(title)}</b><pre><code>$ {esc(shell)}</code></pre>"
                   f"<p>{esc(note)}</p></div>")
    return "\n".join(out)


def render(tree: list[dict[str, Any]], transcript: list[str], capped: str, version: str,
           part_suffixes: str = PART_SUFFIXES_FALLBACK) -> str:
    from avaloka.swarm_limits import limits_for

    commands = [c for c in tree if c["path"] != "avaloka"]
    root = next(c for c in tree if c["path"] == "avaloka")
    oss = limits_for("oss", has_commercial_build=False)

    toc = "".join(f"<li><a href=#{anchor(c['path'])}><code>avaloka {esc(c['path'])}</code></a></li>" for c in commands)
    toc += ("<li><a href=#narration>Debug narration</a></li><li><a href=#swarm-size>Swarm size and the cap</a></li>"
            "<li><a href=#formats>Formats and sources</a></li><li><a href=#llm>Model providers (--llm)</a></li>"
            "<li><a href=#api>Starting the API</a></li>")

    sections = []
    for c in commands:
        first, _, rest = c["help"].partition("\n\n")
        body = f"<p>{esc(first)}</p>" + "".join(f"<p class=lead>{esc(p)}</p>" for p in rest.split("\n\n") if p.strip())
        sections.append(
            f"<section id={anchor(c['path'])}><h2><code>avaloka {esc(c['path'])}</code></h2>{body}"
            f"{render_params(c)}{render_examples(c['path'])}</section>")

    root_opts = render_params(root)
    transcript_html = esc("\n".join(transcript))
    from avaloka.llm.client import DETERMINISTIC_NOTE as deterministic_note
    from avaloka.llm.providers import AUTO_ORDER, LOCAL_SIGNALS, SPECS
    llm_providers = [(name, (" or ".join(LOCAL_SIGNALS) if name == "local" else SPECS[name].key_env),
                      SPECS[name].default_model) for name in AUTO_ORDER]
    levels = f"""
<div class=levels>
<div><b>default</b>Results only: the converged profile, the fleet table, the verdict, the economics and the bundle path.</div>
<div><b>-v, --verbose</b>The team's story as it happens: the brief, the shards, each clone leaving and returning, the merge and the close.</div>
<div><b>--debug</b>Everything <code>-v</code> says, plus every specialist step inside every clone with that step's own one-line result. Local output only; it may show your column names.</div>
</div>"""

    return f"""<!doctype html>
<html lang=en>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="avaloka-version" content="{esc(version)}">
<meta name="generator" content="scripts/gen_cli_guide.py">
<title>Avaloka CLI Guide</title>
<style>{CSS}</style>
<div class=wrap>
<header>
<h1>The <code>avaloka</code> command</h1>
<p class=tag>An AI data scientist for your terminal: point it at a file, ask a question, get a bundle you can rerun and hand to someone else.</p>
<p class=meta>Avaloka {esc(version)} · generated from <code>avaloka --help</code> on {date.today().isoformat()} by <code>scripts/gen_cli_guide.py</code>. Do not edit by hand; rerun the script.</p>
<nav class=toc><b>Contents</b><ol>{toc}</ol></nav>
</header>

<section id=start>
<h2>Start here</h2>
<p>Python 3.11 or 3.12. From a checkout: <code>pip install -e .</code>; then:</p>
<pre><code>$ avaloka analyze app/sample_data/salaries.csv --goal "what drives salary here?"
$ avaloka train   avaloka/examples/churn.csv --target churned
$ avaloka chat    avaloka/examples/churn.csv</code></pre>
<p>Everything runs locally by default and your data stays on your machine. Every run writes a bundle: an executive report, a README of findings and limitations, and the code that reproduces the run.</p>
<h3>Global options</h3>
{root_opts}
</section>

{''.join(sections)}

<section id=narration>
<h2>Debug narration: how the team worked</h2>
<p><code>avaloka coordinate</code> is a team: one coordinating Avaloka reads the data, cuts it into shards and hands each to a clone; each clone runs her own specialists (a data scout, a sampling specialist, a data engineer, an analysis planner, a validator, a reporter) on her shard; the coordinator merges what they bring back. Three levels of narration say how much of that you see.</p>
{levels}
<p>Every sentence is produced from one record of the swarm trace (<code>swarm_trace.jsonl</code>) at the moment it is written, by <code>avaloka/swarm_narration.py</code>. There is no script to fall back on: a line that is not in the records is not printed, and the file and the screen cannot disagree.</p>
<h3>Transcript</h3>
<p class=lead>A real run of <code>avaloka coordinate avaloka/examples/churn.csv --goal "what drives churn" --swarm 2 --debug</code>, produced while generating this page (timings vary between runs; the sentences do not).</p>
<pre><code>{transcript_html}</code></pre>
</section>

<section id=swarm-size>
<h2>Swarm size and the cap</h2>
<p><code>--swarm N</code> (short <code>-s</code>; <code>--workers</code>/<code>-w</code> is the same option under its old name) is the number of clones. Unset, it is 4 or your edition's cap, whichever is lower. The cap lives in <code>avaloka/swarm_limits.py</code>: {esc(oss.summary())} Professional allows 16; Enterprise has no limit. Ask for more than the cap and the run proceeds at the cap, saying so once:</p>
<div class=cap><code>$ AVALOKA_EDITION=oss avaloka coordinate data.csv --goal "…" --swarm 5</code><br>{esc(capped)}</div>
<p>With <code>--debug</code> the narration also states the effective team size and the cap it sits under. Nothing else is said about editions.</p>
</section>

<section id=formats>
<h2>Formats and sources</h2>
<p>Any command that takes a dataset accepts these file formats (<code>avaloka/io/loader.py</code>):</p>
<div class=table><table><thead><tr><th>Format</th><th>Suffixes</th><th>Notes</th></tr></thead><tbody>
{''.join(f'<tr><td>{esc(n)}</td><td><code>{esc(s)}</code></td><td>{esc(note)}</td></tr>' for n, s, note in FORMATS)}
</tbody></table></div>
<p>A <em>directory</em> of part files — one Spark or Daft output written as many files — is read as one dataset when every part has the same suffix, one of <code>{esc(part_suffixes)}</code>; <code>_SUCCESS</code> and hidden files are ignored.</p>
<p>A dataset may also be named by URI; <code>avaloka/io/sources.py</code> materialises it to a local file first and records the provenance:</p>
<div class=table><table><thead><tr><th>URI</th><th>Source</th></tr></thead><tbody>
{''.join(f'<tr><td><code>{esc(u)}</code></td><td>{esc(d)}</td></tr>' for u, d in SOURCES)}
</tbody></table></div>
</section>

<section id=llm>
<h2>Model providers (<code>--llm</code>)</h2>
<p>Without <code>--llm</code> nothing leaves your machine. With it, Avaloka's lines are reworded by a model; the numbers still come from the statistics and a rewording that changes one is discarded. Choose the provider with <code>--llm-provider</code> or <code>AVALOKA_LLM_PROVIDER</code> (the flag implies <code>--llm</code>) and the model with <code>--llm-model</code> or <code>AVALOKA_LLM_MODEL</code>. With neither, the first configured provider in this order is used (<code>avaloka/llm/providers.py</code>):</p>
<div class=table><table><thead><tr><th>Provider</th><th>Configured by</th><th>Default model</th></tr></thead><tbody>
{''.join(f'<tr><td><code>{esc(name)}</code></td><td><code>{esc(how)}</code></td><td><code>{esc(model)}</code></td></tr>' for name, how, model in llm_providers)}
</tbody></table></div>
<p>The first four speak the OpenAI-compatible chat API; Anthropic goes through its own SDK (<code>pip install 'avaloka[llm]'</code>). A provider that is named but unusable, or that fails mid-run, prints one line naming the provider and the fix, and says the rest of the run uses Avaloka's deterministic wording:</p>
<div class=cap><code>LLM error (groq): Groq rejected the API key (HTTP 401). Fix: check GROQ_API_KEY, or choose another provider with --llm-provider. {esc(deterministic_note)}</code></div>
</section>

<section id=api>
<h2>Starting the API</h2>
<p>The CLI is standalone and needs no server. The platform's HTTP API (upload → chat → assets → models, the agent trace, the swarm missions) is a separate process from the same repository:</p>
<pre><code>$ pip install -r requirements.txt
$ uvicorn app.api.server:app --port 9000 --loop asyncio   # the API
$ docker compose up -d                          # optional: the Milvus stack</code></pre>
<p>See <a href="api.md">docs/api.md</a> for the endpoints and <a href="INSTALL.md">docs/INSTALL.md</a> for the full install, including Kubernetes. <code>avaloka benchmark serve</code> above is the only server the CLI itself starts, and it binds loopback only.</p>
</section>

<footer>Avaloka {esc(version)} · <a href="index.html">Documentation index</a> · <a href="../README_CLI.md">README_CLI.md</a> (the narrative guide) · generated by <code>scripts/gen_cli_guide.py</code></footer>
</div>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stdout", action="store_true", help="print the guide instead of writing it")
    args = ap.parse_args()

    from avaloka.version import __version__

    tree = command_tree()
    check_examples(tree)
    part_suffixes = check_formats()
    transcript, capped = debug_transcript()
    page = render(tree, transcript, capped, __version__, part_suffixes)
    if args.stdout:
        sys.stdout.write(page)
        return
    OUT.write_text(page, encoding="utf-8")
    n_opts = sum(len(c["options"]) for c in tree)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(tree) - 1} commands, {n_opts} options, "
          f"{len(transcript)} transcript lines, version {__version__}")


if __name__ == "__main__":
    main()
