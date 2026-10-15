"""Benchmark datasets for the declared upload formats other than CSV.

The API declares nine upload formats (``SUPPORTED_UPLOAD_EXTS`` in
``app/api/server.py``). Every benchmark dataset was a CSV: ``DatasetSpec``
carried a builder that wrote one, and the three source schemes (file, sqlite,
object storage) re-read that same CSV. So eight of the nine declared formats
had no benchmark coverage at all, and the suite scored 1.00 while doing it.

That is not a gap in a nice-to-have. A format the product offers in its upload
dialog and never reads in a test is a format whose failure reaches a user
first. Two of them fail in the worst available way: the CLI loader does not
refuse an XML or an ORC file, it *sniffs it as a delimited text file* and
returns a one-column frame named after the file's magic bytes. Nothing
downstream can tell that from a real table, which is how a profile of a
binary file becomes an insight about a column called ``ORC``.

Each builder here writes one file in one declared format and declares what is
in it, so the comprehension scorers can grade the reader's answer against the
truth rather than against itself.

Writers that are not among the CLI's eight runtime dependencies (``fastavro``
for Avro, ``lxml`` for XML) are imported inside the builder that needs them, so
importing this module never costs the CLI a dependency. They are declared in
the ``benchmark`` extra.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from avaloka.benchmark.spec import GroundTruth

#: One table, written out in every format, so a format difference is the only
#: variable. Deterministic by construction -- a fixed seed and rounded values,
#: so two runs produce byte-identical files and a diff in the scorecard means a
#: diff in the reader.
_N_ROWS = 240
_SEED = 31


def _canonical_frame() -> pd.DataFrame:
    """The one table every format in this module carries.

    Deliberately boring: a text key, two numerics with different scales, a
    date and a low-cardinality category. Nothing here is adversarial -- the
    shape is the *easiest* table a reader could be handed, so a failure is
    unambiguously about the container format rather than about the layout.
    """
    rng = np.random.default_rng(_SEED)
    return pd.DataFrame({
        "region": rng.choice(["north", "south", "east", "west"], _N_ROWS),
        "units_sold": rng.integers(10, 900, _N_ROWS),
        "revenue_usd": rng.normal(5_000.0, 900.0, _N_ROWS).round(2),
        "reported_on": pd.date_range("2024-01-01", periods=_N_ROWS, freq="D").astype(str),
        "channel": rng.choice(["online", "retail"], _N_ROWS),
    })


#: The truth about ``_canonical_frame`` as written to a *self-describing*
#: container (parquet, avro, orc): the type survives the round trip, so the
#: date column is a date on the way back out.
def _ground_truth(*, dates_are_text: bool) -> GroundTruth:
    return GroundTruth(
        columns=["region", "units_sold", "revenue_usd", "reported_on", "channel"],
        n_rows=_N_ROWS,
        # A header *row* is a spreadsheet notion. A CSV has a header line but
        # the loader does not report which one it used, and parquet/avro/orc
        # carry names in a schema rather than in a row, so there is nothing
        # here for a reader to get right or wrong. Left None so the header-row
        # scorer stays out of these tasks instead of failing them for not
        # answering a question that does not apply.
        header_row=None,
        numeric_columns=["units_sold", "revenue_usd"],
        date_columns=[] if dates_are_text else ["reported_on"],
        text_columns=["region", "channel"] + (["reported_on"] if dates_are_text else []),
        shape="the same five-column table in every declared format",
    )


# --------------------------------------------------------------------------
# One builder per declared format. Each writes `dest` and returns it.
# --------------------------------------------------------------------------
def build_tsv(dest: Path) -> Path:
    """Tab-separated. The loader special-cases ``.tsv``; the app's CSV
    connector does not, and returns one column named ``region\\tunits_sold\\t...``.
    """
    _canonical_frame().to_csv(dest, sep="\t", index=False)
    return dest


def build_json(dest: Path) -> Path:
    """Records-oriented JSON, the shape an API export actually has.

    ``avaloka.io.load_dataset`` has no JSON branch. The suffix falls through to
    the delimiter sniffer, which reads the document as one wide CSV row and
    then ``_reject_unusable`` raises "has column headers but no rows".
    """
    _canonical_frame().to_json(dest, orient="records", date_format="iso")
    return dest


def build_xml(dest: Path) -> Path:
    """An XML document with one element per row.

    Needs ``lxml`` to write. The CLI loader has no XML branch, and -- worse
    than refusing -- the delimiter sniffer accepts it, so the frame comes back
    with a single column literally named
    ``<?xml version='1.0' encoding='utf-8'?>``.
    """
    try:
        import lxml  # noqa: F401
    except ImportError as exc:                      # pragma: no cover
        raise ImportError(
            "writing the XML benchmark dataset needs lxml: "
            "pip install 'avaloka[benchmark]'") from exc
    _canonical_frame().to_xml(dest, index=False, root_name="rows", row_name="row")
    return dest


def build_parquet(dest: Path) -> Path:
    """Parquet via pyarrow, which is already a runtime dependency."""
    _canonical_frame().to_parquet(dest, index=False)
    return dest


def build_orc(dest: Path) -> Path:
    """ORC via pyarrow.

    Declared in ``SUPPORTED_UPLOAD_EXTS`` and readable by nothing in the
    product: the app has no ORC connector (``file_handler/handler.py`` has no
    branch for it), and the CLI loader sniffs the binary container as CSV and
    returns a one-column frame named ``ORC`` -- the file's magic bytes.
    """
    _canonical_frame().to_orc(dest, index=False)
    return dest


def build_avro(dest: Path) -> Path:
    """An Avro container with an explicit schema.

    Needs ``fastavro`` to write, which is also what the app's Avro connector
    reads with. The CLI loader has no Avro branch.
    """
    try:
        import fastavro
    except ImportError as exc:                      # pragma: no cover
        raise ImportError(
            "writing the Avro benchmark dataset needs fastavro: "
            "pip install 'avaloka[benchmark]'") from exc
    frame = _canonical_frame()
    schema = fastavro.parse_schema({
        "type": "record", "name": "sales", "fields": [
            {"name": "region", "type": "string"},
            {"name": "units_sold", "type": "long"},
            {"name": "revenue_usd", "type": "double"},
            {"name": "reported_on", "type": "string"},
            {"name": "channel", "type": "string"},
        ],
    })
    records = frame.astype({"units_sold": "int64", "revenue_usd": "float64"}).to_dict("records")
    with open(dest, "wb") as handle:
        fastavro.writer(handle, schema, records)
    return dest


#: name -> (builder, declared upload extension, ground truth)
FORMAT_DATASETS: dict[str, tuple[object, str, GroundTruth]] = {
    "fmt_tsv":     (build_tsv,     "tsv",     _ground_truth(dates_are_text=True)),
    "fmt_json":    (build_json,    "json",    _ground_truth(dates_are_text=True)),
    "fmt_xml":     (build_xml,     "xml",     _ground_truth(dates_are_text=True)),
    "fmt_parquet": (build_parquet, "parquet", _ground_truth(dates_are_text=True)),
    "fmt_orc":     (build_orc,     "orc",     _ground_truth(dates_are_text=True)),
    "fmt_avro":    (build_avro,    "avro",    _ground_truth(dates_are_text=True)),
}

#: Formats declared by the API that this suite deliberately does **not** cover,
#: each with the reason, so the coverage check can tell a deliberate deferral
#: from an oversight. A deferral belongs in code; an absence is how eight
#: formats went uncovered without anyone noticing.
#:
#: ``xls`` is the legacy BIFF format, and nothing in the dependency set can
#: write one. pandas dropped ``xlwt`` (unmaintained since 2017), and
#: ``DataFrame.to_excel(path.xls)`` does not fail -- it silently writes a
#: *zip-based xlsx* under an ``.xls`` name (magic bytes ``PK\x03\x04``), which
#: ``xlrd`` then refuses with "Excel xlsx file; not supported". A task built on
#: such a file would re-test xlsx under a different extension and report a pass
#: for a format it never exercised, which is worse than the gap. Measured
#: 2026-10-10 on pandas 2.3.3.
DEFERRED_FORMATS: dict[str, str] = {
    "xls": (
        "no legacy BIFF writer is available without adding a dependency: pandas "
        "dropped xlwt, and to_excel('.xls') silently writes a zip-based xlsx that "
        "xlrd refuses, so a task built on it would re-test xlsx under another "
        "extension and report a pass for a format it never read"
    ),
}

# --------------------------------------------------------------------------
# The nesting rule, as data the benchmark can grade
# --------------------------------------------------------------------------
# JSON and XML are trees and a dataframe is a rectangle, so a reader has to
# decide what to do with nesting. The rule is stated in
# `avaloka/io/loader.py`; these datasets make each clause of it gradable, so
# the suite checks the rule rather than accepting whatever the reader does.
_NESTED_N = 60


def build_json_nested_scalars(dest: Path) -> Path:
    """Scalar nesting, which must flatten to dotted names.

    ``{"geo": {"country": ...}}`` becomes a column ``geo.country``. Lossless and
    derivable from the document, so a user can predict the name.
    """
    import json
    rng = np.random.default_rng(_SEED)
    records = [{
        "order_id": f"A{index:05d}",
        "geo": {"country": str(rng.choice(["GB", "DE", "FR"])),
                "city": str(rng.choice(["Bath", "Berlin", "Lyon"]))},
        "totals": {"net": round(float(rng.normal(400, 60)), 2),
                   "tax": round(float(rng.normal(80, 12)), 2)},
    } for index in range(_NESTED_N)]
    dest.write_text(json.dumps(records))
    return dest


def build_json_repeated_group(dest: Path) -> Path:
    """A one-to-many, which must be refused by name.

    Each record carries a list of objects. Exploding it multiplies the row
    count, keeping the first drops data, joining it to a string makes the
    numbers unusable -- and none of the three is visible downstream. So the
    reader must refuse and name ``lines``.
    """
    import json
    records = [{
        "invoice": f"INV{index:04d}",
        "customer": f"C{index % 7:03d}",
        "lines": [{"sku": f"S{item}", "qty": item + 1} for item in range(3)],
    } for index in range(_NESTED_N)]
    dest.write_text(json.dumps(records))
    return dest


def build_json_two_candidate_tables(dest: Path) -> Path:
    """Two top-level arrays, which must be refused naming both.

    "Which table is this file" has no single answer here, and taking the longer
    one is a coin toss presented as a fact.
    """
    import json
    dest.write_text(json.dumps({
        "customers": [{"id": i, "name": f"C{i}"} for i in range(10)],
        "orders": [{"id": i, "total": i * 10} for i in range(25)],
    }))
    return dest


def build_xml_repeated_child(dest: Path) -> Path:
    """Repeated child elements inside each record, which must be refused.

    The XML analogue of a JSON array of objects. ``pandas.read_xml`` keeps one
    and drops the rest without saying so.
    """
    rows = []
    for index in range(_NESTED_N):
        items = "".join(f"<item><sku>S{n}</sku><qty>{n + 1}</qty></item>"
                        for n in range(3))
        rows.append(f"<row><invoice>INV{index:04d}</invoice>{items}</row>")
    dest.write_text("<?xml version='1.0' encoding='utf-8'?>\n<rows>"
                    + "".join(rows) + "</rows>")
    return dest


#: name -> (builder, declared extension, ground truth) for the nesting rule.
#: Separate from FORMAT_DATASETS because these grade the *rule*, not whether a
#: container can be opened, and three of the four must be refused rather than
#: read -- a different kind of expectation.
NESTED_DATASETS: dict[str, tuple[object, str, GroundTruth, list[str]]] = {
    # (builder, ext, ground truth, substrings the refusal must name)
    "nested_json_scalars": (
        build_json_nested_scalars, "json",
        GroundTruth(
            columns=["order_id", "geo.country", "geo.city",
                     "totals.net", "totals.tax"],
            n_rows=_NESTED_N,
            numeric_columns=["totals.net", "totals.tax"],
            text_columns=["order_id", "geo.country", "geo.city"],
            shape="scalar nesting: must flatten to dotted names"),
        []),
    "nested_json_repeated_group": (
        build_json_repeated_group, "json",
        GroundTruth(shape="a one-to-many: must be refused, naming 'lines'"),
        ["lines", "repeated group"]),
    "nested_json_two_tables": (
        build_json_two_candidate_tables, "json",
        GroundTruth(shape="two candidate tables: must be refused, naming both"),
        ["customers", "orders", "candidate tables"]),
    "nested_xml_repeated_child": (
        build_xml_repeated_child, "xml",
        GroundTruth(shape="repeated child elements: must be refused, naming 'item'"),
        ["item"]),
}
