"""The CLI loader reads every declared format, and refuses what it must.

Four of the nine declared upload formats had no branch in ``load_dataset``.
Two of those failed loudly; the other two were *accepted* by the delimiter
sniffer and returned a frame named after the file's own bytes -- XML as one
column called ``<?xml version='1.0' encoding='utf-8'?>``, Avro with columns
lifted from its binary schema header. A frame like that profiles and narrates
exactly like a real table, so these tests exist to keep the silent shape from
coming back, not just the loud one.

The refusals are tested as carefully as the successes. For a document whose
shape has no single correct reading, refusing is the right answer and reading
it is the defect -- so a test that only checked "it loaded" would reward the
bug.
"""

from __future__ import annotations

import json

import pytest

from avaloka.benchmark.datasets_formats import (DEFERRED_FORMATS,
                                                FORMAT_DATASETS,
                                                NESTED_DATASETS)
from avaloka.io.loader import UnusableDataset, load_dataset


@pytest.mark.parametrize("name", sorted(FORMAT_DATASETS))
def test_every_declared_format_reads_its_table_back(name, tmp_path):
    builder, ext, truth = FORMAT_DATASETS[name]
    path = tmp_path / f"{name}.{ext}"
    builder(path)

    handle = load_dataset(str(path))
    assert list(handle.frame.columns) == truth.columns, (
        f"{ext}: resolved {list(handle.frame.columns)}, "
        f"the file has {truth.columns}")
    assert handle.frame.shape[0] == truth.n_rows, (
        f"{ext}: read {handle.frame.shape[0]} rows, the file has {truth.n_rows}")
    assert handle.fmt == {"tsv": "csv"}.get(ext, ext), (
        f"{ext}: reported format {handle.fmt!r}")


@pytest.mark.parametrize("ext", sorted({e for _, e, _ in FORMAT_DATASETS.values()}))
def test_no_declared_format_is_read_as_a_delimited_file(ext, tmp_path):
    """The specific regression: a container must not be sniffed as CSV.

    The tell was a single column whose name carried the file's magic bytes or
    its XML declaration. One column from a five-column file is the symptom
    whatever the cause, so that is what this asserts.
    """
    name, (builder, _, truth) = next(
        (n, v) for n, v in FORMAT_DATASETS.items() if v[1] == ext)
    path = tmp_path / f"{name}.{ext}"
    builder(path)
    frame = load_dataset(str(path)).frame
    assert frame.shape[1] == len(truth.columns), (
        f"{ext} came back with {frame.shape[1]} column(s) "
        f"{list(frame.columns)[:3]} instead of {len(truth.columns)}")
    for column in frame.columns:
        text = str(column)
        assert "<?xml" not in text and "avro" not in text.lower(), (
            f"{ext}: column {text!r} is the file's own header, not a column name")


# --------------------------------------------------------------------------
# The nesting rule
# --------------------------------------------------------------------------
def test_scalar_nesting_flattens_to_dotted_names(tmp_path):
    builder, ext, truth, must_name = NESTED_DATASETS["nested_json_scalars"]
    assert not must_name, "this dataset is supposed to be readable"
    path = tmp_path / f"scalars.{ext}"
    builder(path)
    frame = load_dataset(str(path)).frame
    assert list(frame.columns) == truth.columns
    assert frame.shape[0] == truth.n_rows


@pytest.mark.parametrize("name", sorted(
    n for n, v in NESTED_DATASETS.items() if v[3]))
def test_a_document_with_no_single_reading_is_refused_by_name(name, tmp_path):
    """Refusing is the answer, and the message must say what to do about it."""
    builder, ext, _, must_name = NESTED_DATASETS[name]
    path = tmp_path / f"{name}.{ext}"
    builder(path)

    with pytest.raises(UnusableDataset) as caught:
        load_dataset(str(path))
    message = str(caught.value)
    missing = [token for token in must_name if token.lower() not in message.lower()]
    assert not missing, (
        f"the refusal does not name {missing}, so the user has no next step. "
        f"Message: {message}")


# --------------------------------------------------------------------------
# Untrusted XML
# --------------------------------------------------------------------------
_SECRET = "TOP-SECRET-CANARY-VALUE"


def test_an_external_entity_cannot_read_a_local_file(tmp_path):
    """XXE. An upload must not turn a table read into a file read.

    This is not hypothetical for whichever parser runs: the structural scan
    uses the standard library, which rejects undefined external entities, but
    ``pd.read_xml`` prefers ``lxml``, which resolves them. The DOCTYPE refusal
    is what makes the outcome independent of that ordering.
    """
    secret = tmp_path / "secret.txt"
    secret.write_text(_SECRET)
    path = tmp_path / "xxe.xml"
    path.write_text(
        '<?xml version="1.0"?>\n'
        f'<!DOCTYPE rows [ <!ENTITY leak SYSTEM "file://{secret}"> ]>\n'
        "<rows><row><a>&leak;</a><b>1</b></row>"
        "<row><a>x</a><b>2</b></row></rows>")

    with pytest.raises(UnusableDataset) as caught:
        load_dataset(str(path))
    assert "DOCTYPE" in str(caught.value)
    assert _SECRET not in str(caught.value), (
        "the refusal message itself leaked the file contents")


def test_nested_entities_cannot_expand_into_an_out_of_memory_error(tmp_path):
    """Entity expansion. Measured at 10,000 characters from one small cell
    before the DOCTYPE refusal existed; the shape scales."""
    path = tmp_path / "bomb.xml"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE rows [\n'
        ' <!ENTITY a "aaaaaaaaaa">\n'
        ' <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">\n'
        ' <!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">\n'
        ' <!ENTITY d "&c;&c;&c;&c;&c;&c;&c;&c;&c;&c;">\n'
        ']>\n'
        "<rows><row><a>&d;</a></row><row><a>y</a></row></rows>")

    with pytest.raises(UnusableDataset) as caught:
        load_dataset(str(path))
    assert "DOCTYPE" in str(caught.value)


def test_ordinary_xml_still_reads_after_the_doctype_refusal(tmp_path):
    """The guard must not be a ban on XML.

    Without this, refusing every DOCTYPE would look identical to refusing
    every XML file, and the format would be "supported" while unreadable --
    which is the state this branch set out to fix.
    """
    builder, ext, truth = FORMAT_DATASETS["fmt_xml"]
    path = tmp_path / f"plain.{ext}"
    builder(path)
    frame = load_dataset(str(path)).frame
    assert list(frame.columns) == truth.columns
    assert frame.shape[0] == truth.n_rows


# --------------------------------------------------------------------------
# The optional reader
# --------------------------------------------------------------------------
def test_a_missing_avro_reader_names_the_install_command(tmp_path, monkeypatch):
    """Avro is the one format needing a library outside the declared eight.

    Its absence must be actionable. A bare ``ModuleNotFoundError: fastavro``
    from inside pandas tells the user nothing about what to install, and the
    install target has to be one that exists -- ``.xlsx`` spent this release
    pointing at an extra that was never declared.
    """
    import builtins
    real_import = builtins.__import__

    def without_fastavro(name, *args, **kwargs):
        if name.split(".")[0] == "fastavro":
            raise ImportError("blocked by this test")
        return real_import(name, *args, **kwargs)

    builder, ext, _ = FORMAT_DATASETS["fmt_avro"]
    path = tmp_path / f"a.{ext}"
    builder(path)                                  # written while fastavro works
    monkeypatch.setattr(builtins, "__import__", without_fastavro)

    with pytest.raises(UnusableDataset) as caught:
        load_dataset(str(path))
    message = str(caught.value)
    assert "avaloka[formats]" in message or "fastavro" in message, message

    import tomllib
    from pathlib import Path
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    extras = pyproject["project"].get("optional-dependencies", {})
    assert "formats" in extras, (
        "the error tells the user to install 'avaloka[formats]', so that extra "
        "must exist")


def test_the_deferred_format_is_still_deferred_for_a_stated_reason():
    """``xls`` is the one format with no reader, and the reason must stand.

    Legacy BIFF has no writer in the dependency set, and
    ``to_excel('.xls')`` silently writes a zip-based xlsx that ``xlrd``
    refuses -- so a task built on one would report a pass for a format it never
    read. If that ever stops being true the deferral should go, and the
    coverage check will say so.
    """
    assert "xls" in DEFERRED_FORMATS
    assert len(DEFERRED_FORMATS["xls"]) >= 40
