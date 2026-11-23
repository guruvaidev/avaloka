"""B6, B8, B9 and the export privacy floor: HTML, Markdown and chat
renderers against the spec branch's golden examples and built fixtures.

Bites (run by hand, recorded in the PR):
- B6 size: set ``MAX_BYTES = 1_000_000`` -> ``test_export_is_within_the_size_cap`` red;
  separately, inline a CDN ``<script src="https://cdn...">`` in ``_PAGE`` ->
  ``test_export_fetches_nothing`` and ``test_csp_hashes_match_the_inline_scripts`` red.
- B8 parity: render executive KPIs with ``decimals=0`` -> ``test_executive_numbers_are_the_analysts`` red.
- B8 injection: drop the ``<`` escape in ``json_for_script`` / ``md_escape`` -> injection tests red.
- B9 chat: substitute ``str(figure["value"])`` for the formatted figure -> ``test_chat_uses_formatted_figures`` red.
- k-floor: remove the ``< k`` skip in ``geo.region_table`` -> ``test_suppressed_regions_never_reach_the_export`` red.
"""

from __future__ import annotations

import base64
import copy
import hashlib
from html.parser import HTMLParser
import json
import re

import pytest

from avaloka.outcomes import format as fmt
from avaloka.outcomes import render_html as RH
from avaloka.outcomes import spec
from avaloka.outcomes.render_chat import MAX_CHARS, render_chat
from avaloka.outcomes.render_html import ExportTooLarge, render_html
from avaloka.outcomes.render_md import md_escape, render_md
from avaloka.outcomes.templates import build_report

from .conftest import ANALYSIS, CREATED, STATE_ROWS, with_linear_whatif

AUDIENCES = ("executive", "analyst")


class _Page(HTMLParser):
    """Scripts, attributes, figure spans and the text inside data-slot regions."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.scripts: list[tuple[dict, str]] = []
        self.attrs: list[tuple[str, str, str]] = []
        self.figs: dict[str, set[str]] = {}
        self.slot_text: list[str] = []
        self._script: dict | None = None
        self._buf: list[str] = []
        self._stack: list[tuple[str, bool, str | None]] = []   # tag, in_slot, fig id

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        for k, v in attrs:
            self.attrs.append((tag, k, v or ""))
        if tag == "script":
            self._script, self._buf = a, []
            return
        in_slot = (self._stack[-1][1] if self._stack else False) or "data-slot" in a
        fig = a.get("data-fig") or (self._stack[-1][2] if self._stack else None)
        if tag not in ("br", "meta", "img", "input", "hr"):
            self._stack.append((tag, in_slot, fig))

    def handle_endtag(self, tag):
        if tag == "script" and self._script is not None:
            self.scripts.append((self._script, "".join(self._buf)))
            self._script = None
            return
        while self._stack:
            if self._stack.pop()[0] == tag:
                break

    def handle_data(self, data):
        if self._script is not None:
            self._buf.append(data)
            return
        if not self._stack:
            return
        _, in_slot, fig = self._stack[-1]
        if fig:
            self.figs.setdefault(fig, set()).add(data)
        elif in_slot:
            self.slot_text.append(data)


def _parse(html: str) -> _Page:
    p = _Page()
    p.feed(html)
    return p


@pytest.fixture(scope="module")
def built(retail_df, fintech_df):
    return {"retail": build_report(retail_df, "how are sales doing", analysis=ANALYSIS, created_at=CREATED),
            "fintech": build_report(fintech_df, "how is the loan book", analysis=ANALYSIS, created_at=CREATED)}


def with_data_checks(report: dict) -> dict:
    """A 1.1 document: ``report`` plus data checks over its own figures, a recipe and engines."""
    r = copy.deepcopy(report)
    r["schema_version"] = "1.1"
    f1, f2 = r["figures"][0]["id"], r["figures"][1]["id"]
    kind = r["caveats"][0]["kind"]
    r["data_checks"] = [{"gate": "duplicates", "status": "qualified", "figure": f1, "caveat_kind": kind},
                        {"gate": "min_rows", "status": "passed"},
                        {"gate": "period_gaps", "status": "qualified", "figure": f2, "caveat_kind": "small_n"},
                        {"gate": "sample_ratio", "status": "failed", "reason": "sample_ratio_mismatch"}]
    r["provenance"]["recipe"] = {"id": "trend", "version": "1.0"}
    r["provenance"]["engine"] = {"duckdb": "1.2.0", "scipy": "1.14.1", "statsmodels": "0.14.4"}
    spec.assert_valid(r)
    assert spec.unresolved_refs(r) == []
    return r


@pytest.fixture(scope="module")
def reports(golden_retail, golden_fintech, built):
    return {"golden_retail": golden_retail, "golden_fintech": golden_fintech, **built,
            "retail_1_1": with_data_checks(golden_retail)}


# ── B6: size, network, CSP ────────────────────────────────────────────────

def _both_maps(report):
    r = copy.deepcopy(report)
    m = copy.deepcopy(r["maps"][0])
    m.update({"id": "world_map", "geo_kind": "country", "asset": "world_110m"})
    m["option"] = {"series": [{"type": "map", "map": "world_110m", "nameProperty": "iso_a3", "data": []}]}
    r["maps"].append(m)
    return r


@pytest.mark.parametrize("which", ["golden_retail", "retail", "fintech", "both_maps"])
@pytest.mark.parametrize("audience", AUDIENCES)
def test_export_is_within_the_size_cap(reports, which, audience):
    report = _both_maps(reports["retail"]) if which == "both_maps" else reports[which]
    html = render_html(report, audience=audience, enforce_cap=False)
    size = len(html.encode("utf-8"))
    print(f"\nexport {which}/{audience}: {size:,} bytes")
    assert size <= RH.MAX_BYTES


def test_the_cap_is_enforced(reports, monkeypatch):
    monkeypatch.setattr(RH, "MAX_BYTES", 500_000)
    with pytest.raises(ExportTooLarge):
        render_html(reports["retail"])


def test_only_the_maps_a_report_uses_are_inlined(reports):
    html = render_html(reports["fintech"])
    assert 'data-geo=' not in html
    html = render_html(reports["retail"])
    assert 'data-geo="us_states_20m"' in html and 'data-geo="world_110m"' not in html


# XML namespaces and licence references inside echarts.min.js: identifiers, never fetched.
_INERT_URLS = {"http://www.apache.org/licenses/LICENSE-2.0", "http://www.w3.org/1999/xlink",
               "http://www.w3.org/2000/svg", "http://www.w3.org/2000/xmlns/",
               "http://www.w3.org/XML/1998/namespace", "https://github.com/ecomfe/zrender/blob/master/LICENSE.txt"}


@pytest.mark.parametrize("audience", AUDIENCES)
def test_export_fetches_nothing(reports, audience):
    html = render_html(reports["golden_retail"], audience=audience)
    page = _parse(html)
    for tag, k, v in page.attrs:
        assert k not in ("src", "href", "srcset", "action", "poster", "data") or tag == "script" and k == "data", (tag, k, v)
        assert not k.startswith("on"), (tag, k)
    assert all("src" not in a for a, _ in page.scripts)
    assert not re.search(r"@import|url\(\s*['\"]?https?:", html)
    urls = set(re.findall(r"https?://[^\s\"'<>)\\]+", html))
    assert urls <= _INERT_URLS, urls - _INERT_URLS


@pytest.mark.parametrize("audience", AUDIENCES)
def test_csp_hashes_match_the_inline_scripts(reports, audience):
    html = render_html(reports["golden_retail"], audience=audience)
    page = _parse(html)
    import html as _html
    csp = _html.unescape(re.search(r'<meta http-equiv="Content-Security-Policy" content="([^"]+)">', html).group(1))
    directives = dict(d.strip().split(" ", 1) for d in csp.split(";"))
    assert directives["default-src"] == "'none'" and directives["connect-src"] == "'none'"
    assert directives["base-uri"] == "'none'" and directives["form-action"] == "'none'"
    assert "unsafe-inline" not in directives["script-src"] and "unsafe-eval" not in directives["script-src"]
    executable = [body for attrs, body in page.scripts if attrs.get("type") not in ("application/json",)]
    assert len(executable) == 2      # piecewise what-if: no runtime
    want = {"'sha256-" + base64.b64encode(hashlib.sha256(b.encode()).digest()).decode() + "'" for b in executable}
    assert set(directives["script-src"].split()) == want


def test_a_live_whatif_inlines_the_runtime_under_its_own_hash(reports):
    import html as _html
    r = with_linear_whatif(reports["retail"])
    an = render_html(r, audience="analyst")
    page = _parse(an)
    executable = [b for a, b in page.scripts if a.get("type") != "application/json"]
    marker = "Avaloka Outcome Report runtime"
    assert len(executable) == 3 and any(marker in b for b in executable)
    csp = _html.unescape(re.search(r'Content-Security-Policy" content="([^"]+)"', an).group(1))
    script_src = dict(d.strip().split(" ", 1) for d in csp.split(";"))["script-src"].split()
    assert set(script_src) == {RH.csp_hash(b) for b in executable}
    assert 'type="range"' in an and 'aria-valuetext="10.0%"' in an
    assert marker not in render_html(r, audience="executive")
    assert marker not in render_html(reports["golden_retail"], audience="analyst")   # piecewise
    assert "not available for this driver form" in render_html(reports["golden_retail"], audience="analyst")


# ── B8: executive / analyst parity ────────────────────────────────────────

@pytest.mark.parametrize("which", ["golden_retail", "golden_fintech", "retail", "fintech", "retail_1_1"])
def test_executive_numbers_are_the_analysts(reports, which):
    ex = _parse(render_html(reports[which], audience="executive"))
    an = _parse(render_html(reports[which], audience="analyst"))
    assert ex.figs, "the executive page shows no figure at all"
    for fid, strings in ex.figs.items():
        assert strings <= an.figs.get(fid, set()), (fid, strings, an.figs.get(fid))
    assert set(ex.figs) <= set(an.figs)
    figs = {f["id"]: f for f in reports[which]["figures"]}
    for fid, strings in an.figs.items():
        allowed = {fmt.format_figure(figs[fid]), fmt.format_figure(figs[fid], signed=True)}
        assert strings <= allowed, (fid, strings, allowed)


@pytest.mark.parametrize("which", ["golden_retail", "retail", "fintech", "retail_1_1"])
@pytest.mark.parametrize("audience", AUDIENCES)
def test_no_number_in_a_slot_outside_a_figure(reports, which, audience):
    page = _parse(render_html(reports[which], audience=audience))
    loose = [t for t in page.slot_text if re.search(r"\d", t)]
    assert loose == []


def test_the_executive_page_is_a_subset(reports):
    ex = render_html(reports["golden_retail"], audience="executive")
    an = render_html(reports["golden_retail"], audience="analyst")
    assert "<h2>Method</h2>" in an and "<h2>Method</h2>" not in ex
    assert "<h2>The ask</h2>" in ex
    assert ex.count('data-chart="') == 1 and an.count('data-chart="') == 2


# ── B8: injection ─────────────────────────────────────────────────────────

PLANTS = ["<script>alert(1)</script>", "</script><script>alert(2)</script>", "[x](javascript:alert(3))",
          "](javascript:alert(4))", "<img src=x onerror=alert(5)>", "[link](https://evil.example/)"]


def _planted(report):
    r = copy.deepcopy(report)
    bad = " ".join(PLANTS)
    r["headline"]["text"] = ("Net {f1} " + bad)[:160]
    r["headline"]["bullets"][0]["so_what"] = bad[:140]
    for c in r["charts"] + r.get("maps", []):
        c["caption"] = bad[:200]
        c["title"] = bad[:80]
        c["data_table"]["rows"].insert(0, [bad] + [None] * (len(c["data_table"]["columns"]) - 1))
        c["data_table"]["columns"][0] = "</script><b>col</b>"
    r["caveats"].append({"kind": "other", "text": bad[:300]})
    r["steps"][0]["title"] = bad[:80]
    spec.assert_valid(r)
    return r


@pytest.mark.parametrize("audience", AUDIENCES)
def test_html_injection_renders_inert(reports, audience):
    r = _planted(reports["golden_retail"])
    html = render_html(r, audience=audience)
    page = _parse(html)
    assert len([1 for a, _ in page.scripts if a.get("type") != "application/json"]) == 2
    assert not [t for t, k, _ in page.attrs if k.startswith("on")]
    assert not [1 for t, _, _ in page.attrs if t in ("img", "a", "b")]
    assert "javascript:" not in " ".join(v for _, _, v in page.attrs)
    [(_, blob)] = [(a, b) for a, b in page.scripts if a.get("id") == "avaloka-report"]
    assert "<" not in blob
    assert json.loads(blob) == spec.accept(r)
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


@pytest.mark.parametrize("audience", AUDIENCES)
def test_markdown_injection_renders_inert(reports, audience):
    md = render_md(_planted(reports["golden_retail"]), audience=audience)
    assert "<" not in md and ">" not in md.replace("&gt;", "")
    assert not re.search(r"(?<!\\)\]\((?!\\)", md), "an unescaped ](... would start a link"
    assert not re.search(r"(?<!\\)\[", md.replace("\\[", ""))
    assert "javascript:alert" in md          # still readable, as text
    assert "&lt;img src=x onerror=alert\\(5\\)&gt;" in md


def test_md_escape_covers_the_listed_characters():
    raw = "\\ ` * _ { } [ ] ( ) # + - . ! | < > ~"
    out = md_escape(raw)
    for ch in "\\`*_{}[]()#+-.!|~":
        assert "\\" + ch in out
    assert "&lt;" in out and "&gt;" in out and "<" not in out
    assert md_escape("a\x00b\x1bc\nd") == "abc d"


def test_markdown_has_tables_and_no_links(reports):
    md = render_md(reports["retail"], audience="analyst")
    assert "| region | net sales |" in md
    assert "](" not in md.replace("\\]\\(", "")


# ── provenance footer ─────────────────────────────────────────────────────

@pytest.mark.parametrize("audience", AUDIENCES)
def test_footer_has_aid_time_evidence_and_every_caveat(reports, audience):
    r = reports["golden_retail"]
    html = render_html(r, audience=audience)
    footer = html[html.index("<footer"):html.index("</footer>")]
    assert ">a-77<" in footer and r["created_at"] in footer and "associational" in footer
    for c in r["caveats"]:
        assert c["text"] in footer
    md = render_md(r, audience=audience)
    prov = md[md.index("**Provenance**"):]
    assert "a\\-77" in prov and md_escape(r["created_at"]) in prov and "associational" in prov
    for c in r["caveats"]:
        assert md_escape(c["text"]) in prov


def test_an_unverified_report_says_so(reports):
    assert "not verified" in render_html(reports["retail"])


# ── 1.1: the Evidence section and the recipe footer ───────────────────────

@pytest.mark.parametrize("audience", AUDIENCES)
def test_a_1_1_document_renders_its_data_checks_as_the_evidence_section(reports, audience):
    r = reports["retail_1_1"]
    html = render_html(r, audience=audience)
    section = html[html.index("<h2>Evidence</h2>"):html.index("</footer>")]
    assert "Source-data checks" in section
    rows = re.findall(r'<tr data-slot="check">(.*?)</tr>', section)
    assert len(rows) == len(r["data_checks"]) == 4
    # Gate names reach the reader as labels, never as snake_case; the status is a word.
    assert "Duplicate rows" in rows[0] and "Qualified" in rows[0] and "duplicates" not in rows[0]
    assert "Enough rows" in rows[1] and "Passed" in rows[1] and "min_rows" not in rows[1]
    assert "Sample ratio" in rows[3] and "Failed" in rows[3] and "sample_ratio" not in section
    # The deciding number is a figure span showing the figure's own formatting.
    figs = {f["id"]: f for f in r["figures"]}
    for check, row in zip(r["data_checks"], rows):
        if check.get("figure"):
            assert f'data-fig="{check["figure"]}">{fmt.format_figure(figs[check["figure"]])}</span>' in row
        else:
            assert "data-fig" not in row
    # The note is the report's caveat sentence of that kind when it has one.
    assert r["caveats"][0]["text"] in rows[0] and "Small sample" in rows[2]
    # Before the caveats: Risks on the executive page, the footer on the analyst page.
    end = html.index("<h2>Risks</h2>") if audience == "executive" else html.index("<footer")
    assert html.index("<h2>Evidence</h2>") < end
    footer = html[html.index("<footer"):html.index("</footer>")]
    assert "<dt>Recipe</dt><dd>trend 1.0</dd>" in footer
    assert "<dt>Engines</dt><dd>duckdb 1.2.0, scipy 1.14.1, statsmodels 0.14.4</dd>" in footer
    md = render_md(r, audience=audience)
    table = md[md.index("## Evidence"):md.index("**Provenance**")]
    assert "Source-data checks" in table
    assert table.count("\n| ") == 2 + 4
    assert "| Duplicate rows | Qualified | " + md_escape(fmt.format_figure(figs[r["data_checks"][0]["figure"]])) in table
    assert "| Enough rows | Passed |  |  |" in table
    assert "- Recipe: trend 1\\.0" in md and "- Engines: duckdb 1\\.2\\.0, scipy 1\\.14\\.1, statsmodels 0\\.14\\.4" in md
    assert md.index("## Evidence") < (md.index("## Risks") if audience == "executive" else md.index("**Provenance**"))


@pytest.mark.parametrize("audience", AUDIENCES)
def test_a_1_0_document_renders_without_the_evidence_section(reports, audience):
    """Nothing of 1.1 leaks into a 1.0 render: no section, no rows, no recipe or engine line."""
    r = reports["golden_retail"]
    assert r["schema_version"] == "1.0" and "data_checks" not in r
    html = render_html(r, audience=audience)
    for marker in ("Source-data checks", "<h2>Evidence</h2>", 'data-slot="check"', "<dt>Recipe</dt>", "<dt>Engines</dt>"):
        assert marker not in html
    md = render_md(r, audience=audience)
    for marker in ("## Evidence", "Source-data checks", "- Recipe:", "- Engines:"):
        assert marker not in md
    # The section is gated on content, not on the version: an empty list renders the
    # same page (only the embedded spec differs, by the empty list itself).
    empty = copy.deepcopy(r)
    empty["data_checks"] = []
    spec.assert_valid(empty)
    page = lambda h: re.sub(r'<script type="application/json" id="avaloka-report">.*?</script>', "", h)  # noqa: E731
    assert page(render_html(empty, audience=audience)) == page(html)
    assert render_md(empty, audience=audience) == md


def test_check_labels_and_status_words_carry_no_digits():
    """The invariant behind data-slot="check": a check's number is its figure span."""
    for text in list(RH.GATE_LABEL.values()) + list(RH.CHECK_STATUS.values()) + list(RH.CAVEAT_LABEL.values()):
        assert not re.search(r"\d", text), text
        assert "_" not in text, text
    rows = RH.check_rows({"data_checks": [{"gate": "new_gate", "status": "qualified", "caveat_kind": "other",
                                           "reason": "engine_timeout"}]})
    assert rows == [{"label": "New gate", "status": "Qualified", "figure": None, "note": "See caveats; engine timeout"}]


def test_chat_summary_is_unchanged_by_1_1_fields(reports):
    assert render_chat(reports["retail_1_1"]) == render_chat(reports["golden_retail"])


def test_unknown_major_is_refused(golden_retail):
    r = dict(golden_retail, schema_version="2.0")
    for fn in (render_html, render_md, render_chat):
        with pytest.raises(spec.SpecError):
            fn(r)


# ── k floor, end to end ───────────────────────────────────────────────────

def test_suppressed_regions_never_reach_the_export(reports):
    r = reports["retail"]
    [m] = r["maps"]
    small = [s for s, n in STATE_ROWS.items() if n < 10]
    assert m["suppressed_regions"] == len(small) == 2 and m["k_floor"] == 10
    assert any(c["kind"] == "suppressed_geo" for c in r["caveats"])
    names = {"WY": "Wyoming", "VT": "Vermont"}
    for audience in AUDIENCES:
        html = render_html(r, audience=audience)
        blob = html[html.index('id="avaloka-report">'):html.index("</script>", html.index('id="avaloka-report">'))]
        for s in small:
            assert f'"{s}"' not in blob and names[s] not in blob
            assert f"<td>{names[s]}</td>" not in html
        md = render_md(r, audience=audience)
        assert all(names[s] not in md for s in small)


# ── B9: chat summary ──────────────────────────────────────────────────────

@pytest.mark.parametrize("which", ["golden_retail", "golden_fintech", "retail", "fintech"])
def test_chat_is_short_and_points_at_the_report(reports, which):
    text = render_chat(reports[which])
    assert len(text) <= MAX_CHARS
    assert text.endswith(f"avaloka report {reports[which]['analysis']['analysis_id']}")
    assert "{f" not in text


def test_chat_uses_formatted_figures(golden_retail):
    text = render_chat(golden_retail)
    figs = {f["id"]: f for f in golden_retail["figures"]}
    for fid in spec.placeholders(golden_retail["headline"]["text"]):
        assert fmt.format_figure(figs[fid]) in text
    assert "Top move:" in text and "not a tested effect" in text


def test_chat_stays_under_the_limit_with_long_slots(golden_retail):
    r = copy.deepcopy(golden_retail)
    r["headline"]["text"] = ("Long headline {f1} " * 9)[:160]
    for b in r["headline"]["bullets"]:
        b["so_what"], b["now_what"] = "x" * 140, "y" * 140
    r["headline"]["bullets"].append({"so_what": "z" * 140, "now_what": "w" * 140})
    text = render_chat(r)
    first = fmt.format_figure({f["id"]: f for f in r["figures"]}["f1"])
    assert len(text) <= MAX_CHARS and text.startswith(f"Long headline {first}")


def test_e2e_fixture_is_current():
    """ui/tests/e2e/outcome-export.spec.ts runs this file; it must be today's render."""
    from .conftest import E2E_FIXTURE, e2e_export
    assert E2E_FIXTURE.read_text(encoding="utf-8") == e2e_export(), \
        "regenerate: python tests/avaloka/outcomes/conftest.py"


def test_one_bad_chart_does_not_stop_the_rest():
    assert "try {" in RH.INIT_JS and "console.error" in RH.INIT_JS
