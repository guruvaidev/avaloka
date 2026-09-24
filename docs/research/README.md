# Research

The paper behind Avaloka, and its sources.

| File | What it is |
| --- | --- |
| [Avaloka-Research-Paper.pdf](../Avaloka-Research-Paper.pdf) | The compiled paper — start here |
| `avaloka-paper.tex` | LaTeX source |
| `references.bib` | Bibliography |
| `paper-draft.md` | Working draft in markdown |
| `fig/` | Figures |

## ⚠️ The compiled PDF is out of date

[`Avaloka-Research-Paper.pdf`](../Avaloka-Research-Paper.pdf) **predates a
correction and should not be cited until it is rebuilt.** It contains a
reference to *"Agents in the Wild: Safety, Security, and Beyond,
arXiv:2508.05002"*. That paper does not exist: arXiv:2508.05002 is
**"AgenticData: An Agentic Data Analytics System for Heterogeneous Data"**
(Sun et al.), and *Agents in the Wild* is an ICLR 2026 **workshop**, not a
paper. The `.tex`, the `.bib` and `paper-draft.md` in this directory are
corrected; the PDF is not, because it cannot currently be rebuilt — see below.

## Building it

`avaloka-paper.tex` was written against the **ICLR 2026** template and will not
compile without it. Line 3 is `\usepackage{iclr2026_conference,times}` and line
6 is `\input{math_commands.tex}`.

Those files are deliberately **not** vendored here. `iclr2026_conference.sty`,
`iclr2026_conference.bst`, `natbib.sty`, `fancyhdr.sty` and `math_commands.tex`
are third-party LaTeX packages under their own licences and do not belong in an
Apache-2.0 tree.

To build as-is, download the ICLR 2026 author kit and unpack it alongside
`avaloka-paper.tex`:

```bash
# from https://github.com/ICLR/Master-Template (or the ICLR 2026 CFP page)
pdflatex avaloka-paper.tex && bibtex avaloka-paper && pdflatex avaloka-paper.tex
```

Since the paper is no longer being submitted to ICLR, the more useful option is
to reformat it for the venue you are targeting — every venue mandates its own
template anyway, which is the second reason vendoring one would have been
wrong. See [`submissions/`](submissions/) for work already done that way.

## Note on scope

The paper describes the architecture and the evaluation design. Measured
results produced since it was written live in the repository rather than in the
paper:

* [Benchmarks](../benchmarks.md) — the deterministic suite, and what it does not measure
* [Reliability measurements](../test-reports/1.6-reliability-measurements.md) — routing accuracy across models, conversational and ingest results
* [Three-pillar coverage](../test-reports/three-pillar-coverage.md) — per-case evidence, including which cases are blocked

Where the paper proposes an evaluation that has not been run, it says so. Where
a number exists, it is in the documents above.
