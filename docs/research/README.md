# Research

The paper behind Avaloka, and its sources.

| File | What it is |
| --- | --- |
| [Avaloka-Research-Paper.pdf](../Avaloka-Research-Paper.pdf) | The compiled paper — start here |
| `avaloka-paper.tex` | LaTeX source |
| `references.bib` | Bibliography |
| `paper-draft.md` | Working draft in markdown |
| `fig/` | Figures |

## Building it

The source was written against the ICLR 2026 template. Those class and style
files are **not** vendored here — `natbib.sty`, `fancyhdr.sty`, the conference
`.sty` and `.bst`, and `math_commands.tex` were removed because they are
third-party LaTeX packages under their own licences, which do not belong in an
Apache-2.0 tree. A build artefact (`.aux`) and a duplicate of the compiled PDF
went with them.

To build, fetch the template you are submitting to and place `avaloka-paper.tex`
alongside it. Most venues require their own template anyway, so vendoring one
would have been wrong even without the licensing question.

## Note on scope

The paper describes the architecture and the evaluation design. Measured
results produced since it was written live in the repository rather than in the
paper:

* [Benchmarks](../benchmarks.md) — the deterministic suite, and what it does not measure
* [Reliability measurements](../test-reports/1.6-reliability-measurements.md) — routing accuracy across models, conversational and ingest results
* [Three-pillar coverage](../test-reports/three-pillar-coverage.md) — per-case evidence, including which cases are blocked

Where the paper proposes an evaluation that has not been run, it says so. Where
a number exists, it is in the documents above.
