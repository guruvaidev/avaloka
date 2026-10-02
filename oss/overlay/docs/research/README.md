# Research

The paper behind Avaloka, as published PDFs.

| Path | What it is |
|---|---|
| [`../Avaloka-Research-Paper.pdf`](../Avaloka-Research-Paper.pdf) | The paper |
| `iclr2026/` | The ICLR 2026 submission, as submitted |
| `fig/` | Figures |

## Why the sources are not here

The LaTeX sources build against the ICLR 2026 template, and
`iclr2026_conference.sty`, `iclr2026_conference.bst`, `natbib.sty`
(Patrick W Daly) and `fancyhdr.sty` are third-party packages under their own
licences. They cannot ship under Apache-2.0, and a `.tex` that will not
compile is worse than no `.tex`. So this tree carries the compiled papers.

The template is available from the ICLR 2026 author kit if you want to rebuild.

## A correction

arXiv:2508.05002 is **"AgenticData: An Agentic Data Analytics System for
Heterogeneous Data"** (Sun et al.). It is *not* "Agents in the Wild: Safety,
Security, and Beyond" — that is an ICLR 2026 **workshop**, not a paper.

The PDFs here **predate** that correction and should not be cited for it. The
sources have been fixed; these have not yet been rebuilt.
