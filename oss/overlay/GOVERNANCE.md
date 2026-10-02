# Governance

Avaloka is developed in the open under the [Apache License 2.0](LICENSE).
This document describes how decisions get made, so that "who decides?" has a
written answer rather than a conventional one.

## Roles

**Contributor** — anyone who opens an issue or a pull request. No paperwork
beyond a [DCO sign-off](CONTRIBUTING.md#sign-your-work) on each commit.

**Maintainer** — a contributor with commit rights, listed in
[MAINTAINERS.md](MAINTAINERS.md). Maintainers review and merge pull requests,
triage issues, and are accountable for the areas they own.

**Technical Steering Committee (TSC)** — the maintainers acting together on
decisions that cross areas: the roadmap, releases, adding or removing
maintainers, and changes to this document.

## How decisions are made

Day to day, **lazy consensus**. A pull request with an approving review from a
maintainer who did not write it, and no unresolved objection, may be merged.
Silence is assent; an objection is not a veto but does require a reply.

For decisions the TSC owns, a **simple majority of maintainers** decides, on a
thread anyone can read. A maintainer with a conflict of interest abstains and
says so.

Changes to the licence, to this document, or to the project's name require a
**two-thirds majority**.

## Becoming a maintainer

Sustained, high-quality contribution over time, proposed by an existing
maintainer and approved by a majority. What counts is judgement demonstrated in
review and in the tests you write, not commit count.

A maintainer who has been inactive for six months may be moved to emeritus by a
majority, and may return by asking.

## Releases

Releases follow [semantic versioning](https://semver.org). A release requires a
green test suite and a benchmark run, and the release notes must state what was
**not** verified as plainly as what was — an unverified claim in a release note
is the one that costs trust later.

## Code of conduct

Everyone participating is bound by the [Code of Conduct](CODE_OF_CONDUCT.md).
Reports go to the address named there and are handled by the maintainers not
involved in the report.

## Security

Vulnerabilities are reported privately, following [SECURITY.md](SECURITY.md),
never as a public issue.

## Foundation

This project intends to move to a neutral foundation home. Until that is
complete, the copyright holders are the contributors, the trademark position is
unsettled, and this document is the authority on how the project is run.
