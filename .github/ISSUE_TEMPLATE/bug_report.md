---
name: Bug report
about: Something behaves differently from what it claims
labels: bug
---

## What happened

<!-- What you observed. Paste the exact error if there is one -- it is usually
     more useful than a description of it. -->

## What you expected

## How to reproduce

<!-- The smallest thing that shows it. A dataset's shape and column roles are
     often enough; please do NOT attach data you cannot share publicly. -->

```bash
avaloka analyze ...
```

## Environment

<!-- `python scripts/doctor.py` answers all of this at once, and catches the
     environment problems that masquerade as bugs: a mismatched numpy/torch
     ABI, or an x86_64 Python under Rosetta, whose failures look like hangs
     rather than crashes. -->

```
paste the output of: python scripts/doctor.py
```

## Anything else

<!-- If a validation verdict was involved, the bundle's validation_report.json
     names the check that produced it. -->
