## What this changes

<!-- The behaviour that differs afterwards, not a list of files. -->

## Why

<!-- If it fixes a defect, what was wrong and how it showed up. A defect that
     produced plausible-looking output is worth describing carefully -- those
     are the ones that survive review. -->

## How it was verified

<!-- What you ran, and what it said. If a test would have caught this, add it
     and say so; if you could not run something, say that too. An unverified
     claim in a PR is the one that costs trust later. -->

```
paste the relevant output
```

## Checklist

- [ ] Commits are signed off (`git commit -s`) — see [CONTRIBUTING.md](../CONTRIBUTING.md#sign-your-work). There is no CLA.
- [ ] Tests added for new behaviour, or a regression test for a fix
- [ ] `./scripts/ci.sh --fast` passes locally, or the failure is explained above
- [ ] Anything left unverified is stated plainly
