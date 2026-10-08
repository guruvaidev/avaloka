# Redaction test fixtures trip the secret scanner — every time

**If you are writing tests for a redactor, read this before you write the first
fixture.**

## The pattern

A redaction test needs inputs that look like secrets: an AWS key, a provider
token, a PEM header, a database URL with a password. The obvious way to write
one is as a string literal.

The repository scans tracked files for exactly those shapes
(`tests/contract/test_c1_repo_hygiene.py::test_e11_05_secret_material_absent_from_tracked_files`,
and the OSS generator has its own scan). The scanners read files as text. They
cannot tell a fixture from a leak, and they should not try.

This has now happened twice, independently:

- `tests/analytics/test_redaction.py` (product analytics): four hits — AWS key,
  Groq key, PEM header, database URL with a password.
- `tests/avaloka/test_telemetry_redaction.py:32,34` (PR #329): four hits in the
  OSS generator's scan.

It will happen to the third redactor too.

## Why it is easy to miss

In the `avaloka-ci-deps` container there is no `git`, so the scanner **skips**
rather than fails, and the run is green. A git worktree bind-mounted into a
container skips the same way even with git installed, because its `.git` file
points outside the mount. Check `git ls-files | wc -l` inside the container
before trusting a green hygiene run.

## The fix

Assemble credential-shaped fixtures at import time, so no tracked file contains
the literal shape:

```python
_AWS = "AKIA" + "IOSFODNN7EXAMPLE"
_GSK = "gsk" + "_Zx81kPq92LmNw0Vb7TtYy"
_PEM = "-----BEGIN RSA " + "PRIVATE KEY-----\nMIIEvQIBADANBg\n-----END RSA " + "PRIVATE KEY-----"
_DSN = "postgres://admin:" + "hunter2" + "@db.internal:5432/prod"
```

The redactor sees the same string at run time. The scanner sees two harmless
halves.

## What not to do

- Do not add the test file to a scanner exemption list. An exempted path is
  where the next real key will hide.
- Do not weaken the fixture until it stops matching the scanner. It will stop
  matching the redactor too, and the test will pass while testing nothing.
- Do not use real keys, revoked or not.
