"""The telemetry schema version: one number for every producer.

Events are persisted and read back later, by code that did not write them. An
unversioned shape is harmless until there is a year of data in it.

Both producers stamp this on what they emit: process telemetry (this package,
PR #329) and product analytics (``app/analytics``). It lives here, not in either
producer, so there is exactly one.

``MAJOR.MINOR``. MINOR is additive only -- a new event type, a new optional
field, a new enum value. MAJOR is anything else: a rename, a removal, a changed
meaning, a field becoming required. A reader supports the current MAJOR and the
previous one; stored records are never rewritten.

This file is deliberately alone in the directory on develop-1.6: the rest of
``avaloka/telemetry`` arrives with PR #329, and a lone module does not collide
with the ``__init__.py`` that PR adds.
"""

#: 1.0  the version module alone (PR #426). No event shape was published under it.
#: 1.1  the first product-analytics schema. Its exported representation carries
#:      no question text. Drafts of 1.1 on an unmerged branch did; that was
#:      withdrawn before anything merged or ran, so 1.1 is defined without it
#:      rather than spending a MAJOR on a shape nobody ever received. See
#:      "Version history" in docs/analytics/DESIGN.md.
SCHEMA_MAJOR = 1
SCHEMA_MINOR = 1
SCHEMA_VERSION = f"{SCHEMA_MAJOR}.{SCHEMA_MINOR}"
SUPPORTED_MAJORS = frozenset({1})
