"""The telemetry ingest service.

Receives usage batches from Avaloka installations, validates them against an
allowlist, deduplicates them and stores them.

What it holds is pseudonymous, not anonymous. It stores no name, address or
account. It does store identifiers: a random id per installation, a session
id, and -- on product-analytics records -- ``actor_id``, a per-analyst
pseudonym that the exporting deployment re-keys every 28 days with a secret it
never transmits. See ``schema.py`` for exactly what that does and does not let
this service link.

Deliberately self-contained: it shares no code with the ``avaloka`` CLI package
or the product API, so it can be lifted into its own repository without
touching either. The only thing it shares with the client is the *shape* of the
data, and that is restated here rather than imported - the two ends must agree
by contract, not by a shared module that could be changed on one side only.
"""

__version__ = "0.1.0"
