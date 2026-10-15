"""What the server will accept, and nothing else.

This is the second half of a control whose first half lives in the client. The
client drops unknown fields before writing them; the server *rejects* a payload
that contains them. Both are needed: filtering on the server would quietly
accept a client that had started sending something it should not, and nobody
would ever find out.

The lists here are stated independently rather than imported from the client.
The two ends are separate services that will be deployed on different
schedules, and a shared module would let a change on one side silently alter
the other.

Two producers send to this endpoint, and each record is checked against the
keys of *its own* event type, not against everything any type may carry:

* process telemetry -- `install`, `agent.started`, `agent.completed`,
  `dataset` -- from an Avaloka installation;
* product analytics -- the `analytics.*` types -- exported by a deployment
  whose operator switched that export on. Behavioural fields only. The
  question an analyst typed stays in the deployment that recorded it.

Neither set is open. An unknown type is rejected, and so is a known type
carrying a key it does not declare.

And no value is open either. A pattern constrains form; a vocabulary constrains
meaning; and a name a customer chose -- for a file, a column, a model
deployment -- is a meaning problem that no pattern solves. So every accepted
value, from either producer, is one of: a member of a closed vocabulary, a
bounded integer, a boolean, a hex or UUID id, a class name, a version, or a
route template. Where a real value falls outside a vocabulary the sender maps
it to the literal ``other``; the server rejects it, it does not coerce it.

What is stored about a person: nothing that names one, and one thing that
distinguishes one. Analytics records carry ``actor_id``, a per-analyst
pseudonym. Stated exactly, because this is the claim a reader will check:

* It is an HMAC, computed by the exporting deployment, of the identifier that
  deployment stores for the analyst. The key is 32 random bytes the deployment
  generates for itself and keeps in its own database. The key is never
  transmitted, so this service cannot turn a pseudonym back into a user, or
  into the identifier the deployment stores.
* It is re-keyed every 28 days. The window is a fixed 28-day period counted
  from the Unix epoch and chosen by when the deployment received the event, so
  every analyst's pseudonym changes on the same day. This service cannot link
  an analyst in one window to the same analyst in the next.
* Two deployments have different keys, so the same person using both appears
  as two unrelated pseudonyms.
* Within one window it does tie one analyst's events together, and beside
  ``install_id`` it distinguishes one analyst of one deployment for up to 28
  days. Whoever holds that deployment's key and its user table -- its
  operator, not this service -- can recompute it.

So what this service holds is pseudonymous. It is not anonymous, and nothing
here should describe it as such.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# ------------------------------------------------------------------ value rules

MAX_T_MS = 7 * 24 * 3600 * 1000      # a tab left open for a week
MAX_DURATION_MS = 24 * 3600 * 1000

#: Hex digits and hyphens only: admits a UUID in either form, cannot spell a
#: name.
_OPAQUE_ID = re.compile(r"^[0-9a-fA-F][0-9a-fA-F-]{15,63}$")
#: "1.6.2", "1.7.0-rc1", "1.6.2+3f9a1c0". Starts with a digit.
_VERSION = re.compile(r"^\d{1,4}(?:\.\d{1,4}){0,3}(?:[-+][0-9A-Za-z.]{1,16})?$")
#: The version of an interpreter or of an installed build, which is not ours
#: to constrain: Python and packaging tools write "3.13.0rc1", "3.12.3+",
#: "1.6.2.post1", "1.6.2.dev3+g1a2b3c4.d20261007". Rejecting a whole batch over
#: one of those would discard every other event in it. So this is the PEP 440
#: shapes -- release, then optional a/b/rc, .post, .dev, then an optional local
#: part -- and still a bounded pattern: it starts with a digit, is at most 40
#: characters, and holds only digits, ASCII letters, dots, "+" and "-". No
#: space, no slash, no underscore, so not a file name, a path or a sentence.
#:
#: Like _CLASS_NAME it constrains form, not meaning. The widest thing it admits
#: is a digit, a "+" or "-", and up to 24 letters, digits and dots:
#: "1+db01.internal.acme.corp" passes.
_RELEASE_VERSION = re.compile(
    r"^(?=.{1,40}$)\d{1,4}(?:\.\d{1,4}){0,3}"
    r"(?:(?:a|b|rc)\d{1,3})?(?:\.post\d{1,4})?(?:\.dev\d{1,5})?"
    r"(?:[-+][0-9A-Za-z.]{0,24})?$",
    re.ASCII,       # \d would otherwise admit every Unicode digit
)
#: "MAJOR.MINOR", the telemetry schema version stamped on every record.
_SCHEMA_VERSION = re.compile(r"^\d{1,4}\.\d{1,4}$")
#: An exception *class name*: one dotted identifier, never a message. Used for
#: `error_class` from both producers. That field exists so that `error_message`
#: does not have to, which only holds if a message cannot be sent under the
#: permitted name: exception text routinely quotes column names, paths and
#: values, and a class name cannot contain a space.
_CLASS_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,63}$")


class Spec(NamedTuple):
    """What one field may hold. ``kind`` selects the check."""
    kind: str                       # enum | int | bool | id | class | version | release_version | schema_version | counts
    values: frozenset = frozenset()  # enum: the vocabulary. counts: the permitted keys
    lo: int = 0                     # int: the range. counts: the range of each count
    hi: int = 0


def _enum(*values: str) -> Spec:
    return Spec("enum", values=frozenset(values))


def _int(lo: int, hi: int) -> Spec:
    return Spec("int", lo=lo, hi=hi)


_ID = Spec("id")
_BOOL = Spec("bool")
_CLASS = Spec("class")
#: The web UI's version, as the analytics producer declares it. Tied to that
#: producer's own pattern by the contract test, so it is not widened here.
_UI_VERSION = Spec("version")
#: `python_version` and `avaloka_version`: see _RELEASE_VERSION.
_RELEASE = Spec("release_version")
_DURATION = _int(0, MAX_DURATION_MS)
_SEQ = _int(0, 10_000_000)

# Vocabularies both producers use for the same thing. One constant each, so the
# two halves of this schema cannot come to disagree about what a format or an
# outcome is.
_OUTCOMES = _enum("success", "failed", "refused", "timeout", "cancelled")
#: A closed set of formats, so a file *name* can never fit. A format outside it
#: is sent as "other".
_FILE_FORMATS = _enum("csv", "tsv", "xlsx", "parquet", "json", "other")

# Which vocabularies have an "other", and which do not. It is one rule, not
# taste: who controls the values.
#
# * The world controls them -> "other". Any operating system, architecture,
#   cloud, file format, model name or kind of column can turn up on somebody's
#   machine. Refusing a batch because of one nobody listed would punish the
#   installation for something it does not control, so the sender maps it to
#   "other" and the rest of the record still counts. (`os`, `arch`, `cloud`,
#   `file_format`, `llm_model`, the kinds in `column_type_counts`.) Versions
#   are the same case without a vocabulary: the pattern for `python_version`
#   and `avaloka_version` is wide enough for what interpreters and build
#   tools actually write.
# * We control them -> no "other". We decide what an edition is, which agents
#   exist, which providers the router can reach, how a deployment or an
#   outcome is classified. A value missing from one of those lists is our
#   omission, and it must fail loudly here rather than be swallowed as
#   "other" and never noticed. (`edition`, `agent`, `llm_provider`,
#   `deployment`, `outcome`, the size bands.)
#
# The second kind has a cost, and it is an ordering rule for whoever deploys
# this: a closed vocabulary refuses every batch from a sender that is ahead of
# it. A new graph node, default model or edition must reach this service before
# any installation reports it. tests/telemetry_ingest/test_product_contract.py
# fails in the repository the moment one is added, which is the prompt to do so.

# ------------------------------------------------------------ process telemetry

#: The product's editions.
EDITIONS = _enum("oss", "free", "professional", "enterprise")

#: Where inference was routed: the canonical provider names of the product's
#: model-routing layer. Closed, with no "other": a provider the router does not
#: know cannot have served a call.
LLM_PROVIDERS = _enum("groq", "openrouter", "local", "openai", "bedrock", "vertex", "azure")

#: The models the product ships as defaults, and "other". Free text is not
#: acceptable here: an Azure deployment name or a self-hosted model path is
#: chosen by the customer and can carry their name. A sender whose model is not
#: on this list sends the literal "other"; an unlisted string is rejected.
LLM_MODELS = _enum("openai/gpt-oss-120b", "openai/gpt-oss-20b", "other")

#: Every agent that can be named: the nodes of the product's graphs and the CLI
#: fireflies. A node added to a graph must be added here before an installation
#: running it can report.
AGENTS = _enum(
    # the coding sub-graph
    "coder", "validator_syntax", "validator_static", "execute_code",
    "validator_contract", "validator_logical",
    # the main graph
    "memory_injection", "avaloka_agent", "plan_etl", "summarize_etl",
    "generate_planner_graph", "code_etl", "execute_actions", "provision_infra",
    "schedule_task", "execute_on_ray", "execute_locally", "execute_on_k8s",
    "visualize", "narrate_result", "verify_claims", "train_models", "profile_data",
    # CLI fireflies
    "analysis_planner", "data_engineer", "data_scout", "finops", "ml_engineer",
    "model_scientist", "reporter", "sampling_specialist", "validator",
)

#: Kinds of column, as the profiler classifies them, and "other". The keys of
#: `column_type_counts` come from here and nowhere else: a column called
#: `salary` has the same shape as a kind called `numeric`, so only a vocabulary
#: keeps a column name out.
COLUMN_KINDS = frozenset({
    "numeric", "text", "boolean", "datetime", "categorical", "id", "constant", "other",
})
MAX_COLUMNS_COUNTED = 1_000_000

# The size bands process telemetry uses. Product analytics uses different bands
# for the same two fields (see ANALYTICS_EVENTS below). They are deliberately
# not unified here; the difference is tracked separately.
_PROCESS_ROW_BUCKETS = _enum("0-1e3", "1e3-1e4", "1e4-1e5", "1e5-1e6", "1e6-1e7",
                             "1e7-1e8", "1e8+")
_PROCESS_COLUMN_BUCKETS = _enum("1-5", "5-10", "10-50", "50-200", "200+")

#: Sent once per batch. Describes the installation, never the user.
INSTALL_FIELDS = {
    "event_id": _ID,
    "install_id": _ID,
    "avaloka_version": _RELEASE,
    "edition": EDITIONS,
    "deployment": _enum("laptop", "kind", "kubernetes"),
    # "local" means not in a cloud at all. A provider that is not listed is
    # "other", never "local": a deployment on an unlisted cloud counted as a
    # laptop is a wrong value that looks right.
    "cloud": _enum("local", "gcp", "aws", "azure", "other"),
    "os": _enum("linux", "darwin", "windows", "other"),
    "arch": _enum("x86_64", "amd64", "arm64", "aarch64", "other"),
    "python_version": _RELEASE,
    "cpu_count": _int(1, 4096),
    "memory_gib": _int(1, 65_536),
}

#: Present on every record but the install one. `session_id` groups one run,
#: `seq` orders it. Neither is the kind of identifier this design refuses: the
#: client holds the session id in memory and never writes it to disk, so it
#: cannot link a user's activity across sessions the way a mission or
#: conversation id would.
SESSION_FIELDS = {"session_id": _ID, "seq": _SEQ}

#: A pair per agent activation -- `agent.started` then `agent.completed`. A
#: start with no matching completion is the only trace a killed process leaves,
#: which is why the pair exists rather than one record carrying a duration.
AGENT_FIELDS = {
    "event_id": _ID,
    "agent": AGENTS,
    "outcome": _OUTCOMES,
    "error_class": _CLASS,
    # A training run can outlast a day, so this is wider than the bound on an
    # analyst's question.
    "duration_ms": _int(0, MAX_T_MS),
    "retry_count": _int(0, 100),
    "started_seq": _SEQ,
    "llm_provider": LLM_PROVIDERS,
    "llm_model": LLM_MODELS,
    "tokens_in": _int(0, 1_000_000_000),
    "tokens_out": _int(0, 1_000_000_000),
    **SESSION_FIELDS,
}

#: The shape of a dataset, never its content. No question text and no column
#: names: telemetry is behavioural only, and what a customer typed or called
#: their columns is customer data. Both are named in REFUSED_KEYS below.
DATASET_FIELDS = {
    "event_id": _ID,
    "row_count_bucket": _PROCESS_ROW_BUCKETS,
    "column_count_bucket": _PROCESS_COLUMN_BUCKETS,
    "file_format": _FILE_FORMATS,
    "column_type_counts": Spec("counts", values=COLUMN_KINDS, lo=0, hi=MAX_COLUMNS_COUNTED),
    "sampled": _BOOL,
    **SESSION_FIELDS,
}

#: event type -> {field: Spec}. A record is checked against its own type's
#: fields: `agent` on an `install` record is as much a sign of a confused
#: client as a key nobody has heard of.
PROCESS_FIELDS_BY_TYPE = {
    "install": INSTALL_FIELDS,
    "agent.started": AGENT_FIELDS,
    "agent.completed": AGENT_FIELDS,
    "dataset": DATASET_FIELDS,
}

# ----------------------------------------------------------- product analytics
#
# The exported representation of a product-analytics event, restated here from
# the producer's schema rather than imported from it, for the reason given at
# the top of this file. tests/telemetry_ingest/test_analytics_contract.py
# imports both sides and fails when they disagree, in either direction.
#
# Every string below is a member of a closed vocabulary, an opaque id, a class
# name or a version. There is no free-text field, so there is nowhere for a
# question, a column name or a file name to travel. A route is a *template*
# ("/reports/$reportId"), never a concrete path.

#: Keeps these records apart from the four process types above.
ANALYTICS_PREFIX = "analytics."

_ROUTES = _enum(
    "/", "/analysis", "/configurations", "/dashboard", "/database", "/datasets",
    "/db-tables", "/logout", "/proanalysis", "/proanalysis/scheduled-analysis",
    "/reports", "/reports/$reportId", "/reports/generated", "/scheduled-analysis",
    "/settings", "/shared-report/$token", "/shared/$token", "/support",
    "/support-queries", "other",
)
_SOURCE_KINDS = _enum("upload", "database", "gcs", "s3", "azure", "sample", "other")
_CHART_TYPES = _enum("bar", "line", "area", "scatter", "pie", "histogram", "box",
                     "heatmap", "table", "other")
_SURFACES = _enum("analysis", "proanalysis", "datasets", "database", "dashboard",
                  "reports", "scheduled_analysis", "settings", "configurations",
                  "support", "shared", "other")

#: On every analytics record. `actor_id` is not the identifier the deployment
#: stores: it is re-keyed every 28 days with a secret that never leaves the
#: deployment, so it cannot be joined to a user, to the stored id, or to the
#: same analyst in the next window. `install_id` is minted for the export.
ANALYTICS_COMMON = {
    "event_id": _ID,
    "seq": _SEQ,
    "t_ms": _int(0, MAX_T_MS),
    "route": _ROUTES,
    "session_id": _ID,
    "install_id": _ID,
    "actor_id": _ID,
    "schema_version": Spec("schema_version"),
    "ui_version": _UI_VERSION,
}

#: event type (unprefixed) -> the fields of its own it may carry.
ANALYTICS_EVENTS = {
    "session.started": {
        "entry": _enum("direct", "login", "shared_link", "reload"),
        "viewport": _enum("xs", "sm", "md", "lg", "xl"),
    },
    "session.ended": {
        "reason": _enum("pagehide", "logout", "idle_timeout"),
        "active_ms": _int(0, MAX_T_MS),
    },
    "auth.completed": {
        "method": _enum("password", "oauth", "sso", "magic_link"),
    },
    "page.viewed": {
        "from_route": _ROUTES,
    },
    "dataset.connect_started": {
        "source_kind": _SOURCE_KINDS,
    },
    "dataset.connect_completed": {
        "source_kind": _SOURCE_KINDS,
        "outcome": _OUTCOMES,
        "file_format": _FILE_FORMATS,
        "row_count_bucket": _enum("0", "1-1e2", "1e2-1e3", "1e3-1e4", "1e4-1e5",
                                  "1e5-1e6", "1e6-1e7", "1e7-1e8", "1e8+"),
        "column_count_bucket": _enum("0", "1-10", "10-50", "50-200", "200-1000",
                                     "1000+"),
        "duration_ms": _DURATION,
        "error_class": _CLASS,
    },
    "question.submitted": {
        "turn_id": _ID,
        "input_method": _enum("typed", "suggestion", "rerun", "edit_resubmit"),
        "turn_index": _int(0, 10_000),
        "is_retry": _BOOL,
        "retry_of": _ID,
        # All that is said about the question itself: which of six length
        # bands it fell in. Not the text, not a count of what was in it.
        "prompt_chars_bucket": _enum("0", "1-20", "21-80", "81-200", "201-500", "500+"),
    },
    "question.completed": {
        "turn_id": _ID,
        "outcome": _OUTCOMES,
        "duration_ms": _DURATION,
        "first_response_ms": _DURATION,
        "error_class": _CLASS,
        "result_kind": _enum("table", "chart", "text", "model", "file", "none"),
        "chart_type": _CHART_TYPES,
        "retry_count": _int(0, 100),
    },
    "result.interacted": {
        "turn_id": _ID,
        "action": _enum("chart_type_changed", "code_viewed", "table_sorted",
                        "table_paged", "expanded", "copied", "exported", "shared"),
        "chart_type": _CHART_TYPES,
        "export_format": _enum("csv", "xlsx", "png", "pdf", "json", "html"),
    },
    "feedback.submitted": {
        "turn_id": _ID,
        "feedback_type": _enum("positive", "negative"),
        "has_comment": _BOOL,           # never the comment itself
    },
    "error.shown": {
        "surface": _SURFACES,
        "error_class": _CLASS,
        "http_status": _int(100, 599),
        "turn_id": _ID,
    },
    "feature.used": {
        "feature": _enum(
            "auto_insights", "scheduled_analysis", "report_generate", "report_view",
            "share_link_create", "email_report", "dataset_preview", "db_table_browse",
            "model_training", "code_view", "suggestion_chip", "conversation_new",
            "conversation_resume", "settings_change", "support_query",
        ),
        "surface": _SURFACES,
    },
}

#: wire event type -> {field: Spec}, the envelope included.
ANALYTICS_FIELDS_BY_TYPE = {
    ANALYTICS_PREFIX + name: {**ANALYTICS_COMMON, **own}
    for name, own in ANALYTICS_EVENTS.items()
}

#: Every accepted event type, and what each of its fields may hold.
FIELDS_BY_TYPE = {**PROCESS_FIELDS_BY_TYPE, **ANALYTICS_FIELDS_BY_TYPE}

#: Every accepted event type, and the keys a record of that type may carry.
KEYS_BY_TYPE = {
    name: frozenset(fields) | {"event_type"} for name, fields in FIELDS_BY_TYPE.items()
}

#: Every key permitted on any type. Not what a record is checked against --
#: that is its own type's set -- but what the refused list is checked against.
ALLOWED_KEYS = frozenset().union(*KEYS_BY_TYPE.values())

#: Refused by name, with a reason that says why. An unknown key is rejected
#: anyway; these are rejected *first* and separately, for two reasons. A client
#: that sends one is not merely ahead of the server, it is sending customer
#: content, and the rejection should say so. And a name on this list can never
#: be added to an accepted set by accident: the check below stops the service
#: starting if one is. The server must not be able to accept what the client
#: must not send.
#:
#: What they have in common is that a customer or a model wrote the value:
#: question text in any form, the names of columns, tables, files and datasets,
#: error text, output, and anything that identifies a person, an organisation
#: or a machine.
REFUSED_KEYS = frozenset({
    # question text, in every form it has ever had a name for
    "prompt", "prompt_stripped", "prompt_export", "prompt_redactions",
    "prompt_names_exact", "prompt_names_common_word", "prompt_names_substring",
    "question", "query", "goal", "notes", "comment",
    # schema: names describe the business and sometimes the individual
    "column_names", "columns", "schema", "table_name", "tables",
    "dataset_name", "alias",
    # where the data lives
    "file_name", "filename", "file_path", "path", "bucket", "data_source",
    "connection_string", "url", "host", "hostname",
    # what went wrong, in words
    "error_message", "error", "message", "traceback", "stack_trace",
    # what came out
    "generated_code", "code", "output", "result", "rows", "values",
    # who
    "user", "user_id", "username", "email", "name", "tenant", "org",
    "owner_id", "profile_id",
    "ip", "ip_address", "user_agent", "referrer", "locale", "timezone",
    "gcp_project", "aws_account_id", "azure_subscription_id",
    # identifiers that outlive a session and so link one person's activity
    "mission_id", "conversation_id", "thread_id", "analysis_id", "report_id",
    "token",
    # exact sizes: close to a fingerprint for one specific file
    "row_count", "column_count",
})

if ALLOWED_KEYS & REFUSED_KEYS:
    # Not an assert: asserts vanish under -O, and this must hold in production.
    raise RuntimeError(
        "refused field(s) on an accepted list: "
        + ", ".join(sorted(ALLOWED_KEYS & REFUSED_KEYS))
    )

#: A record must declare which kind it is, and the value must be known. An
#: unrecognised type means the client is ahead of the server, which is a
#: deployment problem worth surfacing rather than silently storing.
EVENT_TYPES = frozenset(KEYS_BY_TYPE)

MAX_RECORDS_PER_BATCH = 10_000


class Rejected(Exception):
    """A payload the server refuses. The client must not retry it."""

    def __init__(self, reason: str, record_index: int | None = None) -> None:
        self.reason = reason
        self.record_index = record_index
        super().__init__(reason)

    def as_dict(self) -> dict:
        detail = {"error": "rejected", "reason": self.reason}
        if self.record_index is not None:
            detail["record"] = self.record_index
        return detail


def _check_value(key: str, spec: Spec, value, index: int) -> None:
    """One value against its Spec. Never echoes the value: a rejected string
    may be exactly the thing that must not be logged."""
    kind = spec.kind
    if kind == "bool":
        ok = isinstance(value, bool)
    elif kind == "int":
        # bool is an int in Python; a flag is not a measurement.
        ok = (isinstance(value, int) and not isinstance(value, bool)
              and spec.lo <= value <= spec.hi)
    elif kind == "enum":
        # Type first: an integer must not get past a vocabulary check that a
        # wrong word would fail.
        ok = isinstance(value, str) and value in spec.values
    elif kind == "counts":
        # {"numeric": 12, "text": 3}: how many columns of each kind. The keys
        # are a vocabulary, not a shape -- this is where a column name would
        # otherwise travel.
        ok = isinstance(value, dict) and all(
            isinstance(name, str) and name in spec.values
            and isinstance(count, int) and not isinstance(count, bool)
            and spec.lo <= count <= spec.hi
            for name, count in value.items()
        )
    else:
        pattern = {"id": _OPAQUE_ID, "class": _CLASS_NAME, "version": _VERSION,
                   "release_version": _RELEASE_VERSION,
                   "schema_version": _SCHEMA_VERSION}[kind]
        ok = isinstance(value, str) and bool(pattern.match(value))
    if not ok:
        raise Rejected(f"{key!r} is not a valid {kind}", index)


def validate(record: dict, index: int = 0) -> dict:
    """Return the record if it is acceptable, otherwise raise :class:`Rejected`.

    Rejection rather than filtering is deliberate. A record containing a field
    nobody approved means something upstream is sending more than it should,
    and that must be visible immediately, not quietly discarded.
    """
    if not isinstance(record, dict):
        raise Rejected("record is not an object", index)

    event_type = record.get("event_type")
    if not isinstance(event_type, str):
        # Checked before the lookup: a list or an object here is unhashable,
        # and the lookup raising would turn a malformed record into a 500 that
        # the client retries forever.
        raise Rejected("event_type is missing or not a string", index)
    if event_type not in KEYS_BY_TYPE:
        raise Rejected(f"unknown event_type {event_type[:64]!r}", index)

    if "event_id" not in record:
        raise Rejected("event_id is missing", index)

    refused = set(record) & REFUSED_KEYS
    if refused:
        # Before the general check, and worded differently: this is not a
        # client that is ahead of the server, it is one sending content. Only
        # the names are reported. The values are never read.
        raise Rejected(
            "refused field(s): " + ", ".join(sorted(refused)[:5])
            + " -- customer content is never accepted", index)

    unknown = set(record) - KEYS_BY_TYPE[event_type]
    if unknown:
        # Named so a client bug is diagnosable — but capped and truncated,
        # because these names are attacker-chosen text that ends up in our
        # logs and in the response body.
        shown = ", ".join(name[:64] for name in sorted(unknown)[:5])
        if len(unknown) > 5:
            shown += f" (+{len(unknown) - 5} more)"
        raise Rejected(f"unpermitted field(s): {shown}", index)

    fields = FIELDS_BY_TYPE[event_type]
    for key, value in record.items():
        if key != "event_type":
            _check_value(key, fields[key], value, index)

    return record


def validate_batch(records: list) -> list[dict]:
    """Validate a whole batch. One bad record rejects the batch.

    All-or-nothing keeps the client's accounting simple: a batch is either
    stored or it is not, so a partial success can never leave the client's
    cursor in a state that skips records.
    """
    if not isinstance(records, list):
        raise Rejected("payload is not a list of records")
    if not records:
        raise Rejected("payload is empty")
    if len(records) > MAX_RECORDS_PER_BATCH:
        raise Rejected(f"batch exceeds {MAX_RECORDS_PER_BATCH} records")
    return [validate(record, index) for index, record in enumerate(records)]
