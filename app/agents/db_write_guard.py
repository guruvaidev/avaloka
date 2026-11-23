"""A database write happens through the transfer path, or not at all.

rc10 web check, D1 e_expiry: the user cancelled a transfer, asked an unrelated
question, then said "yes" twice. The sticky-cancel marker (#570) had expired,
the planner took "yes" as consent, and the analysis coder -- which sees the
thread history -- wrote a blueprint that opened a connection to the warehouse
and appended the table with ``to_sql``. No rows were written only because the
model then chose to raise instead of writing. The coder sandbox has the
drivers; nothing else stood in the way, and no confirmation had been shown.

A write to a database is the data transfer agent's job: it shows
"❓ Copy A → B (…)?" and runs only on a plain yes. So the analysis path refuses
one at three places, each deterministic:

1. the planner: a ``generate_code`` plan that describes a database write is not
   sent to the coder (``describes_database_write`` over the plan text);
2. the coder: the same check over its own pseudocode blueprint, which is where
   rc10's write first appeared -- the planner's plan was "stroke rate by gender";
3. the static validator: ``DatabaseWriteVisitor`` rejects code that imports a
   database driver, opens an engine or connection, or runs write SQL.

The prose check (1, 2) reads plans and blueprints, not code, so it keys on what
a plan says when it means a write: ``to_sql``/``to_gbq`` or a MongoDB write
method named, a write statement spelled as SQL (INSERT/UPDATE/DELETE/MERGE/
TRUNCATE/ALTER, or CREATE/DROP of a table aimed at a connection), or a write
verb aimed INTO a database noun. Reading from a database ("read the patients
table from the database") never matches.

rc11 web check: the first version also treated any driver name (psycopg2,
sqlalchemy) and any upper-case CREATE TABLE as a write, and refused 5 of 9 read
questions. The memory plane's session signature ("Use SQL to copy and rename
the table ...") leads the planner to write read plans that open an engine and
``read_sql`` the table, or stage it in a scratch ``CREATE TABLE d1_b AS SELECT``.
Neither writes anything; and the code cannot do either anyway -- the validator's
NoDatabaseWrites rule rejects any driver import, and the executor has no
database. So a driver named, a read, or a scratch table is not a write here.
"""

from __future__ import annotations

import re
from typing import Optional

# The rule's name, as the user and the coder's retry feedback see it.
DB_WRITE_RULE = "NoDatabaseWrites"

# What the user is told when a write is stopped before any code runs. It is a
# question, so the turn ends waiting on the user rather than on a transfer.
DB_WRITE_REFUSAL = (
    "That would write into a database, and I only do that through a transfer you "
    "confirm — nothing was written. Tell me what to copy and where, for example "
    "\"Copy the <table> table to <connection> as <new table>\"."
)

# Calls that write, whatever the driver: pandas and BigQuery loads, MongoDB
# writes. Naming one in a plan or blueprint is the write itself. A driver or ORM
# name alone (psycopg2, sqlalchemy, create_engine) is not: read plans name them
# to read_sql the table, and the validator stops any code that imports one.
_WRITE_CALL_RE = re.compile(
    r"\b(?:to_sql|to_gbq|insert_one|insert_many|bulk_write|update_one|update_many|"
    r"delete_one|delete_many|replace_one)\b",
    re.IGNORECASE,
)

# Write statements as SQL spells them. Upper case only: a plan that says
# "truncate long labels" or "update the chart" is prose, not SQL.
_SQL_WRITE_RE = re.compile(
    r"\b(?:INSERT\s+INTO|UPDATE\s+[\w.\"`]+\s+SET|DELETE\s+FROM|"
    r"ALTER\s+TABLE|TRUNCATE(?:\s+TABLE)?|MERGE\s+INTO|REPLACE\s+INTO)\b"
)

# CREATE/DROP is a write only when what it makes or drops persists somewhere the
# user owns: a database or schema, or a table named through a connection or
# schema ("wh.d1_new", "db://e2ewhgroq/d1"). A scratch "CREATE TABLE d1_b AS
# SELECT ..." or a TEMP table or view is how a read plan stages a query.
_SQL_DDL_RE = re.compile(
    r"\b(?P<verb>CREATE|DROP)\s+(?:OR\s+REPLACE\s+)?(?P<temp>(?:GLOBAL\s+|LOCAL\s+)?TEMP(?:ORARY)?\s+)?"
    r"(?P<kind>TABLE|VIEW|MATERIALIZED\s+VIEW|SCHEMA|DATABASE|INDEX)\s+"
    r"(?:IF\s+(?:NOT\s+)?EXISTS\s+)?(?P<name>[\w\"`\[\]./:-]+)"
)

# Where a write lands: a database, a warehouse, a connection named as the
# target, or a db:// URI.
_DB_NOUNS = (r"database|databases|warehouse|data\s+warehouse|db|postgres|postgresql|mysql|"
             r"mariadb|sql\s+server|snowflake|bigquery|redshift|mongo|mongodb|oracle|sqlite|"
             r"(?:target|destination|remote|warehouse|database|db)\s+(?:table|connection|db|database|schema)|"
             r"db://")

# A CREATE/DROP whose clause names where it lands: "... on the target
# connection", "... in the warehouse".
_DDL_TARGET_RE = re.compile(
    r"\b(?:on|in|into|to)\s+(?:the\s+)?(?:" + _DB_NOUNS + r")",
    re.IGNORECASE,
)

# A write verb aimed into a database noun, within one clause: "append the rows
# to the target table", "load it into the warehouse", "export to Postgres".
# "into"/"to" only, and the words between it and the noun never include "from",
# "of", "using", "via" or "with": "load the rows from the database into a
# DataFrame" and "load it into a DataFrame using the connection" are reads.
# "to keep the database tidy" is a purpose, not a destination.
# "push" is not a write verb here: "push the aggregation down to the database"
# is how a plan describes a read.
_WRITE_VERBS = (r"write|writes|writing|insert|inserts|inserting|load|loads|loading|copy|copies|"
                r"copying|transfer|transfers|transferring|export|exports|exporting|save|saves|"
                r"saving|append|appends|appending|upload|uploads|uploading|replicate|replicates|"
                r"persist|persists|store|stores|migrate|migrates|sync|syncs")
# "to" that opens a purpose, not a destination: "drop the copy table to keep
# the database tidy".
_PURPOSE_VERBS = (r"keep|avoid|free|make|ensure|prevent|clean|tidy|reduce|preserve|match|mirror|"
                  r"reflect|limit|spare|release|protect|leave|stay|let|help|see|check|verify|confirm")
_PROSE_WRITE_RE = re.compile(
    r"\b(?:" + _WRITE_VERBS + r")\b(?:://|[^.;:\n]){0,80}?\b(?:into|to)\s+"
    r"(?!(?:" + _PURPOSE_VERBS + r")\b)"
    r"(?:(?!(?:from|of|using|via|with)\b)[^.;:\n\s]+\s+){0,4}?(?:" + _DB_NOUNS + r")\b",
    re.IGNORECASE,
)

# Opening a connection TO the place a transfer writes: "open a psycopg2
# connection to the warehouse", "a SQLAlchemy engine for the destination". A
# connection to read from is described from the data's side ("read the table
# from the database"), and passes.
_CONNECT_TO_TARGET_RE = re.compile(
    r"\b(?:connection|engine)\s+(?:to|for|on)\s+(?:the\s+)?(?:(?:target|destination)\b|warehouse\b)",
    re.IGNORECASE,
)

_CLAUSE_END = re.compile(r"[.;\n]")
_FROM_RE = re.compile(r"\bfrom\b", re.IGNORECASE)

# Plans put SQL in markdown table cells, where a line break is spelled "<br>"
# or a literal backslash-n: "```sql\\nDROP TABLE wh.t;\\n```". Read both as
# line breaks, or "\\nDROP" has no word boundary before DROP.
_SOFT_BREAK_RE = re.compile(r"<br\s*/?>|\\n", re.IGNORECASE)


def _ddl_write(text: str) -> Optional[str]:
    """A CREATE/DROP that persists into a database or connection, else None."""
    for m in _SQL_DDL_RE.finditer(text):
        if m.group("temp"):
            continue
        kind = m.group("kind").upper()
        name = m.group("name").strip('"`[]')
        if kind in ("SCHEMA", "DATABASE") or "." in name or "://" in name:
            return m.group(0)
        # The rest of the clause, after the object's name and before any FROM
        # (that side is the source): "... on the target connection".
        tail = text[m.end():]
        stop = min((x for x in (_CLAUSE_END.search(tail), _FROM_RE.search(tail)) if x),
                   key=lambda x: x.start(), default=None)
        if _DDL_TARGET_RE.search(tail[: stop.start() if stop else len(tail)]):
            return m.group(0)
    return None


def describes_database_write(text: Optional[str]) -> Optional[str]:
    """The phrase in a plan or blueprint that writes to a database, else None."""
    if not text:
        return None
    text = _SOFT_BREAK_RE.sub("\n", str(text))
    for pattern in (_WRITE_CALL_RE, _SQL_WRITE_RE):
        m = pattern.search(text)
        if m:
            return m.group(0)
    ddl = _ddl_write(text)
    if ddl:
        return ddl
    for pattern in (_PROSE_WRITE_RE, _CONNECT_TO_TARGET_RE):
        m = pattern.search(text)
        if m:
            return m.group(0)
    return None
