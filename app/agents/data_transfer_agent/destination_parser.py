"""Where a database transfer is going: the connection, and the table in it.

One parser for every phrasing that names BOTH the destination connection and
its table, shared by the 1.6 and 2.x planners so the two lines cannot drift:

    transfer to churn_dst table customers_copy       -> churn_dst / customers_copy
    transfer to targetdb new table X                 -> targetdb / X (create)
    copy it into the warehouse db, table sales.orders -> warehouse / sales.orders
    transfer into churn_dst.customers_copy           -> churn_dst / customers_copy
    transfer into churn_dst.sales.customers_copy     -> churn_dst / sales.customers_copy
    move it to a new table "Customers Copy" in dst   -> dst / Customers Copy (create)

The older extractors split this clause in two and got both halves wrong: the
table regex read "to <X> table" as a table named X ("Table churn_dst doesn't
exist in churn_dst"), and the destination regex stopped only at " table", so
"to targetdb new table X" named a connection "targetdb new". Reading the clause
as one unit is what fixes both.

Pure: no I/O, no app imports. ``parse_db_destination`` returns ``None`` for
anything it cannot read whole (a bucket path, a file name, a column phrase like
"move revenue to profit"), and the caller falls back to its older parsers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# A name is quoted (any characters but the quote) or a bare SQL identifier.
_QUOTED = r'"[^"\n]+"|`[^`\n]+`|\'[^\'\n]+\''
_BARE_IDENT = r"[A-Za-z_][A-Za-z0-9_$]*"
_PART = rf"(?:{_QUOTED}|{_BARE_IDENT})"
# Optional schema: "sales.orders", "\"sales\".\"Orders\"".
# It ends at a word boundary; a full stop ends it too ("... table orders.").
_QUALIFIED = rf"(?P<name>{_PART}(?:\.{_PART})?)(?![\w$/]|\.[\w\"'`])"

# Words that are grammar, never a connection-name word or a table name.
_RESERVED = {
    "a", "an", "the", "this", "that", "my", "our", "it", "its", "them",
    "to", "into", "in", "on", "at", "as", "from", "of", "for", "by",
    "table", "new", "fresh", "database", "db", "connection", "schema",
    "and", "or", "with", "where", "keep", "keeping", "only", "filter",
    "filtering", "then", "called", "named", "data", "result", "results",
    "output", "file", "bucket", "folder", "please",
    # Navigation and pronouns: "let's move to the next table please".
    "next", "previous", "prev", "first", "last", "beginning", "start", "end",
    "me", "us", "you", "him", "her",
}
# A bare connection-name word: letters, digits, _ and - (no slashes: those are
# bucket paths). Before an explicit "table", inner dots are part of the name
# ("to db.example.com table x"); without one, "X.Y" is the dotted form below.
_CONN_WORD = r"[A-Za-z0-9_][A-Za-z0-9_\-]*"
_CONN_WORD_DOTS = r"[A-Za-z0-9_](?:[A-Za-z0-9_\-.]*[A-Za-z0-9_])?"
# One to four words, so an unquoted multi-word name ("My Warehouse") still reads.
_CONN = rf"(?P<conn>{_QUOTED}|{_CONN_WORD_DOTS}(?:\s+{_CONN_WORD_DOTS}){{0,3}}?)"
_CONN_NOUN = r"(?:(?:database|db|connection)\s+)"
_NEW = r"(?P<new>(?:a\s+)?(?:new|fresh)\s+)"
_ARTICLE = r"(?:(?:the|a|an|my|this)\s+)"
# Before a connection only "the" is filler: "My Warehouse" is a connection's name.
_CONN_ARTICLE = r"(?:the\s+)"

# File extensions: "into out.csv" is an object path, not <connection>.<table>.
_FILE_EXTS = {"csv", "json", "parquet", "pq", "tsv", "xlsx", "xls", "xml", "avro", "orc", "txt", "jsonl"}

# "to <conn> [db] [, ] [in|into|to|as] [the|a] [new] table <name>"
_CONN_THEN_TABLE = re.compile(
    rf"\b(?:to|into)\s+{_CONN_ARTICLE}?{_CONN_NOUN}?{_CONN}(?:\s+(?:database|db|connection)\b)?"
    rf"\s*,?\s*(?:(?:in|into|to|as)\s+)?{_ARTICLE}?{_NEW}?table\s+{_QUALIFIED}",
    re.IGNORECASE,
)
# "to|into [the|a] [new] table <name> in|on|at [the] [db] <conn>"
_TABLE_THEN_CONN = re.compile(
    rf"\b(?:to|into)\s+{_ARTICLE}?{_NEW}?table\s+{_QUALIFIED}"
    rf"\s+(?:in|on|at|inside)\s+{_CONN_ARTICLE}?{_CONN_NOUN}?(?P<conn>{_QUOTED}|{_CONN_WORD_DOTS})"
    r"(?![\w\-/]|\.[\w])",
    re.IGNORECASE,
)
# "into <conn>.<table>" / "into <conn>.<schema>.<table>". Only after "into":
# "to db1.internal" is far more often a host-style connection name than a table.
_DOTTED = re.compile(
    rf"\binto\s+{_CONN_ARTICLE}?(?P<conn>{_QUOTED}|{_CONN_WORD})\.{_QUALIFIED}"
    # "to db.example.com table x" names the table explicitly: that is the first form.
    r"(?!\s+(?:(?:a\s+)?new\s+)?table\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DbDestination:
    """A destination connection plus a table in it, as the user named them."""

    connection: str
    table: str
    schema: Optional[str] = None
    create_new: bool = False
    #: True for the "<conn>.<table>" form. A registered connection whose own
    #: name contains a dot ("db.example.com") must win over this reading.
    dotted: bool = False
    #: The whole dotted token, e.g. "db.example.com"; "" for the other forms.
    raw: str = ""
    #: Offset just past the connection name in the text.
    connection_end: int = 0

    @property
    def qualified_table(self) -> str:
        return f"{self.schema}.{self.table}" if self.schema else self.table


def _unquote(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'`":
        return token[1:-1].strip()
    return token


def _split_qualified(name: str):
    """'sales.orders' -> ('sales', 'orders'); quotes may hold dots, so split outside them."""
    m = re.fullmatch(rf"({_PART})(?:\.({_PART}))?", name.strip())
    if not m:
        return None, _unquote(name)
    if m.group(2):
        return _unquote(m.group(1)), _unquote(m.group(2))
    return None, _unquote(m.group(1))


def _bare_ok(token: str) -> bool:
    """A bare (unquoted) name must not be a grammar word."""
    return token.strip("\"'`").lower() not in _RESERVED


def _conn_ok(conn: str) -> bool:
    if conn[:1] in "\"'`":
        return bool(_unquote(conn))
    words = conn.split()
    # "My Warehouse" is a name; a lone "my" ("to my db table x") is not.
    if len(words) > 1 and words[0].lower() in ("my", "our"):
        words = words[1:]
    return all(_bare_ok(w) for w in words)


def _name_ok(name: str) -> bool:
    schema, table = _split_qualified(name)
    parts = [p for p in (schema, table) if p is not None]
    if not parts or not all(parts):
        return False
    raw_parts = re.findall(_PART, name)
    return all(p[:1] in "\"'`" or _bare_ok(p) for p in raw_parts)


def _from_match(m: "re.Match", *, dotted: bool = False) -> Optional[DbDestination]:
    conn, name = m.group("conn"), m.group("name")
    if not _conn_ok(conn) or not _name_ok(name):
        return None
    schema, table = _split_qualified(name)
    if dotted:
        last = table.lower()
        if last in _FILE_EXTS:
            return None
    new = m.groupdict().get("new")
    return DbDestination(
        connection=_unquote(conn),
        table=table,
        schema=schema,
        create_new=bool(new),
        dotted=dotted,
        raw=f"{_unquote(conn)}.{name}" if dotted else "",
        connection_end=m.end("conn"),
    )


def parse_db_destination(text: str, start: int = 0) -> Optional[DbDestination]:
    """The destination connection and table named in ``text``, or ``None``.

    ``start`` skips a prefix (a caller that has already consumed the source
    clause). The earliest well-formed clause wins; a clause whose words are
    grammar ("to the table this") is skipped, not accepted.
    """
    text = text or ""
    best = None
    for rx, dotted in ((_CONN_THEN_TABLE, False), (_TABLE_THEN_CONN, False), (_DOTTED, True)):
        pos = start
        while True:
            m = rx.search(text, pos)
            if not m:
                break
            # A dotted token that runs on into a path or URI is a bucket object.
            if dotted and re.match(r"[/:]", text[m.end():m.end() + 1] or " "):
                pos = m.start() + 1
                continue
            found = _from_match(m, dotted=dotted)
            if found:
                if best is None or m.start() < best[0]:
                    best = (m.start(), found)
                break
            pos = m.start() + 1
    return best[1] if best else None
