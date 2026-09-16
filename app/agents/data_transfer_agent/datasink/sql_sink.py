import logging
import os
import uuid
import sqlalchemy as sa
from typing import Iterator
from daft.io import DataSink
from daft.io.sink import WriteResult
from daft.recordbatch import MicroPartition
import daft

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SQLSink")


def _downcast_unsigned_ints(df):
    """Downcast every UNSIGNED integer column to signed int64, in place.

    pandas ``to_sql`` cannot map an unsigned 64-bit integer to a SQL type — it
    raises ``ValueError: Unsigned 64 bit integer datatype is not supported`` while
    building the table. Daft's ``count()`` / ``count_distinct()`` / ``rank()`` /
    ``row_number()`` aggregations all produce ``UInt64``, so any aggregation transfer
    would crash at the sink. This is the runtime backstop for the coder prompt's cast
    rule (AGG-4b) — the write succeeds even if the generated code forgot to cast.
    ``dtype.kind == "u"`` covers uint8/16/32/64; such columns are non-null (nulls make
    arrow→pandas pick a float dtype), so the cast never fails.

    Only values that provably fit are cast: a genuine uint64 >= 2**63 would wrap
    negative SILENTLY, so it keeps its dtype and hits the loud ValueError instead.
    """
    for c in df.columns:
        try:
            col = df[c]
            if col.dtype.kind != "u":
                continue
            if len(col) and col.max() >= 2**63:
                logger.error(
                    "Column %r holds unsigned values >= 2**63; refusing to downcast to "
                    "int64 (it would wrap negative and silently corrupt the data). "
                    "pandas to_sql will reject this column.", c,
                )
                continue
            df[c] = col.astype("int64")
        except Exception:  # never let the guard itself break an otherwise-valid write
            logger.warning("Could not downcast unsigned column %r; leaving as-is.", c)
    return df


class BaseSQLDataSink(DataSink[dict]):
    """SQL sink with an atomic staging swap.

    Instead of every worker appending straight into the target table, each
    partition writes into a private staging table. ``finalize()`` then copies
    the staged rows into the target in a SINGLE transaction. This buys two
    things the old append-in-place sink lacked:

    * ``write_mode="overwrite"`` actually replaces the destination data
      (the target is cleared and refilled inside that one transaction).
    * Atomicity — the target is only ever touched in ``finalize()``, so a
      failure during the distributed write leaves it completely untouched
      (no half-written table).

    If the staging table cannot be prepared (unknown columns, target missing,
    permission issue, a case-mismatch between generated and destination
    columns, …) the sink transparently falls back to the legacy direct append
    so the common path is never worse than before.
    """

    # Identifier quote character — overridden per dialect.
    _ident_quote = '"'

    # A tiny bookkeeping table in the DESTINATION db recording which transfer jobs
    # have already committed, so a whole-pipeline retry (see container_file.py) never
    # re-inserts the same rows. Shared across all jobs; keyed by the container JOB_ID.
    _ledger_table = "_dta_completed_jobs"

    def __init__(self, connection_url: str, table_name: str, connect_args: dict = None,
                 write_mode: str = "append", columns=None, job_id: str = None):
        self.connection_url = connection_url
        self.target_table = table_name
        self.write_mode = (write_mode or "append").strip().lower()
        self.columns = [str(c) for c in columns] if columns else []
        # Idempotency key for retry-safe append. Defaults to the container's JOB_ID
        # (set by the runner) so a re-run of run_pipeline() dedupes against the first
        # committed finalize. Empty string disables the ledger (behavior unchanged).
        self.job_id = (job_id if job_id is not None else os.environ.get("JOB_ID") or "").strip()

        raw_args = connect_args or {}
        self.sslmode = raw_args.pop("sslmode", None)
        self.extra_connect_args = raw_args

        self._result_schema = daft.Schema._from_field_name_and_types([
            ("rows_inserted", daft.DataType.int64()),
            ("status", daft.DataType.string()),
        ])

        # All writes go through a staging table; the target is only touched by
        # the single transaction in finalize(). A failure BEFORE finalize leaves the
        # target untouched, and a job-id ledger stamped INSIDE the finalize
        # transaction makes a retry AFTER a committed finalize a no-op — so a
        # whole-pipeline retry is idempotent in either window (no duplicate append).
        # There is deliberately no silent direct-append fallback (non-atomic,
        # duplicates on retry, ignores write_mode) — we raise instead.
        self._staging_table = None
        if not self.columns:
            # No generated schema — reflect the target's insertable columns so
            # staging can still be used instead of an unsafe direct append.
            self.columns = self._reflect_insertable_columns()
        if not self.columns:
            raise RuntimeError(
                f"[SQLSink] Could not determine any insertable columns for target "
                f"'{self.target_table}'. Aborting instead of appending directly, "
                f"which would duplicate rows on retry and ignore write_mode."
            )
        try:
            self._prepare_staging()
        except Exception as e:
            raise RuntimeError(
                f"[SQLSink] Could not prepare an atomic staging table for target "
                f"'{self.target_table}': {e}. The transfer was aborted instead of "
                f"appending directly (which would duplicate rows if the run is "
                f"retried, and would ignore write_mode). Ensure the destination "
                f"user can CREATE TABLE and that the generated columns match the "
                f"destination table (check column-name case)."
            ) from e
        self._use_staging = True
        self._write_table = self._staging_table

    # ---- helpers -------------------------------------------------------------

    def _qi(self, name: str) -> str:
        """Quote an identifier for this dialect (doubling embedded quotes)."""
        q = self._ident_quote
        return f"{q}{str(name).replace(q, q + q)}{q}"

    def _qt(self, name: str) -> str:
        """Quote a TABLE reference: ``public.orders`` → ``"public"."orders"``.

        Tables only — column names may legitimately contain a dot (use ``_qi``).
        """
        return ".".join(self._qi(part) for part in str(name).split("."))

    def _col_list(self) -> str:
        return ", ".join(self._qi(c) for c in self.columns)

    def _build_connect_args(self) -> dict:
        args = dict(self.extra_connect_args)
        if self.sslmode == "require":
            args["sslmode"] = "require"
        elif self.sslmode == "verify-full":
            args["sslmode"] = "verify-full"
        return args

    def _engine(self):
        return sa.create_engine(
            self.connection_url,
            connect_args=self._build_connect_args(),
            pool_pre_ping=True,
            pool_size=1,
            max_overflow=0,
        )

    # ---- staging lifecycle (driver, once) ------------------------------------

    def _reflect_insertable_columns(self) -> list[str]:
        """Reflect the target's columns, excluding identity / auto-increment /
        computed columns the destination populates itself.

        Used only when the generated output schema wasn't captured (the sink was
        constructed with an empty ``columns`` list). Recovering a column list
        here lets the write still go through the atomic staging table — keeping
        it retry-safe — instead of falling back to an unsafe direct append.
        Returns an empty list if reflection fails (the caller then aborts).
        """
        engine = self._engine()
        try:
            cols = sa.inspect(engine).get_columns(self.target_table)
        except Exception as e:
            logger.warning("[SQLSink] Could not reflect columns for '%s': %s",
                           self.target_table, e)
            return []
        finally:
            engine.dispose()

        insertable: list[str] = []
        for c in cols:
            if c.get("autoincrement") is True or c.get("identity") or c.get("computed"):
                continue
            default = c.get("default")
            if default is not None and "nextval" in str(default).lower():
                continue
            insertable.append(str(c["name"]))
        return insertable

    def _prepare_staging(self):
        self._staging_table = f"_dta_stg_{uuid.uuid4().hex[:12]}"
        tgt = self._qt(self.target_table)
        stg = self._qi(self._staging_table)
        cols = self._col_list()
        engine = self._engine()
        try:
            with engine.begin() as conn:
                conn.exec_driver_sql(f"DROP TABLE IF EXISTS {stg}")
                # Clone ONLY the generated columns, with the target's types but no
                # constraints/defaults — so the destination's own defaults and
                # auto-increment columns still apply during the final insert.
                conn.exec_driver_sql(
                    f"CREATE TABLE {stg} AS SELECT {cols} FROM {tgt} WHERE 1=0"
                )
            logger.info("[SQLSink] Staging table %s ready for target %s.",
                        self._staging_table, self.target_table)
        finally:
            engine.dispose()

    def _ensure_ledger(self, engine) -> None:
        """Create the idempotency ledger table if it doesn't exist.

        Failure to maintain the ledger must not fail the transfer — we log and
        disable the ledger for this run (``self.job_id = ""``), degrading to the
        previous, non-deduped behavior rather than aborting.
        """
        ledger = self._qi(self._ledger_table)
        try:
            with engine.begin() as conn:
                # PK matches the finalize() lookup: one job id per target table.
                conn.exec_driver_sql(
                    f"CREATE TABLE IF NOT EXISTS {ledger} ("
                    f"job_id VARCHAR(128) NOT NULL, "
                    f"target_table VARCHAR(256) NOT NULL, "
                    f"rows_inserted BIGINT, "
                    f"completed_at TIMESTAMP, "
                    f"PRIMARY KEY (job_id, target_table))"
                )
            # An earlier ledger schema had no target_table column and CREATE IF NOT
            # EXISTS won't add it — finalize()'s lookup would then fail every transfer
            # to this destination. Widen in place; old rows keep NULL.
            try:
                with engine.begin() as conn:
                    conn.exec_driver_sql(f"SELECT target_table FROM {ledger} WHERE 1=0")
            except Exception:
                with engine.begin() as conn:
                    conn.exec_driver_sql(
                        f"ALTER TABLE {ledger} ADD COLUMN target_table VARCHAR(256)"
                    )
                logger.info(
                    "[%s] Migrated idempotency ledger %s: added target_table column.",
                    self.name(), self._ledger_table,
                )
        except Exception as e:
            logger.warning(
                "[%s] Could not ensure idempotency ledger %s: %s. Proceeding "
                "WITHOUT retry-dedup for job %s.",
                self.name(), self._ledger_table, e, self.job_id,
            )
            self.job_id = ""

    # ---- per-partition write -------------------------------------------------

    def write(self, micropartitions: Iterator[MicroPartition]) -> Iterator[WriteResult[dict]]:
        connect_args = self._build_connect_args()

        for i, mp in enumerate(micropartitions):
            # Downcast UInt64 (count/rank aggregations) → int64: pandas to_sql cannot
            # write an unsigned 64-bit integer. Runtime backstop for AGG-4b.
            df = _downcast_unsigned_ints(mp.to_arrow().to_pandas())
            row_count = len(df)
            logger.info(f"[{self.name()}] Partition {i}: writing {row_count} rows to {self._write_table}...")

            engine = sa.create_engine(
                self.connection_url,
                connect_args=connect_args,
                pool_pre_ping=True,
                pool_size=1,
                max_overflow=0,
            )
            try:
                # Pass the Engine (not a Connection): pandas calls .connect() on
                # `con`, which a SQLAlchemy 2.0 Connection lacks.
                df.to_sql(
                    name=self._write_table,
                    con=engine,
                    if_exists="append",
                    index=False,
                    chunksize=500,
                )
                logger.info(f"[{self.name()}] Partition {i}: wrote {row_count} rows.")
                yield WriteResult(
                    result={"status": "success", "partition": i},
                    rows_written=row_count,
                    bytes_written=0,
                )
            except Exception as e:
                logger.error(f"[{self.name()}] Partition {i} FAILED. Error: {e}")
                raise
            finally:
                engine.dispose()

    # ---- finalize: atomic swap staging -> target -----------------------------

    def finalize(self, write_results: list[WriteResult[dict]]) -> MicroPartition:
        total_rows = sum(wr.rows_written for wr in write_results)

        if self._use_staging:
            tgt = self._qt(self.target_table)
            stg = self._qi(self._staging_table)
            cols = self._col_list()
            engine = self._engine()

            # An empty result must never destroy the target. `overwrite` clears the
            # table before copying the staged rows in, so zero staged rows would
            # silently delete every existing row — the classic "filter matched
            # nothing" mistake. The pre-flight COUNT(*) guard only runs for database
            # sources, so file/cloud sources (C2D) reach here unprotected. Enforce it
            # where the destructive statement actually lives.
            if total_rows == 0 and self.write_mode == "overwrite":
                try:
                    with engine.begin() as conn:
                        conn.exec_driver_sql(f"DROP TABLE IF EXISTS {stg}")
                except Exception as e:  # best-effort cleanup; the abort matters more
                    logger.warning(f"[{self.name()}] Could not drop staging table: {e}")
                finally:
                    engine.dispose()
                raise RuntimeError(
                    f"[{self.name()}] Refusing to overwrite {self.target_table}: the "
                    f"transformation produced 0 rows, which would delete every existing "
                    f"row. Check the filter in your request, or use write_mode='append'."
                )
            # Idempotency guard against a whole-pipeline retry. container_file.py
            # retries run_pipeline() up to 3× on ANY exception; if a prior attempt's
            # finalize already COMMITTED but then something failed downstream, the
            # retry re-reads the source and re-stages every row — and an append would
            # insert them a SECOND time (duplicates, still reported as success). We
            # record each job in a ledger inside the SAME transaction as the insert,
            # so a committed job is durable together with its ledger row; a retry
            # sees the row and skips the re-insert. (Overwrite is already idempotent,
            # but the ledger costs nothing and keeps the two modes consistent.)
            ledger = self._qi(self._ledger_table)
            if self.job_id:
                self._ensure_ledger(engine)  # may clear self.job_id on failure
            already_committed = False
            try:
                # One transaction: idempotency check, (optionally) clear the target,
                # copy the staged rows in, and stamp the ledger — all or nothing.
                # Because the target is only touched here, a failure during the
                # distributed write leaves it untouched.
                with engine.begin() as conn:
                    prior = None
                    if self.job_id:
                        # Scope to the target table too — the ledger is shared and never
                        # pruned, so a bare id collision would cancel an unrelated insert.
                        prior = conn.execute(
                            sa.text(
                                f"SELECT rows_inserted FROM {ledger} "
                                f"WHERE job_id = :jid AND target_table = :tbl"
                            ),
                            {"jid": self.job_id, "tbl": self.target_table},
                        ).fetchone()
                    if prior is not None:
                        # This job already committed on an earlier attempt — do not
                        # insert again. Report the originally-committed row count.
                        already_committed = True
                        total_rows = int(prior[0])
                    else:
                        if self.write_mode == "overwrite":
                            conn.exec_driver_sql(f"DELETE FROM {tgt}")
                        conn.exec_driver_sql(
                            f"INSERT INTO {tgt} ({cols}) SELECT {cols} FROM {stg}"
                        )
                        if self.job_id:
                            conn.execute(
                                sa.text(
                                    f"INSERT INTO {ledger} "
                                    f"(job_id, target_table, rows_inserted, completed_at) "
                                    f"VALUES (:jid, :tbl, :rows, CURRENT_TIMESTAMP)"
                                ),
                                {"jid": self.job_id, "tbl": self.target_table, "rows": int(total_rows)},
                            )
                # Best-effort cleanup (DDL, outside the data transaction).
                try:
                    with engine.begin() as conn:
                        conn.exec_driver_sql(f"DROP TABLE IF EXISTS {stg}")
                except Exception as e:
                    logger.warning(f"[{self.name()}] Could not drop staging table {self._staging_table}: {e}")
            finally:
                engine.dispose()
            if already_committed:
                logger.warning(
                    f"[{self.name()}] Job {self.job_id} already committed {total_rows} rows "
                    f"to {self.target_table}; skipped re-insert (idempotent retry)."
                )
            else:
                logger.info(f"[{self.name()}] Finalize: swapped {total_rows} rows into "
                            f"{self.target_table} (mode={self.write_mode}).")
        else:
            # Invariant: __init__ guarantees staging (or raises). Reaching here
            # would mean staged rows were never copied — fail loudly.
            raise RuntimeError(
                f"[{self.name()}] finalize() reached without a staging table; "
                f"{total_rows} staged rows were not committed to {self.target_table}."
            )

        # Machine-readable marker so the runner (Docker/GKE) can surface the row
        # count and target table back to the user. Printed to stdout (always
        # captured in container logs) and flushed so it isn't lost on exit.
        print(
            f"AVALOKA_RESULT rows_inserted={total_rows} table={self.target_table}",
            flush=True,
        )

        return MicroPartition.from_pydict({
            "rows_inserted": [total_rows],
            "status": ["complete"],
        })

    def name(self) -> str:
        return f"Generic SQL Sink ({self.target_table})"

    def schema(self) -> daft.Schema:
        return self._result_schema


class MySQLDataSink(BaseSQLDataSink):
    _ident_quote = "`"

    def name(self) -> str:
        return "MySQL Sink"


class PostgresDataSink(BaseSQLDataSink):
    _ident_quote = '"'

    def name(self) -> str:
        return "PostgreSQL Sink"
