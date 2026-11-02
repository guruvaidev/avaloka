#!/bin/sh
# Applies the chart's Supabase migrations to the in-cluster database, once each.
#
# Run by the supabase-migrate hook Job (templates/supabase.yaml) on every install
# and upgrade. It used to re-run EVERY .sql file on EVERY deploy with errors
# ignored, so a role grant in an old migration was re-granted on each upgrade and
# a non-idempotent migration was a hazard. Now each file's version (its file
# name without .sql) is recorded in avaloka_meta.schema_migrations and a recorded
# file never runs again.
#
# Connection comes from the standard libpq variables (PGHOST, PGPORT, PGUSER,
# PGDATABASE, PGPASSWORD). Nothing here prints them.
#
#   MIGRATIONS_DIR  directory holding the *.sql files            (/migrations)
#   BASELINE_FILE   versions that existed before tracking began  ($MIGRATIONS_DIR/baseline.txt)
#   ADMIN_EMAILS    comma-separated; granted 'admin' if the account exists (empty: none)
#   GOTRUE_MAILER_AUTOCONFIRM  the in-cluster gotrue's setting; the admin bootstrap
#                   runs only when it is explicitly "false"
#
# Three cases on the first run of this script against a database:
#
#   tracked   the version table exists: apply only unrecorded files.
#   existing  no version table, but the objects created by the newest baseline
#             migrations are all present. The old Job re-ran every file on every
#             deploy, so all baseline files have been applied already: record them
#             WITHOUT running them again, then apply only files outside the baseline.
#   untracked anything else -- a fresh database, or an install from a chart older
#             than the newest baseline file. Each baseline file runs ONCE the old way
#             (statement by statement, errors reported but not fatal), which is exactly
#             what the previous Job would have done on this deploy, and is then recorded
#             whatever its errors. On a fresh supabase/postgres database about a fifth
#             of the baseline files report errors, all of them objects 0-schema.sql
#             already created; re-running them on every deploy is the hazard this
#             script exists to remove, and it never made them succeed.
#
# Files outside the baseline (every migration added from now on) run inside one
# transaction with ON_ERROR_STOP, are recorded in that same transaction, and fail
# the Job when they fail. Put "-- avaloka:no-transaction" in a file that cannot run
# inside a transaction (CREATE INDEX CONCURRENTLY and the like).
set -eu

MIGRATIONS_DIR="${MIGRATIONS_DIR:-/migrations}"
BASELINE_FILE="${BASELINE_FILE:-$MIGRATIONS_DIR/baseline.txt}"
ADMIN_EMAILS="${ADMIN_EMAILS:-}"
GOTRUE_MAILER_AUTOCONFIRM="${GOTRUE_MAILER_AUTOCONFIRM:-}"

PSQL="psql -X -q -v ON_ERROR_STOP=1"

valid_version() {
  case "$1" in
    "" | *[!A-Za-z0-9._-]*) return 1 ;;
  esac
  return 0
}

in_baseline() {
  [ -f "$BASELINE_FILE" ] && grep -qxF "$1" "$BASELINE_FILE"
}

echo "waiting for db..."
until pg_isready -q >/dev/null 2>&1; do sleep 3; done

# ---- 1. version table, and the one-time backfill for existing installs ----
baseline=""
if [ -f "$BASELINE_FILE" ]; then
  while IFS= read -r v || [ -n "$v" ]; do
    case "$v" in "" | "#"*) continue ;; esac
    valid_version "$v" || { echo "invalid version in baseline: refusing to continue"; exit 2; }
    baseline="${baseline:+$baseline,}$v"
  done < "$BASELINE_FILE"
fi

# Detection runs in the same transaction that creates the table, so a second run
# (or a retried pod) always takes the "tracked" branch. The marker objects are the
# ones created by the newest baseline migrations, plus the early user_roles table:
# all of them exist only if the old Job ran a chart that already carried those files.
# Query output is discarded (set_config echoes its value); NOTICEs go to stderr.
$PSQL -v baseline="$baseline" >/dev/null <<'SQL'
BEGIN;
SELECT set_config('avaloka.baseline', :'baseline', true);
DO $$
DECLARE
  n integer := 0;
BEGIN
  IF to_regclass('avaloka_meta.schema_migrations') IS NOT NULL THEN
    RAISE NOTICE 'migration state: tracked';
    RETURN;
  END IF;

  CREATE SCHEMA IF NOT EXISTS avaloka_meta;
  -- Not exposed through PostgREST (PGRST_DB_SCHEMAS lists public, storage and
  -- graphql_public) and no API role gets USAGE: a browser must not be able to
  -- mark a migration applied or erase the record.
  REVOKE ALL ON SCHEMA avaloka_meta FROM PUBLIC;
  CREATE TABLE avaloka_meta.schema_migrations (
    version    text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now(),
    how        text NOT NULL DEFAULT 'applied'   -- applied | backfilled
  );
  REVOKE ALL ON avaloka_meta.schema_migrations FROM PUBLIC;

  IF to_regclass('public.ray_jobs') IS NOT NULL
     AND to_regclass('public.notifications') IS NOT NULL
     AND to_regclass('public.user_roles') IS NOT NULL
     AND current_setting('avaloka.baseline') <> '' THEN
    INSERT INTO avaloka_meta.schema_migrations (version, how)
    SELECT v, 'backfilled'
    FROM unnest(string_to_array(current_setting('avaloka.baseline'), ',')) AS v
    ON CONFLICT (version) DO NOTHING;
    GET DIAGNOSTICS n = ROW_COUNT;
    RAISE NOTICE 'migration state: existing install, backfilled % baseline migration(s) without re-applying them', n;
  ELSE
    RAISE NOTICE 'migration state: untracked, applying everything in order';
  END IF;
END
$$;
COMMIT;
SQL

is_recorded() {
  [ "$($PSQL -tA -v ver="$1" <<'SQL'
SELECT count(*) FROM avaloka_meta.schema_migrations WHERE version = :'ver';
SQL
)" = "1" ]
}

record() {
  $PSQL -v ver="$1" <<'SQL'
INSERT INTO avaloka_meta.schema_migrations (version) VALUES (:'ver') ON CONFLICT (version) DO NOTHING;
SQL
}

# ---- 2. migrations, in file-name order ----
applied=0
skipped=0
errfile="$(mktemp)"
trap 'rm -f "$errfile"' EXIT
for f in $(ls "$MIGRATIONS_DIR"/*.sql 2>/dev/null | sort); do
  v="$(basename "$f" .sql)"
  valid_version "$v" || { echo "unsupported migration file name: $(basename "$f")"; exit 2; }
  if is_recorded "$v"; then
    skipped=$((skipped + 1))
    continue
  fi

  if in_baseline "$v"; then
    # Pre-tracking file on an untracked database: exactly the previous behaviour.
    echo "applying $v (baseline)"
    psql -X -q -v ON_ERROR_STOP=0 -f "$f" 2>"$errfile" >/dev/null || true
    if grep -q "ERROR:" "$errfile"; then
      grep "ERROR:" "$errfile" >&2 || true
      echo "  $(grep -c "ERROR:" "$errfile") statement error(s) in baseline migration $v (pre-tracking behaviour); recorded, it will not run again"
    fi
    record "$v"
    applied=$((applied + 1))
  elif grep -q "avaloka:no-transaction" "$f"; then
    echo "applying $v (no transaction)"
    psql -X -q -v ON_ERROR_STOP=1 -f "$f"
    record "$v"
    applied=$((applied + 1))
  else
    echo "applying $v"
    # One transaction: the migration and its record commit together or not at all.
    printf '%s\n' "INSERT INTO avaloka_meta.schema_migrations (version) VALUES (:'ver');" |
      psql -X -q -v ON_ERROR_STOP=1 --single-transaction -v ver="$v" -f "$f" -f -
    applied=$((applied + 1))
  fi
done
echo "migrations complete: $applied applied, $skipped already recorded"

# ---- 3. bootstrap admins from configuration ----
# Grants 'admin' to the configured addresses that already have a confirmed
# account. It only ever adds: an existing grant is never revoked, and an empty
# ADMIN_EMAILS grants nothing (there is no fallback and no "first user is admin").
# The addresses are not printed.
has_admin_emails=no
case "$ADMIN_EMAILS" in *[!\ ,]*) has_admin_emails=yes ;; esac
autoconfirm="$(printf '%s' "$GOTRUE_MAILER_AUTOCONFIRM" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
if [ "$has_admin_emails" = yes ] && [ "$autoconfirm" != false ]; then
  # Autoconfirm on (or not stated): gotrue confirms new accounts without the
  # owner clicking a link, so a listed address with no account yet could be
  # registered by anyone. The chart refuses to render this combination; this is
  # the second line of defence.
  echo "WARNING: admin bootstrap skipped: GOTRUE_MAILER_AUTOCONFIRM is not false, so a listed address could be claimed without proving ownership. Set supabase.auth.autoconfirm=false to enable it. Nothing granted." >&2
elif [ "$has_admin_emails" = yes ]; then
  # >/dev/null: set_config would echo the address list. The count comes back as a NOTICE.
  $PSQL -v admin_emails="$ADMIN_EMAILS" >/dev/null <<'SQL'
BEGIN;
SELECT set_config('avaloka.admin_emails', :'admin_emails', true);
DO $$
DECLARE
  n integer := 0;
BEGIN
  IF to_regclass('public.user_roles') IS NULL OR to_regclass('auth.users') IS NULL THEN
    RAISE NOTICE 'admin bootstrap: user_roles or auth.users missing, nothing granted';
    RETURN;
  END IF;
  -- The DB image's own auth.users predates gotrue's migrations and has no
  -- email_confirmed_at. Until gotrue has run there can be no real account, so wait
  -- for the next deploy rather than grant to an unconfirmed address.
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_schema = 'auth' AND table_name = 'users'
                   AND column_name = 'email_confirmed_at') THEN
    RAISE NOTICE 'admin bootstrap: auth.users not migrated by gotrue yet, nothing granted';
    RETURN;
  END IF;
  EXECUTE $q$
    INSERT INTO public.user_roles (user_id, role)
    SELECT u.id, 'admin'::public.app_role
    FROM auth.users u
    WHERE u.email_confirmed_at IS NOT NULL
      AND lower(btrim(u.email)) IN (
        SELECT lower(btrim(e))
        FROM unnest(string_to_array(current_setting('avaloka.admin_emails'), ',')) AS e
        WHERE btrim(e) <> ''
      )
    ON CONFLICT (user_id, role) DO NOTHING
  $q$;
  GET DIAGNOSTICS n = ROW_COUNT;
  RAISE NOTICE 'admin bootstrap: % new grant(s)', n;
EXCEPTION WHEN OTHERS THEN
  -- Never fail the deploy over the bootstrap; SQLSTATE only, so no value is echoed.
  RAISE WARNING 'admin bootstrap failed (SQLSTATE %), nothing granted', SQLSTATE;
END
$$;
COMMIT;
SQL
else
  echo "admin bootstrap: ADMIN_EMAILS is empty, nothing granted"
fi
