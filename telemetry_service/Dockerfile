# The telemetry ingest service.
#
# Small on purpose: this is a public, unauthenticated endpoint, so every extra
# package in the image is another thing that can have a vulnerability in it. It
# needs the web server, the Postgres driver, and nothing else — none of Avaloka
# itself, which is a different program entirely.

FROM python:3.12-slim AS runtime

# Fail fast, do not buffer logs, and do not litter the image with .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# psycopg2-binary rather than psycopg2 so no compiler is needed in the image.
RUN pip install --no-cache-dir \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.29" \
        "psycopg2-binary>=2.9"

COPY telemetry_service/ /srv/telemetry_service/

# A writable home for the default SQLite fallback. Without it, uid 10001 has
# nowhere it can write, and the container in its own documented default
# configuration dies on the first insert.
RUN mkdir /data && chown 10001 /data
ENV TELEMETRY_DATABASE_URL=/data/telemetry.db

# Runs as a non-root user. A public endpoint should not be root, and there is
# nothing in this container that needs to be.
RUN useradd --system --uid 10001 --no-create-home telemetry
USER 10001

# Cloud Run supplies PORT; the default is for running it by hand.
ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8080)}/health').read()" || exit 1

# One worker per container, scaled horizontally rather than vertically: the
# work is IO-bound and Cloud Run adds instances on its own.
# --no-access-log is a privacy control, not a preference: uvicorn's access
# log prints the client address of every request, and those lines persist in
# the platform's log storage. The design's promise is that the sender's IP is
# used for rate limiting and then discarded — a log line is storage.
CMD ["sh", "-c", "exec uvicorn telemetry_service.app:create_app --factory --host 0.0.0.0 --port ${PORT} --workers 1 --no-server-header --no-access-log"]
