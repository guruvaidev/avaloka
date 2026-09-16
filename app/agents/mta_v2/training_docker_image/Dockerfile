FROM rayproject/ray:2.49.2-py311

ENV PYTHONUNBUFFERED=1
ENV RAY_BACKEND_LOG_LEVEL=error
ENV RAY_DISABLE_IMPORT_WARNING=1

WORKDIR /app

# ---- System deps ----
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- App code ----
COPY . /app

WORKDIR /app
ENV PYTHONPATH="/app:${PYTHONPATH}"
