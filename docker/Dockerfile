# SatQuery AI backend — architecture.md Section 6 ("Local-first,
# Docker-based deployment") / Section 8 (docker/ + docker-compose.yml).
#
# Build context is the repo root, not this docker/ directory, since
# requirements.txt and backend/ both live there — see docker-compose.yml's
# `context: ..` / `dockerfile: docker/Dockerfile`.
#
# Image is large (torch + transformers + rasterio/fiona/GDAL + the rest of
# requirements.txt) because backend/model_registry's real (non-mock)
# inference path needs all of it once a checkpoint is attached (see the
# root README's "Attaching a trained checkpoint later") — none of it is
# optional-at-build-time given requirements.txt is a single consolidated
# file for the whole backend, mock mode included.
FROM python:3.12-slim-bookworm

# rasterio/fiona ship manylinux wheels with GDAL already bundled for the
# platforms this normally builds for, so this usually doesn't need a
# separate system GDAL install -- build-essential stays as a fallback so a
# package without a prebuilt wheel for this exact interpreter/platform
# combo can still compile from source instead of failing the whole build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies in their own layer: a code-only change afterward doesn't
# reinstall this entire (large, slow) stack on every rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/

# Matches README.md's documented launch command exactly: `cd backend &&
# python -m uvicorn api.main:app`. Not cosmetic -- backend/api/main.py's
# own top-of-file sys.path bootstrap (see its comments, and
# docs/SYSTEM_DESIGN.md's note on the import-root drift between Part 2 and
# Parts 4/5) is written assuming this exact working directory.
WORKDIR /app/backend

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
