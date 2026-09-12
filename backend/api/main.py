"""
Part 2 — Backend Core: the FastAPI app implementing Part 1's contract
verbatim from Section 3.2:

    POST /api/upload   -> ImageMetadata
    POST /api/query    -> {"job_id": string}
    GET  /api/jobs/{id} -> {"status", "progress", "result", "error"}
    GET  /api/jobs/{id}/trace -> {"steps": [...]}
    GET  /api/models   -> list[ModelRegistryEntry]
    GET  /api/health   -> {"status": "ok", "models_loaded": [...]}

Section 2/5 addition (projects + multi-file upload), added later and not
in Part 1's original contract above -- see the "NEW" block near the end
of this file:

    POST /api/projects              -> Project
    GET  /api/projects              -> list[Project]
    GET  /api/projects/{id}         -> Project
    POST /api/projects/{id}/files   -> {"project_id", "files": list[ProjectFile]}
    POST /api/jobs/{id}/cancel      -> {"job_id", "cancelled": bool}          (Section 14)
    GET  /api/jobs/{id}/report      -> a PDF (Section 15/16)

Everything below the endpoint bodies is stitching, not logic: intent
classification lives in agent/intent.py, sequencing in agent/planner.py,
concurrency/decoupling in agent/queue.py. This file's own job is the HTTP
surface and the chunked-upload bookkeeping, which is specific to this
endpoint and doesn't belong in agent/.

Integration (Section 5, step 3): `_funcs` below is the one place the mock
Part 3/4/5 imports get swapped for the real ones — everything downstream
(JobQueue, Planner) only knows about the Part345Functions/validate_and_prepare
call shape, not that they're mocks. As of this merge, all three of Part 3,
Part 4, and Part 5 are wired to their real implementations — none of Part 2's
own mocks in agent/mocks.py are on the live call path anymore. They're kept
in the tree for backend/tests/test_agent.py, which still exercises the
planner/queue logic against them directly.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# --- environment / secrets --------------------------------------------------
# Loads the repo-root .env (SATQUERY_GOOGLE_SEARCH_API_KEY, etc.) into
# os.environ. Has to happen before the bootstrap/imports below: modules like
# preprocessing/config.py read os.environ.get(...) once, at import time, so
# anything imported before load_dotenv() runs would never see .env values.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# --- integration bootstrap (added while merging parts 1/2/4/5/6) -----------
# Part 2 (this file included) imports the shared schema bare, as
# `shared.schemas`, which resolves when `backend/` itself sits on
# sys.path. Parts 4 and 5 import the identical file as
# `backend.shared.schemas`, which resolves when the repo ROOT sits on
# sys.path instead. Both conventions are needed at once, so both roots
# go on sys.path here.
#
# That alone isn't quite enough: left alone, Python would load
# backend/shared/schemas.py twice, once under each name, producing two
# non-identical `Evidence`/`TaskType`/etc. classes. Pydantic validates
# nested-model fields by class identity, so a real Part 4 `Evidence`
# passed into a Part 2 `FinalResponse` would fail validation the
# instant real code replaced the mocks below. Importing the module
# once under its real name and aliasing the second name to the same
# module object keeps one identity everywhere. (Same block, for the
# test session, in /conftest.py at the repo root.)
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
for _p in (str(_REPO_ROOT), str(_BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import shared.schemas as _schemas_mod  # noqa: E402
import shared as _shared_pkg  # noqa: E402
import backend as _backend_pkg  # noqa: E402

sys.modules.setdefault("backend.shared", _shared_pkg)
sys.modules.setdefault("backend.shared.schemas", _schemas_mod)
_backend_pkg.shared = _shared_pkg
# --- end integration bootstrap ----------------------------------------------

import asyncio

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from shared.schemas import (
    AnalysisMode,
    FileCategory,
    ImageMetadata,
    Modality,
    Project,
    ProjectFile,
    WebSource,
)
from agent import mocks
from agent.planner import Part345Functions
from agent.queue import JobQueue
from agent.store import ImageStore, UploadSessionStore
from files import (
    detect_category,
    extract_document_metadata,
    extract_photo_metadata,
    extract_scientific_metadata,
    extract_tabular_metadata,
    extract_vector_metadata,
    unsupported_reason,
)
from projects import ProjectStore
from search import google_custom_search
from reports import generate_pdf_report

# Real Part 4 (Section 3.4) and Part 5 (Section 3.5) — wired in during this
# merge. Both expose sync functions; Part 2's Part345Functions bundle wants
# Awaitable callables (see agent/mocks.py's own docstring: "if a real
# implementation ends up sync ... wrap it with asyncio.to_thread(...) at the
# call site rather than changing these signatures"), so each is wrapped
# below rather than changing either part's code.
from backend.model_registry.inference import (
    configure_engine as _configure_inference_engine,
)
from backend.model_registry.inference import health_check as _real_health_check
from backend.model_registry.inference import (
    list_available_models as _real_list_available_models,
)
from backend.model_registry.inference import run_inference as _real_run_inference
from backend.evidence.service import validate_and_respond as _real_validate_and_respond

# Real Part 3 (Section 3.3) — wired in once it was uploaded. Also plain sync
# functions (rasterio/GDAL calls), same asyncio.to_thread treatment as
# Part 4/5 above. Imported via the package's public re-export
# (backend/preprocessing/__init__.py), matching its own documented interface.
from backend.preprocessing import (
    validate_and_prepare as _real_validate_and_prepare,
)
from backend.preprocessing import config as _preprocessing_config
from backend.preprocessing import check_coregistration as _real_check_coregistration
from backend.preprocessing.thumbnail import get_or_generate_thumbnail as _get_or_generate_thumbnail
from backend.preprocessing import tile_image as _real_tile_image

app = FastAPI(title="SatQuery AI — Backend Core (Part 2)")

# CORS — added during the merge. The frontend (Part 1) is a static file with
# no build step; depending on how it's served (opened directly as file://,
# or via a plain local file server on its own port — see start_program.bat)
# its origin won't match this API's http://localhost:8000, and browsers
# block cross-origin fetches by default. This is a local single-user
# hackathon deployment (Section 6: "runs entirely on local
# infrastructure... no live internet dependency"), not a multi-tenant
# public service, so a permissive allow-list is the right tradeoff here —
# tighten this before deploying it anywhere that isn't localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("/tmp/satquery_uploads")
CHUNK_DIR = Path("/tmp/satquery_upload_chunks")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

image_store = ImageStore()
upload_sessions = UploadSessionStore(CHUNK_DIR)

# --- Part 3/4/5 wiring (Section 3.2 Mocking strategy / Section 5 step 3) --
# All five calls are real as of this merge (Part 3 landed last). Part 2's
# own mocks in agent/mocks.py stay in the tree, unused on this call path,
# for backend/tests/test_agent.py.
#
# Part 6 (Training) hasn't produced any real checkpoints yet — models.yaml's
# checkpoint_path entries are still placeholders (see backend/model_registry/
# config/models.yaml) — so Part 4's engine stays on its own mock inference
# engine per Section 5 step 4's fallback branch ("otherwise leave it on a
# pretrained-only fallback entry"). Flip use_mock=False here once real
# checkpoints land and this box has the GPU deps from
# backend/model_registry's requirements installed.
_configure_inference_engine(use_mock=True)


def _safe_upload_name(filename: str | None) -> str:
    """Keep uploaded bytes inside the configured storage directory."""
    name = Path(filename or "upload").name.replace("\x00", "")
    return name or "upload"


def _validate_project_id(project_id: str) -> None:
    try:
        uuid.UUID(project_id)
    except (ValueError, AttributeError):
        raise HTTPException(400, "project_id must be a valid UUID")


async def _run_inference_async(task, tiles, model_hint=None, **kwargs):
    return await asyncio.to_thread(_real_run_inference, task, tiles, model_hint, **kwargs)


async def _validate_and_respond_async(query, evidence):
    return await asyncio.to_thread(_real_validate_and_respond, query, evidence)


async def _list_available_models_async():
    return await asyncio.to_thread(_real_list_available_models)


async def _health_check_async():
    return await asyncio.to_thread(_real_health_check)


async def _validate_and_prepare_async(file_path, declared_modality, declared_timestamp):
    return await asyncio.to_thread(
        _real_validate_and_prepare, file_path, declared_modality, declared_timestamp
    )


async def _tile_image_async(image_id, task, tile_size=1024, overlap_pct=0.15):
    return await asyncio.to_thread(_real_tile_image, image_id, task, tile_size, overlap_pct)


async def _check_coregistration_async(image_a_id, image_b_id):
    return await asyncio.to_thread(_real_check_coregistration, image_a_id, image_b_id)


# NEW (Section 4) -- the app-level web-search dependency JobQueue passes down
# to every job's Planner.execute() call (see agent/planner.py's WebSearchFn).
# Maps backend/search/google_search.py's SearchResult onto the shared
# WebSource schema at this boundary, exactly like the Part 3/4/5 wrappers
# above map real functions onto Part345Functions' call shape -- agent/ only
# ever depends on shared.schemas types, never on search/'s own internal
# SearchResult class.
async def _web_search_fn(query: str, num_results: int) -> list[WebSource]:
    results = await google_custom_search(query, num_results=num_results)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    return [
        WebSource(title=r.title, url=r.url, snippet=r.snippet, retrieved_at=retrieved_at)
        for r in results
    ]


_validate_and_prepare = _validate_and_prepare_async  # Part 3 — real
_funcs = Part345Functions(
    tile_image=_tile_image_async,  # Part 3 — real
    check_coregistration=_check_coregistration_async,  # Part 3 — real
    run_inference=_run_inference_async,  # Part 4 — real
    validate_and_respond=_validate_and_respond_async,  # Part 5 — real
)
job_queue = JobQueue(funcs=_funcs, max_concurrent_inference=1, web_search_fn=_web_search_fn)


class QueryRequest(BaseModel):
    query: str
    image_ids: list[str]
    web_search: bool = False  # NEW (Section 4) -- off by default; see README before enabling
    mode: AnalysisMode = AnalysisMode.DEEP  # NEW (Section 3) -- DEEP reproduces pre-mode behavior


@app.post("/api/upload")
async def upload_image(
    file: UploadFile = File(...),
    modality: Modality = Form(...),
    timestamp: Optional[str] = Form(None),
    upload_id: Optional[str] = Form(None),
    chunk_index: Optional[int] = Form(None),
    total_chunks: Optional[int] = Form(None),
):
    chunk_bytes = await file.read(_preprocessing_config.MAX_FILE_SIZE_BYTES + 1)
    if len(chunk_bytes) > _preprocessing_config.MAX_FILE_SIZE_BYTES:
        raise HTTPException(413, "uploaded file exceeds the configured size limit")

    if total_chunks is not None and total_chunks > 1:
        # Chunked path (Section 3.2 hardening: resumable chunked handling).
        if chunk_index is None:
            raise HTTPException(400, "chunk_index is required when total_chunks > 1")
        session_id = upload_id or str(uuid.uuid4())
        session = upload_sessions.get_or_create(session_id, total_chunks, modality, timestamp)
        try:
            session.write_chunk(
                chunk_index,
                chunk_bytes,
                max_total_bytes=_preprocessing_config.MAX_FILE_SIZE_BYTES,
            )
        except ValueError as exc:
            raise HTTPException(413 if "size limit" in str(exc) else 400, str(exc))

        if not session.is_complete():
            return JSONResponse(status_code=202, content={
                "status": "chunk_received",
                "upload_id": session_id,
                "received_chunks": sorted(session.received),
                "total_chunks": total_chunks,
            })

        final_path = session.assemble()
        upload_sessions.discard(session_id)
    else:
        final_path = UPLOAD_DIR / f"{uuid.uuid4()}_{_safe_upload_name(file.filename)}"
        final_path.write_bytes(chunk_bytes)

    try:
        metadata = await _validate_and_prepare(str(final_path), modality, timestamp)
    except Exception as e:
        raise HTTPException(422, f"validation failed: {e}")

    image_store.put(metadata)
    return metadata.model_dump()


@app.post("/api/query")
async def submit_query(payload: QueryRequest):
    images = []
    for image_id in payload.image_ids:
        img = image_store.get(image_id)
        if img is None:
            raise HTTPException(404, f"unknown image_id: {image_id}")
        images.append(img)

    job_id = job_queue.submit(
        payload.query, images, web_search_enabled=payload.web_search, mode=payload.mode
    )
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    record = job_queue.get(job_id)
    if record is None:
        raise HTTPException(404, "job not found")
    return {
        "status": record.status,
        "progress": record.progress,
        "result": record.result.model_dump() if record.result is not None else None,
        "error": record.error,
    }


@app.get("/api/jobs/{job_id}/trace")
async def get_job_trace(job_id: str):
    record = job_queue.get(job_id)
    if record is None:
        raise HTTPException(404, "job not found")
    return record.trace.as_dict()


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Section 14 (Stop/Cancel) -- see JobQueue.cancel's own docstring for
    exactly which race conditions this is (and isn't) safe against."""
    if job_queue.get(job_id) is None:
        raise HTTPException(404, "job not found")
    cancelled = job_queue.cancel(job_id)
    return {"job_id": job_id, "cancelled": cancelled}


REPORTS_DIR = Path(os.environ.get("SATQUERY_REPORTS_DIR", str(Path.cwd() / "data" / "reports")))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/api/jobs/{job_id}/report")
async def get_job_report(job_id: str):
    """Section 15/16 (Save Report). See reports/pdf_report.py's own
    docstring for exactly which report sections this does and doesn't
    cover yet. Generated once per job then cached on disk -- re-requesting
    the same finished job's report doesn't re-run reportlab every time."""
    record = job_queue.get(job_id)
    if record is None:
        raise HTTPException(404, "job not found")
    if record.status != "done" or record.result is None:
        raise HTTPException(409, f"job is not finished yet (status: {record.status})")

    pdf_path = REPORTS_DIR / f"{job_id}.pdf"
    if not pdf_path.exists():
        await asyncio.to_thread(generate_pdf_report, record.query, record.result, str(pdf_path))
    return FileResponse(
        str(pdf_path), media_type="application/pdf",
        filename=f"satquery-report-{job_id[:8]}.pdf",
    )


@app.get("/api/images/{image_id}/thumbnail")
async def get_image_thumbnail(image_id: str, max_dim: int = 512):
    """Real PNG generated from the actual raster (decimated read — never
    loads the full-resolution array, see thumbnail.py's own docstring on
    why that matters for GB-scale sources), not a placeholder image.
    Cached on disk after the first request for a given (image_id,
    max_dim) pair."""
    from backend.preprocessing.store import ImageNotFoundError

    try:
        path = await asyncio.to_thread(_get_or_generate_thumbnail, image_id, max_dim)
    except ImageNotFoundError:
        raise HTTPException(404, f"no stored image with id={image_id!r}")
    except Exception as exc:
        raise HTTPException(422, f"couldn't generate a thumbnail for this image: {exc}")
    return FileResponse(path, media_type="image/png")


@app.get("/api/models")
async def get_models():
    models = await _list_available_models_async()
    health_state = await _health_check_async()
    capabilities = health_state.get("capabilities", {})
    return [
        {
            **model.model_dump(),
            "availability": next(
                (capabilities.get(task.value) for task in model.tasks if capabilities.get(task.value)),
                "AVAILABLE",
            ),
        }
        for model in models
    ]


@app.get("/api/health")
async def health():
    return await _health_check_async()


# ---------------------------------------------------------------------------
# NEW — Projects / multi-file upload (Section 2 + Section 5 of the platform
# upgrade brief). Kept in its own block after the original Part 1-6 contract
# above, rather than interleaved with it, so the two are easy to tell apart.
# Only touches image_store and the shared schema types from above — nothing
# else up there had to change.
# ---------------------------------------------------------------------------

project_store = ProjectStore()

PROJECT_FILES_DIR = Path(
    os.environ.get("SATQUERY_PROJECT_FILES_DIR", str(Path.cwd() / "data" / "project_files"))
)
PROJECT_FILES_DIR.mkdir(parents=True, exist_ok=True)

# image_store is memory-only (see agent/store.py's own docstring) and
# resets on every restart, but a Project's JSON file on disk does not --
# without this, POST /api/query would 404 on an image_id that a user can
# still see sitting in their project from a previous session.
for _project in project_store.list():
    for _pf in _project.files:
        if _pf.category == FileCategory.SATELLITE_IMAGE and _pf.image_id:
            try:
                image_store.put(ImageMetadata(**_pf.metadata))
            except Exception:
                pass  # won't be query-able until re-uploaded; doesn't block startup


class CreateProjectRequest(BaseModel):
    name: str


@app.post("/api/projects")
async def create_project(payload: CreateProjectRequest):
    project = project_store.create(payload.name)
    return project.model_dump()


@app.get("/api/projects")
async def list_projects():
    return [p.model_dump() for p in project_store.list()]


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    _validate_project_id(project_id)
    project = project_store.get(project_id)
    if project is None:
        raise HTTPException(404, "project not found")
    return project.model_dump()


# NEW (UI redesign): the sidebar's per-project "..." menu needs these --
# neither existed under the old <select>-based Projects panel.
class RenameProjectRequest(BaseModel):
    name: str


@app.patch("/api/projects/{project_id}")
async def rename_project(project_id: str, payload: RenameProjectRequest):
    _validate_project_id(project_id)
    if project_store.get(project_id) is None:
        raise HTTPException(404, "project not found")
    project = project_store.rename(project_id, payload.name)
    return project.model_dump()


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    _validate_project_id(project_id)
    if project_store.get(project_id) is None:
        raise HTTPException(404, "project not found")
    project_store.delete(project_id)
    # A deleted project shouldn't leave its uploaded files behind on disk
    # with nothing left in the UI to ever reference them.
    project_dir = PROJECT_FILES_DIR / project_id
    if project_dir.exists():
        shutil.rmtree(project_dir, ignore_errors=True)
    return {"project_id": project_id, "deleted": True}


@app.post("/api/projects/{project_id}/files")
async def upload_project_files(
    project_id: str,
    files: list[UploadFile] = File(...),
    modalities_json: Optional[str] = Form(
        None,
        description=(
            'JSON object mapping filename -> "OPTICAL"|"SAR". Required only '
            "for .tif/.tiff files in this batch — modality is never guessed "
            "(see shared.schemas.Modality's own comment: set explicitly at "
            "upload, never inferred from pixels)."
        ),
    ),
):
    """
    Multi-file upload (Section 5): unlike /api/upload above, this takes any
    number of files of mixed formats in one request. Each file is handled
    independently — one bad/unsupported file in the batch produces a
    per-file warning, not a failed request, so a 9-good/1-bad batch still
    keeps its 9 good files.
    """
    _validate_project_id(project_id)
    if project_store.get(project_id) is None:
        raise HTTPException(404, "project not found")

    modality_map: dict[str, str] = {}
    if modalities_json:
        try:
            modality_map = json.loads(modalities_json)
        except json.JSONDecodeError:
            raise HTTPException(400, "modalities_json must be valid JSON")

    results = []
    for upload in files:
        filename = upload.filename or "upload"
        raw = await upload.read(_preprocessing_config.MAX_FILE_SIZE_BYTES + 1)
        if len(raw) > _preprocessing_config.MAX_FILE_SIZE_BYTES:
            raise HTTPException(413, f"file {upload.filename or 'upload'} exceeds the configured size limit")
        category = detect_category(filename, raw)

        project_file = ProjectFile(
            file_id=str(uuid.uuid4()),
            filename=filename,
            category=category,
            size_bytes=len(raw),
            uploaded_at=datetime.now(timezone.utc).isoformat(),
        )

        dest = PROJECT_FILES_DIR / project_id / f"{project_file.file_id}_{_safe_upload_name(filename)}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)

        try:
            if category == FileCategory.SATELLITE_IMAGE:
                declared = modality_map.get(filename)
                if declared not in ("OPTICAL", "SAR"):
                    project_file.warnings = [
                        "modality not specified for this satellite image — "
                        f'include "{filename}" in modalities_json ("OPTICAL" '
                        'or "SAR"); file stored but not yet validated'
                    ]
                else:
                    image_metadata = await _validate_and_prepare(
                        str(dest), Modality(declared), None
                    )
                    image_store.put(image_metadata)
                    project_file.image_id = image_metadata.image_id
                    project_file.metadata = image_metadata.model_dump()
                    if not image_metadata.is_valid:
                        project_file.warnings = image_metadata.validation_errors
            elif category == FileCategory.VECTOR:
                project_file.metadata = await asyncio.to_thread(
                    extract_vector_metadata, str(dest)
                )
            elif category == FileCategory.TABULAR:
                project_file.metadata = await asyncio.to_thread(
                    extract_tabular_metadata, str(dest), filename
                )
            elif category == FileCategory.DOCUMENT:
                project_file.metadata = await asyncio.to_thread(
                    extract_document_metadata, str(dest), filename
                )
            elif category == FileCategory.PHOTO:
                project_file.metadata = await asyncio.to_thread(
                    extract_photo_metadata, str(dest)
                )
            elif category == FileCategory.SCIENTIFIC_DATA:
                project_file.metadata = await asyncio.to_thread(
                    extract_scientific_metadata, str(dest), filename
                )
            else:
                project_file.warnings = [unsupported_reason(filename, raw) or "unsupported file type"]
        except Exception as e:
            project_file.warnings = project_file.warnings + [f"extraction failed: {e}"]

        project_store.add_file(project_id, project_file)
        results.append(project_file.model_dump())

    return {"project_id": project_id, "files": results}
