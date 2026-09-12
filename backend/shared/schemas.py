# Copied verbatim from architecture.md Section 4 (single source of truth).
# Do not redefine these types locally anywhere else -- every part imports
# from here. (p2/p4/p5 each independently shipped their own copy of this
# file; diffed byte-identical on every non-comment line during merge, so
# this one canonical copy, taken straight from the architecture doc,
# replaces all three.)
#
# shared/schemas.py — single source of truth

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel

class TaskType(str, Enum):
    SINGLE_IMAGE_VQA = "SINGLE_IMAGE_VQA"
    CAPTIONING = "CAPTIONING"
    GROUNDING = "GROUNDING"                # boxes and/or masks — see Detection.mask_rle
    CHANGE_DETECTION = "CHANGE_DETECTION"
    CHANGE_VQA = "CHANGE_VQA"
    OPTICAL_SAR_FUSION = "OPTICAL_SAR_FUSION"
    UNSUPPORTED = "UNSUPPORTED"
    # Trimmed from the original ontology: object-counting / spatial / statistical
    # queries are handled as VQA or GROUNDING plus a post-processing step, not
    # separate top-level tasks. Extend this enum only if a real query genuinely
    # doesn't fit one of the above.

class Modality(str, Enum):
    OPTICAL = "OPTICAL"
    SAR = "SAR"
    # Set explicitly at upload. Never inferred from pixels — see Part 3 hardening.

class ImageMetadata(BaseModel):
    image_id: str                 # content hash of the uploaded bytes
    modality: Modality
    crs: Optional[str]
    bounds: Optional[list[float]]     # [minx, miny, maxx, maxy]
    width: int
    height: int
    band_count: int
    dtype: str
    resolution_m: Optional[float]
    timestamp: Optional[str]
    cog_path: str
    is_valid: bool
    validation_errors: list[str] = []
    # NEW (bug fix, post-merge) -- False when the source had neither a real
    # affine geotransform nor GCPs, so Part 3 assigned a synthetic
    # placeholder (see backend/preprocessing/cog.py) just to let the COG
    # driver run. cog_path/crs/bounds still get populated in that case, but
    # crs/bounds are the placeholder's, not a real position -- callers
    # (map display, report coordinates, cross-image comparisons) should
    # treat them as pixel-space only, never a real-world location.
    georeferenced: bool = True

class Tile(BaseModel):
    tile_id: str
    image_id: str
    col_off: int
    row_off: int
    width: int
    height: int
    affine_transform: list[float]     # [a, b, c, d, e, f]
    array_path: str                   # path to pixel data on disk — never inline raw arrays here

class CoregistrationResult(BaseModel):
    aligned: bool
    offset_px: float
    auto_corrected: bool
    reason: Optional[str]

class Detection(BaseModel):
    label: str
    box_px: list[float]               # [x0, y0, x1, y1], always full-image pixel space
    mask_rle: Optional[str]
    score: float

class ChangeMap(BaseModel):
    probability_raster_path: Optional[str]   # path, not inline pixels
    changed_area_px: int
    changed_area_pct: float
    mean_confidence: float

class Confidence(BaseModel):
    value: Optional[float]            # only set if a real calibrated number exists
    band: str                         # "LOW" | "MEDIUM" | "HIGH"
    basis: str                        # human-readable: what signal this came from

class WebSource(BaseModel):
    # NEW (Section 4/17 addition) -- one Google Custom Search result kept as
    # a citation. See agent/search_decision.py for when this gets populated.
    title: str
    url: str
    snippet: str
    retrieved_at: str                 # ISO 8601 -- Section 17: "retrieval timestamp"

class Evidence(BaseModel):
    task: TaskType
    model_used: str
    modality_used: list[Modality]
    detections: list[Detection] = []
    change_map: Optional[ChangeMap] = None
    vqa_answer_raw: Optional[str] = None
    stats: dict[str, float] = {}
    confidence: Confidence
    warnings: list[str] = []
    web_sources: list[WebSource] = []  # NEW -- additive, defaults empty; see above

class FinalResponse(BaseModel):
    answer_text: str
    confidence: Confidence
    evidence: Evidence
    abstained: bool
    abstain_reason: Optional[str] = None

class ModelRegistryEntry(BaseModel):
    name: str
    version: str
    tasks: list[TaskType]
    modalities: list[Modality]
    checkpoint_path: str
    quantization: str                 # "none" | "8bit" | "4bit"
    requires_coregistration: bool
    max_input_px: int

# ---------------------------------------------------------------------------
# NEW — Projects / multi-file upload. Not part of the original Part 1-6
# architecture.md contract above (added when extending SatQuery into a
# persistent, multi-format research platform) but living in this file for
# the same reason everything above does: more than one module needs the
# identical class (api/main.py, backend/files/*.py, backend/projects/*.py).
# ---------------------------------------------------------------------------

class FileCategory(str, Enum):
    SATELLITE_IMAGE = "SATELLITE_IMAGE"   # .tif/.tiff — real Part 3 pipeline
    VECTOR = "VECTOR"                     # .geojson
    TABULAR = "TABULAR"                   # .csv, .xlsx
    DOCUMENT = "DOCUMENT"                 # .pdf, .docx, .txt, .md
    PHOTO = "PHOTO"                       # .jpg/.jpeg/.png — a field photo, not satellite data
    SCIENTIFIC_DATA = "SCIENTIFIC_DATA"   # .nc, .hdf, .h5 — gridded/array data, not row/column tabular
    UNSUPPORTED = "UNSUPPORTED"           # known-but-not-yet-built or unrecognized extension

class ProjectFile(BaseModel):
    file_id: str
    filename: str
    category: FileCategory
    size_bytes: int
    uploaded_at: str                   # ISO 8601
    image_id: Optional[str] = None     # set only when category == SATELLITE_IMAGE
    metadata: dict[str, Any] = {}      # category-specific — see backend/files/extractors.py
    warnings: list[str] = []

class Project(BaseModel):
    project_id: str
    name: str
    created_at: str
    updated_at: str
    files: list[ProjectFile] = []

class AnalysisMode(str, Enum):
    """
    Section 3: Fast/Deep/Research. Genuinely different planner behavior,
    not just a label -- see agent/planner.py's mode handling. DEEP is the
    default because it reproduces exactly the pre-mode behavior (search
    if the heuristic finds a signal), so a request that doesn't specify
    mode at all keeps working unchanged.
    """
    FAST = "FAST"
    DEEP = "DEEP"
    RESEARCH = "RESEARCH"

