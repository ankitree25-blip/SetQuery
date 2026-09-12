"""
Part 4 — fallback registry (Section 3.4's mandated mocking strategy,
`configure_engine(use_mock=True)` — the default until a trained checkpoint
exists for a given task, see models.yaml / docs/SYSTEM_DESIGN.md).

Not a single tier anymore. Two, by task:

  - CHANGE_DETECTION / CHANGE_VQA: real, deterministic computation —
    Change Vector Analysis + Otsu thresholding (deterministic_change.py),
    genuinely computed from whatever pixels are actually uploaded, not a
    fixed fixture. This is real, established remote-sensing science (a
    classical *baseline* method predating deep learning), not a trained
    model, and not fake — different image pairs give different, honestly-
    computed answers. See deterministic_change.py's own docstring for
    exactly what it can and can't tell you.
  - SINGLE_IMAGE_VQA / CAPTIONING / GROUNDING / OPTICAL_SAR_FUSION: still
    a placeholder, unavoidably — answering "what is this" or finding
    "buildings" needs a trained model, and there's no honest classical
    substitute for that. Labeled as a placeholder explicitly in the
    returned Evidence (not just a `[mock]` string easy to skim past), with
    real per-band pixel statistics attached alongside it — genuinely
    computed from the actual uploaded tile, just not a semantic
    understanding of what's in it.

Bug fix (this pass): `query`/`image_modalities`/`image_order` were declared
on InferenceEngine.run_inference's real signature (inference.py) but
agent/planner.py's call site never actually passed them, and this class's
old signature only read `query` out of a **kwargs bucket nothing ever
populated — so this engine never saw the user's real question text or a
reliable before/after image order at all, mock or (eventually) real. Fixed
in both places; see planner.py's own call site for the other half.
"""
from __future__ import annotations

import time
from typing import Optional

from backend.model_registry.adapters.change_detection_adapter import ChangeDetectionAdapter
from backend.model_registry.deterministic_change import detect_change
from backend.preprocessing.stitching import merge_change_maps, mosaic_and_recount_change_maps
from backend.shared.schemas import (
    ChangeMap, Confidence, Evidence, Modality,
    ModelRegistryEntry, TaskType, Tile,
)

_MOCK_REGISTRY: list[ModelRegistryEntry] = [
    ModelRegistryEntry(
        name="mock-vlm", version="0.0.0-mock",
        tasks=[TaskType.SINGLE_IMAGE_VQA, TaskType.CAPTIONING],
        modalities=[Modality.OPTICAL, Modality.SAR],
        checkpoint_path="(mock)", quantization="none",
        requires_coregistration=False, max_input_px=1024,
    ),
    ModelRegistryEntry(
        name="mock-grounding", version="0.0.0-mock",
        tasks=[TaskType.GROUNDING],
        modalities=[Modality.OPTICAL, Modality.SAR],
        checkpoint_path="(mock)", quantization="none",
        requires_coregistration=False, max_input_px=1024,
    ),
    ModelRegistryEntry(
        name="deterministic-change", version="cva-otsu-1.0",
        tasks=[TaskType.CHANGE_DETECTION, TaskType.CHANGE_VQA],
        modalities=[Modality.OPTICAL, Modality.SAR],
        checkpoint_path="(none — classical CV, not a trained model)", quantization="none",
        requires_coregistration=True, max_input_px=1024,
    ),
    ModelRegistryEntry(
        name="mock-fusion", version="0.0.0-mock",
        tasks=[TaskType.OPTICAL_SAR_FUSION],
        modalities=[Modality.OPTICAL, Modality.SAR],
        checkpoint_path="(mock)", quantization="none",
        requires_coregistration=True, max_input_px=1024,
    ),
]

_MODEL_FOR_TASK: dict[TaskType, str] = {
    TaskType.SINGLE_IMAGE_VQA: "mock-vlm",
    TaskType.CAPTIONING: "mock-vlm",
    TaskType.GROUNDING: "mock-grounding",
    TaskType.CHANGE_DETECTION: "deterministic-change",
    TaskType.CHANGE_VQA: "deterministic-change",
    TaskType.OPTICAL_SAR_FUSION: "mock-fusion",
}
_PLACEHOLDER_NOTICE = (
    "This capability needs a fine-tuned vision-language model that hasn't "
    "been trained yet (see docs/SYSTEM_DESIGN.md) — this response is a "
    "labeled placeholder to demonstrate the system's wiring, not a real "
    "analysis of this image's content."
)


class MockInferenceEngine:
    """Drop-in stand-in for InferenceEngine — same public interface,
    including the query/image_modalities/image_order keywords real
    InferenceEngine.run_inference already declared (inference.py) — see
    this module's docstring on why those matter even here."""

    def __init__(self, artificial_latency_s: float = 0.0, simulate_oom_for: Optional[set] = None):
        self.artificial_latency_s = artificial_latency_s
        self.simulate_oom_for = simulate_oom_for or set()

    def run_inference(
        self, task: TaskType, tiles: list[Tile], model_hint: Optional[str] = None, *,
        query: Optional[str] = None,
        image_modalities: Optional[dict[str, Modality]] = None,
        image_order: Optional[list[str]] = None,
    ) -> Evidence:
        if self.artificial_latency_s:
            time.sleep(self.artificial_latency_s)
        if task in self.simulate_oom_for:
            raise RuntimeError("CUDA out of memory (simulated by MockInferenceEngine for downstream testing)")
        if task not in _MODEL_FOR_TASK:
            raise ValueError(f"Mock registry has no fixture for task={task!r}")
        if not tiles:
            raise ValueError("run_inference requires at least one tile.")

        model_name = model_hint or _MODEL_FOR_TASK[task]
        modality_used = _resolve_modalities(tiles, image_modalities)

        if task in (TaskType.CHANGE_DETECTION, TaskType.CHANGE_VQA):
            return _real_change_evidence(task, model_name, tiles, query, image_order, modality_used)
        return _placeholder_evidence(task, model_name, tiles, query, modality_used)

    def list_available_models(self) -> list[ModelRegistryEntry]:
        return list(_MOCK_REGISTRY)

    def health_check(self) -> dict:
        return {
            "status": "degraded",
            "models_loaded": [],
            "inference_mode": "DETERMINISTIC_FALLBACK",
            "capabilities": {
                TaskType.SINGLE_IMAGE_VQA.value: "MODEL_NOT_AVAILABLE",
                TaskType.CAPTIONING.value: "MODEL_NOT_AVAILABLE",
                TaskType.GROUNDING.value: "MODEL_NOT_AVAILABLE",
                TaskType.CHANGE_DETECTION.value: "DETERMINISTIC_FALLBACK",
                TaskType.CHANGE_VQA.value: "DETERMINISTIC_FALLBACK",
                TaskType.OPTICAL_SAR_FUSION.value: "MODEL_NOT_AVAILABLE",
            },
            "vram_used_mb": 0.0,
            "vram_budget_mb": 0.0,
        }


def build_mock_engine(**kwargs) -> MockInferenceEngine:
    return MockInferenceEngine(**kwargs)


def _resolve_modalities(tiles: list[Tile], image_modalities: Optional[dict[str, Modality]]) -> list[Modality]:
    if not image_modalities:
        return [Modality.OPTICAL]  # honest last resort — see shared.schemas.Modality's own comment on never guessing this from pixels
    ids = {t.image_id for t in tiles}
    used = [image_modalities[i] for i in ids if i in image_modalities]
    return used or [Modality.OPTICAL]


def _real_change_evidence(
    task: TaskType, model_name: str, tiles: list[Tile],
    query: Optional[str], image_order: Optional[list[str]], modality_used: list[Modality],
) -> Evidence:
    """Real computation — see deterministic_change.py's own docstring.
    Pairing/merging reuses change_detection_adapter.py's own helpers
    directly (not a second copy of the same logic) so the mock and real
    paths aggregate multi-tile-pair results identically; only the
    per-tile-pair computation itself differs (this calls
    deterministic_change.detect_change, the real path calls
    model.detect_change)."""
    by_image: dict[str, list[Tile]] = {}
    for t in tiles:
        by_image.setdefault(t.image_id, []).append(t)

    if len(by_image) != 2:
        return Evidence(
            task=task, model_used=model_name, modality_used=modality_used,
            warnings=[f"Change detection needs tiles from exactly 2 images, got {len(by_image)}."],
            confidence=Confidence(value=None, band="LOW", basis="input error, no computation performed"),
        )

    ordered_ids = [iid for iid in (image_order or []) if iid in by_image]
    if len(ordered_ids) != 2:
        ordered_ids = sorted(by_image)  # same documented best-effort fallback as change_detection_adapter.py
    before_id, after_id = ordered_ids

    pairs = ChangeDetectionAdapter._pair_by_offset(by_image[before_id], by_image[after_id])
    if not pairs:
        return Evidence(
            task=task, model_used=model_name, modality_used=modality_used,
            warnings=["No spatially-matching tile pairs found between the two images (different tile grids?)."],
            confidence=Confidence(value=None, band="LOW", basis="input error, no computation performed"),
        )

    per_pair = []
    computation_warnings: list[str] = []
    for before, after in pairs:
        try:
            raw = detect_change(before.array_path, after.array_path)
        except Exception as exc:
            computation_warnings.append(f"tile at ({before.col_off},{before.row_off}): {exc}")
            continue
        if raw.get("warning"):
            computation_warnings.append(raw["warning"])
        per_pair.append(raw)

    if not per_pair:
        return Evidence(
            task=task, model_used=model_name, modality_used=modality_used,
            warnings=computation_warnings or ["Change computation failed for every tile pair."],
            confidence=Confidence(value=None, band="LOW", basis="computation failed, see warnings"),
        )

    change_dicts = [
        {"probability_raster_path": r.get("probability_raster_path"), "changed_area_px": r["changed_area_px"],
         "changed_area_pct": r["changed_area_pct"], "mean_confidence": r["mean_confidence"]}
        for r in per_pair
    ]
    total_area_px = sum(b.width * b.height for b, _ in pairs) or 1
    merged = merge_change_maps(change_dicts, total_image_px=total_area_px)

    raster_paths = [c["probability_raster_path"] for c in change_dicts]
    if len(pairs) > 1 and all(raster_paths):
        mosaic_result = mosaic_and_recount_change_maps(raster_paths, after_id, total_area_px)
        if mosaic_result is not None:
            merged.update(mosaic_result)
    elif len(pairs) == 1:
        merged["probability_raster_path"] = raster_paths[0]

    change_map = ChangeMap(**merged)
    answer = _change_answer(query, merged)

    return Evidence(
        task=task, model_used=model_name, modality_used=modality_used, change_map=change_map,
        vqa_answer_raw=answer,
        confidence=Confidence(
            value=merged["mean_confidence"],
            band=_band(merged["mean_confidence"]),
            basis="Otsu threshold separation (between-class/total variance ratio) on this image pair's own change-magnitude histogram — a statistical separation proxy, not a calibrated probability",
        ),
        warnings=computation_warnings,
    )


def _change_answer(query: Optional[str], merged: dict) -> str:
    """Answer change questions from measured evidence without inventing causes."""
    pct = merged["changed_area_pct"]
    pixels = merged["changed_area_px"]
    confidence = merged["mean_confidence"] * 100
    current_query = (query or "").split("CURRENT USER QUESTION:")[-1].strip()
    q = current_query.lower()

    if any(word in q for word in ("why", "cause", "reason", "because")):
        focus = (
            "The pixel comparison cannot determine the cause. New construction, harvest, "
            "flooding, cloud effects, or seasonal variation can produce similar changes."
        )
    elif any(word in q for word in ("where", "location", "area", "region")):
        focus = (
            "The change map shows the spatial distribution of the changed pixels. "
            "A geographic place or feature label requires a trained semantic model."
        )
    elif any(word in q for word in ("road", "roads", "building", "buildings", "river", "water", "development")):
        focus = (
            "This pixel comparison cannot verify whether a specific feature such as roads, "
            "buildings, rivers, or development changed. The heatmap shows pixel shifts; "
            "feature-level identification requires a trained grounding or change-VQA model."
        )
    elif any(word in q for word in ("how much", "percentage", "percent", "extent", "amount")):
        focus = f"The measured extent is {pct:.2f}% of the valid analyzed area ({pixels} pixels)."
    elif any(word in q for word in ("deep", "detail", "analysis", "summary")):
        focus = (
            f"The measured extent is {pct:.2f}% of the valid analyzed area ({pixels} pixels), "
            f"with an Otsu separation score of {confidence:.2f}%."
        )
    else:
        focus = f"The measured extent is {pct:.2f}% of the valid analyzed area ({pixels} pixels)."

    return (
        f"{focus} This is deterministic pixel-change analysis: it identifies where pixel "
        "values shifted between the images, not what caused the shift. "
        f"The statistical separation score is {confidence:.2f}%, not a calibrated probability."
    )


def _placeholder_evidence(
    task: TaskType, model_name: str, tiles: list[Tile], query: Optional[str], modality_used: list[Modality],
) -> Evidence:
    """SINGLE_IMAGE_VQA / CAPTIONING / GROUNDING / OPTICAL_SAR_FUSION —
    see this module's docstring on why these stay placeholders. Real,
    honestly-labeled per-band statistics from the actual tile are
    attached (via `stats`) alongside the placeholder note rather than in
    place of it — real numbers, just not a semantic description."""
    tile = tiles[0]
    stats = _real_tile_stats(tile)

    if task in (TaskType.SINGLE_IMAGE_VQA, TaskType.CAPTIONING):
        return Evidence(
            task=task, model_used=model_name, modality_used=modality_used,
            vqa_answer_raw=_PLACEHOLDER_NOTICE + (f" (question asked: '{query}')" if query else ""),
            stats=stats,
            confidence=Confidence(value=None, band="LOW", basis="no model loaded — see vqa_answer_raw"),
        )

    if task == TaskType.GROUNDING:
        return Evidence(
            task=task, model_used=model_name, modality_used=modality_used,
            vqa_answer_raw=_PLACEHOLDER_NOTICE + (f" (looking for: '{query}')" if query else ""),
            detections=[], stats=stats,
            confidence=Confidence(value=None, band="LOW", basis="no model loaded — see vqa_answer_raw"),
        )

    if task == TaskType.OPTICAL_SAR_FUSION:
        return Evidence(
            task=task, model_used=model_name, modality_used=modality_used,
            vqa_answer_raw=_PLACEHOLDER_NOTICE, detections=[], stats=stats,
            confidence=Confidence(value=None, band="LOW", basis="no model loaded — see vqa_answer_raw"),
        )

    raise AssertionError(f"unreachable: task {task!r} not handled")


def _real_tile_stats(tile: Tile) -> dict:
    """Genuinely computed from the actual tile on disk — per-band mean
    and standard deviation. Real, differentiated by actual input; not a
    claim about what's depicted, just its raw pixel statistics."""
    try:
        import rasterio

        with rasterio.open(tile.array_path) as src:
            arr = src.read().astype("float64")
        return {
            f"band_{i}_mean": float(arr[i].mean()) for i in range(arr.shape[0])
        } | {
            f"band_{i}_std": float(arr[i].std()) for i in range(arr.shape[0])
        }
    except Exception:
        return {}


def _band(score: Optional[float]) -> str:
    if score is None:
        return "LOW"
    if score >= 0.75:
        return "HIGH"
    if score >= 0.4:
        return "MEDIUM"
    return "LOW"
