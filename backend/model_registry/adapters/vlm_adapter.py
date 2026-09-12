"""
VLM adapter — Section 3.4: single-image VQA + captioning.

Wraps a HF-style vision-language model (the specific base checkpoint is
one of Part 6's open decisions, Section 3.6 — this adapter only knows the
*shape* of that model's input/output, not which checkpoint is loaded).

Unlike grounding/change-detection, which genuinely need per-tile precision
(see those adapters' own docstrings on why), VQA/captioning needs the
model to see the *whole* scene, not one 1024x1024 corner of it — a VLM
resizes whatever it's given down to its own fixed input resolution anyway,
so seeing the full extent at lower effective detail beats seeing one tile
at full detail. On a single-tile image (the image fit in one tile to begin
with) this is a no-op: tiles[0].array_path is already the whole scene.

Expected model interface (whatever `loader.get()` returns must provide):
    model.generate(image_path: str, prompt: str) -> {"text": str, "score": float}
"""
from __future__ import annotations

from typing import Any

from backend.model_registry.adapters.base import BaseAdapter, confidence_band
from backend.shared.schemas import Confidence, Evidence, Modality, ModelRegistryEntry, TaskType, Tile

_DEFAULT_CAPTION_PROMPT = "Describe this satellite image in one factual sentence."


class VLMAdapter(BaseAdapter):
    handles = (TaskType.SINGLE_IMAGE_VQA, TaskType.CAPTIONING)

    def __init__(self, query: str | None = None):
        # query set -> SINGLE_IMAGE_VQA; query unset -> CAPTIONING.
        self.query = query

    def preprocess(self, tiles: list[Tile], entry: ModelRegistryEntry) -> Any:
        if not tiles:
            raise ValueError("VLMAdapter.preprocess requires at least one tile.")
        return {"image_path": _build_scene_preview(tiles), "prompt": self.query or _DEFAULT_CAPTION_PROMPT}

    def infer(self, model: Any, model_input: Any) -> Any:
        return model.generate(image_path=model_input["image_path"], prompt=model_input["prompt"])

    def postprocess(
        self, raw_output: Any, tiles: list[Tile], entry: ModelRegistryEntry, modality_used: list[Modality],
    ) -> Evidence:
        task = TaskType.SINGLE_IMAGE_VQA if self.query else TaskType.CAPTIONING
        text = raw_output.get("text", "") if isinstance(raw_output, dict) else str(raw_output)
        score = float(raw_output.get("score", 0.0)) if isinstance(raw_output, dict) else 0.0
        return Evidence(
            task=task,
            model_used=entry.name,
            modality_used=modality_used,
            vqa_answer_raw=text,
            stats={"generation_score": score} if score else {},
            confidence=Confidence(
                value=score or None,
                band=confidence_band(score),
                basis="VLM decoder confidence score",
            ),
            warnings=[] if text else ["Model returned an empty response."],
        )


def _build_scene_preview(tiles: list[Tile]) -> str:
    """
    Single tile -> its array_path directly, no mosaic needed. Multiple
    tiles -> reassembled into one array at each tile's (row_off, col_off)
    and written out as one preview image.

    Tiles overlap by design (tiling.py's overlap_pct), so a later tile in
    the loop simply overwrites an earlier one across the overlap margin --
    fine for a downsampled scene preview, unlike the pixel-exact dedup
    grounding/change-detection need for *their* per-tile outputs (handled
    there via NMS / raster mosaicking instead, which this doesn't need).

    Relies on Part 3's ingest-time pixel-count cap to keep the
    reconstructed canvas a reasonable size — this does hold the whole
    scene in memory at once, which is fine for anything that already
    passed that cap, but is deliberately not re-implementing a second,
    separate size limit here.
    """
    if len(tiles) == 1:
        return tiles[0].array_path

    import numpy as np
    import rasterio

    from backend.preprocessing import store

    canvas_width = max(t.col_off + t.width for t in tiles)
    canvas_height = max(t.row_off + t.height for t in tiles)

    canvas = None
    for tile in tiles:
        with rasterio.open(tile.array_path) as src:
            arr = src.read()  # (bands, height, width)
        if canvas is None:
            canvas = np.zeros((arr.shape[0], canvas_height, canvas_width), dtype=arr.dtype)
        canvas[:, tile.row_off : tile.row_off + tile.height, tile.col_off : tile.col_off + tile.width] = arr

    out_path = f"{store.stitched_dir_for(tiles[0].image_id)}/vqa_scene_preview.png"
    _write_preview_png(canvas, out_path)
    return out_path


def _write_preview_png(canvas: "Any", out_path: str) -> None:
    """canvas: (bands, height, width). Tiles are already per-modality
    normalized (tiling.py), but "normalized" doesn't guarantee uint8
    specifically -- scaling to 0-255 here is defensive, not redundant."""
    import numpy as np
    from PIL import Image

    if canvas.dtype != np.uint8:
        finite = canvas[np.isfinite(canvas)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        span = (hi - lo) or 1.0
        canvas = np.clip((canvas.astype(np.float64) - lo) / span * 255.0, 0, 255).astype(np.uint8)

    band_count = canvas.shape[0]
    if band_count == 1:
        img = Image.fromarray(canvas[0], mode="L")
    elif band_count >= 3:
        img = Image.fromarray(np.transpose(canvas[:3], (1, 2, 0)), mode="RGB")
    else:  # exactly 2 bands -- pad rather than guess which single band to show
        padded = np.zeros((3, *canvas.shape[1:]), dtype=np.uint8)
        padded[:2] = canvas
        img = Image.fromarray(np.transpose(padded, (1, 2, 0)), mode="RGB")

    img.save(out_path)
