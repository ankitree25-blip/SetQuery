"""
Real, deterministic (non-ML) bi-temporal change detection — Change Vector
Analysis (CVA) + Otsu thresholding, band-agnostic on purpose: never
assumes which band means what (shared.schemas.Modality's own comment on
why band semantics are never inferred from pixel values applies here too).

This is what `configure_engine(use_mock=True)` now actually runs for
CHANGE_DETECTION/CHANGE_VQA — the default until a trained checkpoint
exists for `satquery-change-bitemporal` (see models.yaml). Unlike the
other three mock task types (VQA/captioning, grounding, fusion — see
mock_registry.py's own module docstring), this doesn't return the same
fixed numbers regardless of input: CVA + Otsu is real, established
remote-sensing science (a standard *baseline* change-detection technique
that predates deep learning, not a replacement for a trained one),
computed from the actual uploaded pixels every time. Two different image
pairs genuinely produce two different results.

What this can and can't honestly do:
  - CAN: say how much changed and roughly where, from real pixel math,
    for any band count/combination (multispectral optical, SAR, whatever
    coregistered pair it's given) — no assumption about which band is
    "red" or "NIR" anywhere in here.
  - CAN'T: say WHAT changed semantically. A new building, bare soil after
    harvest, and a dry lakebed all look like "these pixels changed a
    lot" to CVA — telling them apart needs a trained model.
    change_detection_adapter.py's optional CHANGE_VQA text path already
    only calls model.answer() when an actual model provides one — this
    module doesn't implement it, so a CHANGE_VQA query against the mock
    engine gets the real numeric change_map above plus an honest note
    that semantic interpretation needs a trained model, not a guessed
    label dressed up as one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def detect_change(before_path: str, after_path: str) -> dict:
    """Real file I/O (rasterio) around _compute_change's pure-numpy math
    — kept as a thin wrapper specifically so the math can be tested
    against synthetic arrays without needing real GeoTIFFs on disk."""
    import rasterio

    with rasterio.open(before_path) as src:
        before = src.read().astype("float64")
        before_nodata = src.nodata
    with rasterio.open(after_path) as src:
        after = src.read().astype("float64")
        after_nodata = src.nodata

    result = _compute_change(before, after, before_nodata, after_nodata)

    if result["probability_array"] is not None:
        try:
            result["probability_raster_path"] = _write_probability_raster(
                result["probability_array"], before_path
            )
        except Exception:
            result["probability_raster_path"] = None
    else:
        result["probability_raster_path"] = None
    del result["probability_array"]

    return result


def _compute_change(before, after, before_nodata: Optional[float], after_nodata: Optional[float]) -> dict:
    """
    before, after: (bands, H, W) float arrays, same shape (tiling.py tiles
    both images on one identical grid for exactly this reason — see its
    own module docstring — so a shape mismatch here means the two source
    images genuinely have different resolutions, not a tiling bug).

    Returns changed_area_px / changed_area_pct / mean_confidence (all
    real numbers derived from these actual arrays) plus
    probability_array (a same-shape float array in [0,1], or None) for
    the caller to write out as a raster — kept separate from file I/O so
    this whole function runs on synthetic arrays with no rasterio needed.
    """
    import numpy as np

    if before.shape != after.shape:
        raise ValueError(
            f"detect_change got mismatched tile shapes: before={before.shape}, "
            f"after={after.shape}. Tiles are supposed to come from the same "
            f"grid — tiling.py tiles both images identically for exactly "
            f"this reason — so this means the two source images genuinely "
            f"have different resolutions, not a tiling bug."
        )

    valid_mask = np.ones(before.shape[1:], dtype=bool)
    if before_nodata is not None:
        valid_mask &= ~np.any(before == before_nodata, axis=0)
    if after_nodata is not None:
        valid_mask &= ~np.any(after == after_nodata, axis=0)

    # Basic saturation heuristic for cloud/glare (all bands near the
    # observed max) and deep shadow (all bands near zero in both images)
    # — NOT a real cloud-mask model, just excludes the most obviously
    # unusable pixels from the change statistics so "cloud today, clear
    # yesterday" doesn't get counted as land-cover change. A real
    # cloud-detection model would do much better; this is the "where
    # possible" floor, not a claim of robust cloud handling.
    band_max = float(max(before.max(initial=0.0), after.max(initial=0.0))) or 1.0
    likely_cloud = np.all(after > 0.97 * band_max, axis=0) | np.all(before > 0.97 * band_max, axis=0)
    likely_deep_shadow = np.all(after < 0.02 * band_max, axis=0) & np.all(before < 0.02 * band_max, axis=0)
    valid_mask &= ~likely_cloud & ~likely_deep_shadow

    total_valid = int(valid_mask.sum())
    if total_valid == 0:
        return {
            "changed_area_px": 0, "changed_area_pct": 0.0, "mean_confidence": 0.0,
            "probability_array": None,
            "warning": "Every pixel in this tile pair was nodata/cloud/deep-shadow — no valid pixels to compare.",
        }

    # Change Vector Analysis: per-pixel Euclidean distance across
    # whatever bands are present. Band-agnostic on purpose — see this
    # module's docstring.
    diff = after - before
    magnitude = np.sqrt(np.sum(diff ** 2, axis=0))
    valid_magnitudes = magnitude[valid_mask]

    if float(valid_magnitudes.max()) == float(valid_magnitudes.min()):
        # Every valid pixel identical (e.g. the same file uploaded twice
        # as both before and after) — correctly "zero change" rather than
        # an arbitrary Otsu split of a zero-variance histogram.
        changed_mask = np.zeros_like(valid_mask)
        confidence = 0.0
        probability_array = np.zeros(before.shape[1:], dtype="float64")
        probability_array[~valid_mask] = float("nan")
    else:
        from skimage.filters import threshold_otsu

        threshold = float(threshold_otsu(valid_magnitudes))
        changed_mask = valid_mask & (magnitude > threshold)

        # Confidence proxy: Otsu's own between-class/total variance ratio
        # at its chosen split — literally the criterion Otsu's method
        # maximizes when picking the threshold, reused here as "how
        # cleanly bimodal is this image pair's own change-magnitude
        # histogram". A crisp separation (obvious real change somewhere
        # in the scene) scores high; a fuzzy, barely-bimodal one (subtle
        # or noise-driven differences) scores low. Real signal computed
        # from this pair's own histogram, not a fixed number.
        below = valid_magnitudes[valid_magnitudes <= threshold]
        above = valid_magnitudes[valid_magnitudes > threshold]
        total_var = float(valid_magnitudes.var())
        if below.size and above.size and total_var > 0:
            between_class_var = (below.size * above.size * (above.mean() - below.mean()) ** 2) / (valid_magnitudes.size ** 2)
            confidence = float(min(max(between_class_var / total_var, 0.0), 1.0))
        else:
            confidence = 0.0

        # Soft threshold (sigmoid centered on Otsu's cut) -> a genuine
        # [0,1] probability-like value per pixel, not just the binary
        # changed_mask -- consistent with mosaic_and_recount_change_maps'
        # own >0.5 convention when multiple tiles' rasters get merged.
        spread = float(valid_magnitudes.std()) or 1.0
        probability_array = 1.0 / (1.0 + np.exp(-(magnitude - threshold) / spread))
        probability_array[~valid_mask] = float("nan")

    changed_px = int(changed_mask.sum())
    changed_pct = 100.0 * changed_px / total_valid

    return {
        "changed_area_px": changed_px,
        "changed_area_pct": changed_pct,
        "mean_confidence": confidence,
        "probability_array": probability_array,
    }


def _write_probability_raster(probability_array, reference_path: str) -> str:
    """Writes the per-pixel change-probability array out as a single-band
    GeoTIFF, copying CRS/transform from the (already-real) before-image
    tile it was computed from, next to where change_detection_adapter.py
    already knows to look (backend/preprocessing/store.py's
    stitched_dir_for — same convention mosaic_and_recount_change_maps
    already uses for its own merged output)."""
    import rasterio

    from backend.preprocessing import store

    with rasterio.open(reference_path) as ref:
        profile = {
            "driver": "GTiff", "height": ref.height, "width": ref.width, "count": 1,
            "dtype": "float32", "crs": ref.crs, "transform": ref.transform, "compress": "DEFLATE",
        }
        image_id = ref.tags().get("SATQUERY_IMAGE_ID") or Path(reference_path).stem

    out_path = f"{store.stitched_dir_for(image_id)}/change_probability_{Path(reference_path).stem}.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(probability_array.astype("float32"), 1)
    return out_path
