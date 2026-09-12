"""
Real server-side thumbnail generation for satellite imagery — GeoTIFF
(and anything else rasterio opens) can't be shown in a browser <img> tag
directly, and frontend/index.html already says so honestly instead of
faking a client-side preview (see its own comment where previewUrl is
left unset for these). This is the actual fix: a real PNG rendered from
the actual pixels, server-side, that the frontend can point an <img> at
like any other image.

Reads at reduced resolution directly (rasterio's `out_shape` decimated
read) rather than reading the full array and downsampling in Python —
GB-scale sources stay GB-scale on disk, not in this process's memory,
same "large file -> tiling/indexing, never a full in-memory load"
principle Part 3's other modules already follow (see tiling.py,
cog.py).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from backend.preprocessing import store

_DEFAULT_MAX_DIM = 512
_CACHE_SUBDIR = "thumbnail"


def thumbnail_path_for(image_id: str, max_dim: int = _DEFAULT_MAX_DIM) -> str:
    return os.path.join(store.image_dir(image_id), _CACHE_SUBDIR, f"thumb_{max_dim}.png")


def get_or_generate_thumbnail(image_id: str, max_dim: int = _DEFAULT_MAX_DIM) -> str:
    """Returns a real PNG path, generating (and caching to disk) it on
    first request. Safe to call on every page load — subsequent calls
    for the same image_id/max_dim just return the cached file instantly
    instead of re-reading the source raster."""
    cached = thumbnail_path_for(image_id, max_dim)
    if os.path.isfile(cached):
        return cached

    import numpy as np
    import rasterio

    source_path = store.cog_path_for(image_id)
    with rasterio.open(source_path) as src:
        scale = min(1.0, max_dim / max(src.width, src.height))
        out_height = max(1, round(src.height * scale))
        out_width = max(1, round(src.width * scale))
        # Decimated read: rasterio/GDAL does the downsampling while
        # reading, so this never materializes the full-resolution array —
        # the whole point on a multi-GB source.
        arr = src.read(out_shape=(src.count, out_height, out_width)).astype("float64")
        nodata = src.nodata

    Path(cached).parent.mkdir(parents=True, exist_ok=True)
    _write_thumbnail_png(arr, nodata, cached)
    return cached


def _write_thumbnail_png(arr, nodata: Optional[float], out_path: str) -> None:
    """arr: (bands, H, W). Same band-count dispatch and dtype-to-uint8
    rescaling as model_registry/adapters/vlm_adapter.py's
    _write_preview_png — kept as its own copy rather than a shared import
    since these two live in different architectural layers (Part 3 here,
    Part 4 there) for different purposes, and the logic is genuinely
    small enough that duplicating it costs less than coupling those two
    layers together for it."""
    import numpy as np
    from PIL import Image

    valid = np.ones(arr.shape[1:], dtype=bool)
    if nodata is not None:
        valid &= ~np.any(arr == nodata, axis=0)

    finite = arr[:, valid][np.isfinite(arr[:, valid])] if valid.any() else arr[np.isfinite(arr)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    span = (hi - lo) or 1.0
    scaled = np.clip((arr - lo) / span * 255.0, 0, 255).astype("uint8")
    scaled[:, ~valid] = 0  # nodata rendered as black rather than an arbitrary stretched value

    band_count = scaled.shape[0]
    if band_count == 1:
        img = Image.fromarray(scaled[0], mode="L")
    elif band_count >= 3:
        img = Image.fromarray(np.transpose(scaled[:3], (1, 2, 0)), mode="RGB")
    else:  # exactly 2 bands -- pad rather than guess which single band to show
        padded = np.zeros((3, *scaled.shape[1:]), dtype="uint8")
        padded[:2] = scaled
        img = Image.fromarray(np.transpose(padded, (1, 2, 0)), mode="RGB")

    img.save(out_path)
