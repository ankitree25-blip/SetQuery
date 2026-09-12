"""
Cloud-Optimized GeoTIFF conversion on ingest (architecture.md 3.3 "In scope").

Uses GDAL's native COG driver (GDAL >= 3.1) via rasterio.shutil.copy, rather
than adding rio-cogeo as an extra dependency — the COG driver already
handles internal tiling and overview generation in one pass. This is I/O
only and depends on rasterio, so it isn't executable in the build sandbox
(see PART3_BUILD_NOTES.md); the logic below follows rasterio's documented
COG-driver usage pattern.

Bug fix (post-merge, first real-hardware run): the COG driver's internal
overview/tiling pass always builds a pixel/line -> geographic transformer,
even when no reprojection is actually requested. A source with neither an
affine geotransform nor GCPs — a bare scientific TIFF, e.g. a raw exported
band with no CRS tags, as opposed to a real satellite product, which is
essentially always shipped georeferenced — makes that transformer
uncomputable, and GDAL raises:

    Unable to compute a transformation between pixel/line and
    georeferenced coordinates for <path>. There is no affine
    transformation and no GCPs. Specify transformation option
    SRC_METHOD=NO_GEOTRANSFORM to bypass this check.

uncaught, straight out of rio_copy. has_real_georeference() below is the
check; convert_to_cog() now stages a placeholder transform/CRS onto a
*copy* of the source (never src_path itself) when that check fails, so the
COG driver always has something valid to compute with. This also
pre-empts the identical failure mode in coregistration.py's WarpedVRT,
which needs a real dataset.crs on both sides to pick a target CRS.
"""

from __future__ import annotations

# Clearly-synthetic placeholder — Web Mercator, anchored at the CRS's own
# origin, at a plausible-looking but arbitrary pixel size. It exists only so
# GDAL has *something* valid to compute a transform with; it is never a real
# position. convert_to_cog()'s return value (and, via it,
# ImageMetadata.georeferenced) is how callers know a given image is running
# on this placeholder rather than a genuine geotransform, so nothing
# downstream — map display, coordinates in a report, a coregistration check
# against another image — mistakes it for one.
PLACEHOLDER_CRS = "EPSG:3857"
PLACEHOLDER_PIXEL_SIZE_M = 10.0


def _is_identity_transform(transform) -> bool:
    """True for GDAL/rasterio's default placeholder affine (a,b,c,d,e,f) ==
    (1,0,0,0,1,0) — what a dataset reports when no geotransform was ever
    set, as opposed to a real (even if trivial) one someone assigned."""
    return tuple(transform)[:6] == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def has_real_georeference(dataset) -> bool:
    """
    Whether `dataset` carries enough georeferencing for GDAL to place its
    pixels in real-world space: a genuine affine geotransform, or ground
    control points. See this module's docstring for what happens without
    either.
    """
    has_transform = not _is_identity_transform(dataset.transform)
    has_gcps = bool(dataset.gcps[0])
    return has_transform or has_gcps


def _assign_placeholder_georeference(staging_path: str) -> None:
    """
    Tags the TIFF at staging_path — a working copy, never the original
    upload — with the placeholder transform/CRS above. Tries the cheap
    in-place tag update first (pixel data untouched); falls back to a full
    read + rewrite for the rare TIFF variant that won't update geo tags
    in place.
    """
    import rasterio
    from affine import Affine

    placeholder_transform = Affine(
        PLACEHOLDER_PIXEL_SIZE_M, 0.0, 0.0,
        0.0, -PLACEHOLDER_PIXEL_SIZE_M, 0.0,
    )
    try:
        with rasterio.open(staging_path, "r+") as staged:
            staged.transform = placeholder_transform
            staged.crs = PLACEHOLDER_CRS
    except Exception:
        with rasterio.open(staging_path) as src:
            profile = src.profile.copy()
            data = src.read()
        profile.update(transform=placeholder_transform, crs=PLACEHOLDER_CRS)
        with rasterio.open(staging_path, "w", **profile) as dst:
            dst.write(data)


def convert_to_cog(src_path: str, dst_path: str, compress: str = "DEFLATE", blocksize: int = 512) -> bool:
    """
    Converts the raster at src_path into a Cloud-Optimized GeoTIFF at
    dst_path. Never modifies src_path.

    Returns True if src_path already carried real georeferencing; False if
    it had neither a transform nor GCPs and the placeholder above had to be
    used instead — callers (validate_and_prepare) put this straight onto
    ImageMetadata.georeferenced.
    """
    import os
    import shutil
    import tempfile

    import rasterio
    from rasterio.shutil import copy as rio_copy

    creation_options = {
        "driver": "COG",
        "compress": compress,
        "blocksize": blocksize,
        "overview_resampling": "average",
        "bigtiff": "IF_SAFER",
    }

    with rasterio.open(src_path) as src:
        georeferenced = has_real_georeference(src)

    if georeferenced:
        with rasterio.open(src_path) as src:
            rio_copy(src, dst_path, **creation_options)
        return True

    staging_fd, staging_path = tempfile.mkstemp(suffix=".tif")
    os.close(staging_fd)
    try:
        shutil.copyfile(src_path, staging_path)
        _assign_placeholder_georeference(staging_path)
        with rasterio.open(staging_path) as staged:
            rio_copy(staged, dst_path, **creation_options)
    finally:
        if os.path.exists(staging_path):
            os.remove(staging_path)
    return False
