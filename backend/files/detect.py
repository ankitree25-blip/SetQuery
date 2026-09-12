"""
File-type detection for Section 5 (multi-file upload): a thin extension-
based router that decides which category an uploaded file falls into, so
api/main.py knows whether to hand it to the existing Part 3 satellite-image
pipeline or to one of the extractors in extractors.py.

Deliberately extension-based, not content-sniffed -- the same tradeoff
Part 3 already makes for SUPPORTED_EXTENSIONS in preprocessing/config.py:
simple and predictable beats "clever" for a classification step that
decides what parsing code touches the file next.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Optional

from shared.schemas import FileCategory

_SATELLITE_EXTENSIONS = {".tif", ".tiff", ".geotiff"}
# .shp deliberately NOT in this set -- see _is_shapefile_bundle_zip()'s
# docstring for why a bare .shp can't be handled the same way .geojson is.
_VECTOR_EXTENSIONS = {".geojson", ".kml", ".kmz"}
_TABULAR_EXTENSIONS = {".csv", ".xlsx", ".parquet"}
_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
_PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_SCIENTIFIC_EXTENSIONS = {".nc", ".hdf", ".h5"}

# Formats still genuinely not covered. HDF4 (classic .hdf without a NetCDF-4/
# HDF5 container) is the one real gap left in _SCIENTIFIC_EXTENSIONS above:
# h5py can only open HDF5, and most modern ".hdf" files in practice *are*
# HDF5-based (NetCDF-4 is itself HDF5 underneath), so .hdf is accepted and
# routed to h5py -- extract_scientific_metadata's error path documents the
# fallback if that guess is wrong for a given file.
_KNOWN_NOT_YET_IMPLEMENTED: dict[str, str] = {}


def _is_shapefile_bundle_zip(raw_bytes: Optional[bytes]) -> bool:
    """True if a .zip contains at least a .shp member (plus, normally,
    .dbf/.shx alongside it).

    Why this exists instead of just adding ".shp" to _VECTOR_EXTENSIONS:
    a Shapefile is 3-4 sibling files (.shp/.dbf/.shx/.prj) that only work
    together, and api/main.py's upload_project_files() saves every file in
    a batch as f"{file_id}_{filename}" -- a fresh uuid prefix *per file*.
    Even if a user selects all four parts in one upload, they'd land on
    disk as e.g. "a1b2_data.shp" and "c3d4_data.dbf": different basenames,
    so GDAL's automatic sidecar lookup (same basename, different
    extension) can't find them next to each other. A single .zip
    containing all parts together is one upload with one file_id, so it
    doesn't hit that problem -- which is also just the normal way
    Shapefiles get shared in practice (ArcGIS Online, Mapbox Studio, and
    most GIS tools all accept "shapefile.zip" for exactly this reason).

    Takes raw bytes, not a path: main.py's upload_project_files() decides
    the category *before* the upload is written to disk (the destination
    filename needs project_file.file_id, generated from the category-
    tagged ProjectFile itself), so there's no path to peek at yet at this
    point -- only the bytes already read via `await upload.read()`.
    zipfile reads an in-memory buffer the same way it reads a file on
    disk, so this never needs to touch disk just to decide the category.
    """
    if not raw_bytes:
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            return any(name.lower().endswith(".shp") for name in zf.namelist())
    except zipfile.BadZipFile:
        return False


def detect_category(filename: str, raw_bytes: Optional[bytes] = None) -> FileCategory:
    """`raw_bytes`, when given, is the upload's actual content -- needed
    only to peek inside a .zip and confirm it's a Shapefile bundle before
    claiming VECTOR for it (see _is_shapefile_bundle_zip). Every other
    format is still decided from the filename alone, unchanged from
    before."""
    ext = Path(filename).suffix.lower()
    if ext in _SATELLITE_EXTENSIONS:
        return FileCategory.SATELLITE_IMAGE
    if ext == ".shp":
        # A bare .shp on its own is never actually openable -- see
        # _is_shapefile_bundle_zip's docstring. Routing it to VECTOR here
        # anyway (rather than UNSUPPORTED) means unsupported_reason() below
        # can give the specific "zip it up" guidance instead of a generic
        # unsupported-format message, and extract_vector_metadata's own
        # .shp branch raises a clear error if the sidecars really are
        # missing when it actually tries to open the file.
        return FileCategory.VECTOR
    if ext == ".zip":
        return FileCategory.VECTOR if _is_shapefile_bundle_zip(raw_bytes) else FileCategory.UNSUPPORTED
    if ext in _VECTOR_EXTENSIONS:
        return FileCategory.VECTOR
    if ext in _TABULAR_EXTENSIONS:
        return FileCategory.TABULAR
    if ext in _DOCUMENT_EXTENSIONS:
        return FileCategory.DOCUMENT
    if ext in _PHOTO_EXTENSIONS:
        return FileCategory.PHOTO
    if ext in _SCIENTIFIC_EXTENSIONS:
        return FileCategory.SCIENTIFIC_DATA
    return FileCategory.UNSUPPORTED


def unsupported_reason(filename: str, raw_bytes: Optional[bytes] = None) -> Optional[str]:
    """None for a handled category; otherwise a human-readable reason.
    `raw_bytes` is forwarded to detect_category for the .zip/Shapefile
    check -- see its docstring."""
    ext = Path(filename).suffix.lower()
    if ext == ".shp":
        return ("a bare .shp can't be read on its own (it needs its .dbf/.shx "
                "files, and this upload path can't keep sidecar files paired "
                "up) -- zip the .shp together with its .dbf/.shx/.prj and "
                "upload the .zip instead")
    if ext == ".zip" and not _is_shapefile_bundle_zip(raw_bytes):
        return "zip archive without a .shp inside -- only Shapefile bundles are supported as .zip uploads"
    known = _KNOWN_NOT_YET_IMPLEMENTED.get(ext)
    if known:
        return f"{known} ({ext}) is on the Section 5 format list but not implemented yet"
    if detect_category(filename, raw_bytes) == FileCategory.UNSUPPORTED:
        return f"unrecognized file extension: {ext or '(none)'}"
    return None
