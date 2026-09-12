"""
Real (non-mocked) metadata extractors for Section 5's non-satellite file
categories. Satellite images (.tif/.tiff) do NOT go through here -- those
keep using the existing, far more rigorous Part 3 pipeline
(preprocessing.validate_and_prepare); duplicating that validation here
would be exactly the kind of unnecessary rewrite the upgrade brief asked
to avoid.

Every function below either returns a real, correctly-computed metadata
dict or raises -- none of them return a plausible-looking placeholder on
failure. Callers (api/main.py) are expected to catch the exception and
record it as a per-file warning rather than fail the whole batch upload.

KML/KMZ and Shapefile (via fiona), NetCDF and HDF/HDF5 (via netCDF4/h5py),
and Parquet (via pandas) are covered below too now. One caveat carried over
from files/detect.py: a bare .shp can't actually be opened (see
detect_category's docstring on why) -- extract_vector_metadata's .shp
branch exists mainly to raise a clear, specific error for that case rather
than let a confusing fiona driver error surface instead.
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def extract_vector_metadata(path: str) -> dict[str, Any]:
    """Dispatches on extension. GeoJSON keeps the original pure-stdlib
    path below (no reason to route it through fiona too). KML/KMZ/
    Shapefile/zip-bundle all go through fiona, added just below."""
    ext = Path(path).suffix.lower()
    if ext == ".geojson":
        return _extract_geojson_metadata(path)
    if ext in (".kml", ".kmz", ".shp", ".zip"):
        return _extract_fiona_metadata(path, ext)
    raise ValueError(f"extract_vector_metadata got an unexpected extension: {ext}")


def _extract_fiona_metadata(path: str, ext: str) -> dict[str, Any]:
    """KML, KMZ, Shapefile (bare .shp, sidecars alongside it), and Shapefile
    zip bundles, all via fiona/GDAL. Unlike GeoJSON, these formats enforce
    a single geometry type per file/layer, and fiona exposes the overall
    bounds directly (`collection.bounds`) instead of needing to walk raw
    coordinate arrays by hand."""
    if ext == ".kmz":
        # KMZ is a zip wrapping a doc.kml (+ resources) -- GDAL's KMZ
        # support varies by build, so extracting the inner .kml first and
        # reading that normally is more portable than trusting a KMZ
        # driver to be present. (Same "extract, don't rely on a zip
        # virtual-filesystem driver" choice as the .zip Shapefile case
        # below, and for the same reason: known to be version-dependent,
        # not just theoretically so -- see the extractor's docstring on
        # the .zip branch.)
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(path) as zf:
                kml_members = [n for n in zf.namelist() if n.lower().endswith(".kml")]
                if not kml_members:
                    raise ValueError("this .kmz has no .kml inside it")
                zf.extract(kml_members[0], tmp)
            return _read_fiona_collection(str(Path(tmp) / kml_members[0]))

    if ext == ".zip":
        # Shapefile bundle -- see detect.py's _is_shapefile_bundle_zip for
        # why this is how Shapefiles arrive at all. Extracted to a real
        # temp directory rather than opened via GDAL's zip://
        # virtual-filesystem support: that path is known to be flaky for
        # locally-stored zips across GDAL/fiona versions (unlike the
        # remote zip+https:// form, which is the one fiona's own docs
        # actually demonstrate) -- a plain extract-then-open sidesteps it
        # entirely instead of depending on exactly which GDAL build this
        # runs against.
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(path) as zf:
                zf.extractall(tmp)
                shp_members = [n for n in zf.namelist() if n.lower().endswith(".shp")]
            if not shp_members:
                raise ValueError("this .zip has no .shp inside it")
            return _read_fiona_collection(str(Path(tmp) / shp_members[0]))

    if ext == ".shp":
        try:
            return _read_fiona_collection(path)
        except Exception as exc:
            # The common case landing here is the one detect.py's
            # unsupported_reason() already warns about at upload time: a
            # lone .shp with its .dbf/.shx saved under different generated
            # filenames (see detect.py's _is_shapefile_bundle_zip
            # docstring), so fiona can't find them next to it. Re-raising
            # with that guidance inline means it's still the message that
            # reaches the user even via this function's own exception
            # path (api/main.py records str(exception) as the per-file
            # warning, not detect.py's unsupported_reason() text, since
            # detect_category() returned VECTOR here rather than
            # UNSUPPORTED -- see detect_category's docstring).
            raise ValueError(
                f"couldn't open this Shapefile ({exc}) -- if its .dbf/.shx/.prj "
                "weren't uploaded alongside it, zip all the Shapefile's files "
                "together and upload the .zip instead"
            ) from exc

    return _read_fiona_collection(path)  # .kml


def _read_fiona_collection(path: str) -> dict[str, Any]:
    import fiona

    with fiona.open(path) as collection:
        geometry_types: set[str] = set()
        feature_count = 0
        for feature in collection:
            feature_count += 1
            gtype = (feature.get("geometry") or {}).get("type")
            if gtype:
                geometry_types.add(gtype)
        bounds = collection.bounds if feature_count else None  # (minx, miny, maxx, maxy)

    return {
        "feature_count": feature_count,
        "geometry_types": sorted(geometry_types),
        "bbox": list(bounds) if bounds else None,
        "driver": collection.driver,
        "crs": str(collection.crs) if collection.crs else None,
    }


def _extract_geojson_metadata(path: str) -> dict[str, Any]:
    """GeoJSON only. Pure stdlib json -- no shapely/fiona dependency for
    this much: feature count, which geometry types are present (including
    inside a GeometryCollection), and a bounding box computed straight
    from the raw coordinate arrays."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gj_type = data.get("type")
    if gj_type == "FeatureCollection":
        features = data.get("features") or []
    elif gj_type == "Feature":
        features = [data]
    elif gj_type:
        # A bare geometry (Point, Polygon, ...) used as the whole file.
        features = [{"type": "Feature", "geometry": data, "properties": {}}]
    else:
        features = []

    geometry_types: set[str] = set()
    minx = miny = float("inf")
    maxx = maxy = float("-inf")

    def _walk_coords(coords) -> None:
        nonlocal minx, miny, maxx, maxy
        if not coords:
            return
        if isinstance(coords[0], (int, float)):
            x, y = coords[0], coords[1]
            minx, maxx = min(minx, x), max(maxx, x)
            miny, maxy = min(miny, y), max(maxy, y)
        else:
            for c in coords:
                _walk_coords(c)

    def _collect(geom: dict) -> None:
        if not geom:
            return
        gtype = geom.get("type")
        if gtype == "GeometryCollection":
            for g in geom.get("geometries") or []:
                _collect(g)
            return
        if gtype:
            geometry_types.add(gtype)
        _walk_coords(geom.get("coordinates"))

    for feat in features:
        _collect((feat or {}).get("geometry") or {})

    bbox = [minx, miny, maxx, maxy] if minx != float("inf") else None

    return {
        "feature_count": len(features),
        "geometry_types": sorted(geometry_types),
        "bbox": bbox,
    }


def extract_tabular_metadata(path: str, filename: str) -> dict[str, Any]:
    """CSV, XLSX, or Parquet via pandas: row/column counts, per-column
    dtype, and describe()-based summary stats for numeric columns only --
    this is metadata for a file browser, not a full profiling report
    (Section 11: "statistical summaries where useful", not exhaustive
    ones)."""
    import pandas as pd

    ext = Path(filename).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".xlsx":
        df = pd.read_excel(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)  # needs pyarrow (requirements.txt) as the read engine
    else:
        raise ValueError(f"extract_tabular_metadata got an unexpected extension: {ext}")

    numeric_summary: dict[str, Any] = {}
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        described = numeric_df.describe().to_dict()
        numeric_summary = {
            str(col): {
                str(stat): (None if pd.isna(val) else round(float(val), 4))
                for stat, val in stats.items()
            }
            for col, stats in described.items()
        }

    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns],
        "numeric_summary": numeric_summary,
    }


def extract_document_metadata(path: str, filename: str) -> dict[str, Any]:
    """PDF (pypdf), DOCX (python-docx), or plain TXT/MD: page/paragraph
    count where the format has one, plus a short text preview."""
    ext = Path(filename).suffix.lower()
    preview_chars = 500

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        first_page_text = reader.pages[0].extract_text() if reader.pages else ""
        return {
            "page_count": len(reader.pages),
            "text_preview": (first_page_text or "")[:preview_chars],
        }

    if ext == ".docx":
        import docx  # python-docx

        parsed = docx.Document(path)
        paragraphs = [p.text for p in parsed.paragraphs if p.text.strip()]
        return {
            "paragraph_count": len(paragraphs),
            "text_preview": "\n".join(paragraphs)[:preview_chars],
        }

    if ext in (".txt", ".md"):
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return {
            "char_count": len(text),
            "line_count": text.count("\n") + 1,
            "text_preview": text[:preview_chars],
        }

    raise ValueError(f"extract_document_metadata got an unexpected extension: {ext}")


def extract_photo_metadata(path: str) -> dict[str, Any]:
    """Plain JPG/PNG -- a field photo, not satellite data. Pillow is
    already a Part 3 dependency, so this adds nothing new to requirements.txt."""
    from PIL import Image

    with Image.open(path) as img:
        return {
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode,
        }


def extract_scientific_metadata(path: str, filename: str) -> dict[str, Any]:
    """NetCDF (.nc) via netCDF4, HDF/HDF5 (.hdf/.h5) via h5py: dimension
    names/sizes, a preview of variables/datasets (name, dtype, shape, and
    up to a handful of their own attributes), and global/root attributes.
    A preview, not an exhaustive dump -- same "Section 11: useful, not
    exhaustive" scope as extract_tabular_metadata's numeric_summary.

    HDF4 caveat: h5py only opens HDF5. Most contemporary ".hdf" files are
    HDF5-based in practice (NetCDF-4 is itself an HDF5 container), so .hdf
    is routed here rather than treated as unsupported -- but a genuine
    classic-HDF4 file will fail h5py.File() with an OSError, surfaced
    below as a clear message rather than a bare traceback.
    """
    ext = Path(filename).suffix.lower()
    max_variables_listed = 50  # preview cap -- see docstring

    if ext == ".nc":
        import netCDF4

        with netCDF4.Dataset(path, "r") as ds:
            dimensions = [
                {"name": name, "size": (None if dim.isunlimited() else dim.size),
                 "unlimited": dim.isunlimited()}
                for name, dim in ds.dimensions.items()
            ]
            variable_names = list(ds.variables.keys())
            variables = [
                {
                    "name": name,
                    "dtype": str(var.dtype),
                    "dimensions": list(var.dimensions),
                    "shape": list(var.shape),
                    "attrs": {a: _jsonable(var.getncattr(a)) for a in var.ncattrs()},
                }
                for name, var in list(ds.variables.items())[:max_variables_listed]
            ]
            global_attrs = {a: _jsonable(ds.getncattr(a)) for a in ds.ncattrs()}

        return {
            "format": "NetCDF",
            "dimensions": dimensions,
            "variable_count": len(variable_names),
            "variables": variables,
            "variables_truncated": len(variable_names) > max_variables_listed,
            "global_attrs": global_attrs,
        }

    if ext in (".hdf", ".h5"):
        import h5py

        try:
            f = h5py.File(path, "r")
        except OSError as exc:
            raise ValueError(
                f"couldn't open this as HDF5 ({exc}) -- if this is a classic "
                "HDF4 file (pre-HDF5 format), that variant isn't supported here"
            ) from exc

        datasets: list[dict[str, Any]] = []
        with f:
            def _collect(name: str, obj) -> None:
                if isinstance(obj, h5py.Dataset) and len(datasets) < max_variables_listed:
                    datasets.append({
                        "name": name,
                        "dtype": str(obj.dtype),
                        "shape": list(obj.shape),
                        "attrs": {k: _jsonable(v) for k, v in obj.attrs.items()},
                    })

            f.visititems(_collect)
            total_dataset_count = sum(1 for _, obj in _walk(f) if isinstance(obj, h5py.Dataset))
            global_attrs = {k: _jsonable(v) for k, v in f.attrs.items()}

        return {
            "format": "HDF5",
            "dataset_count": total_dataset_count,
            "datasets": datasets,
            "datasets_truncated": total_dataset_count > max_variables_listed,
            "global_attrs": global_attrs,
        }

    raise ValueError(f"extract_scientific_metadata got an unexpected extension: {ext}")


def _walk(group):
    """h5py's own visititems() doesn't hand back a plain iterable, and
    counting datasets separately from previewing the first N (above) is
    cheaper than building the full list just to len() it on a file with
    thousands of variables."""
    for name, obj in group.items():
        yield name, obj
        if hasattr(obj, "items"):  # it's a Group, recurse
            yield from _walk(obj)


def _jsonable(value: Any) -> Any:
    """netCDF4/h5py attribute values are frequently numpy scalars/arrays,
    which json.dumps (ProjectFile.metadata is serialized to disk as JSON
    via Project.model_dump_json) can't serialize directly."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
