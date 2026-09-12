"""
Tests for the Section 5 (multi-file upload) addition: files/detect.py and
files/extractors.py. Run with `pytest` from the repo root, same as the
rest of the suite.

These write small real fixture files to tmp_path and run the actual
extractors against them (pandas/pypdf/python-docx/Pillow really parsing a
real file) rather than mocking those libraries — the whole point of this
module is that it isn't a placeholder, so its tests shouldn't be either.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from PIL import Image

from files.detect import detect_category, unsupported_reason
from files.extractors import (
    extract_document_metadata,
    extract_photo_metadata,
    extract_tabular_metadata,
    extract_vector_metadata,
)
from shared.schemas import FileCategory


# --- detect_category / unsupported_reason -----------------------------------

@pytest.mark.parametrize("filename,expected", [
    ("scene.tif", FileCategory.SATELLITE_IMAGE),
    ("scene.TIFF", FileCategory.SATELLITE_IMAGE),
    ("aoi.geojson", FileCategory.VECTOR),
    ("measurements.csv", FileCategory.TABULAR),
    ("measurements.xlsx", FileCategory.TABULAR),
    ("report.pdf", FileCategory.DOCUMENT),
    ("report.docx", FileCategory.DOCUMENT),
    ("notes.md", FileCategory.DOCUMENT),
    ("field_photo.jpg", FileCategory.PHOTO),
    ("field_photo.png", FileCategory.PHOTO),
    ("boundary.kml", FileCategory.VECTOR),
    ("weird.xyz123", FileCategory.UNSUPPORTED),
])
def test_detect_category(filename, expected):
    assert detect_category(filename) == expected


def test_unsupported_reason_distinguishes_known_from_unknown():
    known = unsupported_reason("boundary.shp")
    unknown = unsupported_reason("mystery.xyz123")
    assert known is not None and "can't be read" in known
    assert unknown is not None and "unrecognized" in unknown
    assert unsupported_reason("scene.tif") is None  # handled category -> no reason


# --- extract_vector_metadata -------------------------------------------------

def test_extract_vector_metadata_feature_collection(tmp_path):
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [10.0, 20.0]}, "properties": {}},
            {"type": "Feature", "geometry": {"type": "Polygon",
                                              "coordinates": [[[0, 0], [0, 5], [5, 5], [5, 0], [0, 0]]]},
             "properties": {}},
        ],
    }
    path = tmp_path / "aoi.geojson"
    path.write_text(json.dumps(geojson))

    result = extract_vector_metadata(str(path))

    assert result["feature_count"] == 2
    assert result["geometry_types"] == ["Point", "Polygon"]
    assert result["bbox"] == [0, 0, 10.0, 20.0]


def test_extract_vector_metadata_bare_geometry(tmp_path):
    """A file that's just a Geometry, not a Feature/FeatureCollection --
    valid GeoJSON, and a real edge case a user's file could actually be."""
    path = tmp_path / "point.geojson"
    path.write_text(json.dumps({"type": "Point", "coordinates": [1.0, 2.0]}))

    result = extract_vector_metadata(str(path))

    assert result["feature_count"] == 1
    assert result["geometry_types"] == ["Point"]


# --- extract_tabular_metadata -------------------------------------------------

def test_extract_tabular_metadata_csv(tmp_path):
    path = tmp_path / "measurements.csv"
    path.write_text("name,score\nalice,10\nbob,20\ncarol,30\n")

    result = extract_tabular_metadata(str(path), "measurements.csv")

    assert result["row_count"] == 3
    assert result["column_count"] == 2
    assert result["numeric_summary"]["score"]["mean"] == 20.0
    assert "name" not in result["numeric_summary"]  # non-numeric column excluded


def test_extract_tabular_metadata_xlsx(tmp_path):
    path = tmp_path / "measurements.xlsx"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_excel(path, index=False)

    result = extract_tabular_metadata(str(path), "measurements.xlsx")

    assert result["row_count"] == 3
    assert "a" in result["numeric_summary"]


# --- extract_document_metadata -----------------------------------------------

def test_extract_document_metadata_docx(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "report.docx"
    d = docx.Document()
    d.add_paragraph("Hello world from a test document.")
    d.save(path)

    result = extract_document_metadata(str(path), "report.docx")

    assert result["paragraph_count"] == 1
    assert "Hello world" in result["text_preview"]


def test_extract_document_metadata_txt(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("line one\nline two\n")

    result = extract_document_metadata(str(path), "notes.txt")

    assert result["line_count"] == 3
    assert result["char_count"] == len("line one\nline two\n")


def test_extract_document_metadata_pdf(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    path = tmp_path / "blank.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)

    result = extract_document_metadata(str(path), "blank.pdf")

    assert result["page_count"] == 1


# --- extract_photo_metadata --------------------------------------------------

def test_extract_photo_metadata_png(tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (37, 51), color=(255, 0, 0)).save(path)

    result = extract_photo_metadata(str(path))

    assert result["width"] == 37
    assert result["height"] == 51
    assert result["format"] == "PNG"
