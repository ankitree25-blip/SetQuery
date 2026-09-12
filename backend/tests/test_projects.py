"""
Tests for the Section 2 (project/workspace) addition: projects/store.py.
Run with `pytest` from the repo root, same as the rest of the suite.

The one test that matters most here is test_persistence_across_restart:
the entire point of ProjectStore (vs. agent/store.py's in-memory
ImageStore) is surviving a process restart, so that's exercised directly
by pointing two separate ProjectStore instances at the same directory
rather than trusting a single long-lived instance's own cache.
"""

from __future__ import annotations

import pytest

from projects.store import ProjectStore
from shared.schemas import FileCategory, ProjectFile


def make_file(file_id="f1", filename="scene.tif",
              category=FileCategory.SATELLITE_IMAGE) -> ProjectFile:
    return ProjectFile(
        file_id=file_id, filename=filename, category=category,
        size_bytes=1234, uploaded_at="2026-01-01T00:00:00+00:00",
    )


def test_create_and_get_roundtrip(tmp_path):
    store = ProjectStore(base_dir=str(tmp_path))

    project = store.create("Agricultural Change Study")

    fetched = store.get(project.project_id)
    assert fetched is not None
    assert fetched.project_id == project.project_id
    assert fetched.name == "Agricultural Change Study"
    assert fetched.files == []
    assert fetched.updated_at >= fetched.created_at


def test_get_unknown_project_returns_none(tmp_path):
    store = ProjectStore(base_dir=str(tmp_path))
    assert store.get("does-not-exist") is None


def test_add_file_appends_and_persists(tmp_path):
    store = ProjectStore(base_dir=str(tmp_path))
    project = store.create("Flood Study")

    updated = store.add_file(project.project_id, make_file())

    assert len(updated.files) == 1
    assert updated.files[0].file_id == "f1"
    assert updated.updated_at >= updated.created_at

    # add_file's return value shouldn't be the only place the change landed
    refetched = store.get(project.project_id)
    assert len(refetched.files) == 1


def test_add_file_to_unknown_project_raises(tmp_path):
    store = ProjectStore(base_dir=str(tmp_path))
    with pytest.raises(KeyError):
        store.add_file("does-not-exist", make_file())


def test_list_sorted_most_recently_updated_first(tmp_path):
    store = ProjectStore(base_dir=str(tmp_path))
    older = store.create("Older project")
    newer = store.create("Newer project")
    # Touch "older" so it becomes the most recently *updated* -- list()
    # sorts by updated_at, not created_at.
    store.add_file(older.project_id, make_file())

    listed_ids = [p.project_id for p in store.list()]

    assert listed_ids[0] == older.project_id
    assert newer.project_id in listed_ids


def test_list_skips_a_corrupt_project_file(tmp_path):
    store = ProjectStore(base_dir=str(tmp_path))
    good = store.create("Good project")
    (tmp_path / "corrupt.json").write_text("{not valid json")

    listed_ids = [p.project_id for p in store.list()]

    assert listed_ids == [good.project_id]  # corrupt file skipped, not raised


def test_persistence_across_restart(tmp_path):
    """The actual point of this feature: a second, independent
    ProjectStore pointed at the same directory (standing in for the
    process having restarted) sees exactly what the first one wrote."""
    store_before_restart = ProjectStore(base_dir=str(tmp_path))
    project = store_before_restart.create("Agricultural Change Study")
    store_before_restart.add_file(project.project_id, make_file())

    store_after_restart = ProjectStore(base_dir=str(tmp_path))
    recovered = store_after_restart.get(project.project_id)

    assert recovered is not None
    assert recovered.name == "Agricultural Change Study"
    assert len(recovered.files) == 1
    assert recovered.files[0].filename == "scene.tif"


# NEW (UI redesign): the sidebar's per-project "..." menu needed somewhere
# to actually rename/delete a project.

def test_rename_persists_and_bumps_updated_at(tmp_path):
    store = ProjectStore(base_dir=str(tmp_path))
    project = store.create("Flood Study")

    renamed = store.rename(project.project_id, "Flood Study 2024")

    assert renamed.name == "Flood Study 2024"
    assert renamed.updated_at >= project.updated_at
    assert store.get(project.project_id).name == "Flood Study 2024"


def test_rename_unknown_project_raises(tmp_path):
    store = ProjectStore(base_dir=str(tmp_path))
    with pytest.raises(KeyError):
        store.rename("does-not-exist", "New name")


def test_delete_removes_project(tmp_path):
    store = ProjectStore(base_dir=str(tmp_path))
    project = store.create("Scratch project")

    assert store.delete(project.project_id) is True
    assert store.get(project.project_id) is None
    assert project.project_id not in [p.project_id for p in store.list()]


def test_delete_unknown_project_returns_false(tmp_path):
    store = ProjectStore(base_dir=str(tmp_path))
    assert store.delete("does-not-exist") is False
