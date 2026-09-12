"""
Section 2 (PROJECT / WORKSPACE SYSTEM): a persistent Project store.

Deliberately disk-backed JSON, one file per project -- NOT the in-memory
dicts agent/store.py uses for ImageStore/UploadSessionStore. Those are
explicitly scoped to "resolve within one running process" (see that
file's docstring); a Project's entire purpose is surviving a server
restart ("the architecture must support returning later and continuing
the investigation" -- Section 2), so it has to actually hit disk, not
just RAM.

No database: the same "single-deployment demo" tradeoff the rest of this
repo already makes (agent/store.py; preprocessing/config.py's DATA_DIR).
One JSON file per project, named by project_id, is easy to inspect or
back up by hand and is more than fast enough at the project counts a
single local user will ever create. Swap this for a real database later
without any caller (api/main.py) needing to change.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from shared.schemas import Project, ProjectFile

PROJECTS_DIR = os.environ.get(
    "SATQUERY_PROJECTS_DIR", os.path.join(os.getcwd(), "data", "projects")
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectStore:
    def __init__(self, base_dir: Optional[str] = None) -> None:
        self._dir = Path(base_dir or PROJECTS_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, project_id: str) -> Path:
        return self._dir / f"{project_id}.json"

    def create(self, name: str) -> Project:
        now = _now_iso()
        project = Project(
            project_id=str(uuid.uuid4()), name=name, created_at=now, updated_at=now, files=[],
        )
        self._save(project)
        return project

    def get(self, project_id: str) -> Optional[Project]:
        path = self._path(project_id)
        if not path.exists():
            return None
        return Project.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[Project]:
        projects = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                projects.append(Project.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue  # a corrupt/partial file shouldn't take the whole list down
        return sorted(projects, key=lambda p: p.updated_at, reverse=True)

    def add_file(self, project_id: str, project_file: ProjectFile) -> Project:
        project = self.get(project_id)
        if project is None:
            raise KeyError(f"unknown project_id: {project_id}")
        project.files.append(project_file)
        project.updated_at = _now_iso()
        self._save(project)
        return project

    # NEW (UI redesign): the sidebar's per-project "..." menu needs somewhere
    # to actually rename/delete a project -- neither existed before, since
    # the old <select>-based UI never offered either action.
    def rename(self, project_id: str, new_name: str) -> Project:
        project = self.get(project_id)
        if project is None:
            raise KeyError(f"unknown project_id: {project_id}")
        project.name = new_name
        project.updated_at = _now_iso()
        self._save(project)
        return project

    def delete(self, project_id: str) -> bool:
        path = self._path(project_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def _save(self, project: Project) -> None:
        self._path(project.project_id).write_text(
            project.model_dump_json(indent=2), encoding="utf-8"
        )
