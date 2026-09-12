# SatQuery AI

SIH 2026, problem 26167. Full spec: `docs/architecture.md`. What was actually
built, how it fits together, and every integration decision made while
merging it: `docs/SYSTEM_DESIGN.md`. This file is just: how to run it, and
what to do if something goes wrong.

## Quick start (Windows)

Two double-click scripts, at the repo root:

1. **`Setup.bat`** — once. Creates a virtual environment, detects your GPU,
   installs the right PyTorch build for it, installs everything else.
2. **`Start_program.bat`** — runs the app. Opens two windows (backend API,
   frontend server) and your browser, pointed at the upload/query UI.

Not on Windows, or prefer a terminal? Same steps, manually:

```bash
python -m venv .venv && source .venv/bin/activate      # macOS/Linux
pip install torch --index-url https://download.pytorch.org/whl/cu121  # see note below
pip install -r requirements.txt

cd backend && python -m uvicorn api.main:app --port 8000    # terminal 1
cd frontend && python -m http.server 5500                    # terminal 2, then open http://localhost:5500
```

**PyTorch + CUDA:** plain `pip install torch` gives you a CPU-only build on
Windows (works fine on Linux, where it bundles CUDA by default). `Setup.bat`
handles this automatically; doing it manually, use the `--index-url` above,
or check https://pytorch.org/get-started/locally/ for the current command —
that page is more likely to be up to date than this README by the time you
read it.

**Or, Docker — one command, no local Python/GDAL setup at all:**
```bash
docker-compose up --build
```
Then open http://localhost:5500. Runs on CPU / mock inference out of the
box; GPU passthrough for a real attached checkpoint is a commented-out
block in `docker-compose.yml` (needs `nvidia-container-toolkit` on the
host — uncomment once there's a real checkpoint to run, see "Attaching a
trained checkpoint later" below). Copy `.env.example` to `.env` first for
optional features like web search — the app runs fine without it either
way, just with that one feature quietly unavailable rather than the whole
thing failing to start.

## What's real right now

| Part | Status |
|---|---|
| 1–3, Frontend/Backend/Preprocessing | Real. Upload, query routing, tiling, co-registration all run against actual code. |
| 4, Model Registry & Inference | Real code, running on its own **mock engine** — no fine-tuned checkpoint exists yet. |
| 5, Evidence & Response | Real, deterministic template mode (no LLM wired up — that's its zero-setup default). |

The app works end-to-end today — upload images, ask questions, get
schema-valid responses with confidence scores and an execution trace — the
responses just aren't model-grounded until a real checkpoint is attached (see
"Attaching a trained checkpoint later" below). Training itself is deliberately
not part of this repo — the model gets fine-tuned separately, and only the
resulting checkpoint comes back here. See `docs/SYSTEM_DESIGN.md` for the
full gap analysis.

## Attaching a trained checkpoint later

Once you have a checkpoint from training done separately — a PEFT adapter
directory plus a `checkpoint_metadata.json` in the shape
`backend/model_registry/README.md` documents (`base_model`, `adapter_type`,
`dataset`, `eval`, `checkpoint_path`) — wiring it in is:

1. Point the matching entry's `checkpoint_path` in
   `backend/model_registry/config/models.yaml` at that adapter directory.
2. In `backend/api/main.py`, change `configure_engine(use_mock=True)` to
   `configure_engine(use_mock=False)`.
3. `torch`/`transformers`/`peft`/`bitsandbytes`/`accelerate` are already in
   `requirements.txt` for this — nothing extra to install.
4. Restart `Start_program.bat` / the backend.

See `backend/model_registry/README.md`'s "Switching from mock to real" for
the full detail, including the `default_model_factory` seam in `loader.py`
each adapter's docstring specifies the exact handle interface for.

## Tests

```bash
pip install -r requirements.txt
pytest                              # backend/tests/ + tests/ (incl. Part 3's) in one run
python tests/test_core_logic.py     # Part 3's pure-logic subset — no rasterio/GDAL/pydantic needed
```

`tests/test_core_logic.py`'s 45 tests were actually re-executed while
building this merge (numpy/scipy/scikit-image were available in that
sandbox) — genuine, current coverage. `pytest tests/test_integration.py -v`
is the single most valuable thing to run once dependencies are installed —
it's the one part of the system (Part 3's rasterio-dependent I/O layer)
that couldn't be executed at all while merging this (no network access
there to install rasterio). See `docs/SYSTEM_DESIGN.md`'s "Honest
verification" section for exactly what was and wasn't checked, and how.

## Troubleshooting

- **"Python was not found"** (Setup.bat) — install from
  python.org/downloads, ticking "Add python.exe to PATH" during install.
- **rasterio/GDAL fails to install** — see
  `backend/preprocessing/README.md`, or
  https://rasterio.readthedocs.io/en/stable/installation.html
- **Port 8000 or 5500 already in use** — `Start_program.bat` closes any
  leftover process on those ports automatically before starting; if you're
  running something else on one of them, close it first or edit the port
  numbers in `Start_program.bat` and `frontend/index.html`'s `BASE_URL`
  together.
- **Frontend loads but nothing happens when you upload/query** — check the
  "SatQuery AI Backend" window for errors; also check the connection status
  at the bottom of the sidebar in the app itself (green dot = connected to
  `http://localhost:8000`, red = it isn't running or isn't reachable).
- Anything not covered here: `docs/SYSTEM_DESIGN.md`'s "Known gaps" section
  lists every limitation that's a known, deliberate scope decision rather
  than a bug.
