# Patch — real change detection, real thumbnails, real query threading

## How to apply

Extract this zip's contents directly into your project root (the folder
that already has your `backend/`, `frontend/`, `docs/` in it) and let it
overwrite the 5 existing files it replaces. 2 files are new
(`backend/model_registry/deterministic_change.py`,
`backend/preprocessing/thumbnail.py`) — no other setup needed, they use
libraries already in your `requirements.txt` (`numpy`, `scikit-image`,
`rasterio`, `Pillow`).

No dependency changes. No config changes. No database/migration steps.

## What changed and why

**`backend/model_registry/mock_registry.py`** (rewritten) +
**`backend/model_registry/deterministic_change.py`** (new)
The mock engine's change-detection path returned the exact same hardcoded
numbers (48211 px, 6.4%, 69% confidence — the numbers in your screenshot)
no matter what images you fed it. That's the bug behind "same result for
every data." Replaced with real Change Vector Analysis + Otsu thresholding
— genuine image-differencing math computed from the actual pixels you
upload, every time. Real remote-sensing science (a classical baseline
method, not a trained deep model — see the file's own docstring for
exactly what it can and can't tell you), not a placeholder. Different
image pairs now genuinely produce different, honestly-computed results.

VQA/captioning/grounding/fusion still can't give you real semantic
answers ("what is this," "find the buildings") without a trained model —
there's no honest way to fake that with classical methods, so these stay
placeholders. But they now say so explicitly in the response itself, and
carry real per-band pixel statistics (genuinely computed) alongside the
placeholder note instead of nothing.

**`backend/agent/planner.py` + `backend/api/main.py`** (both touched for
one connected bug)
Your actual question text, each image's real modality, and which image is
"before" vs "after" were declared as real parameters on the inference
engine's interface but were never actually being passed from the planner
— silently dropped at three separate points in the chain. Fixed all
three. This is why the system never seemed to actually use your question:
it wasn't receiving it.

**`backend/preprocessing/thumbnail.py`** (new) + a new endpoint in
`backend/api/main.py` + `frontend/index.html`
"No inline preview for this file type" — browsers can't render GeoTIFF
directly, and your frontend already knew that and said so honestly
instead of faking a preview (there was already a code comment saying
exactly this). Built the real fix: the backend reads a downsampled — never
full-resolution, so this stays fine on GB-scale files too — version of the
actual raster and generates a real PNG from it. Frontend now fetches and
shows that.

**`docs/SYSTEM_DESIGN.md`** (updated)
Documents all of the above in the same place the rest of your project's
real-vs-placeholder status already lives, so it stays accurate.

## What this patch does NOT do

Uploaded CSV/PDF/Excel/GeoJSON files are still extracted (real metadata:
row/column stats, page counts, feature counts, all real) but not yet
pulled into query answering — the "other file types aren't used in
analysis yet" label in your UI is still accurate and wasn't changed.
Wiring that in properly (retrieving only the query-relevant data per file,
not injecting entire spreadsheets into every prompt) is a real feature on
the same scale as the change-detection fix above, not something to rush
alongside it.
