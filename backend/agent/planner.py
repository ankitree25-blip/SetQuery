"""
Planner.execute() sequences Part 3 -> Part 4 -> Part 5 calls for a single
classified task (Section 3.2: "A planner that sequences Part 3 -> Part 4 ->
Part 5 calls per task type").

Two deliberate design choices beyond the literal endpoint/function list:

1. Co-registration gate. Section 3.3's hardening says check_coregistration
   "returns aligned: false above the offset threshold; Part 2 is responsible
   for refusing the downstream task on that result." Applied here to every
   task that combines two images pixel-for-pixel — CHANGE_DETECTION,
   CHANGE_VQA, and OPTICAL_SAR_FUSION — not just the change tasks, since
   fusing misaligned optical/SAR is exactly as unreliable as diffing
   misaligned before/after images. On failure, Part 2 does NOT call Part 4
   at all (that is what "refusing" means here) — it builds a minimal
   Evidence carrying the failure reason in `warnings` and routes straight to
   validate_and_respond, so Part 5's existing abstain path (Section 3.5:
   "If ... evidence.warnings contains a co-registration refusal, sets
   abstained: true") produces one consistent, well-formatted decline instead
   of Part 2 inventing a second message format.

2. UNSUPPORTED short-circuit. There is no Evidence to generate a response
   from, and Section 3.2's hardening wants "a plain-language response, not
   a forced wrong answer" — so this builds the FinalResponse directly, with
   a fixed template sentence, rather than manufacturing empty Evidence just
   to route it through Part 5. Nothing here is free-generated text, so it
   doesn't cross into Part 5's "answer-text generation" territory.

`inference_gate` is the bounded-concurrency semaphore from Section 3.2's
hardening ("Bounded-concurrency queue around every Part 4 (GPU) call"). It
is acquired around run_inference only — tiling and co-registration aren't
GPU work and queuing them too would just add latency for no benefit.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from shared.schemas import (
    AnalysisMode,
    Confidence,
    CoregistrationResult,
    Evidence,
    FinalResponse,
    ImageMetadata,
    TaskType,
    Tile,
    WebSource,
)
from agent.search_decision import should_search_web
from agent.trace import Trace

_CROSS_IMAGE_TASKS = (TaskType.CHANGE_DETECTION, TaskType.CHANGE_VQA, TaskType.OPTICAL_SAR_FUSION)

TileImageFn = Callable[[str, TaskType, int, float], Awaitable[list[Tile]]]
CheckCoregFn = Callable[[str, str], Awaitable[CoregistrationResult]]
RunInferenceFn = Callable[..., Awaitable[Evidence]]  # see call site below for the real keyword args this takes
ValidateRespondFn = Callable[[str, Evidence], Awaitable[FinalResponse]]

# NEW (Section 4 addition) -- deliberately NOT part of Part345Functions below:
# that bundle is specifically "the 4 planner-owed calls" from the original
# contract, and web search is a later, optional capability, not a Part 3/4/5
# call. Takes (query, num_results); returns already-shaped WebSource records
# (not this module's business to know about Google Custom Search specifically
# -- see api/main.py's _web_search_fn for the real implementation).
WebSearchFn = Callable[[str, int], Awaitable[list[WebSource]]]


@dataclass
class Part345Functions:
    """Dependency-injection bundle for the 4 planner-owed calls (validate_and_prepare
    is only used at upload time, in api/main.py, so it isn't part of this bundle)."""
    tile_image: TileImageFn
    check_coregistration: CheckCoregFn
    run_inference: RunInferenceFn
    validate_and_respond: ValidateRespondFn


def _unsupported_response(reason: str) -> FinalResponse:
    evidence = Evidence(
        task=TaskType.UNSUPPORTED, model_used="none", modality_used=[],
        confidence=Confidence(value=None, band="LOW", basis="no matching capability"),
    )
    return FinalResponse(
        answer_text=f"I can't help with that: {reason}",
        confidence=evidence.confidence, evidence=evidence,
        abstained=True, abstain_reason=reason,
    )


def _refusal_evidence(task: TaskType, images: list[ImageMetadata], reason: str) -> Evidence:
    return Evidence(
        task=task, model_used="none", modality_used=[img.modality for img in images],
        confidence=Confidence(value=None, band="LOW", basis="co-registration check failed"),
        warnings=[f"co-registration refusal: {reason}"],
    )


# NEW (bug fix, post-merge): tiling calls into Part 3's rasterio/GDAL layer,
# which -- like every other Part 3/4/5 call -- can fail in ways nobody
# anticipated (the "no geotransform or GCPs" COG-driver crash that motivated
# this fix was exactly one such case, on real hardware, that never showed up
# while Part 3 was written against docs alone in a sandbox without rasterio
# installed). Section 3.2's "one failed job must never take the process
# down" applies here exactly as it already does to co-registration refusals
# below -- this builds the same shape of clean refusal instead of letting a
# library-level exception surface to the user as a raw traceback.
def _preprocessing_failure_evidence(task: TaskType, images: list[ImageMetadata], reason: str) -> Evidence:
    return Evidence(
        task=task, model_used="none", modality_used=[img.modality for img in images],
        confidence=Confidence(value=None, band="LOW", basis="image preprocessing failed"),
        warnings=[f"preprocessing refusal: {reason}"],
    )


class Planner:
    async def execute(
        self,
        task: TaskType,
        query: str,
        images: list[ImageMetadata],
        trace: Trace,
        inference_gate: asyncio.Semaphore,
        funcs: Part345Functions,
        web_search_enabled: bool = False,
        web_search_fn: Optional[WebSearchFn] = None,
        mode: AnalysisMode = AnalysisMode.DEEP,
    ) -> FinalResponse:
        if task == TaskType.UNSUPPORTED:
            return _unsupported_response(
                "this doesn't match a supported analysis for the image(s) and question given"
            )

        if task in _CROSS_IMAGE_TASKS and len(images) == 2:
            async with trace.stage("coregistration_check"):
                coreg = await funcs.check_coregistration(images[0].image_id, images[1].image_id)
            if not coreg.aligned:
                evidence = _refusal_evidence(task, images, coreg.reason or "offset too large")
                async with trace.stage("response_generation"):
                    return await funcs.validate_and_respond(query, evidence)

        all_tiles: list[Tile] = []
        try:
            async with trace.stage("tiling"):
                for img in images:
                    all_tiles.extend(await funcs.tile_image(img.image_id, task))
        except Exception as e:
            # trace.stage() above already recorded "tiling: failed" and
            # re-raised (see trace.py) -- this is the catch that keeps that
            # re-raise from reaching JobQueue._run() as a raw exception.
            evidence = _preprocessing_failure_evidence(task, images, str(e))
            async with trace.stage("response_generation"):
                return await funcs.validate_and_respond(query, evidence)

        async with trace.stage("inference"):
            async with inference_gate:
                if all(img.timestamp for img in images):
                    ordered_images = sorted(images, key=lambda img: img.timestamp)
                else:
                    ordered_images = images  # upload/selection order -- still more meaningful than an alphabetical id sort
                evidence = await funcs.run_inference(
                    task, all_tiles,
                    query=query,
                    image_modalities={img.image_id: img.modality for img in images},
                    image_order=[img.image_id for img in ordered_images],
                )

        # NEW (Section 4, mode-aware since Section 3's Fast/Deep/Research) --
        # additive: only runs with the toggle on, only searches when there's
        # a real signal, and a search failure becomes a warning on the
        # evidence rather than an exception. trace.stage() re-raises whatever
        # it sees (trace.py) straight into JobQueue._run(), which would fail
        # the *whole* job -- a flaky search must never take down an
        # otherwise-successful image analysis, so the try/except has to sit
        # inside the stage, not around it.
        #
        # Mode is what actually makes Fast/Deep/Research differ here, not
        # just their names: FAST never spends the extra round-trip on search
        # ("minimal retrieval" -- Section 3); RESEARCH gathers supporting
        # evidence whenever search is enabled at all, skipping the "is this
        # even needed" heuristic gate DEEP still applies ("repeated searches
        # where justified" -- Section 3), and asks for more sources per
        # search. This one axis (web evidence) is real end-to-end today;
        # the rest of Section 3's per-mode asks (multi-file cross-comparison,
        # alternative-hypothesis generation, uncertainty depth) need
        # reasoning Part 4's mock engine can't do yet -- see the upgrade
        # notes rather than pretending those exist here too.
        if web_search_enabled and web_search_fn is not None and mode != AnalysisMode.FAST:
            if mode == AnalysisMode.RESEARCH:
                should_search, num_results = True, 5
            else:
                should_search, _reason = should_search_web(query)
                num_results = 3
            if should_search:
                async with trace.stage("web_search"):
                    try:
                        evidence.web_sources = await web_search_fn(query, num_results)
                    except Exception as e:
                        evidence.warnings = evidence.warnings + [f"web search attempted but failed: {e}"]

        async with trace.stage("response_generation"):
            return await funcs.validate_and_respond(query, evidence)
