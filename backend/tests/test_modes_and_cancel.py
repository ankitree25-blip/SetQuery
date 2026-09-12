"""
Tests for the Section 3 (Fast/Deep/Research) and Section 14 (Stop/Cancel)
additions to Planner/JobQueue. Same fixtures as test_web_search_planner.py
(not re-imported from there to keep each test file independently
runnable, matching how test_agent.py itself is self-contained).
"""

from __future__ import annotations

import asyncio

import pytest

from agent import mocks
from agent.planner import Part345Functions, Planner
from agent.queue import JobQueue
from agent.trace import Trace
from shared.schemas import AnalysisMode, ImageMetadata, Modality, TaskType, WebSource


def make_image(modality=Modality.OPTICAL, image_id="img-1") -> ImageMetadata:
    return ImageMetadata(
        image_id=image_id, modality=modality, crs="EPSG:4326",
        bounds=[0.0, 0.0, 1.0, 1.0], width=512, height=512, band_count=3,
        dtype="uint8", resolution_m=10.0, timestamp="2026-01-01T00:00:00Z",
        cog_path=f"/data/images/{image_id}.tif", is_valid=True, validation_errors=[],
    )


def make_funcs() -> Part345Functions:
    return Part345Functions(
        tile_image=mocks.mock_tile_image,
        check_coregistration=mocks.mock_check_coregistration,
        run_inference=mocks.mock_run_inference,
        validate_and_respond=mocks.mock_validate_and_respond,
    )


def _counting_search_fn():
    calls = []

    async def search(query, num_results):
        calls.append(num_results)
        return [WebSource(title="t", url="https://example.com", snippet="s",
                           retrieved_at="2026-01-01T00:00:00+00:00")]
    return search, calls


# --- Section 3: modes actually change behavior, not just their label -------

async def test_fast_mode_never_searches_even_with_a_matching_query():
    search_fn, calls = _counting_search_fn()
    resp = await Planner().execute(
        TaskType.SINGLE_IMAGE_VQA, "what is the current vegetation coverage?",
        [make_image()], Trace(), asyncio.Semaphore(2), make_funcs(),
        web_search_enabled=True, web_search_fn=search_fn, mode=AnalysisMode.FAST,
    )
    assert resp.abstained is False
    assert calls == []  # never called at all


async def test_deep_mode_still_gates_on_the_heuristic():
    search_fn, calls = _counting_search_fn()
    resp = await Planner().execute(
        TaskType.SINGLE_IMAGE_VQA, "where are the buildings?",  # no signal
        [make_image()], Trace(), asyncio.Semaphore(2), make_funcs(),
        web_search_enabled=True, web_search_fn=search_fn, mode=AnalysisMode.DEEP,
    )
    assert resp.abstained is False
    assert calls == []  # Deep respects should_search_web same as the default


async def test_research_mode_always_searches_and_asks_for_more_results():
    search_fn, calls = _counting_search_fn()
    resp = await Planner().execute(
        TaskType.SINGLE_IMAGE_VQA, "where are the buildings?",  # no signal, but Research ignores the gate
        [make_image()], Trace(), asyncio.Semaphore(2), make_funcs(),
        web_search_enabled=True, web_search_fn=search_fn, mode=AnalysisMode.RESEARCH,
    )
    assert resp.abstained is False
    assert calls == [5]  # called once, asking for more sources than Deep's 3


# --- Section 14: Stop/Cancel and its race conditions -----------------------

async def _wait_until_terminal(queue: JobQueue, job_id: str, timeout=2.0):
    elapsed = 0.0
    while elapsed < timeout:
        status = queue.get(job_id).status
        if status in ("done", "failed", "cancelled"):
            return status
        await asyncio.sleep(0.01)
        elapsed += 0.01
    raise AssertionError(f"job {job_id} never reached a terminal status")


async def test_cancel_before_the_task_has_started_marks_the_job_cancelled():
    queue = JobQueue(funcs=make_funcs())
    job_id = queue.submit("where are the buildings?", [make_image()])

    cancelled_now = queue.cancel(job_id)  # no await between submit() and this -- task hasn't run yet
    assert cancelled_now is True

    status = await _wait_until_terminal(queue, job_id)
    assert status == "cancelled"
    assert queue.get(job_id).result is None


async def test_cancel_twice_is_idempotent_not_an_error():
    queue = JobQueue(funcs=make_funcs())
    job_id = queue.submit("where are the buildings?", [make_image()])

    first = queue.cancel(job_id)
    assert first is True
    await _wait_until_terminal(queue, job_id)  # let the first cancellation actually resolve

    second = queue.cancel(job_id)  # "stop clicked twice" -- genuinely nothing left to cancel now
    assert second is False


async def test_cancel_after_the_job_already_finished_is_a_harmless_no_op():
    queue = JobQueue(funcs=make_funcs())
    job_id = queue.submit("where are the buildings?", [make_image()])
    status = await _wait_until_terminal(queue, job_id)  # let it actually finish first
    assert status == "done"

    # "analysis completes at the same time as stop" -- Task.cancel() on an
    # already-done task is a documented asyncio no-op; this asserts our
    # wrapper surfaces that as False rather than raising.
    assert queue.cancel(job_id) is False
    assert queue.get(job_id).status == "done"  # the real result is untouched


def test_cancel_unknown_job_id_returns_false():
    queue = JobQueue(funcs=make_funcs())
    assert queue.cancel("does-not-exist") is False
