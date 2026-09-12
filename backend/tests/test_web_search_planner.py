"""
Tests for the Section 4 (web search) addition to Planner.execute /
JobQueue: agent/search_decision.py's heuristic, and the planner-level
wiring around it. Matches test_agent.py's own fixtures/style (make_image,
make_funcs) rather than reinventing them.

The single most important test here is
test_planner_web_search_failure_does_not_fail_the_job: trace.stage()
re-raises anything it sees (trace.py) straight into JobQueue._run(),
which turns *any* propagating exception into a failed job -- so a search
provider hiccup must never surface past Planner.execute() as an
exception, or a perfectly good image analysis would get reported as
"failed" over an unrelated, optional side-lookup.
"""

from __future__ import annotations

import asyncio

import pytest

from agent import mocks
from agent.planner import Part345Functions, Planner
from agent.queue import JobQueue
from agent.search_decision import should_search_web
from agent.trace import Trace
from shared.schemas import ImageMetadata, Modality, TaskType, WebSource


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


# --- should_search_web (see search_decision.py for the full rationale) ------

@pytest.mark.parametrize("query,expected", [
    ("where are the buildings?", False),
    ("what changed between these two images?", False),
    ("describe this image", False),
    ("what is the current vegetation coverage?", True),
    ("could flooding explain the vegetation decrease?", True),
    ("is this rainfall typical for this region?", True),
])
def test_should_search_web(query, expected):
    should_search, reason = should_search_web(query)
    assert should_search is expected
    assert isinstance(reason, str) and reason  # always explains itself either way


# --- Planner wiring -----------------------------------------------------

async def test_web_search_off_by_default_no_stage_even_for_a_matching_query():
    """web_search_enabled defaults to False -- existing callers (and the
    whole pre-Section-4 test suite) get byte-for-byte the same behavior."""
    poisoned = _poisoned_search_fn()
    trace = Trace()
    resp = await Planner().execute(
        TaskType.SINGLE_IMAGE_VQA, "what is the current vegetation coverage?",
        [make_image()], trace, asyncio.Semaphore(2), make_funcs(),
        web_search_fn=poisoned,  # enabled=False (default) -- must never be called
    )
    assert resp.abstained is False
    assert "web_search" not in [s.stage for s in trace.steps]


async def test_web_search_enabled_but_heuristic_says_no_signal_skips_search():
    poisoned = _poisoned_search_fn()
    trace = Trace()
    resp = await Planner().execute(
        TaskType.SINGLE_IMAGE_VQA, "where are the buildings?",
        [make_image()], trace, asyncio.Semaphore(2), make_funcs(),
        web_search_enabled=True, web_search_fn=poisoned,
    )
    assert resp.abstained is False
    assert "web_search" not in [s.stage for s in trace.steps]


async def test_web_search_runs_and_populates_evidence_when_signal_present():
    async def stub_search(query, num_results):
        return [WebSource(title="t", url="https://example.com", snippet="s",
                           retrieved_at="2026-01-01T00:00:00+00:00")]

    trace = Trace()
    resp = await Planner().execute(
        TaskType.SINGLE_IMAGE_VQA, "what is the current vegetation coverage?",
        [make_image()], trace, asyncio.Semaphore(2), make_funcs(),
        web_search_enabled=True, web_search_fn=stub_search,
    )

    assert resp.abstained is False
    assert [s.stage for s in trace.steps] == ["tiling", "inference", "web_search", "response_generation"]
    assert len(resp.evidence.web_sources) == 1
    assert resp.evidence.web_sources[0].url == "https://example.com"


async def test_planner_web_search_failure_does_not_fail_the_job():
    async def failing_search(query, num_results):
        raise RuntimeError("quota exceeded")

    trace = Trace()
    resp = await Planner().execute(
        TaskType.SINGLE_IMAGE_VQA, "what is the current vegetation coverage?",
        [make_image()], trace, asyncio.Semaphore(2), make_funcs(),
        web_search_enabled=True, web_search_fn=failing_search,
    )

    assert resp.abstained is False  # the image analysis itself still succeeded
    assert resp.evidence.web_sources == []
    assert any("web search" in w and "quota exceeded" in w for w in resp.evidence.warnings)
    # the stage itself is still recorded (as success -- the *step* completed,
    # even though the search inside it didn't); see planner.py's comment.
    assert "web_search" in [s.stage for s in trace.steps]


async def test_queue_threads_web_search_toggle_through_to_the_planner():
    async def stub_search(query, num_results):
        return [WebSource(title="t", url="https://example.com", snippet="s",
                           retrieved_at="2026-01-01T00:00:00+00:00")]

    queue = JobQueue(funcs=make_funcs(), max_concurrent_inference=1, web_search_fn=stub_search)
    job_id = queue.submit(
        "what is the current vegetation coverage?", [make_image()], web_search_enabled=True
    )

    for _ in range(100):
        if queue.get(job_id).status == "done":
            break
        await asyncio.sleep(0.01)

    record = queue.get(job_id)
    assert record.status == "done"
    assert len(record.result.evidence.web_sources) == 1


def _poisoned_search_fn():
    async def poisoned(query, num_results):
        raise AssertionError("web_search_fn must not be called when search is off/skipped")
    return poisoned
