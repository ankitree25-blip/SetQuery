"""
JobQueue owns the three queue-related hardening items from Section 3.2:

  - "A single bounded-concurrency job queue"
  - "Bounded-concurrency queue around every Part 4 (GPU) call — two
    simultaneous requests must queue, never collide"
  - "Job execution decoupled from the HTTP connection — closing the tab
    must not kill the job"
  - "try/except around every Part 3/4/5 call — one failed job must never
    take the process down"

submit() only registers a JobRecord and schedules a background asyncio task
— it never awaits the job itself, so the HTTP handler that calls it (POST
/api/query in api/main.py) returns immediately regardless of how long the
job takes or whether the client is still connected. The concurrency bound
on run_inference lives in the asyncio.Semaphore passed into Planner.execute
(see planner.py) — this class owns creating and sharing that one semaphore
across every job, which is what actually makes "two simultaneous requests
queue instead of colliding" true.

Trace-step-level failures are logged and re-raised by Trace.stage() (see
trace.py); _run() is the single place that turns an exception, wherever it
came from, into job status "failed" instead of letting it propagate into
asyncio's default (silent, process-surviving but easy to lose track of)
unhandled-task-exception handling.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Optional

from shared.schemas import AnalysisMode, FinalResponse, ImageMetadata, TaskType
from agent.intent import classify_intent
from agent.planner import Part345Functions, Planner, WebSearchFn
from agent.trace import Trace


@dataclass
class JobRecord:
    job_id: str
    query: str = ""  # NEW -- so /api/jobs/{id}/report doesn't need the client to resend it
    image_ids: list[str] = field(default_factory=list)
    status: str = "queued"  # "queued" | "running" | "done" | "failed" | "cancelled"
    progress: list[str] = field(default_factory=list)
    result: Optional[FinalResponse] = None
    error: Optional[str] = None
    trace: Trace = field(default_factory=Trace)


class JobQueue:
    def __init__(
        self,
        funcs: Part345Functions,
        max_concurrent_inference: int = 1,
        web_search_fn: Optional[WebSearchFn] = None,
    ):
        self._jobs: dict[str, JobRecord] = {}
        self._funcs = funcs
        self._inference_gate = asyncio.Semaphore(max_concurrent_inference)
        self._planner = Planner()
        self._background_tasks: set[asyncio.Task] = set()
        self._tasks: dict[str, asyncio.Task] = {}  # NEW -- needed so cancel() can find the right task
        self._web_search_fn = web_search_fn  # NEW (Section 4) -- None if search isn't configured

    def submit(
        self,
        query: str,
        images: list[ImageMetadata],
        web_search_enabled: bool = False,
        mode: AnalysisMode = AnalysisMode.DEEP,
        execution_query: Optional[str] = None,
    ) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = JobRecord(
            job_id=job_id,
            query=query,
            image_ids=[image.image_id for image in images],
        )
        task = asyncio.create_task(
            self._run(job_id, execution_query or query, images, web_search_enabled, mode)
        )
        self._tasks[job_id] = task
        self._background_tasks.add(task)
        task.add_done_callback(lambda t, jid=job_id: self._on_task_done(jid, t))
        return job_id

    def get(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """
        Section 14 (Stop/Cancel), including its own race-condition list:

          - "stop clicked twice" -- the second call sees status already
            "cancelled" (or "done"/"failed") and returns False; harmless.
          - "analysis completes at the same time as stop" -- asyncio.Task
            .cancel() on an already-done task is a documented no-op that
            returns False on its own, so this can't race into cancelling a
            result that's already been delivered.
          - "duplicate requests" -- same as "clicked twice": idempotent.
          - "network disconnect" / "browser refresh" -- not this method's
            concern at all: JobQueue already decouples job execution from
            the HTTP connection (see this module's own docstring), so a
            client vanishing doesn't touch a running job either way.

        NOT handled (and not pretended to be): a backend process crash.
        There's no persisted job state to recover from that with the
        current single-process, in-memory JobQueue -- same "single-
        deployment demo" tradeoff as ProjectStore's own docstring notes
        for *why* project data (unlike jobs) had to go to disk instead.
        """
        record = self._jobs.get(job_id)
        if record is None or record.status in ("done", "failed", "cancelled"):
            return False
        task = self._tasks.get(job_id)
        if task is None or task.done():
            return False
        return task.cancel()

    def _on_task_done(self, job_id: str, task: asyncio.Task) -> None:
        # The one place record.status becomes "cancelled" -- deliberately
        # NOT inside _run()'s own except-clause. Verified empirically (not
        # just reasoned about) that calling task.cancel() before the task
        # has been scheduled even once means asyncio throws the
        # CancelledError in *before* any of _run()'s own code -- including
        # its try/except -- ever runs, so _run() alone can't be trusted to
        # record the transition. task.cancelled() is reliable in every
        # case (before the task started, or mid-flight) because asyncio
        # sets it centrally, so that's the source of truth here instead.
        self._background_tasks.discard(task)
        record = self._jobs.get(job_id)
        if record is None:
            return
        if task.cancelled():
            record.status = "cancelled"
            if not record.progress or record.progress[-1] != "cancelled":
                record.progress.append("cancelled")

    async def _run(
        self,
        job_id: str,
        query: str,
        images: list[ImageMetadata],
        web_search_enabled: bool = False,
        mode: AnalysisMode = AnalysisMode.DEEP,
    ) -> None:
        record = self._jobs[job_id]
        record.status = "running"
        try:
            record.progress.append("classifying intent")
            task_type: TaskType = await classify_intent(query, images)
            record.progress.append(f"classified as {task_type.value}")

            result = await self._planner.execute(
                task_type, query, images, record.trace, self._inference_gate, self._funcs,
                web_search_enabled=web_search_enabled, web_search_fn=self._web_search_fn, mode=mode,
            )

            record.result = result
            record.status = "done"
            record.progress.append("done")
        except asyncio.CancelledError:
            # Re-raised, not swallowed: asyncio only marks task.cancelled()
            # True if the CancelledError actually propagates out of the
            # coroutine (verified -- see _on_task_done's comment); catching
            # it here without re-raising would make the task look like it
            # completed normally, which is the wrong signal for a cancel.
            record.progress.append("cancelled")
            raise
        except Exception as e:  # one failed job must never take the process down
            record.status = "failed"
            record.error = str(e)
            record.progress.append("failed")
