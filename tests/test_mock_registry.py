"""
Proves Definition of Done #1 (Section 3.4): "schema-valid Evidence for
every mandatory capability." Also proves the mock's artificial-latency and
OOM-simulation knobs actually work, since Parts 2 and 5 build against
those directly.
"""
from __future__ import annotations

import time

import pytest

from backend.model_registry.mock_registry import MockInferenceEngine, _change_answer
from backend.shared.schemas import Evidence, Modality, TaskType

MANDATORY_TASKS = [
    TaskType.SINGLE_IMAGE_VQA,
    TaskType.CAPTIONING,
    TaskType.GROUNDING,
    TaskType.CHANGE_DETECTION,
    TaskType.CHANGE_VQA,
    TaskType.OPTICAL_SAR_FUSION,
]


def _tiles_for(task: TaskType, single_tile, before_after_tiles, optical_sar_tiles):
    if task in (TaskType.CHANGE_DETECTION, TaskType.CHANGE_VQA):
        return before_after_tiles
    if task == TaskType.OPTICAL_SAR_FUSION:
        return optical_sar_tiles
    return [single_tile]


@pytest.mark.parametrize("task", MANDATORY_TASKS)
def test_mock_returns_schema_valid_evidence_for_every_mandatory_task(task, single_tile, before_after_tiles, optical_sar_tiles):
    engine = MockInferenceEngine()
    tiles = _tiles_for(task, single_tile, before_after_tiles, optical_sar_tiles)

    evidence = engine.run_inference(task, tiles, query="what is here?")

    assert isinstance(evidence, Evidence)
    assert evidence.task == task
    assert evidence.model_used
    assert evidence.confidence.band in ("LOW", "MEDIUM", "HIGH")


def test_grounding_fixture_has_detections(single_tile):
    evidence = MockInferenceEngine().run_inference(TaskType.GROUNDING, [single_tile], query="find buildings")
    assert evidence.detections == []
    assert evidence.confidence.value is None
    assert "hasn't been trained yet" in evidence.vqa_answer_raw


def test_change_detection_fixture_has_change_map(before_after_tiles):
    evidence = MockInferenceEngine().run_inference(TaskType.CHANGE_DETECTION, before_after_tiles)
    assert evidence.change_map is None
    assert evidence.confidence.value is None
    assert any("does not exist" in warning for warning in evidence.warnings)


def test_change_vqa_answers_follow_question_focus():
    evidence = {"changed_area_pct": 8.5, "changed_area_px": 1200, "mean_confidence": 0.7}
    deep = _change_answer("give a deep analysis", evidence)
    extent = _change_answer("how much area changed?", evidence)
    cause = _change_answer("why did it change?", evidence)
    roads = _change_answer("changes in roads", evidence)

    assert deep != extent
    assert extent != cause
    assert roads != extent
    assert "cause" in cause.lower()
    assert "calibrated probability" in cause.lower()
    assert "specific feature" in roads.lower()


def test_change_answer_uses_current_question_after_chat_context():
    evidence = {"changed_area_pct": 8.5, "changed_area_px": 1200, "mean_confidence": 0.7}
    answer = _change_answer(
        "Conversation context from earlier turns:\nUSER: why did it change?\n"
        "ASSISTANT: cause is uncertain\n\nCURRENT USER QUESTION: how much changed?",
        evidence,
    )
    assert "measured extent" in answer.lower()
    assert "cannot determine the cause" not in answer.lower()


def test_vqa_fixtures_have_answer_text(single_tile):
    evidence = MockInferenceEngine().run_inference(TaskType.SINGLE_IMAGE_VQA, [single_tile], query="how many buildings?")
    assert evidence.vqa_answer_raw


def test_fusion_fixture_reports_both_modalities(optical_sar_tiles):
    evidence = MockInferenceEngine().run_inference(
        TaskType.OPTICAL_SAR_FUSION,
        optical_sar_tiles,
        image_modalities={"img-optical-1": Modality.OPTICAL, "img-sar-1": Modality.SAR},
    )
    assert set(evidence.modality_used) == {Modality.OPTICAL, Modality.SAR}


def test_list_available_models_covers_every_mandatory_task():
    models = MockInferenceEngine().list_available_models()
    covered = {t for entry in models for t in entry.tasks}
    assert set(MANDATORY_TASKS).issubset(covered)


def test_health_check_shape():
    health = MockInferenceEngine().health_check()
    assert health["status"] == "degraded"
    assert health["models_loaded"] == []
    assert health["capabilities"][TaskType.CHANGE_DETECTION.value] == "DETERMINISTIC_FALLBACK"
    assert health["capabilities"][TaskType.GROUNDING.value] == "MODEL_NOT_AVAILABLE"


def test_artificial_latency_actually_delays(single_tile):
    engine = MockInferenceEngine(artificial_latency_s=0.05)
    start = time.monotonic()
    engine.run_inference(TaskType.CAPTIONING, [single_tile])
    assert time.monotonic() - start >= 0.045


def test_simulate_oom_for_raises_oom_style_error(single_tile):
    engine = MockInferenceEngine(simulate_oom_for={TaskType.CAPTIONING})
    with pytest.raises(RuntimeError, match="(?i)out of memory"):
        engine.run_inference(TaskType.CAPTIONING, [single_tile])


def test_run_inference_rejects_empty_tiles():
    with pytest.raises(ValueError):
        MockInferenceEngine().run_inference(TaskType.CAPTIONING, [])
