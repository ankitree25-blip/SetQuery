"""
Real per-architecture checkpoint loading — the seam loader.py's
default_model_factory dispatches into once an architecture actually has a
checkpoint (Section 3.6). Only the VLM (`satquery-vlm-base`) case is
implemented here: it's the only one with a training pipeline built
(RSVQA-LR + PaliGemma LoRA/QLoRA — see the standalone training repo's own
README). Grounding/change-detection/fusion don't have a training pipeline
yet — see docs/SYSTEM_DESIGN.md's Known gaps — so loader.py's dispatch
still raises its existing honest "not ready" error for those.

Kept in its own module rather than inside loader.py so loader.py's own
"zero hard dependency on torch/transformers" property (its own docstring)
stays true for anyone building/testing Part 4's eviction/budget logic
without a GPU box.

The confidence score `generate()` returns comes from the model's own
actual per-token generation probabilities (HF's documented
`compute_transition_scores` pattern:
https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075)
— a geometric mean of the generated tokens' probabilities, not an
invented number. Same "real signals only" rule Part 5's confidence
scoring already follows for the rest of the pipeline (architecture.md
Section 3.5).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.model_registry.exceptions import ModelLoadError
from backend.shared.schemas import ModelRegistryEntry

_DEFAULT_MAX_NEW_TOKENS = 64

# What checkpoint_metadata.json must contain — mirrors, by hand, the
# training pipeline's own tests/test_checkpoint_contract.py validate().
# Kept as a second copy rather than a shared import on purpose: the two
# repos are deliberately decoupled (root README's "Attaching a trained
# checkpoint later" — training runs standalone, possibly on a completely
# different machine, and only the checkpoint file + this metadata file
# ever cross that boundary).
_REQUIRED_METADATA_KEYS = ("base_model", "adapter_type", "dataset", "eval", "checkpoint_path")


def load_vlm_checkpoint(entry: ModelRegistryEntry, device: str, quantization_override: str | None) -> "VLMHandle":
    metadata, adapter_dir = _resolve_checkpoint(entry)
    quantization = quantization_override or entry.quantization

    if quantization not in ("4bit", "8bit", "none"):
        raise ModelLoadError(
            f"'{entry.name}': unrecognized quantization {quantization!r} "
            f"(expected '4bit', '8bit', or 'none')."
        )
    if quantization in ("4bit", "8bit") and device == "cpu":
        raise ModelLoadError(
            f"'{entry.name}': {quantization} quantization needs a CUDA GPU "
            f"(bitsandbytes doesn't run on CPU) — pass "
            f"quantization_override='none' to force a (slow) full-precision "
            f"CPU load instead, or run this on a machine with a GPU."
        )

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, PaliGemmaForConditionalGeneration

    base_model_id = metadata["base_model"]

    quantization_config = None
    if quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    try:
        base = PaliGemmaForConditionalGeneration.from_pretrained(
            base_model_id, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map="auto",
        )
    except Exception as exc:
        raise ModelLoadError(
            f"'{entry.name}': failed to load base model {base_model_id!r} "
            f"({exc}). If this is a network/auth error, the base model may "
            f"be gated on Hugging Face — run `huggingface-cli login` with "
            f"an account that's accepted PaliGemma's license."
        ) from exc

    try:
        # train.py saves the processor alongside the adapter (same repo,
        # training/train.py's build_model_and_processor) — fall back to
        # the base model's copy if that's missing for some reason; it's
        # the same processor either way, train.py never modifies it.
        processor = AutoProcessor.from_pretrained(str(adapter_dir))
    except Exception:
        processor = AutoProcessor.from_pretrained(base_model_id)

    try:
        model = PeftModel.from_pretrained(base, str(adapter_dir))
    except Exception as exc:
        raise ModelLoadError(
            f"'{entry.name}': failed to load the LoRA adapter from "
            f"{adapter_dir} onto base model {base_model_id!r} ({exc}). A "
            f"common cause: the adapter was trained against a different "
            f"base model than checkpoint_metadata.json's base_model field says."
        ) from exc
    model.eval()

    return VLMHandle(
        model=model, processor=processor,
        vram_mb=_estimate_loaded_vram_mb(model),
        max_new_tokens=_DEFAULT_MAX_NEW_TOKENS,
    )


def _resolve_checkpoint(entry: ModelRegistryEntry) -> tuple[dict, Path]:
    """
    entry.checkpoint_path is meant to point directly at the adapter
    directory (root README's "Attaching a trained checkpoint later"),
    with checkpoint_metadata.json alongside it (Section 3.6: "alongside
    the checkpoint file") — i.e. as its sibling, one level up. Also
    accepts metadata living *inside* checkpoint_path itself (adapter
    files and metadata flattened into the same directory), since that's
    an easy, reasonable variation to get right by hand when copying files
    off a training box, and getting the nesting wrong would be a
    frustrating way to lose an hour after a multi-hour training run.
    """
    checkpoint_dir = Path(entry.checkpoint_path)
    candidates = [
        checkpoint_dir / "checkpoint_metadata.json",         # flattened: metadata inside checkpoint_path
        checkpoint_dir.parent / "checkpoint_metadata.json",  # nested: checkpoint_path IS the adapter dir
    ]
    metadata_path = next((c for c in candidates if c.is_file()), None)
    if metadata_path is None:
        raise ModelLoadError(
            f"'{entry.name}': no checkpoint_metadata.json found at "
            f"{candidates[0]} or {candidates[1]}. Expected either right "
            f"inside checkpoint_path, or as its sibling one level up "
            f"(checkpoint_path itself being the adapter directory) — see "
            f"root README.md's \"Attaching a trained checkpoint later\"."
        )

    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelLoadError(f"'{entry.name}': couldn't read {metadata_path} ({exc}).") from exc

    missing = [k for k in _REQUIRED_METADATA_KEYS if k not in metadata]
    if missing:
        raise ModelLoadError(
            f"'{entry.name}': {metadata_path} is missing required field(s) "
            f"{missing} — see architecture.md Section 3.6's contract."
        )
    if metadata["adapter_type"] not in ("LoRA", "QLoRA"):
        raise ModelLoadError(
            f"'{entry.name}': {metadata_path}'s adapter_type is "
            f"{metadata['adapter_type']!r}, expected 'LoRA' or 'QLoRA'."
        )

    # The adapter directory is checkpoint_dir itself in both layouts this
    # accepts: the flattened one (metadata found via candidates[0], right
    # inside checkpoint_dir alongside the adapter files) and the nested
    # one (metadata found via candidates[1], one level above checkpoint_dir
    # -- because checkpoint_dir *is* the adapter directory in that layout).
    # NOT metadata_path.parent generally -- that's only right for the
    # flattened case; for the nested case it'd resolve one level too high,
    # to checkpoint_dir's own parent instead of checkpoint_dir.
    adapter_dir = checkpoint_dir
    if not (adapter_dir / "adapter_config.json").is_file():
        raise ModelLoadError(
            f"'{entry.name}': found {metadata_path} but no adapter_config.json "
            f"next to it at {adapter_dir} — expected a PEFT adapter directory "
            f"(adapter_config.json + adapter weights) alongside the metadata file."
        )
    return metadata, adapter_dir


def _estimate_loaded_vram_mb(model: Any) -> float:
    """Real figure from the model actually loaded (summed parameter +
    buffer byte sizes) — takes priority over loader.py's static
    per-quantization-tier guess (_estimate_vram_mb), which only runs when
    a factory's handle doesn't report its own .vram_mb."""
    try:
        total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
        return total_bytes / (1024 * 1024)
    except Exception:
        return 0.0  # loader.py falls back to its own static estimate when this is falsy


class VLMHandle:
    """What vlm_adapter.py's `model.generate(image_path, prompt)` calls
    into — `satquery-vlm-base` is the only registry entry that resolves
    to this today (see this module's docstring)."""

    def __init__(self, model: Any, processor: Any, vram_mb: float, max_new_tokens: int):
        self.model = model
        self.processor = processor
        self.vram_mb = vram_mb
        self._max_new_tokens = max_new_tokens

    def generate(self, image_path: str, prompt: str) -> dict:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=self._max_new_tokens,
                return_dict_in_generate=True, output_scores=True,
            )

        decoded = self.processor.decode(output.sequences[0], skip_special_tokens=True)
        # Same prompt-echo strip training/evaluate.py's identical case
        # uses, same fallback for the same reason — see that file's comment
        # on why the fallback exists rather than assuming the prefix always matches.
        text = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()

        try:
            score = self._sequence_confidence(output, input_len)
        except Exception:
            # A real answer with no confidence score beats no answer at
            # all — vlm_adapter.py already treats score=0.0 as "no signal"
            # (Confidence(value=score or None, ...)), so this degrades
            # honestly rather than failing the whole request over a
            # scoring nicety.
            score = 0.0

        return {"text": text, "score": score}

    def _sequence_confidence(self, output: Any, input_len: int) -> float:
        """Geometric mean of the actually-generated tokens' probabilities.
        -inf entries (padding after an early EOS) are excluded rather than
        allowed to zero out the whole score — see this module's docstring
        for the HF-documented pattern this follows."""
        import torch

        if output.sequences.shape[1] <= input_len:
            return 0.0  # generated nothing at all

        log_probs = self.model.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )[0]
        finite = log_probs[torch.isfinite(log_probs)]
        if finite.numel() == 0:
            return 0.0
        return float(torch.exp(finite.mean()).item())

    def close(self) -> None:
        """Called by loader.py's ModelLoader.unload() on LRU eviction."""
        import torch

        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
