"""The default model, and the facts that justify the figures next to it.

Every number here was read from the Hugging Face API and the repo's own
``config.json`` on 2026-08-08, not estimated:

* 15 safetensors shards, 55.6 GB in bf16, ungated
* ``model_type`` ``qwen3_5``, architecture ``Qwen3_5ForConditionalGeneration``
* 64 layers, hidden 5120, GQA 24 heads / 4 KV heads
* ``max_position_embeddings`` 262144
* a vision tower alongside the text tower (image and video token ids present)
* Q4_K_S GGUF measured at 16.1 GB; Ollama publishes ``qwen3.6:27b``

Nothing in this file touches the network — the point is to pin what the
catalogue *claims*, so a future edit cannot quietly drift from the weights.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from jarvis.core.config import Config, LLMConfig, load_config
from jarvis.llm.models import (
    DEFAULT_ALIAS,
    KNOWN_MODELS,
    UnreleasedModelError,
    recommend,
    resolve,
)


DEFAULT_REPO = "Qwen/Qwen3.6-27B"


def test_the_default_is_qwen3_6_27b():
    assert LLMConfig().model == DEFAULT_REPO
    assert DEFAULT_ALIAS == "qwen3.6-27b"
    assert resolve(DEFAULT_ALIAS).id == DEFAULT_REPO


def test_the_default_is_selectable():
    """It exists, so unlike Qwen3.8 it must resolve rather than raise."""
    spec = resolve(DEFAULT_ALIAS)
    assert spec.exists is True
    assert spec.gated is False


def test_the_verified_specification():
    spec = resolve(DEFAULT_ALIAS)
    assert spec.params == pytest.approx(27.0)
    assert spec.context == 262144
    assert spec.quantised_size_gb == pytest.approx(16.1, abs=0.5)
    assert spec.ollama_tag == "qwen3.6:27b"


def test_it_is_dense_not_mixture_of_experts():
    """The whole performance story turns on this.

    A dense 27B reads every parameter per token; qwen3-30b-a3b reads ~3.3B. If
    someone ever sets active_params here, the CPU speed guidance in the README
    and in recommend() stops being true.
    """
    spec = resolve(DEFAULT_ALIAS)
    assert spec.is_moe is False
    assert spec.effective_params == pytest.approx(27.0)


def test_the_notes_warn_about_the_dense_cpu_cost():
    notes = resolve(DEFAULT_ALIAS).notes.lower()
    assert "dense" in notes
    assert "think" in notes, "thinking-by-default is the other latency trap"


def test_qwen3_8_is_still_listed_but_unselectable():
    """It is not released; listing it is useful, selecting it is not."""
    assert "qwen3.8-27b" in KNOWN_MODELS
    assert KNOWN_MODELS["qwen3.8-27b"].exists is False
    with pytest.raises(UnreleasedModelError):
        resolve("qwen3.8-27b")


def test_recommend_still_prefers_the_moe_for_cpu_only_work():
    """Making the dense 27B the default must not corrupt the sizing advice.

    recommend() ranks by parameters *active per token*, so on a CPU-only box it
    should still steer towards the mixture-of-experts model rather than simply
    echoing whatever the default happens to be.
    """
    spec = recommend(ram_gb=32, has_gpu=False, purpose="interactive")
    assert spec.effective_params < 10.0, (
        f"recommended {spec.id} with {spec.effective_params}B active per token "
        f"for interactive CPU use — that will not feel like conversation"
    )


# --------------------------------------------------------------------------- #
#  The declared default must agree everywhere it is written down
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent


def test_example_config_matches_the_dataclass_default():
    example = ROOT / "config.example.yaml"
    if not example.exists():
        pytest.skip("config.example.yaml not present")
    text = example.read_text(encoding="utf-8")

    match = re.search(r"^\s+model:\s*(\S+)", text, re.M)
    assert match, "config.example.yaml declares no llm.model"
    assert match.group(1) == LLMConfig().model


def test_readme_names_the_current_default():
    readme = ROOT / "README.md"
    if not readme.exists():
        pytest.skip("README.md not present")
    text = readme.read_text(encoding="utf-8")
    assert DEFAULT_REPO in text, "README does not mention the default model"


def test_transformers_pin_covers_the_architecture():
    """Qwen3_5 landed in transformers 4.57; an older pin fails at load time."""
    for name in ("requirements-full.txt", "pyproject.toml"):
        path = ROOT / name
        if not path.exists():
            continue
        # The lookbehind matters: `sentence-transformers>=2.3.0` is a different
        # package and pins a different version series.
        pattern = r"(?<![-\w])transformers>=(\d+)\.(\d+)"
        for match in re.finditer(pattern, path.read_text(encoding="utf-8")):
            major, minor = int(match.group(1)), int(match.group(2))
            assert (major, minor) >= (4, 57), (
                f"{name} pins transformers>={major}.{minor}, but the Qwen3_5 "
                f"architecture needs >= 4.57"
            )


def test_a_full_config_still_builds_with_the_new_default():
    cfg = load_config(use_env=False)
    assert cfg.llm.model == DEFAULT_REPO
    assert isinstance(cfg, Config)
