"""The default model, and the facts that justify the figures next to it.

The default is Qwen3.8-27B, released 2026-08-14 under Apache 2.0:

* 27,781,427,952 stored parameters (marketed as 27B), dense
* 64 decoder layers: 48 Gated DeltaNet linear-attention + 16 full-attention
* hidden 5120, FFN 17408, GQA 24 query heads / 4 KV heads
* ``max_position_embeddings`` 262144, extensible to ~1M via YaRN
* a vision tower alongside the text tower (image and video in)
* Ollama publishes an 18 GB package as ``qwen3.8:27b``
* thinking is ON by default, which JARVIS disables for interactive turns

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
    THINKING_DEFAULT_ON,
    recommend,
    resolve,
)


DEFAULT_REPO = "Qwen/Qwen3.8-27B"


def test_the_default_is_qwen3_8_27b():
    assert LLMConfig().model == DEFAULT_REPO
    assert DEFAULT_ALIAS == "qwen3.8-27b"
    assert resolve(DEFAULT_ALIAS).id == DEFAULT_REPO


def test_the_default_is_selectable():
    """It is released, so it must resolve rather than raise."""
    spec = resolve(DEFAULT_ALIAS)
    assert spec.exists is True
    assert spec.gated is False


def test_the_verified_specification():
    spec = resolve(DEFAULT_ALIAS)
    assert spec.params == pytest.approx(27.0)
    assert spec.context == 262144
    assert spec.quantised_size_gb == pytest.approx(18.0, abs=0.5)
    assert spec.ollama_tag == "qwen3.8:27b"


def test_the_ollama_tag_matches_the_configured_one():
    """The catalogue tag and the config default must not drift apart."""
    assert LLMConfig().ollama_model == resolve(DEFAULT_ALIAS).ollama_tag


def test_the_default_family_is_known_to_reason_by_default():
    """The thinking-by-default list is what stops silent replies.

    If the default model's family ever falls off THINKING_DEFAULT_ON, the
    runtime stops disabling chain-of-thought and every spoken reply becomes an
    unclosed <think> block — i.e. silence. Pin it.
    """
    assert any(f in DEFAULT_REPO.lower() for f in THINKING_DEFAULT_ON)


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


def test_the_previous_default_is_still_available():
    """Qwen3.6-27B remains in the catalogue as a fallback for anyone pinned to it."""
    spec = resolve("qwen3.6-27b")
    assert spec.id == "Qwen/Qwen3.6-27B"
    assert spec.exists is True


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


# --------------------------------------------------------------------------- #
#  No stale copy of the default may survive anywhere
# --------------------------------------------------------------------------- #
def test_no_module_hardcodes_a_superseded_ollama_tag():
    """A literal fallback that drifts silently pulls the *previous* model.

    `jarvis.runtime.ollama.model_tag` ends in a hardcoded tag for the case
    where the catalogue cannot be read at all. When the default moved from
    Qwen3.6 to Qwen3.8 that literal was left behind, so any machine hitting
    the fallback quietly downloaded 18 GB of the wrong model. Pin it.
    """
    import re
    from pathlib import Path as _Path

    current = LLMConfig().ollama_model
    package = _Path(__file__).resolve().parent.parent / "jarvis"

    offenders = []
    for path in package.rglob("*.py"):
        if path.name == "models.py":
            continue        # the catalogue legitimately names every model
        for match in re.finditer(r'"(qwen[0-9.]*:[0-9a-zA-Z._-]+)"', path.read_text()):
            tag = match.group(1)
            # Only the 27B flagship tags are the "default" that can go stale;
            # smaller tags are deliberate references to specific models.
            if tag.endswith(":27b") and tag != current:
                offenders.append(f"{path.relative_to(package.parent)}: {tag}")

    assert not offenders, (
        "these still name a superseded default instead of "
        f"{current!r}: {offenders}"
    )


def test_the_installer_default_matches_the_catalogue():
    """install.sh carries its own fallback tag; it must not drift either."""
    import re

    text = (ROOT / "install.sh").read_text(encoding="utf-8")
    match = re.search(r'^FALLBACK_MAIN_TAG="([^"]+)"', text, re.M)
    assert match, "install.sh no longer declares FALLBACK_MAIN_TAG"
    assert match.group(1) == LLMConfig().ollama_model


def test_the_installer_can_report_the_current_default():
    """The upgrade warning depends on this action existing."""
    text = (ROOT / "install.sh").read_text(encoding="utf-8")
    assert '"default-tag": action_default_tag' in text
    assert "def action_default_tag" in text
    # It must read the catalogue rather than repeat the tag in shell.
    assert "models.KNOWN_MODELS[models.DEFAULT_ALIAS].ollama_tag" in text


def test_the_installer_warns_when_config_pins_an_older_model():
    """Honouring config.yaml is right; doing it silently on an upgrade is not.

    Without this the user runs ./install.sh expecting the new model, gets the
    previous one, and is given no reason why.
    """
    text = (ROOT / "install.sh").read_text(encoding="utf-8")
    assert 'DEFAULT_TAG="$(runtime_capture default-tag)"' in text
    assert 'if [ -n "$DEFAULT_TAG" ] && [ "$MAIN_TAG" != "$DEFAULT_TAG" ]' in text
    assert "./install.sh --model $DEFAULT_TAG" in text, (
        "the warning must include the command that fixes it"
    )
