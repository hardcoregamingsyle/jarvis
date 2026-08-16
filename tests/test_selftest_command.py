"""`jarvis selftest`: the command that answers "which stage is broken".

``doctor`` reports what is installed. This one runs each stage of the pipeline
for real, so the two failure modes that look identical from the outside — "the
model is slow" and "the model returns nothing" — are told apart by name.

The contract worth protecting: **it never raises, and one broken stage never
hides the others.** Somebody runs this precisely when things are already going
wrong, so a traceback in the diagnostic is worse than useless.
"""

from __future__ import annotations

import pytest

from jarvis import cli


@pytest.fixture
def run_selftest(monkeypatch, capsys, tmp_path):
    """Run the command against a config rooted in tmp_path, return its output."""

    def _run(**config_overrides):
        from jarvis.core.config import load_config

        cfg = load_config(use_env=False)
        cfg.data_dir = str(tmp_path)
        cfg.memory.db_path = str(tmp_path / "memory.db")
        cfg.tts.enabled = False
        for dotted, value in config_overrides.items():
            section, _, field = dotted.partition(".")
            setattr(getattr(cfg, section), field, value)

        monkeypatch.setattr(cli, "_load", lambda args: cfg)
        code = cli.cmd_selftest(_Args())
        return code, capsys.readouterr().out

    return _run


class _Args:
    config = None
    verbose = False
    model = None
    backend = None
    no_speech = True


def test_it_reports_every_stage(run_selftest):
    _code, out = run_selftest()

    for heading in ("Hardware", "Language model", "Speech", "Full turn"):
        assert heading in out
    for stage in ("CPU", "Backend", "Speech to text", "Microphone", "Text to speech"):
        assert stage in out


def test_a_missing_backend_is_named_not_hidden(run_selftest):
    """With nothing reachable the LLM stage must fail loudly and say why."""
    _code, out = run_selftest(**{"llm.backend": "ollama", "llm.allow_fallback": True})

    assert "FAIL" in out
    assert "ollama serve" in out, "it must say how to fix it, not merely that it broke"


def test_it_exits_nonzero_when_a_stage_fails(run_selftest):
    code, out = run_selftest()
    if "FAIL" in out:
        assert code == 1
    else:
        assert code == 0


def test_one_exploding_stage_does_not_hide_the_rest(run_selftest, monkeypatch):
    """The whole point is a complete picture, so a raising probe is caught."""

    def boom(*_args, **_kwargs):
        raise RuntimeError("detection exploded")

    monkeypatch.setattr("jarvis.core.hardware.detect", boom)

    code, out = run_selftest()

    assert "Hardware detection" in out
    # Everything after the failed stage still ran.
    assert "Language model" in out and "Speech" in out and "Full turn" in out
    assert code in (0, 1)


def test_a_silent_model_is_diagnosed_as_the_thinking_trap(run_selftest, monkeypatch):
    """A model that returns nothing is the bug this whole release is about.

    "Generation produced no text" is useless on its own; the message has to
    point at thinking mode, which is what actually causes it.
    """
    from jarvis.core.contracts import LLMResult

    class SilentModel:
        name = "ollama"

        def is_available(self):
            return True

        def load(self):
            return None

        def generate(self, messages, config=None):
            return LLMResult(text="")

        def stream(self, messages, config=None):
            yield ""

    monkeypatch.setattr("jarvis.llm.create_llm", lambda cfg, **k: SilentModel())

    code, out = run_selftest()

    assert "returned NOTHING" in out
    assert "thinking" in out.lower()
    assert code == 1


def test_the_command_is_registered_in_the_parser():
    args = cli.build_parser().parse_args(["selftest"])
    assert args.func is cli.cmd_selftest
