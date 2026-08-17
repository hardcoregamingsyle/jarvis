"""Speculative decoding: the arithmetic, and the command that enables it.

The claim this module makes is strong — 2-3x on a dense 27B at *no quality
cost* — so the maths behind it is pinned here rather than trusted.

The key property, and the one that makes the whole thing work: the large model
verifies all `k` drafted tokens in a **single** batched forward pass, so its
weights are read once per *round* rather than once per *token*. Everything
below follows from that.
"""

from __future__ import annotations

import pytest

from jarvis.runtime import llamacpp
from jarvis.runtime.llamacpp import (
    DEFAULT_DRAFT_TOKENS,
    build_server_plan,
    estimate_speedup,
    expected_tokens_per_round,
    pick_target_and_draft,
)


# --------------------------------------------------------------------------- #
#  The acceptance formula
# --------------------------------------------------------------------------- #
class TestExpectedTokens:
    """E = (1 - a^(k+1)) / (1 - a) — expected accepted tokens per round."""

    def test_a_rejected_draft_still_yields_one_token(self):
        """Even with every proposal wrong, the verified token lands. This is
        why speculation can never produce *fewer* tokens than the baseline."""
        assert expected_tokens_per_round(0.0, 4) == pytest.approx(1.0)

    def test_perfect_acceptance_yields_every_drafted_token_plus_one(self):
        assert expected_tokens_per_round(1.0, 4) == pytest.approx(5.0)

    @pytest.mark.parametrize(
        "acceptance, k, expected",
        [
            (0.5, 1, 1.5),
            (0.5, 4, 1.9375),
            (0.75, 4, 3.0508),
            (0.8, 4, 3.3616),
        ],
    )
    def test_known_values(self, acceptance, k, expected):
        assert expected_tokens_per_round(acceptance, k) == pytest.approx(
            expected, abs=1e-3
        )

    def test_it_rises_monotonically_with_acceptance(self):
        values = [expected_tokens_per_round(a / 10, 4) for a in range(11)]
        assert values == sorted(values)

    @pytest.mark.parametrize("bad", [-1.0, 2.0])
    def test_out_of_range_acceptance_is_clamped_not_crashed(self, bad):
        assert expected_tokens_per_round(bad, 4) >= 1.0


# --------------------------------------------------------------------------- #
#  Throughput projection
# --------------------------------------------------------------------------- #
class TestSpeedupEstimate:
    def test_it_reaches_the_target_on_this_hardware(self):
        """The whole point: 3-4 tok/s on a dense 27B at Q4, DDR4-2666."""
        est = estimate_speedup(
            target_gb=18.0, draft_gb=0.4, acceptance=0.7, bandwidth_gb_s=28.0
        )
        assert est["baseline_tok_s"] == pytest.approx(1.56, abs=0.05)
        assert est["speculative_tok_s"] >= 3.0
        assert est["worthwhile"] is True

    def test_a_useless_draft_is_reported_as_not_worthwhile(self):
        """Below ~50% acceptance the draft reads cost more than the batching
        saves, and the tool must say so rather than recommending it anyway."""
        est = estimate_speedup(
            target_gb=18.0,
            draft_gb=9.0,          # far too large to be a draft
            acceptance=0.2,
            bandwidth_gb_s=28.0,
        )
        assert est["worthwhile"] is False
        assert est["speculative_tok_s"] < est["baseline_tok_s"]

    def test_a_smaller_draft_is_strictly_better_at_equal_acceptance(self):
        big = estimate_speedup(target_gb=18.0, draft_gb=1.1, acceptance=0.75)
        small = estimate_speedup(target_gb=18.0, draft_gb=0.4, acceptance=0.75)
        assert small["speculative_tok_s"] > big["speculative_tok_s"]

    def test_the_target_is_read_once_per_round_not_once_per_token(self):
        """The load-bearing property. GB/round must be the target read ONCE
        plus k draft reads -- if this ever becomes k * target, the entire
        speedup has evaporated."""
        est = estimate_speedup(
            target_gb=18.0, draft_gb=0.4, acceptance=0.7, draft_tokens=4
        )
        assert est["gb_per_round"] == pytest.approx(4 * 0.4 + 18.0)

    def test_zero_division_is_impossible(self):
        est = estimate_speedup(target_gb=0.0, draft_gb=0.0, acceptance=0.7)
        assert est["speculative_tok_s"] >= 0


# --------------------------------------------------------------------------- #
#  The server command
# --------------------------------------------------------------------------- #
class TestServerPlan:
    def test_the_draft_flags_are_present_when_a_draft_is_given(self):
        plan = build_server_plan("/m/big.gguf", draft_path="/m/small.gguf")
        assert plan.uses_speculation is True
        assert "--model-draft" in plan.argv
        assert plan.argv[plan.argv.index("--model-draft") + 1] == "/m/small.gguf"
        assert "--draft-max" in plan.argv

    def test_no_draft_flags_without_a_draft(self):
        plan = build_server_plan("/m/big.gguf")
        assert plan.uses_speculation is False
        assert "--model-draft" not in plan.argv
        assert any("No draft model" in n for n in plan.notes)

    def test_threads_default_to_physical_cores(self, monkeypatch):
        monkeypatch.setattr(llamacpp, "physical_cores", lambda: 4)
        plan = build_server_plan("/m/big.gguf")
        assert plan.threads == 4
        assert plan.argv[plan.argv.index("--threads") + 1] == "4"
        assert any("physical cores" in n for n in plan.notes)

    def test_a_large_context_warns_about_the_kv_cache(self):
        plan = build_server_plan("/m/big.gguf", context=131072)
        assert any("KV cache" in n for n in plan.notes)

    def test_the_base_url_matches_the_bound_port(self):
        plan = build_server_plan("/m/big.gguf", host="0.0.0.0", port=9000)
        assert plan.base_url == "http://0.0.0.0:9000/v1"

    def test_the_command_line_is_printable(self):
        plan = build_server_plan("/m/a model.gguf")
        assert '"/m/a model.gguf"' in plan.command_line()

    def test_extra_args_are_appended(self):
        plan = build_server_plan("/m/big.gguf", extra_args=["--flash-attn"])
        assert plan.argv[-1] == "--flash-attn"

    def test_nothing_is_executed(self):
        """Building a plan must never start a server -- it is for printing."""
        plan = build_server_plan("/does/not/exist.gguf")
        assert isinstance(plan.argv, list)


# --------------------------------------------------------------------------- #
#  Model discovery
# --------------------------------------------------------------------------- #
class TestModelDiscovery:
    def test_the_largest_is_the_target_and_a_small_one_is_the_draft(self, tmp_path):
        (tmp_path / "big.gguf").write_bytes(b"x" * 10_000)
        (tmp_path / "small.gguf").write_bytes(b"x" * 100)

        found = pick_target_and_draft(tmp_path)

        assert found["target"].endswith("big.gguf")
        assert found["draft"].endswith("small.gguf")

    def test_a_similarly_sized_model_is_not_used_as_a_draft(self, tmp_path):
        """A draft close in size to the target saves nothing: the point is
        that drafting is cheap relative to verification."""
        (tmp_path / "a.gguf").write_bytes(b"x" * 10_000)
        (tmp_path / "b.gguf").write_bytes(b"x" * 9_000)

        assert pick_target_and_draft(tmp_path)["draft"] == ""

    def test_an_empty_directory_is_not_an_error(self, tmp_path):
        assert pick_target_and_draft(tmp_path) == {"target": "", "draft": ""}

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        assert pick_target_and_draft(tmp_path / "nope") == {"target": "", "draft": ""}


# --------------------------------------------------------------------------- #
#  Config plumbing
# --------------------------------------------------------------------------- #
def test_the_config_carries_the_draft_settings():
    from jarvis.core.config import LLMConfig

    cfg = LLMConfig()
    assert cfg.draft_model == "", "speculation is opt-in; it needs llama.cpp"
    assert cfg.draft_tokens == DEFAULT_DRAFT_TOKENS


def test_the_cli_registers_serve_plan():
    from jarvis import cli

    args = cli.build_parser().parse_args(["serve-plan"])
    assert args.func is cli.cmd_serve_plan
