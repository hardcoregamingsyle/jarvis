"""The model catalogue: resolution, sizing, credentials, and access checks.

No network, no models, no tokens: ``urlopen`` is monkeypatched, and the token
lookup is pointed at ``tmp_path`` so the developer's real
``~/.cache/huggingface/token`` is neither read nor written.
"""

from __future__ import annotations

import dataclasses
import json
import urllib.error
from pathlib import Path
from typing import Optional

import pytest

from jarvis.core.config import LLMConfig, load_config
from jarvis.llm import models
from jarvis.llm.models import (
    DEFAULT_ALIAS,
    KNOWN_MODELS,
    ModelSpec,
    UnknownModelError,
    UnreleasedModelError,
    all_models,
    check_access,
    estimate_footprint,
    hf_token,
    is_hf_repo_id,
    local_models,
    lookup,
    recommend,
    redact,
    resolve,
)

# A token shaped exactly like the real thing, and used nowhere but here.
FAKE_TOKEN = "hf_QRSTuvwx0123456789ABCDefghIJKLmnop"

# The "announced but not shipped" machinery has to keep working, but pinning it
# to a real catalogue entry means these tests break every time that model
# actually ships (as qwen3.8-27b just did). So the fixture below installs a
# purely fictional entry instead, and the behaviour is tested against that.
UNRELEASED_ALIAS = "test-unreleased-99b"
UNRELEASED_REPO = "Qwen/Qwen-Test-Unreleased-99B"


@pytest.fixture(autouse=True)
def unreleased_entry(monkeypatch: pytest.MonkeyPatch):
    """Add a fictional unreleased model to the catalogue for the duration."""
    spec = models.ModelSpec(
        id=UNRELEASED_REPO,
        label="Qwen Test Unreleased 99B",
        params=99.0,
        family="qwen3",
        context=32768,
        quantised_size_gb=59.0,
        notes="Fictional entry used only to exercise the unreleased-model path.",
        backends=("ollama", "transformers"),
        exists=False,
        alias=UNRELEASED_ALIAS,
    )
    monkeypatch.setitem(KNOWN_MODELS, UNRELEASED_ALIAS, spec)
    monkeypatch.setitem(models._ID_INDEX, UNRELEASED_REPO.lower(), UNRELEASED_ALIAS)
    return spec


@pytest.fixture(autouse=True)
def clean_hf_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """No inherited HF credentials, and a home directory that is not the real one.

    Every path this module consults for a cached token is redirected under
    ``tmp_path``; nothing here deletes or writes outside it.
    """
    for name in (
        "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH",
        "HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME",
    ):
        monkeypatch.delenv(name, raising=False)
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    return fake_home


# --------------------------------------------------------------------------- #
#  Resolution
# --------------------------------------------------------------------------- #
def test_alias_resolves_to_the_curated_spec():
    spec = resolve("qwen3-30b-a3b")
    assert spec.id == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert spec.alias == "qwen3-30b-a3b"
    assert spec.is_moe
    assert spec.effective_params < spec.params


def test_alias_lookup_ignores_case_and_surrounding_space():
    assert resolve("  Qwen3-8B ").id == "Qwen/Qwen3-8B"
    assert resolve("QWEN3-4B").id == "Qwen/Qwen3-4B-Instruct-2507"


def test_exact_hugging_face_id_resolves_to_the_catalogue_entry():
    spec = resolve("Qwen/Qwen3-4B-Instruct-2507")
    assert spec.alias == "qwen3-4b"
    assert spec.context == 262144


def test_ollama_tag_resolves_to_the_same_model():
    spec = resolve("qwen3:4b-instruct-2507-q4_K_M")
    assert spec.alias == "qwen3-4b"
    assert "ollama" in spec.backends


def test_an_unknown_repo_id_is_synthesised_rather_than_rejected():
    spec = resolve("SomeLab/Nova-42B-A5B-Instruct-2999")
    assert spec.id == "SomeLab/Nova-42B-A5B-Instruct-2999"
    assert spec.alias == ""
    assert spec.params == 42.0
    assert spec.active_params == 5.0
    assert spec.quantised_size_gb > 0
    assert "not in the built-in catalogue" in spec.notes.lower()


def test_an_unknown_repo_without_a_size_in_its_name_admits_it_does_not_know():
    spec = resolve("acme/mystery-model")
    assert spec.params == 0.0
    assert spec.quantised_size_gb == 0.0
    assert "unknown" in spec.notes.lower()


@pytest.mark.parametrize("bad", ["not a repo", "", "   ", None, "a/b/c"])
def test_a_string_that_is_not_a_model_reference_is_refused(bad):
    with pytest.raises(UnknownModelError):
        resolve(bad)


def test_the_unreleased_model_is_listed_but_cannot_be_selected():
    listed = {spec.id for spec in all_models()}
    assert UNRELEASED_REPO in listed, "an unavailable target must still be visible"

    known = lookup(UNRELEASED_ALIAS)
    assert known is not None and known.exists is False

    with pytest.raises(UnreleasedModelError) as excinfo:
        resolve(UNRELEASED_ALIAS)
    message = str(excinfo.value).lower()
    assert "not released" in message
    assert UNRELEASED_ALIAS in message
    # The suggested alternative must itself be real.
    assert resolve("qwen3-30b-a3b").exists is True


def test_the_unreleased_model_is_refused_by_its_full_repo_id_too():
    with pytest.raises(UnreleasedModelError):
        resolve(UNRELEASED_REPO)


def test_unreleased_models_can_still_be_inspected_deliberately():
    spec = resolve(UNRELEASED_ALIAS, allow_unreleased=True)
    assert spec.exists is False
    assert spec.id == UNRELEASED_REPO


def test_flipping_the_exists_flag_is_the_entire_release_change(monkeypatch):
    """The promise: when it ships, one flag makes it selectable."""
    shipped = dataclasses.replace(KNOWN_MODELS[UNRELEASED_ALIAS], exists=True)
    monkeypatch.setitem(KNOWN_MODELS, UNRELEASED_ALIAS, shipped)
    assert resolve(UNRELEASED_ALIAS).id == UNRELEASED_REPO


def test_all_models_can_hide_the_unreleased_entries():
    released = {spec.id for spec in all_models(include_unreleased=False)}
    assert UNRELEASED_REPO not in released
    assert "Qwen/Qwen3-8B" in released


def test_the_catalogue_covers_the_qwen3_line():
    for alias in ("qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b", "qwen3-30b-a3b"):
        spec = KNOWN_MODELS[alias]
        assert spec.id.startswith("Qwen/")
        assert spec.ollama_tag, f"{alias} should carry an ollama tag"


def test_resolve_config_follows_the_configured_model(config):
    config.llm.model = "Qwen/Qwen3-14B"
    assert models.resolve_config(config).alias == "qwen3-14b"
    config.llm.backend = "ollama"
    config.llm.ollama_model = "qwen3:8b"
    assert models.resolve_config(config).alias == "qwen3-8b"


# --------------------------------------------------------------------------- #
#  Repo-id shape
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("good", [
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "a/b",
    "org.name/model_name-v2",
])
def test_is_hf_repo_id_accepts_real_repo_ids(good):
    assert is_hf_repo_id(good) is True


@pytest.mark.parametrize("bad", [
    "not a repo", "a/b/c", "", "   ", None, 42, ["Qwen/Qwen3-8B"],
    "Qwen/", "/Qwen3-8B", "Qwen3-8B", "org/../etc", "org/name with space",
])
def test_is_hf_repo_id_rejects_everything_else(bad):
    assert is_hf_repo_id(bad) is False


# --------------------------------------------------------------------------- #
#  Token discovery
# --------------------------------------------------------------------------- #
def test_config_token_beats_the_environment(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_from_the_environment")
    cfg = LLMConfig(hf_token=FAKE_TOKEN)
    assert hf_token(cfg) == FAKE_TOKEN


def test_an_empty_config_token_does_not_shadow_the_environment(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    assert hf_token(LLMConfig(hf_token="")) == FAKE_TOKEN
    assert hf_token(LLMConfig(hf_token="   ")) == FAKE_TOKEN


def test_hf_token_reads_the_nested_llm_section(config):
    config.llm.hf_token = FAKE_TOKEN
    assert hf_token(config) == FAKE_TOKEN


def test_hf_token_prefers_hf_token_over_the_longer_variable(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_the_other_one")
    assert hf_token(LLMConfig()) == FAKE_TOKEN


def test_hf_token_falls_back_to_hugging_face_hub_token(monkeypatch):
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", FAKE_TOKEN)
    assert hf_token(LLMConfig()) == FAKE_TOKEN


def test_hf_token_falls_back_to_the_cli_cache_file(monkeypatch, tmp_path):
    hf_home = tmp_path / "hfhome"
    hf_home.mkdir()
    (hf_home / "token").write_text(FAKE_TOKEN + "\n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(hf_home))
    assert hf_token(LLMConfig()) == FAKE_TOKEN


def test_hf_token_honours_an_explicit_token_path(monkeypatch, tmp_path):
    token_file = tmp_path / "elsewhere.token"
    token_file.write_text(f"  {FAKE_TOKEN}  ", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN_PATH", str(token_file))
    assert hf_token(LLMConfig()) == FAKE_TOKEN


def test_the_environment_beats_the_cli_cache_file(monkeypatch, tmp_path):
    hf_home = tmp_path / "hfhome"
    hf_home.mkdir()
    (hf_home / "token").write_text("hf_stale_cached_token", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    assert hf_token(LLMConfig()) == FAKE_TOKEN


def test_hf_token_returns_none_rather_than_an_empty_string():
    assert hf_token(LLMConfig()) is None
    assert hf_token(None) is None


def test_an_empty_token_file_counts_as_no_token(monkeypatch, tmp_path):
    hf_home = tmp_path / "hfhome"
    hf_home.mkdir()
    (hf_home / "token").write_text("\n  \n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(hf_home))
    assert hf_token(LLMConfig()) is None


def test_hf_token_accepts_an_explicit_environment_mapping():
    assert hf_token(LLMConfig(), environ={"HF_TOKEN": FAKE_TOKEN}) == FAKE_TOKEN


# --------------------------------------------------------------------------- #
#  Redaction
# --------------------------------------------------------------------------- #
def test_redact_masks_a_token_in_a_url():
    url = f"https://huggingface.co/api/models/Qwen/Qwen3-8B?token={FAKE_TOKEN}"
    masked = redact(url)
    assert FAKE_TOKEN not in masked
    assert "REDACTED" in masked
    assert "huggingface.co/api/models/Qwen/Qwen3-8B" in masked


def test_redact_masks_a_token_in_an_error_string():
    error = f"HTTPError 401: Invalid credentials in Authorization: Bearer {FAKE_TOKEN}"
    masked = redact(error)
    assert FAKE_TOKEN not in masked
    assert "401" in masked


def test_redact_masks_a_bare_token():
    assert FAKE_TOKEN not in redact(f"the token is {FAKE_TOKEN}, do not share it")


def test_redact_leaves_our_own_advice_readable():
    advice = "Set HF_TOKEN or llm.hf_token; see https://huggingface.co/settings/tokens"
    assert redact(advice) == advice


def test_redact_handles_empty_and_none():
    assert redact(None) == ""
    assert redact("") == ""


# --------------------------------------------------------------------------- #
#  Redaction of credentials WITHOUT a recognisable prefix
#
#  The tests above all use an ``hf_``-prefixed token, which the dedicated
#  Hugging Face pattern catches on its own — so they passed while a real bug sat
#  underneath. A credential with no known prefix (``llm.api_key``: a vLLM
#  --api-key value, or a hosted OpenAI-compatible key) took a different path:
#  the generic key/value pattern matched "Authorization: Bearer" and masked the
#  literal word "Bearer" as though it were the value, then, having consumed it,
#  stopped the standalone Bearer pattern from ever matching. The secret survived
#  in cleartext in exactly the header openai_compat.py builds.
# --------------------------------------------------------------------------- #
OPAQUE_SECRET = "sk_longenoughsecret123"


@pytest.mark.parametrize("text", [
    f"Authorization: Bearer {OPAQUE_SECRET}",
    f"authorization=Bearer {OPAQUE_SECRET}",
    f"Bearer {OPAQUE_SECRET}",
    f"Authorization: Token {OPAQUE_SECRET}",
    f"api_key={OPAQUE_SECRET}",
    f"apikey: {OPAQUE_SECRET}",
    f"POST /v1/chat/completions -- headers: {{'Authorization': 'Bearer {OPAQUE_SECRET}'}}",
])
def test_redact_masks_a_prefixless_credential(text):
    masked = redact(text)
    assert OPAQUE_SECRET not in masked, f"credential leaked: {masked}"
    assert "REDACTED" in masked


def test_redact_keeps_the_scheme_word_so_the_message_stays_readable():
    """Masking the value must not destroy the diagnostic context around it."""
    masked = redact(f"Authorization: Bearer {OPAQUE_SECRET}")
    assert masked.startswith("Authorization: Bearer ")
    assert OPAQUE_SECRET not in masked


def test_redact_does_not_mangle_ordinary_prose():
    plain = "the model is Qwen/Qwen3-8B and the key is in your config"
    assert redact(plain) == plain


# --------------------------------------------------------------------------- #
#  Access checks
# --------------------------------------------------------------------------- #
class FakeResponse:
    """Minimal stand-in for the object ``urlopen`` yields."""

    def __init__(self, status: int = 200, body: str = "{}") -> None:
        self.status = status
        self._body = body.encode("utf-8")

    def read(self, amount: Optional[int] = None) -> bytes:
        return self._body if amount is None else self._body[:amount]

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def urlopen(monkeypatch):
    """Install a scripted ``urlopen``; returns the ``(url, auth-header)`` log.

    Nothing in this module may touch the network, so the log doubles as proof
    that a request was — or was not — attempted.
    """
    calls: list = []

    def install(outcome):
        def fake(request, timeout=None, **kwargs):
            calls.append((request.full_url, request.get_header("Authorization")))
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr("urllib.request.urlopen", fake)
        return calls

    return install


def test_check_access_reports_ok_for_a_public_repo(urlopen):
    urlopen(FakeResponse(200, '{"id": "Qwen/Qwen3-8B", "gated": false}'))
    result = check_access(resolve("qwen3-8b"))
    assert result["status"] == "ok"
    assert result["ok"] is True
    assert result["http_status"] == 200


def test_check_access_asks_for_a_token_on_401_without_one(urlopen):
    calls = urlopen(urllib.error.HTTPError(
        "https://huggingface.co/api/models/x", 401, "Unauthorized", {}, None))
    result = check_access(resolve("qwen3-8b"))
    assert result["status"] == "needs_token"
    assert "HF_TOKEN" in result["message"]
    assert calls[0][1] is None, "no token was supplied, so no header may be sent"


def test_check_access_calls_a_rejected_token_gated(urlopen):
    calls = urlopen(urllib.error.HTTPError(
        "https://huggingface.co/api/models/x", 401, "Unauthorized", {}, None))
    result = check_access(resolve("llama3.1-8b"), FAKE_TOKEN)
    assert result["status"] == "gated"
    assert result["authenticated"] is True
    assert calls[0][1] == f"Bearer {FAKE_TOKEN}", "the token must actually be used"


def test_check_access_explains_a_gated_repo_on_403(urlopen):
    urlopen(urllib.error.HTTPError(
        "https://huggingface.co/api/models/x", 403, "Forbidden", {}, None))
    result = check_access(resolve("llama3.1-8b"))
    assert result["status"] == "gated"
    assert result["gated"] is True
    assert "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct" in result["message"]
    assert "licence" in result["message"]


def test_check_access_reports_not_found_on_404(urlopen):
    urlopen(urllib.error.HTTPError(
        "https://huggingface.co/api/models/x", 404, "Not Found", {}, None))
    result = check_access("Qwen/Qwen3-Typo-8B")
    assert result["status"] == "not_found"
    assert result["ok"] is False
    assert "HF_TOKEN" in result["message"], "a 404 can also mean 'private'"


def test_check_access_reports_offline_on_a_connection_error(urlopen):
    urlopen(urllib.error.URLError("[Errno -3] Temporary failure in name resolution"))
    result = check_access(resolve("qwen3-8b"))
    assert result["status"] == "offline"
    assert result["http_status"] is None
    assert "name resolution" in result["message"]


def test_check_access_survives_an_unexpected_exception(urlopen):
    urlopen(RuntimeError("the socket layer exploded"))
    result = check_access(resolve("qwen3-8b"))
    assert result["status"] == "offline"


def test_check_access_treats_a_server_error_as_offline(urlopen):
    urlopen(FakeResponse(503, ""))
    result = check_access(resolve("qwen3-8b"))
    assert result["status"] == "offline"
    assert result["http_status"] == 503


def test_check_access_never_leaks_the_token(urlopen):
    urlopen(urllib.error.URLError(f"bad handshake using Bearer {FAKE_TOKEN}"))
    result = check_access(resolve("qwen3-8b"), FAKE_TOKEN)
    assert FAKE_TOKEN not in json.dumps(result)
    assert "REDACTED" in result["message"]


def test_check_access_does_not_hit_the_network_for_a_non_repo(urlopen):
    calls = urlopen(RuntimeError("must not be called"))
    result = check_access("qwen3:8b")
    assert result["status"] == "not_found"
    assert calls == []


def test_check_access_short_circuits_an_unreleased_model(urlopen):
    calls = urlopen(RuntimeError("must not be called"))
    result = check_access(resolve(UNRELEASED_ALIAS, allow_unreleased=True))
    assert result["status"] == "not_found"
    assert "not released" in result["message"]
    assert calls == []


def test_check_access_never_puts_the_token_in_the_url(urlopen):
    calls = urlopen(FakeResponse(200))
    check_access(resolve("qwen3-8b"), FAKE_TOKEN)
    assert FAKE_TOKEN not in calls[0][0]


# --------------------------------------------------------------------------- #
#  Sizing
# --------------------------------------------------------------------------- #
def _synthetic(params: float) -> ModelSpec:
    return ModelSpec(
        id=f"test/model-{params}B",
        label=f"test {params}B",
        params=params,
        family="test",
        context=32768,
        quantised_size_gb=params * 0.6,
    )


def test_estimate_footprint_is_monotonic_in_parameter_count():
    sizes = [_synthetic(p) for p in (0.6, 1.7, 4.0, 8.0, 14.0, 32.0, 70.0)]
    footprints = [estimate_footprint(s, "q4") for s in sizes]
    downloads = [f["download_gb"] for f in footprints]
    rams = [f["ram_gb"] for f in footprints]
    assert downloads == sorted(downloads) and len(set(downloads)) == len(downloads)
    assert rams == sorted(rams) and len(set(rams)) == len(rams)


def test_estimate_footprint_is_monotonic_across_the_real_dense_ladder():
    ladder = ["qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b"]
    downloads = [estimate_footprint(resolve(a))["download_gb"] for a in ladder]
    assert downloads == sorted(downloads)


def test_estimate_footprint_shrinks_with_heavier_quantisation():
    spec = resolve("qwen3-8b")
    fp16 = estimate_footprint(spec, "fp16")["download_gb"]
    eight = estimate_footprint(spec, "8bit")["download_gb"]
    four = estimate_footprint(spec, "q4")["download_gb"]
    assert fp16 > eight > four > 0


def test_estimate_footprint_accepts_a_plain_string():
    result = estimate_footprint("Qwen/Qwen3-8B", "q4")
    assert result["model"] == "Qwen/Qwen3-8B"
    assert result["ram_gb"] > result["download_gb"]


def test_a_moe_costs_ram_like_30b_but_kv_cache_like_3b():
    moe = estimate_footprint(resolve("qwen3-30b-a3b"), "q4")
    dense = estimate_footprint(resolve("qwen3-32b"), "q4")
    assert moe["kv_cache_gb"] < dense["kv_cache_gb"] / 4
    assert moe["weights_gb"] > dense["weights_gb"] * 0.8


def test_estimate_footprint_reports_an_error_instead_of_raising():
    result = estimate_footprint("not a repo")
    assert result["download_gb"] == 0.0
    assert "error" in result


# --------------------------------------------------------------------------- #
#  Recommendation
# --------------------------------------------------------------------------- #
def test_recommend_fits_a_32gb_cpu_only_laptop():
    spec = recommend(32, False, "chat")
    assert spec.exists and not spec.gated
    footprint = estimate_footprint(spec, "q4")
    assert footprint["ram_gb"] < 32.0
    assert spec.effective_params <= 6.0, "an interactive CPU model must stay small per token"
    assert spec.params >= 8.0, "a 32 GB machine deserves better than a toy model"


def test_recommend_does_not_pick_a_dense_32b_for_interactive_cpu_use():
    spec = recommend(32, False, "chat")
    assert spec.id != KNOWN_MODELS["qwen3-32b"].id
    assert spec.id == KNOWN_MODELS["qwen3-30b-a3b"].id


def test_recommend_scales_down_for_a_small_machine():
    spec = recommend(8, False, "chat")
    assert spec.quantised_size_gb <= 8 * 0.6


def test_recommend_never_raises_on_an_absurd_budget():
    spec = recommend(0.1, False, "chat")
    assert isinstance(spec, ModelSpec)
    assert spec.params <= 1.0


def test_recommend_prefers_the_coder_model_for_code():
    assert "coder" in recommend(32, False, "code").id.lower()


def test_recommend_will_use_a_dense_model_when_a_gpu_is_present():
    spec = recommend(64, True, "quality")
    assert spec.params >= 30


@pytest.mark.parametrize("ram", [4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("gpu", [True, False])
@pytest.mark.parametrize("purpose", ["chat", "code", "quality", "tiny", "nonsense"])
def test_recommend_only_ever_suggests_a_real_ungated_model(ram, gpu, purpose):
    spec = recommend(ram, gpu, purpose)
    assert spec.exists is True
    assert spec.gated is False


# --------------------------------------------------------------------------- #
#  What is on disk
# --------------------------------------------------------------------------- #
def test_local_models_finds_downloads_in_both_locations(config, monkeypatch, tmp_path):
    shard_dir = config.models_dir() / "Qwen3-30B-A3B-Instruct-2507"
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / "model-00001.safetensors").write_bytes(b"x" * 2048)

    cache = tmp_path / "hfcache" / "hub"
    repo = cache / "models--Qwen--Qwen3-8B" / "snapshots" / "abc"
    repo.mkdir(parents=True)
    (repo / "model.safetensors").write_bytes(b"y" * 512)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))

    found = local_models(config)
    by_id = {entry["id"]: entry for entry in found}

    assert "Qwen3-30B-A3B-Instruct-2507" in by_id
    assert by_id["Qwen3-30B-A3B-Instruct-2507"]["source"] == "jarvis"
    assert by_id["Qwen3-30B-A3B-Instruct-2507"]["size_bytes"] == 2048

    assert "Qwen/Qwen3-8B" in by_id, "hf cache directories must be un-mangled"
    cached = by_id["Qwen/Qwen3-8B"]
    assert cached["source"] == "hf-cache"
    assert cached["size_bytes"] == 512
    assert cached["label"] == "Qwen3 8B", "known repos get their catalogue label"

    assert [e["size_bytes"] for e in found] == sorted(
        (e["size_bytes"] for e in found), reverse=True
    )


def test_local_models_picks_up_loose_gguf_files(config):
    gguf = config.models_dir() / "some-model-q4_k_m.gguf"
    gguf.write_bytes(b"g" * 128)
    ids = {entry["id"] for entry in local_models(config)}
    assert "some-model-q4_k_m" in ids


def test_local_models_is_empty_and_calm_when_nothing_is_downloaded(config, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "does-not-exist"))
    assert local_models(config) == []
    assert local_models(None) == []


def test_local_models_returns_no_credentials(config, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    (config.models_dir() / "a-model").mkdir(parents=True, exist_ok=True)
    assert FAKE_TOKEN not in json.dumps(local_models(config))


# --------------------------------------------------------------------------- #
#  Config integration
# --------------------------------------------------------------------------- #
def test_the_new_llm_fields_have_the_documented_defaults():
    llm = LLMConfig()
    assert llm.vllm_host == "http://127.0.0.1:8000/v1"
    assert llm.api_key == ""
    assert llm.hf_token == ""
    assert llm.max_concurrent_requests == 8
    assert llm.model_revision == ""
    assert llm.trust_remote_code is False
    # The default model is asserted in tests/test_default_model.py, which
    # also pins the figures behind it. Checking the identity here keeps the
    # two in step without duplicating the justification.
    assert llm.model == resolve(DEFAULT_ALIAS).id


def test_load_config_honours_the_standard_hugging_face_variables():
    assert load_config(environ={"HF_TOKEN": FAKE_TOKEN}).llm.hf_token == FAKE_TOKEN
    assert load_config(
        environ={"HUGGING_FACE_HUB_TOKEN": FAKE_TOKEN}
    ).llm.hf_token == FAKE_TOKEN


def test_the_jarvis_specific_variable_wins_over_the_standard_one():
    cfg = load_config(environ={
        "HF_TOKEN": "hf_the_generic_one",
        "JARVIS_LLM_HF_TOKEN": FAKE_TOKEN,
    })
    assert cfg.llm.hf_token == FAKE_TOKEN


def test_a_config_file_token_is_not_clobbered_by_the_environment(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"llm": {"hf_token": FAKE_TOKEN}}), encoding="utf-8")
    cfg = load_config(path, environ={"HF_TOKEN": "hf_the_generic_one"})
    assert cfg.llm.hf_token == FAKE_TOKEN


def test_no_token_anywhere_leaves_the_field_empty():
    assert load_config(environ={}).llm.hf_token == ""


def test_the_new_fields_take_typed_environment_overrides():
    cfg = load_config(environ={
        "JARVIS_LLM_VLLM_HOST": "http://10.0.0.4:8000/v1",
        "JARVIS_LLM_MAX_CONCURRENT_REQUESTS": "2",
        "JARVIS_LLM_TRUST_REMOTE_CODE": "true",
        "JARVIS_LLM_MODEL_REVISION": "e1f2a3b",
    })
    assert cfg.llm.vllm_host == "http://10.0.0.4:8000/v1"
    assert cfg.llm.max_concurrent_requests == 2
    assert cfg.llm.trust_remote_code is True
    assert cfg.llm.model_revision == "e1f2a3b"


def test_switching_model_is_a_single_environment_variable():
    cfg = load_config(environ={"JARVIS_LLM_MODEL": "Qwen/Qwen3-14B"})
    assert models.resolve_config(cfg).alias == "qwen3-14b"


# --------------------------------------------------------------------------- #
#  Presentation helpers
# --------------------------------------------------------------------------- #
def test_describe_flags_the_unreleased_and_the_gated():
    unreleased = models.describe(lookup(UNRELEASED_ALIAS))
    assert "NOT RELEASED" in unreleased
    assert "gated" in models.describe(lookup("llama3.1-8b"))
    moe = models.describe(lookup("qwen3-30b-a3b"))
    assert "MoE" in moe and "3.3B active" in moe


def test_with_revision_records_the_pin_without_changing_the_repo():
    spec = resolve("qwen3-8b")
    pinned = models.with_revision(spec, "abc1234")
    assert pinned.id == spec.id
    assert "abc1234" in pinned.label
    assert models.with_revision(spec, "") is spec


def test_iter_aliases_lists_the_catalogue_keys():
    aliases = models.iter_aliases()
    assert "qwen3-30b-a3b" in aliases
    assert aliases == sorted(aliases)


@pytest.mark.parametrize("name,bits", [
    ("q4_K_M", 4.8), ("4bit", 4.8), ("8bit", 8.5), ("fp16", 16.0),
    ("", 16.0), ("nonsense", 16.0), (None, 16.0),
])
def test_quantisation_bits_maps_the_usual_names(name, bits):
    assert models.quantisation_bits(name) == bits


@pytest.mark.parametrize("text,expected", [
    ("qwen3:8b", True), ("llama3.1:8b-instruct-q4_K_M", True),
    ("Qwen/Qwen3-8B", False), ("qwen3", False), ("", False), (None, False),
])
def test_is_ollama_tag(text, expected):
    assert models.is_ollama_tag(text) is expected
