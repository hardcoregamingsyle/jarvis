"""Configuration: dataclasses + YAML/JSON file + environment overrides.

Load order (later wins):
  1. dataclass defaults (tuned for the target i5-10210U / 32 GB laptop)
  2. config file (``config.yaml`` next to the project, or in the user config dir)
  3. environment variables prefixed ``JARVIS_``

The file parser accepts YAML when PyYAML is installed and falls back to JSON,
so configuration never becomes a hard dependency.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, get_args, get_origin

from .platform_utils import config_dir, data_dir, ensure_dir


# --------------------------------------------------------------------------- #
#  Sections
# --------------------------------------------------------------------------- #
@dataclass
class LLMConfig:
    """Which model to run, where to run it, and how to reach it.

    Nothing in JARVIS hardcodes a model name: ``model`` is the single source of
    truth and every field below is overridable from the config file or from
    ``JARVIS_LLM_<FIELD>`` in the environment, so moving to a different Hugging
    Face repo is a one-line change.

    Connection / credentials:
      ``vllm_host``  Base URL of an OpenAI-compatible server (vLLM, llama.cpp
        ``--api``, TGI, LM Studio…).  Includes the ``/v1`` suffix because that
        is what the OpenAI wire protocol expects.
      ``api_key``  Bearer token for that endpoint — vLLM's ``--api-key`` value,
        or the key of a hosted OpenAI-compatible service.  Empty means the
        endpoint is unauthenticated, which is the normal case for a local vLLM.
      ``hf_token``  Hugging Face access token, needed only for gated or private
        repositories (Llama, Gemma, and anything behind a licence click-through).
        Left empty, ``load_config`` picks up the standard ``HF_TOKEN`` or
        ``HUGGING_FACE_HUB_TOKEN`` environment variables instead, and
        :func:`jarvis.llm.models.hf_token` additionally falls back to the token
        the ``huggingface-cli login`` command caches on disk.

    Model selection:
      ``model_revision``  Pin a specific commit SHA, branch or tag of the HF
        repo.  Empty means "whatever ``main`` is today", which is convenient
        until an upstream re-upload silently changes the weights underneath a
        working deployment.
      ``trust_remote_code``  Allow the repo's own Python modelling code to run
        at load time.  Required by some architectures (and by brand-new ones
        that ship ahead of a ``transformers`` release); off by default because
        it executes code from the repo, not because anything is forbidden.

    Resource management:
      ``max_concurrent_requests``  Ceiling on in-flight generation calls.  A
        CPU-only box has one set of cores: letting a swarm of subagents issue
        unbounded parallel requests turns a slow answer into no answer at all
        and can exhaust memory.  ``0`` disables the cap.

    Two backends, one opt-in:
      By default there is exactly one backend — everything below describes
      it, unchanged from how JARVIS has always worked — and ``spawn_task``
      simply reuses it, because ``task_backend`` and friends are empty.

      Setting the ``task_*`` fields turns that one backend into two, with two
      different jobs. ``backend``/``model``/``ollama_model`` above then
      becomes the ROUTER: the model that holds the conversation and answers
      directly, small and fast enough that its own tokens can drive live
      speech (see :mod:`jarvis.speech.streaming_tts`) — its latency IS the
      assistant's latency, so this is where ``qwen3:4b-instruct-2507-q4_K_M``
      belongs, not a dense 27B. The ``task_*`` fields then describe the model
      ``spawn_task`` dispatches a background
      :class:`~jarvis.agent.subagent.SubAgent` to — a dense 27B, or a larger
      model served elsewhere as an OpenAI-compatible endpoint (llama.cpp's
      ``llama-server``, vLLM, ...). Its latency does not matter the same way:
      the router has already answered and is not waiting on it, and the
      result is relayed through TTS when it lands. See
      ``config.example.yaml`` for a worked two-tier example.
    """

    # "airllm", "ollama", "transformers", "openai-compat", "stub", or "auto".
    backend: str = "auto"
    # Qwen3.8-27B: dense 27B, vision-language, 262K native context, Apache 2.0.
    # The most capable model that still fits 32 GB at Q4 (~18 GB); it needs
    # transformers >= 4.57.
    #
    # Being DENSE is the trade-off: every one of the 27B parameters is read per
    # token, where Qwen3-30B-A3B activates only ~3B. On a CPU-only box that is
    # roughly 0.5-1 tok/s against 4-8. This is why `voice_model` below exists:
    # the big model reasons, a small one speaks, and the conversation stays
    # responsive regardless of how long the thinking takes -- and, when this
    # same model is also busy enough to be worth keeping off the interactive
    # path entirely, `task_model` further below hands `spawn_task` a separate
    # backend so `model` here can instead be something genuinely fast.
    model: str = "Qwen/Qwen3.8-27B"
    # AirLLM streams layers from disk; this is where they get cached.
    compression: str = ""            # "" | "4bit" | "8bit"  (needs bitsandbytes)
    layer_shards_dir: str = ""       # defaults to <data>/models
    device: str = "auto"             # "auto" | "cpu" | "cuda" | "mps"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    # Context window. Two costs, and the second is the one that bites:
    #   RAM  -- KV cache. Cheap on Qwen3.8: only 16 of its 64 layers are full
    #           attention, so ~64 KB/token. 32k costs ~2.1 GB.
    #   TIME -- PREFILL. Reading a prompt is compute-bound (matrix-matrix), not
    #           bandwidth-bound like generation, and on 4 cores a 27B ingests
    #           roughly 8-15 tok/s. A COLD 32k prompt is 35-70 minutes.
    #
    # Prefix caching is what makes a large window usable: llama.cpp reuses the
    # KV of an unchanged prefix, so a long-lived session pays prefill ONCE and
    # subsequent turns only ingest what changed. Set this high for deep work
    # sessions; keep the conversation going rather than restarting it.
    #
    # 32768 suits real engineering work (a large source file plus history).
    # Raise to 131072 for whole-repository or datasheet work and accept the
    # first-turn cost. See docs/PERFORMANCE.md.
    context_tokens: int = 32768
    # Hard ceiling on what one tool result may contribute. Without this a
    # single large file read silently blows the window and the server drops
    # the top of the prompt -- losing the system prompt or the question.
    max_tool_result_tokens: int = 8000
    # Ollama fallback settings
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.8:27b"
    # OpenAI-compatible server (vLLM and friends); see the class docstring.
    vllm_host: str = "http://127.0.0.1:8000/v1"
    api_key: str = ""
    # Hugging Face credentials and repo pinning.
    hf_token: str = ""
    model_revision: str = ""
    trust_remote_code: bool = False
    # 0 = unlimited; caps in-flight LLM calls so subagents cannot swamp the CPU.
    max_concurrent_requests: int = 8
    # Fail over to the next available backend instead of raising.
    allow_fallback: bool = True
    request_timeout: float = 600.0

    # -- Thinking / reasoning ---------------------------------------------- #
    # Qwen3.x ships with chain-of-thought ON. A dense 27B on a CPU spends the
    # entire token budget inside <think> and returns an empty answer, which is
    # the single most common cause of "JARVIS never replies". "auto" turns it
    # off for model families known to reason by default (see
    # jarvis.llm.models.THINKING_DEFAULT_ON); "on" forces it; "off" always
    # disables it. Background subagents may still reason — see
    # AgentConfig.subagent_thinking.
    thinking: str = "auto"           # "auto" | "on" | "off"

    # -- The voice model ---------------------------------------------------- #
    # The trick that makes a slow local brain feel instant. The big model above
    # does the thinking and the tool work; this small one turns its result into
    # the sentence that actually gets spoken, and it starts speaking while the
    # big model is still working. A 1.7-3B model runs at 15-30 tok/s on a CPU,
    # so the reply begins in well under a second.
    #
    # Empty voice_model disables the split entirely and the main model speaks
    # for itself.
    voice_model: str = "qwen3:1.7b"
    voice_model_enabled: bool = True
    # Voice replies are one or two spoken sentences; they never need more.
    voice_max_new_tokens: int = 160
    voice_temperature: float = 0.5
    # Spoken immediately, before the main model has finished. Keeps the
    # conversation alive instead of leaving dead air.
    voice_ack_enabled: bool = True

    # -- CPU tuning --------------------------------------------------------- #
    # Dense inference on a CPU is memory-bandwidth bound. 0 = use physical
    # cores, which is what you want: hyperthreaded siblings share one memory
    # port and contend for the exact resource that is already the bottleneck.
    num_threads: int = 0
    # Map weights from the page cache rather than copying them in. Leave on.
    use_mmap: bool = True
    # Pin weights in RAM. Only with headroom to spare -- on a tight machine
    # this causes swapping, which is far slower than not pinning.
    use_mlock: bool = False

    # -- Speculative decoding (llama.cpp only) ------------------------------ #
    # The one change that makes a dense 27B genuinely usable on this CPU. A
    # small draft model proposes N tokens; the big model verifies all N in a
    # SINGLE batched pass, so its 18 GB of weights are read once per round
    # instead of once per token. Output is identical to running the big model
    # alone -- rejected drafts are discarded -- so unlike dropping to Q2 this
    # costs no quality at all.
    #
    # Roughly 2-3x on an i5-10210U: ~1.5 tok/s becomes ~3.3-4.8 depending on
    # how often the draft agrees. Requires llama.cpp; Ollama does not expose
    # --model-draft. See docs/PERFORMANCE.md.
    draft_model: str = ""            # path to a small GGUF, or "" to disable
    draft_tokens: int = 4            # proposals per round; >6 rarely pays
    llamacpp_host: str = "http://127.0.0.1:8080/v1"

    # -- Routing ------------------------------------------------------------ #
    # When true the small model triages every turn and only wakes the big one
    # for work that genuinely needs it. See jarvis.agent.router.
    routing_enabled: bool = True

    # -- Task / delegated model ---------------------------------------------- #
    # A DIFFERENT axis from voice_model/routing above: those make the *same*
    # big model above feel responsive (a small model phrases for it, triages
    # around it). This instead gives `spawn_task` a SEPARATE backend to
    # delegate to, so `model`/`ollama_model` above can be the fast one doing
    # the actual talking. Empty (the default) = spawn_task reuses `model`
    # above, i.e. today's single-backend behaviour, exactly unchanged.
    #
    # Worked example -- a MiniLLM-hosted llama-server elsewhere serving a much
    # larger model, with THIS machine's model/ollama_model turned fast:
    #   model: Qwen/Qwen3-4B-Instruct-2507   (router, fast, conversational)
    #   ollama_model: qwen3:4b-instruct-2507-q4_K_M
    #   task_backend: openai-compat
    #   task_base_url: http://<the other machine>:8080/v1
    task_backend: str = ""
    task_model: str = ""
    task_ollama_model: str = ""
    task_base_url: str = ""          # e.g. a llama.cpp llama-server's http://host:port/v1
    task_api_key: str = ""
    task_max_new_tokens: int = 1024
    task_temperature: float = 0.7
    task_request_timeout: float = 900.0


@dataclass
class STTConfig:
    """Speech in. Defaults tuned for accuracy on a 4-core CPU with 32 GB RAM.

    ``model`` is the single biggest lever on transcription quality.
    ``small.en`` is roughly 3x the compute of ``base.en`` but makes far fewer
    word errors, and at int8 on four cores it still transcribes a five-second
    utterance in well under a second -- comfortably real-time. On a machine
    with less headroom, drop to ``base.en``.
    """

    engine: str = "auto"             # "auto" | "faster-whisper" | "whisper" | "vosk" | "stub"
    # tiny.en < base.en < small.en < medium.en. small.en is the accuracy sweet
    # spot when there is RAM to spare; base.en if you need the speed back.
    model: str = "small.en"
    device: str = "auto"
    compute_type: str = "int8"       # int8 is ~2x faster on CPU with negligible loss
    language: str = "en"
    sample_rate: int = 16000
    vad_filter: bool = True

    # -- Decoding quality --------------------------------------------------- #
    # Greedy decoding (beam_size=1) is what makes small Whisper models sound
    # "trash": it commits to a bad first token and never recovers. 5 is the
    # reference default and costs little on short utterances.
    beam_size: int = 5
    # Whisper conditions on its own previous output by default, which makes a
    # single mistake propagate through the rest of the session and is the
    # classic cause of runaway repeated text.
    condition_on_previous_text: bool = False
    # Biases the decoder towards the words it will actually hear. Proper nouns
    # are exactly what a small model gets wrong, and "Jarvis" is the one word
    # that must survive -- misheard, the wake word never fires.
    initial_prompt: str = "Jarvis, the British AI assistant."
    # 0 = use every core. Left at 0 the CTranslate2 default is often 1 thread.
    cpu_threads: int = 0
    # Segments the model is this unsure about are dropped rather than guessed.
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    # Whisper reliably hallucinates stock phrases over silence and noise
    # ("Thank you.", "Thanks for watching!"). Filter them out of short
    # transcripts; see jarvis.speech.stt.HALLUCINATIONS.
    filter_hallucinations: bool = True

    # -- Voice activity / recording behaviour ------------------------------- #
    silence_threshold: float = 0.015
    # 0.9s cuts people off mid-thought; anyone who pauses to think gets their
    # sentence truncated and half-transcribed.
    silence_duration: float = 1.2
    max_utterance_seconds: float = 30.0
    # Audio kept from *before* speech was detected, so the first syllable of
    # the wake word is not clipped off.
    preroll_seconds: float = 0.5
    # Shorter bursts are coughs, clicks and door slams, not speech.
    min_speech_seconds: float = 0.3
    input_device: Optional[int] = None


@dataclass
class TTSConfig:
    # "auto" tries piper -> edge -> pyttsx3 -> espeak and uses the first that works.
    engine: str = "auto"
    # Piper: a crisp RP-English voice; the closest offline match to film-JARVIS.
    piper_voice: str = "en_GB-alan-medium"
    piper_model_path: str = ""       # auto-downloaded into <data>/voices when empty
    # edge-tts (needs internet, but the most "polished British" of the lot).
    edge_voice: str = "en-GB-RyanNeural"
    edge_rate: str = "+0%"
    edge_pitch: str = "+0Hz"
    # pyttsx3 / SAPI fallback: prefer a UK English system voice.
    sapi_voice_hint: str = "United Kingdom"
    speed: float = 1.0
    volume: float = 1.0
    output_device: Optional[int] = None
    enabled: bool = True

    # Speak the router's reply live, sentence by sentence, as it streams --
    # instead of waiting for the whole reply and synthesizing it as one
    # block. See jarvis.speech.streaming_tts.StreamingSpeaker. Off falls back
    # to today's whole-utterance SpeechQueue.
    streaming: bool = True
    # A sentence shorter than this is held and merged into the next one,
    # rather than firing a synth+play round trip for one short word.
    stream_min_chars: int = 12
    # No sentence-ending punctuation for this many characters -> speak what
    # has arrived on a whitespace boundary anyway, so a reply without
    # punctuation is still heard incrementally rather than staying silent
    # until the whole thing is done.
    stream_max_buffer: int = 220


@dataclass
class MemoryConfig:
    db_path: str = ""                # defaults to <data>/memory.db
    embedder: str = "auto"           # "auto" | "sentence-transformers" | "hash" | "none"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384
    # How much of the recalled memory to inject into each prompt.
    recall_k: int = 8
    recall_min_score: float = 0.15
    # Conversation summarisation keeps context bounded while retaining detail.
    summarize_after_turns: int = 20
    keep_recent_turns: int = 8
    # Never delete anything — the "forgets nothing" guarantee.
    prune: bool = False


@dataclass
class AgentConfig:
    name: str = "JARVIS"
    user_title: str = "Sir"
    max_tool_iterations: int = 8
    max_concurrent_tasks: int = 4
    subagent_timeout: float = 900.0
    # Announce subagent completions as soon as the main agent is idle.
    announce_updates: bool = True
    # Let the agent write brand-new tools when a capability is missing.
    allow_tool_creation: bool = True
    # Anything the router escalates to the big model runs as a BACKGROUND
    # subagent instead of blocking the turn. The small model answers
    # immediately and the subagent's report is relayed when it lands.
    #
    # This is the difference between an assistant and a batch job on a
    # CPU-only box: a dense 27B at ~0.5-1 tok/s takes minutes on real work,
    # and blocking the conversation for that long reads as a crash. Set False
    # to make every turn wait for its own answer (which is what one-shot
    # `jarvis ask` does, since there is no later turn to relay a report on).
    background_heavy_work: bool = True
    # THE HARD CEILING ON HOW LONG ANY TURN MAY TAKE. A dispatched turn is
    # waited on for this long; whatever has not finished by then is left to
    # run in the background and relayed later. So a reply always lands within
    # this many seconds, no matter how slow the model or how large the job.
    #
    # It doubles as the line between quick and slow work: a single tool call
    # often settles inside it and answers within the turn, while a research
    # task will not and gets handed off. 0 defers everything immediately.
    background_after_seconds: float = 5.0


@dataclass
class SecurityConfig:
    """Access policy for full-PC control.  Unrestricted out of the box.

    JARVIS is built to drive the machine it runs on, and by default it does so
    without asking: ``mode="open"`` permits every path, every command and every
    tool, no confirmation prompts, no protected locations.  Nothing here blocks
    anything until you configure it to.

    The policy engine is still fully present for anyone who wants restrictions
    back — set ``mode="guarded"`` to have dangerous operations ask first, or
    ``mode="readonly"`` to forbid mutation entirely, and populate
    ``protected_paths`` / ``dangerous_patterns`` to give those modes something
    to act on.  Both lists ship empty; they are opt-in, not opt-out.

    ``audit_log`` stays on by default.  A log is a record of what happened, not
    a restriction on what may happen — it never blocks an action.
    """

    # "open" = run anything (default); "guarded" = confirm dangerous ops; "readonly".
    mode: str = "open"
    allow_shell: bool = True
    allow_file_write: bool = True
    allow_process_control: bool = True
    allow_network: bool = True
    # What to do when something needs confirmation but no interactive callback
    # is wired up:
    #   "allow"     — proceed, but audit it (default; keeps headless work going)
    #   "deny"      — refuse
    unattended_policy: str = "allow"
    # Paths that "guarded"/"readonly" refuse to write or delete.  Empty by
    # default: no location is off limits.  Populate to opt in.
    protected_paths: list = field(default_factory=list)
    # Substrings that make "guarded" ask before running a command.  Empty by
    # default: no command needs confirmation.  Populate to opt in.
    dangerous_patterns: list = field(default_factory=list)
    command_timeout: float = 120.0
    audit_log: bool = True
    confirm_callback: Optional[str] = None   # reserved for UI integration


@dataclass
class VoiceConfig:
    """Hands-free behaviour.

    Every field here is read by :mod:`jarvis.voice`. The defaults must stay in
    step with the fallbacks in that module — a field missing from this dataclass
    is silently dropped from both the config file and the environment, because
    ``_apply_mapping`` and ``_env_overrides`` only set attributes that already
    exist.
    """

    # How an utterance is deemed to be addressed to JARVIS:
    #   wake       — only when it opens with a wake word (plus the follow-up window)
    #   continuous — open conversation until `continuous_timeout` of silence
    #   push       — records only between begin_utterance() and end_utterance()
    mode: str = "wake"

    wake_words: list = field(default_factory=lambda: ["jarvis", "hey jarvis"])
    require_wake_word: bool = True

    # Barge-in: stop speaking when the user starts talking.
    allow_interrupt: bool = True
    # How much louder than the measured ambient floor a sound must be to count
    # as an interruption. Too low and JARVIS interrupts itself on its own voice
    # coming back through the speakers.
    interrupt_margin: float = 2.5

    # After a reply, how long the wake word may be omitted.
    follow_up_seconds: float = 15.0
    # In continuous mode, silence this long re-arms the wake word.
    continuous_timeout: float = 120.0

    # Audio retained from *before* speech was detected, so the opening syllable
    # is not clipped — detection necessarily trips after the speaker has begun.
    preroll_seconds: float = 0.5
    # Bursts shorter than this are coughs, clicks and door slams, not speech.
    min_speech_seconds: float = 0.3

    # Spoken when addressed by name with no request. Empty means "Yes, <title>?"
    acknowledge: str = ""
    greeting: str = "Good day. All systems are online and at your disposal."


@dataclass
class WindowsConfig:
    """Windows desktop integration (:mod:`jarvis.win`).

    Ignored on other platforms. Global hotkeys may require an elevated process
    on Windows, and are not grabbable at all by an ordinary client under Wayland
    on Linux — :func:`jarvis.win.is_windows_integration_available` reports what
    actually works on the running machine.
    """

    tray_icon: bool = True
    hotkey_toggle: str = "ctrl+alt+j"
    hotkey_push_to_talk: str = "ctrl+alt+space"
    autostart: bool = False
    notifications: bool = True


@dataclass
class HardwareConfig:
    """Manual hardware overrides. Auto-detected by default (see jarvis.core.hardware and
    jarvis.llm.planner) -- every field here exists to be overridden, not to be filled in by a
    typical user. mode="manual" disables auto-detection entirely and trusts these values as-is,
    which matters on a machine where detection guesses wrong (a headless VM passing through a
    GPU that nvidia-smi cannot see, a container with a fake /proc/meminfo, etc.)."""

    mode: str = "auto"              # "auto" | "manual"
    accelerator: str = "auto"       # "auto" | "cuda" | "rocm" | "mps" | "tpu" | "cpu"
    vram_gb: float = 0.0            # 0 = auto-detect
    ram_gb: float = 0.0             # 0 = auto-detect
    gpu_count: int = 0              # 0 = auto-detect


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    windows: WindowsConfig = field(default_factory=WindowsConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    data_dir: str = ""
    log_level: str = "INFO"

    # -- derived paths ----------------------------------------------------- #
    def home(self) -> Path:
        return Path(self.data_dir) if self.data_dir else data_dir()

    def path(self, *parts: str) -> Path:
        p = self.home().joinpath(*parts)
        ensure_dir(p.parent)
        return p

    def db_file(self) -> Path:
        if self.memory.db_path:
            p = Path(self.memory.db_path).expanduser()
            ensure_dir(p.parent)
            return p
        return self.path("memory.db")

    def tools_dir(self) -> Path:
        return ensure_dir(self.home() / "tools")

    def voices_dir(self) -> Path:
        return ensure_dir(self.home() / "voices")

    def models_dir(self) -> Path:
        if self.llm.layer_shards_dir:
            return ensure_dir(Path(self.llm.layer_shards_dir).expanduser())
        return ensure_dir(self.home() / "models")

    def logs_dir(self) -> Path:
        return ensure_dir(self.home() / "logs")

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self) -> dict:
        return _to_dict(self)

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (config_dir() / "config.json")
        ensure_dir(path.parent)
        payload = self.to_dict()
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
                path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
                return path
            except ImportError:
                path = path.with_suffix(".json")
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


# --------------------------------------------------------------------------- #
#  (de)serialisation helpers
# --------------------------------------------------------------------------- #
def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _coerce(value: Any, target_type: Any) -> Any:
    """Best-effort coercion of a scalar into the dataclass field's type."""
    origin = get_origin(target_type)
    if origin is not None:
        args = [a for a in get_args(target_type) if a is not type(None)]  # noqa: E721
        if origin is list:
            inner = args[0] if args else str
            return [_coerce(v, inner) for v in (value if isinstance(value, list) else [value])]
        # Optional[X] / Union[...] -> use the first non-None member.
        if args:
            if value is None:
                return None
            return _coerce(value, args[0])
        return value
    if target_type is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if target_type is int:
        return int(float(value))
    if target_type is float:
        return float(value)
    if target_type is str:
        return str(value)
    return value


def _apply_mapping(target: Any, data: dict) -> None:
    """Recursively overlay ``data`` onto a dataclass instance."""
    type_map = {f.name: f.type for f in fields(target)}
    for key, value in (data or {}).items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_mapping(current, value)
        else:
            ftype = type_map.get(key, str)
            # Dataclass field types can be strings under `from __future__ import
            # annotations`; fall back to the current value's type in that case.
            if isinstance(ftype, str):
                ftype = type(current) if current is not None else str
            try:
                setattr(target, key, _coerce(value, ftype))
            except (TypeError, ValueError):
                setattr(target, key, value)


def _load_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
            return yaml.safe_load(text) or {}
        except ImportError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse config file {path}: {exc}") from exc


# Standard Hugging Face token variables, in descending precedence.  These are
# the names every other tool in the ecosystem uses, so a machine that is already
# logged in should not have to learn a JARVIS-specific one.
HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def _apply_hf_token_env(cfg: Config, environ: dict) -> None:
    """Fill an empty ``llm.hf_token`` from the ecosystem-standard variables.

    Runs after the generic ``JARVIS_*`` pass, so an explicit
    ``JARVIS_LLM_HF_TOKEN`` (or a value from the config file) always wins.
    """
    llm = getattr(cfg, "llm", None)
    if llm is None or getattr(llm, "hf_token", "").strip():
        return
    for name in HF_TOKEN_ENV_VARS:
        value = (environ.get(name) or "").strip()
        if value:
            llm.hf_token = value
            return


def _env_overrides(cfg: Config, environ: Optional[dict] = None) -> None:
    """Apply ``JARVIS_<SECTION>_<FIELD>`` environment variables.

    Example: ``JARVIS_LLM_MODEL=Qwen/Qwen3-8B`` or ``JARVIS_LOG_LEVEL=DEBUG``.

    Two non-``JARVIS_`` names are honoured as well: ``HF_TOKEN`` and
    ``HUGGING_FACE_HUB_TOKEN`` seed ``llm.hf_token`` when it is otherwise empty.
    """
    environ = environ if environ is not None else dict(os.environ)
    section_names = {f.name for f in fields(cfg) if is_dataclass(getattr(cfg, f.name))}
    for raw_key, raw_value in environ.items():
        if not raw_key.startswith("JARVIS_"):
            continue
        body = raw_key[len("JARVIS_"):].lower()
        if body in ("home", "config_dir"):        # handled by platform_utils
            continue
        matched = False
        for section in section_names:
            prefix = f"{section}_"
            if body.startswith(prefix):
                field_name = body[len(prefix):]
                sub = getattr(cfg, section)
                if hasattr(sub, field_name):
                    _apply_mapping(sub, {field_name: raw_value})
                    matched = True
                break
        if not matched and hasattr(cfg, body):
            _apply_mapping(cfg, {body: raw_value})

    _apply_hf_token_env(cfg, environ)


def default_config_paths() -> list:
    """Candidate config locations, in ascending priority."""
    paths = [
        config_dir() / "config.yaml",
        config_dir() / "config.yml",
        config_dir() / "config.json",
        Path.cwd() / "config.yaml",
        Path.cwd() / "config.yml",
        Path.cwd() / "config.json",
    ]
    env_path = os.environ.get("JARVIS_CONFIG")
    if env_path:
        paths.append(Path(env_path).expanduser())
    return paths


def load_config(
    path=None,
    *,
    environ: Optional[dict] = None,
    use_env: bool = True,
) -> Config:
    """Build a :class:`Config` from defaults + file + environment."""
    cfg = Config()

    candidates: list
    if path is not None:
        candidates = [Path(path).expanduser()]
        if not candidates[0].exists():
            raise FileNotFoundError(f"Config file not found: {candidates[0]}")
    else:
        candidates = [p for p in default_config_paths() if p.exists()]

    for candidate in candidates:
        _apply_mapping(cfg, _load_file(candidate))

    if use_env:
        _env_overrides(cfg, environ)
    return cfg


__all__ = [
    "Config", "LLMConfig", "STTConfig", "TTSConfig", "MemoryConfig",
    "AgentConfig", "SecurityConfig", "VoiceConfig", "WindowsConfig",
    "HardwareConfig",
    "load_config", "default_config_paths", "HF_TOKEN_ENV_VARS",
]
