# JARVIS configuration.
#
# Copy to `config.yaml` (next to this file, or in your user config directory)
# and edit. Every value here is the default, so delete anything you do not wish
# to change. JSON works too if PyYAML is not installed.
#
# Any setting can also be set by environment variable:
#   JARVIS_LLM_BACKEND=ollama
#   JARVIS_VOICE_MODE=continuous
#   JARVIS_SECURITY_MODE=guarded

# ---------------------------------------------------------------------------
# Language model
# ---------------------------------------------------------------------------
llm:
  # auto | ollama | transformers | airllm | stub
  #   "auto" probes ollama -> transformers -> airllm and takes the first that
  #   works. AirLLM is probed LAST on purpose: its availability check is only an
  #   import test, and it is ~100x slower than Ollama for a model that fits RAM.
  backend: auto

  # Any Hugging Face repo id. `jarvis model list` shows the known aliases.
  #   Qwen/Qwen3.8-27B                  — DEFAULT. Dense 27B, vision+text, 262K
  #                                       context, ~18 GB at Q4. The strongest
  #                                       model that fits 32 GB — but dense, so
  #                                       ~0.5-1 tok/s on CPU, and it thinks
  #                                       before answering. Needs
  #                                       transformers >= 4.57.
  #   Qwen/Qwen3.6-27B                  — the previous default; same shape.
  #   Qwen/Qwen3-30B-A3B-Instruct-2507  — MoE, ~3B active/token, ~4-8 tok/s on CPU.
  #   Qwen/Qwen3-4B-Instruct-2507       — the one that feels like conversation.
  #   Qwen/Qwen3-32B                    — dense, smarter, considerably slower.
  #
  # NOTE: on a CPU-only box you do NOT have to choose between "capable" and
  # "responsive" — see `voice_model` below, which puts a small fast model in
  # front of this one so replies begin speaking immediately.
  model: Qwen/Qwen3.8-27B

  compression: ""            # "" | 4bit | 8bit  (needs bitsandbytes + CUDA)
  layer_shards_dir: ""       # defaults to <data dir>/models
  device: auto               # auto | cpu | cuda | mps

  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  # Context window. Two costs, and the second is the one that bites:
  #   RAM  -- KV cache. Cheap on Qwen3.8 (only 16 of 64 layers are full
  #           attention): 32k costs ~2.1 GB.
  #   TIME -- PREFILL, which is compute-bound, not bandwidth-bound. On 4 cores
  #           a 27B ingests ~8-15 tok/s, so a COLD 32k prompt is 35-70 minutes.
  #
  # Prefix caching is what rescues this: llama.cpp reuses the KV of an
  # unchanged prefix, so a long-lived session pays prefill ONCE and later
  # turns ingest only what changed. Keep the session alive rather than
  # restarting it between questions.
  #
  #    8192 -- voice conversation
  #   32768 -- engineering work (the default): a large source file + history
  #  131072 -- whole-repo or datasheet work; 2-5 hours cold, use a subagent
  #  262144 -- the model's capability, not a runtime setting here: ~16.8 GB of
  #            KV on top of 18 GB of weights does not fit in 32 GB.
  context_tokens: 32768      # also sent to Ollama as num_ctx

  # Hard cap on what ONE tool result may contribute. Reading a 40,000-line
  # Verilog file is ordinary work; without this it evicts the system prompt
  # and the question along with it.
  max_tool_result_tokens: 8000

  # Install from https://ollama.com  (winget install Ollama.Ollama)
  ollama_host: http://127.0.0.1:11434
  #   qwen3.8:27b                         ~18 GB,  ~0.5-1 tok/s  (most capable)
  #   qwen3:30b-a3b-instruct-2507-q4_K_M  ~19 GB,  ~4-8 tok/s    (background)
  #   qwen3:4b-instruct-2507-q4_K_M       ~2.5 GB, ~15-25 tok/s  (interactive)
  ollama_model: qwen3.8:27b

  # Chain-of-thought. Qwen3.x reasons by DEFAULT and will happily spend the
  # entire max_new_tokens budget inside a <think> block, returning nothing at
  # all — the single most common cause of "JARVIS never answers".
  #   auto — off for families known to reason by default (recommended)
  #   on   — force it on; expect long silences on CPU
  #   off  — always off
  thinking: auto

  # -- The voice model ------------------------------------------------------
  # The trick that makes a slow local brain feel instant. The big model above
  # does the thinking and the tool work; this small one turns the result into
  # the sentence that is actually spoken. A 1.7B model runs at 15-30 tok/s on a
  # CPU, so speech begins in well under a second while the big model is still
  # working.
  #
  # It never decides anything: it is given the finished answer and asked only
  # to phrase it. If it returns something implausibly longer than what it was
  # given, the original wording is spoken instead.
  #
  # Set voice_model_enabled: false (or voice_model: "") to have the main model
  # speak for itself.
  #   qwen3:1.7b   ~1.1 GB, very fast, quite sufficient for phrasing
  #   qwen3:0.6b   ~0.5 GB, faster still, occasionally clumsy
  #   qwen3:4b-instruct-2507-q4_K_M  ~2.5 GB, best phrasing, still quick
  voice_model: qwen3:1.7b
  voice_model_enabled: true
  voice_max_new_tokens: 160
  voice_temperature: 0.5
  # Speak a short holding line ("One moment, Sir") the instant a request is
  # understood, so the wait reads as deliberation rather than a crash.
  voice_ack_enabled: true

  # -- CPU tuning -----------------------------------------------------------
  # Dense inference on a CPU is memory-bandwidth bound. See docs/PERFORMANCE.md
  # for the arithmetic; the short version is that these are worth 20-40%, not
  # a multiple.
  #
  # 0 = physical cores. Do NOT set this to your logical-processor count: two
  # hyperthreads on one core share a memory port and contend for the exact
  # resource that is already the bottleneck.
  num_threads: 0
  use_mmap: true             # map weights from the page cache; leave on
  use_mlock: false           # pin in RAM; only with headroom, else it swaps

  # -- Speculative decoding (llama.cpp only) ---------------------------------
  # The single largest speedup available on a bandwidth-bound CPU, and it
  # costs NOTHING in quality: a small draft model proposes N tokens, the big
  # model verifies all N in ONE batched pass, and rejected drafts are thrown
  # away. Output is identical to running the big model alone.
  #
  # ~1.5 tok/s becomes ~3.3-4.8 on an i5-10210U depending on how often the
  # draft agrees. Requires llama.cpp -- Ollama does not expose --model-draft.
  # Run `jarvis serve-plan` for the exact command. See docs/PERFORMANCE.md.
  draft_model: ""            # path to a small GGUF (e.g. qwen3-0.6b-q4_k_m)
  draft_tokens: 4            # proposals per round; beyond ~6 rarely pays
  llamacpp_host: http://127.0.0.1:8080/v1

  # -- Routing ---------------------------------------------------------------
  # The small model triages every turn: greetings, status questions and
  # pause/resume are answered without waking the big model at all. Anything it
  # cannot classify confidently escalates -- slow beats confidently wrong.
  routing_enabled: true

  allow_fallback: true
  request_timeout: 600.0

# ---------------------------------------------------------------------------
# Speech to text
# ---------------------------------------------------------------------------
stt:
  # auto | faster-whisper | whisper | vosk | windows | stub
  #   "windows" uses the speech recogniser built into Windows — no download at
  #   all, but noticeably less accurate than whisper. It exists so voice works
  #   on a machine with nothing installed.
  engine: auto
  # tiny.en | base.en | small.en | medium.en
  # small.en is the accuracy sweet spot: ~3x the compute of base.en but far
  # fewer word errors, and still comfortably real-time at int8 on four cores.
  # Drop to base.en if the machine is busy.
  model: small.en
  device: auto
  compute_type: int8         # int8 is roughly twice as fast on CPU
  language: en
  sample_rate: 16000
  vad_filter: true

  # -- Decoding quality -----------------------------------------------------
  # Greedy decoding (beam_size 1) is what makes small Whisper models sound
  # useless: they commit to a bad first token and never recover.
  beam_size: 5
  # Whisper conditions on its own previous output by default, so one mistake
  # propagates through the whole session. This is the classic cause of
  # runaway repeated text.
  condition_on_previous_text: false
  # Biases the decoder towards words it will actually hear. Proper nouns are
  # what a small model gets wrong, and if "Jarvis" is misheard the wake word
  # never fires at all.
  initial_prompt: "Jarvis, the British AI assistant."
  cpu_threads: 0             # 0 = use every core
  no_speech_threshold: 0.6
  log_prob_threshold: -1.0
  # Whisper reliably invents "Thank you." / "Thanks for watching!" over silence
  # and background noise. Discard those when they are the whole transcript.
  filter_hallucinations: true

  # Voice activity detection — how the end of your sentence is detected.
  # MEASURED ON THIS MACHINE: the room's noise floor is about 0.007, so 0.021
  # (three times the floor) is a safer threshold than the 0.015 default.
  # Run `jarvis calibrate` to measure your own room.
  silence_threshold: 0.015
  # 0.9s cuts people off mid-thought: anyone who pauses to think gets their
  # sentence truncated and only half of it transcribed.
  silence_duration: 1.2      # seconds of quiet that end an utterance
  max_utterance_seconds: 30.0
  # Audio kept from *before* speech was detected, so the first syllable of the
  # wake word survives instead of arriving as "-arvis".
  preroll_seconds: 0.5
  min_speech_seconds: 0.3    # shorter bursts are coughs and clicks
  input_device: null         # null = system default; `jarvis doctor` lists them

# ---------------------------------------------------------------------------
# The voice
# ---------------------------------------------------------------------------
tts:
  # auto | piper | edge | sapi | pyttsx3 | espeak | null
  #   auto order: piper -> edge -> sapi -> pyttsx3 -> espeak -> null
  #   "sapi" is the Windows built-in voice: no download, but this machine only
  #   has en-US voices installed (David, Zira), so it will not sound British.
  #   Add a British voice: Settings > Time & language > Speech > Add voices.
  engine: auto

  # Piper: offline, neural, CPU-friendly. The default British voice.
  #   en_GB-alan-medium     — male RP, measured. Closest to the films.
  #   en_GB-northern_english_male-medium
  #   en_GB-jenny_dioco-medium — female RP
  piper_voice: en_GB-alan-medium
  piper_model_path: ""       # defaults to <data dir>/voices/<voice>.onnx

  # edge-tts: needs internet, and is the most convincing British voice.
  #   en-GB-RyanNeural | en-GB-ThomasNeural | en-GB-SoniaNeural
  # Returns MP3; playback and transcription both handle that fine.
  edge_voice: en-GB-RyanNeural
  edge_rate: "+0%"
  edge_pitch: "+0Hz"

  sapi_voice_hint: United Kingdom
  speed: 1.0
  volume: 1.0
  output_device: null
  enabled: true

# ---------------------------------------------------------------------------
# Memory — the "forgets nothing" store
# ---------------------------------------------------------------------------
memory:
  db_path: ""                # defaults to <data dir>/memory.db (always absolute)
  embedder: auto             # auto | sentence-transformers | hash | none
  embed_model: sentence-transformers/all-MiniLM-L6-v2
  embed_dim: 384

  recall_k: 8                # recollections injected per turn
  recall_min_score: 0.15

  summarize_after_turns: 20
  keep_recent_turns: 8
  prune: false               # never delete anything

# ---------------------------------------------------------------------------
# Agent behaviour
# ---------------------------------------------------------------------------
agent:
  name: JARVIS
  user_title: Sir
  max_tool_iterations: 8
  max_concurrent_tasks: 4
  subagent_timeout: 900.0
  announce_updates: true
  allow_tool_creation: true

# ---------------------------------------------------------------------------
# Access policy — OFF BY DEFAULT
# ---------------------------------------------------------------------------
# JARVIS ships unrestricted: it will run any command, write any file, and delete
# anything you ask it to, without prompting. That is the intended behaviour for
# a personal assistant on your own machine.
#
# The machinery below exists for anyone who wants restrictions. It does nothing
# until you configure it.
security:
  # open     — run anything, no confirmation (THE DEFAULT)
  # guarded  — confirm anything matching dangerous_patterns; refuse writes under
  #            protected_paths
  # readonly — observation only
  mode: open

  allow_shell: true
  allow_file_write: true
  allow_process_control: true
  allow_network: true

  # Only consulted in "guarded" mode when nothing can be asked (a headless run).
  #   allow — proceed   |   deny — refuse
  unattended_policy: allow

  # Both lists are EMPTY by default: nothing is protected, nothing is flagged.
  # Populate them only if you switch to guarded/readonly mode. For example:
  # protected_paths:
  #   - C:\Windows
  #   - C:\Program Files
  # dangerous_patterns:
  #   - format
  #   - diskpart
  protected_paths: []
  dangerous_patterns: []

  command_timeout: 120.0
  audit_log: true            # a record, not a restriction: <data>/logs/audit.jsonl

# ---------------------------------------------------------------------------
# Hands-free behaviour
# ---------------------------------------------------------------------------
voice:
  # wake       — respond only when addressed by name (the default)
  # continuous — open conversation, no wake word, until continuous_timeout of
  #              silence re-arms it. Best when you are alone at the desk.
  # push       — push-to-talk: records only while the hotkey is held
  mode: wake

  wake_words:
    - jarvis
    - hey jarvis
  require_wake_word: true

  # Barge-in: start talking and JARVIS stops speaking mid-sentence.
  allow_interrupt: true
  # How much louder than the room a sound must be to count as you interrupting.
  # Too low and JARVIS interrupts itself by hearing its own voice through the
  # speakers; raise it if that happens, lower it if barge-in feels unresponsive.
  interrupt_margin: 2.5

  # After a reply, this many seconds in which you need not repeat the wake word.
  follow_up_seconds: 15.0
  # In continuous mode, silence this long re-arms the wake word.
  continuous_timeout: 120.0

  # Audio kept from *before* speech was detected, so the first syllable of
  # "Jarvis" is not clipped. Lower it only if latency matters more than accuracy.
  preroll_seconds: 0.5
  # Bursts shorter than this are coughs, clicks and door slams — not speech.
  min_speech_seconds: 0.3

  # Spoken when addressed by name with no request. Empty = "Yes, <user_title>?"
  acknowledge: ""
  greeting: Good day. All systems are online and at your disposal.

# ---------------------------------------------------------------------------
# Windows desktop integration
# ---------------------------------------------------------------------------
windows:
  tray_icon: true                  # system-tray presence with a state indicator
  hotkey_toggle: ctrl+alt+j        # start/stop listening
  hotkey_push_to_talk: ctrl+alt+space
  autostart: false                 # start JARVIS when you log in
  notifications: true              # toast when a background task finishes

# ---------------------------------------------------------------------------
# Hardware — manual overrides for auto-detection
# ---------------------------------------------------------------------------
# JARVIS probes the machine it is on (see `jarvis hardware`) and picks a backend
# and model accordingly: RAM-only, NVIDIA, AMD/ROCm, Apple Silicon (MPS), or
# Google TPU (detected but not yet usable by any backend here — falls back to
# CPU). Everything below is OFF by default (mode: auto); set mode: manual only
# when detection gets this machine wrong (a VM with passed-through GPU that
# nvidia-smi cannot see, a container with a fake /proc/meminfo, etc.) and you
# want to state the truth yourself instead.
hardware:
  mode: auto            # auto | manual
  accelerator: auto     # auto | cuda | rocm | mps | tpu | cpu
  vram_gb: 0.0           # 0 = auto-detect (GPU/accelerator memory, in GB)
  ram_gb: 0.0            # 0 = auto-detect (system RAM, in GB)
  gpu_count: 0           # 0 = auto-detect (number of GPUs/accelerators)

# ---------------------------------------------------------------------------
data_dir: ""                 # "" = platform default; see `jarvis config`
log_level: INFO              # DEBUG | INFO | WARNING | ERROR
