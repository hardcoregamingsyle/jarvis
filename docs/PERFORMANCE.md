# Speed on a CPU: getting a dense 27B to 3-4 tok/s

**You can have the 27B at 3-4 tok/s, at full Q4 quality.** Not by tuning
threads — by changing how tokens are generated. This page shows the arithmetic.

> **Correction.** An earlier version of this document said 4 tok/s was
> unreachable on this hardware. That was wrong: it modelled speculative
> decoding with a bad formula and dismissed it. The corrected numbers are
> below, and they hit the target.

---

## Two different resources

The most common confusion, and worth separating before anything else:

| | | your machine |
|---|---|---|
| **Capacity** | Can the weights *fit*? | 32.6 GB total, 18 GB model — **yes, comfortably** |
| **Bandwidth** | How fast can they be *read*? | ~28 GB/s achieved — **this sets tok/s** |

Having 12 GB spare is genuinely useful: it holds the KV cache, the draft
model, and the page cache. But **free RAM does not add bandwidth.** Whether
you fill 20 GB or 30 GB of a dual-channel DDR4-2666 bus, you still read at
~28 GB/s.

---

## The baseline, and why it is slow

A **dense** model reads *every* weight per token. Qwen3.8-27B at Q4 is 18 GB:

```
28 GB/s ÷ 18 GB = 1.56 tok/s
```

The CPU is essentially idle throughout — it is waiting on memory. That idle
compute is what the next section spends.

*(A mixture-of-experts model like `qwen3:30b-a3b` activates only ~3.3B
parameters per token, so it reads ~2.2 GB and reaches ~12 tok/s. Still the
fastest option — but no longer the only one that clears 3 tok/s.)*

---

## Speculative decoding: 2-3x, no quality cost

A small **draft** model proposes `k` tokens cheaply. The large model then
verifies **all k in one batched forward pass**.

That is the whole trick: **verification costs one read of the 18 GB
regardless of `k`.** The expensive resource is touched once per *round*
instead of once per *token*, and the idle CPU does the extra arithmetic for
free.

Expected accepted tokens per round, with acceptance rate `a` and depth `k`:

```
E = (1 - a^(k+1)) / (1 - a)
```

With a 0.6B draft (0.4 GB), `k=4`:

| Acceptance | Tokens/round | GB/round | **tok/s** | Speedup |
|---|---|---|---|---|
| 60% | 2.31 | 19.6 | **3.3** | 2.1× |
| 70% | 2.77 | 19.6 | **4.0** | 2.5× |
| 80% | 3.36 | 19.6 | **4.8** | 3.1× |

Same-family drafts agree often — Qwen3 0.6B drafting for Qwen3.8-27B typically
lands at 70-80% on ordinary prose. **That is your 3-4 tok/s.**

**The output is bit-identical to running the 27B alone.** Rejected drafts are
discarded; accepted ones are exactly what the big model would have produced.
This is pure latency, unlike Q2 quantisation which buys similar speed by
making the model measurably worse.

### The catch

Acceptance is workload-dependent. Predictable prose accepts well; dense code,
unusual identifiers and long numbers accept poorly. Below ~50% the draft
overhead outweighs the batching win — `estimate_speedup()` computes the
crossover, and `jarvis selftest` measures the rate you actually get.

### Setting it up

Ollama does not expose `--model-draft`, so this needs llama.cpp:

```bash
# 1. Build llama.cpp (or grab a release binary)
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
cmake -B build -DGGML_NATIVE=ON && cmake --build build -j4

# 2. Fetch both GGUFs
huggingface-cli download Qwen/Qwen3.8-27B-GGUF   qwen3.8-27b-q4_k_m.gguf --local-dir ~/models
huggingface-cli download Qwen/Qwen3-0.6B-GGUF    qwen3-0.6b-q4_k_m.gguf  --local-dir ~/models

# 3. Serve, with the draft attached
llama-server \
  --model       ~/models/qwen3.8-27b-q4_k_m.gguf \
  --model-draft ~/models/qwen3-0.6b-q4_k_m.gguf \
  --draft-max 4 --draft-min 1 \
  --ctx-size 8192 --threads 4 --threads-batch 4 \
  --host 127.0.0.1 --port 8080
```

`jarvis serve-plan` prints this command filled in for your machine.

Then point JARVIS at it:

```yaml
llm:
  backend: openai-compat
  vllm_host: http://127.0.0.1:8080/v1
  draft_model: ~/models/qwen3-0.6b-q4_k_m.gguf
  draft_tokens: 4
```

---

## Context: the second cost nobody mentions

`context_tokens` now defaults to **32768**, not 8192. But a big window has two
costs and only one of them is RAM.

### Cost 1 — KV cache (RAM). Cheap here.

Qwen3.8's hybrid attention means only **16 of its 64 layers** are full
attention; the other 48 are Gated DeltaNet, whose state does not grow with
context. So ~64 KB/token, not the ~256 KB a uniform model would need:

| Context | KV cache | Total with 18 GB weights |
|---|---|---|
| 8,192 | 0.52 GB | 18.5 GB |
| **32,768** | **2.1 GB** | **20.1 GB** ← default |
| 131,072 | 8.4 GB | 26.4 GB |
| 262,144 | 16.8 GB | 34.8 GB — **does not fit** |

### Cost 2 — prefill (TIME). This is the one that bites.

Generation is bandwidth-bound. **Prefill is compute-bound** — it is
matrix-matrix, not matrix-vector — and your 4 cores have roughly 100-130
GFLOP/s of usable AVX2 throughput against 54 GFLOPs per token.

Real llama.cpp `pp` figures for a ~30B Q4 on 4-core mobile land at **8-15
tok/s**. So ingesting a **cold** prompt costs:

| Context | at 8 tok/s | at 15 tok/s |
|---|---|---|
| 8,192 | 17 min | 9 min |
| 32,768 | 68 min | 36 min |
| 131,072 | 4.6 hours | 2.4 hours |
| 262,144 | 9 hours | 5 hours |

That is why 262k is not a runtime setting on this machine. It is the model's
*capability*, and it assumes a GPU.

### What makes a large window usable anyway: prefix caching

llama.cpp reuses the KV of an unchanged prefix. So a long-lived session pays
prefill **once**, and each later turn only ingests what changed — a few
hundred tokens, seconds not hours.

The practical rule:

> **Keep the session alive.** A 32k context is 35-70 minutes cold and
> near-instant warm. Restarting JARVIS between questions pays that cost every
> single time; leaving it running pays it once a day.

This is also why background subagents matter for the work you describe. A
subagent doing a four-hour chip audit is not something you sit in front of;
it reports when done.

### Choosing a window

| Work | Setting | First-turn cost |
|---|---|---|
| Voice conversation | 8192 | ~10 min, then instant |
| **Engineering (default)** | **32768** | **~35-70 min cold, instant warm** |
| Whole-repo / datasheet | 131072 | 2-5 hours cold. Use a subagent |
| 262144 | — | Does not fit in 32 GB alongside the weights |

Set it per-session rather than globally if your work varies:

```bash
JARVIS_LLM_CONTEXT_TOKENS=131072 jarvis chat
```

### One tool result cannot eat the window

`max_tool_result_tokens` (default 8000) caps what a single tool contributes.
Reading a 40,000-line Verilog file is ordinary; without a cap it would evict
the system prompt and the question along with it.

If the transcript still exceeds the window, JARVIS trims it **before** sending
— oldest turns first, system prompt always kept, and an oversized single
message elided from the *middle* so its head and tail both survive. The
alternative is the server truncating from the front, silently, taking the
system prompt with it.

---

## Memory budget

Qwen3.8-27B's hybrid attention keeps long context cheap: only **16 of its 64
layers** are full attention (the other 48 are Gated DeltaNet, whose state does
not grow with context). KV cache is ~64 KB/token, not the ~256 KB a uniform
64-layer model would need:

| Context | KV cache | Total with weights |
|---|---|---|
| 4,096 | 0.27 GB | 18.3 GB |
| 8,192 | 0.54 GB | **18.5 GB** ← recommended |
| 32,768 | 2.15 GB | 20.1 GB |
| 131,072 | 8.59 GB | 26.6 GB |

Plus 0.4 GB for the draft and ~1.1 GB for the voice model: **~20 GB in use,
~12 GB free.** Comfortable. Even 32k context fits.

Do not chase the full 262,144-token window — that is the model's *capability*,
not a sensible runtime setting. It would need ~27 GB and leave nothing for the
page cache the weights are mapped from, and swapping costs an order of
magnitude, not a percentage.

---

## Ordinary tuning: 20-40%

```yaml
llm:
  num_threads: 0      # 0 = physical cores (4 on an i5-10210U)
  use_mmap: true
  use_mlock: false    # only with headroom; otherwise it swaps
  context_tokens: 8192
```

**Use 4 threads, not 8.** Two hyperthreads on one physical core share a single
memory port, so they contend for the resource that is already saturated.
Ollama defaults to all logical processors; `num_threads: 0` now detects
physical cores.

---

## Perceived speed: already handled

Independently of raw throughput, JARVIS hides the latency:

- **Routing** — greetings, status questions and control verbs are answered by
  the 1.7B model or read from the task tree. The 27B never wakes for most turns.
- **The voice model** — when it does wake, a small model speaks its answer at
  15-30 tok/s.
- **Sentence streaming** — audio starts on the first full stop.
- **Background subagents** — long work runs detached.

So the 27B at 4 tok/s with speculation, fronted by a 1.7B voice, feels
immediate. Which is what you described wanting.

---

## Summary

| Want | Do | Result |
|---|---|---|
| **27B at 3-4 tok/s, full quality** | llama.cpp + 0.6B draft model | **3.3-4.8 tok/s** |
| **Fastest overall** | `qwen3:30b-a3b` (MoE) | ~12 tok/s |
| **Simplest** | Ollama, no draft | ~1.5 tok/s |
| **Feels instant regardless** | Already on by default | <1s to first word |

Every figure here is arithmetic from published bandwidth specs.
**`jarvis selftest` measures your real numbers — trust those over mine.**
