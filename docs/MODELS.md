# Models

Everything JARVIS knows about weights lives in `jarvis/llm/models.py`. This
document explains what is in there, why the defaults are what they are, and how
to change them.

The short version is in the README's
[Choosing and changing the model](../README.md#choosing-and-changing-the-model)
section. This is the long version: the family, the arithmetic, and how to add
something that is not on the list.

Reference machine throughout: **i5-10210U class, 4 cores / 8 threads, 32 GB RAM,
no usable CUDA, Linux.** Numbers on a GPU box look entirely different.

> **Units.** `GB` here means 10⁹ bytes — what Hugging Face and Ollama report —
> not GiB. A "32 GB" laptop has about 34.4 GB by this measure; the estimates
> below deliberately budget against the smaller, honest number.

---

## 1. The Qwen3 family

JARVIS defaults to Qwen3 because it tool-calls natively in the
`<tool_call>{...}</tool_call>` form `jarvis/agent/protocol.py` parses, it has
strong open weights at every size from 0.6B to 235B, and the MoE variants are
unusually well suited to a CPU-only machine. Nothing depends on it — see
[§6](#6-running-something-that-is-not-qwen).

### What `-2507` means

Qwen shipped a refreshed instruct checkpoint in **July 2025**, tagged `-2507`.
It matters for two reasons:

1. **It replaces hybrid thinking with a dedicated instruct model.** The original
   Qwen3 release used one checkpoint that switched between a "thinking" mode and
   a direct mode. The `-2507` line splits those apart: `-Instruct-2507` answers
   directly, and a separate `-Thinking-2507` exists for chain-of-thought work.
   For an assistant that must respond in speech, the instruct variant is the
   right one — no reasoning preamble to strip, lower latency, fewer tokens.
2. **The context window jumps from 32k to 262k.** Compare
   `Qwen/Qwen3-8B` (32,768) with `Qwen/Qwen3-4B-Instruct-2507` (262,144).

JARVIS still strips any `<think>` block it sees (`llm/base.py:strip_thinking`),
so a thinking checkpoint works — it is just slower and more verbose.

**Only some sizes got a `-2507` refresh: 4B, 30B-A3B and 235B-A22B.** The dense
8B, 14B and 32B aliases therefore still point at the original Qwen3 releases,
which remain the current weights for those sizes. If a `-2507` dense checkpoint
appears, changing the `id` in `KNOWN_MODELS` is the entire migration.

### The catalogue

`KNOWN_MODELS` maps a short alias to a `ModelSpec`. It is a **convenience, never
a whitelist** — `resolve()` will happily synthesise a spec for a repo it has
never heard of (see [§5](#5-adding-a-model-to-the-catalogue)).

| alias | repo | params | active | ctx | ~q4 size | notes |
|---|---|---|---|---|---|---|
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | 0.6B | dense | 32k | 0.5 GB | Smoke-test size |
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B` | 1.7B | dense | 32k | 1.1 GB | Weak hardware |
| `qwen3-4b` | `Qwen/Qwen3-4B-Instruct-2507` | 4B | dense | 256k | 2.5 GB | Safe default on a small or busy machine |
| `qwen3-8b` | `Qwen/Qwen3-8B` | 8.2B | dense | 32k | 4.9 GB | Borderline for CPU voice latency |
| `qwen3-14b` | `Qwen/Qwen3-14B` | 14.8B | dense | 32k | 8.9 GB | Background work, not conversation |
| `qwen3-32b` | `Qwen/Qwen3-32B` | 32.8B | dense | 32k | 19.7 GB | Smartest dense Qwen3; wants a GPU |
| `qwen3-30b-a3b` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 30.5B | **3.3B** | 256k | 18.3 GB | Best speed/quality trade-off on 32 GB CPU |
| `qwen3-coder-30b-a3b` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | 30.5B | 3.3B | 256k | 18.3 GB | Same shape, tuned for code and tool calls |
| `qwen3-235b-a22b` | `Qwen/Qwen3-235B-A22B-Instruct-2507` | 235B | 22B | 256k | 141 GB | Far past one laptop |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | 8B | dense | 128k | 4.9 GB | **Gated** — accept Meta's licence first |
| **`qwen3.6-27b`** | `Qwen/Qwen3.6-27B` | 27B | dense | **256k** | 16.1 GB | **The default.** Strongest that fits 32 GB; multimodal; dense, so ~1 tok/s on CPU |
| `qwen3.8-27b` | `Qwen/Qwen3.8-27B` | 27B | — | 256k | 16.2 GB | **NOT RELEASED.** Placeholder figures |

### `Qwen3.8-27B` is not a real model

It is in the catalogue with `exists=False` on purpose. No such repository exists
on Hugging Face; the parameter count, context and size above are placeholders.
It is listed so that `jarvis` can say *"that model is not released yet"* instead
of pretending the name was a typo. Selecting it raises `UnreleasedModelError`
with the full explanation:

```
Qwen3.8 27B (unreleased) (Qwen/Qwen3.8-27B) is not released yet: no such model
exists, so it cannot be selected. Not released yet — no such repository exists on
Hugging Face. Figures are placeholders. Use qwen3-30b-a3b until it ships. When it
is published, set exists=True for the 'qwen3.8-27b' entry in jarvis/llm/models.py
and it becomes selectable immediately.
```

When it ships, the migration is: set `exists=True`, correct the numbers, done.

---

## 2. Dense vs mixture-of-experts

This is the single most important idea for CPU-only inference.

A **dense** model multiplies every input by every weight. A **mixture-of-experts**
model splits its feed-forward layers into many experts and a router picks a
handful per token. `Qwen3-30B-A3B` holds 30.5B parameters and activates about
**3.3B** of them per token — the `A3B` in the name.

Two different numbers therefore matter:

| number | governs |
|---|---|
| **Total** parameters | how much **memory** you need — all experts must be resident |
| **Active** parameters | how much **arithmetic** per token, and therefore how fast it feels |

On a GPU with plenty of VRAM, memory bandwidth dominates and the gap narrows. On
a CPU with 4 cores, arithmetic dominates completely, and 3.3B of arithmetic
against 32.8B is roughly a **tenfold** difference in tokens per second for models
of almost identical size on disk.

### Why 30B-A3B beats 32B dense on this hardware

| | `Qwen3-30B-A3B-Instruct-2507` | `Qwen3-32B` |
|---|---|---|
| Total parameters | 30.5B | 32.8B |
| Active per token | **3.3B** | 32.8B |
| q4 weights | 18.3 GB | 19.7 GB |
| KV cache @ 8k ctx | **0.49 GB** | 4.84 GB |
| Estimated RAM @ 8k | **20.95 GB** | 26.79 GB |
| Native context | 256k | 32k |
| Feels like | a conversation | a batch job |

Nearly the same download, nearly the same knowledge, ten times the arithmetic
per token. The KV cache difference is the same effect from the other end: it
scales with *active* parameters, so the MoE model's cache is an order of
magnitude smaller and its 262k context window is actually usable.

`models.recommend(ram_gb=32, has_gpu=False, purpose="chat")` encodes exactly this
and returns `Qwen/Qwen3-30B-A3B-Instruct-2507`. It refuses to recommend anything
whose *active* parameter count exceeds 6B for interactive use on CPU
(`_INTERACTIVE_CPU_ACTIVE_B`), which is why the dense 32B is never offered for
conversation even though it would technically almost fit.

```python
>>> from jarvis.llm import models
>>> models.recommend(32.0, False, "chat").id
'Qwen/Qwen3-30B-A3B-Instruct-2507'
>>> models.recommend(8.0, False, "chat").id
'Qwen/Qwen3-4B-Instruct-2507'
>>> models.recommend(32.0, False, "code").id
'Qwen/Qwen3-Coder-30B-A3B-Instruct'
```

---

## 3. Quantisation and what it really costs

Quantisation stores each weight in fewer bits. The advertised bit-width is not
the whole story: every k-quant keeps fp16 scale factors alongside the packed
values, so "4-bit" is really about **4.8 bits per weight** in practice. JARVIS
uses the honest numbers (`models._BITS`):

| name | effective bits/weight | typical use |
|---|---|---|
| `fp16` / `bf16` | 16.0 | reference quality; what the repo ships |
| `q8` / `q8_0` | 8.5 | indistinguishable from fp16 in practice |
| `q6_k` | 6.6 | very safe |
| `q5_k_m` | 5.5 | safe |
| **`q4_k_m`** | **4.8** | **the standard choice** — small, measurable quality loss |
| `awq` | 4.3 | GPU-oriented 4-bit |
| `nf4` / `int4` | 4.5 | bitsandbytes 4-bit |
| `q3_k_m` | 3.4 | noticeably degraded |
| `q2_k` | 2.6 | usually not worth running |

Real cost for `Qwen3-30B-A3B` (30.5B parameters, 8k context), from
`models.estimate_footprint`:

| quantisation | weights | estimated RAM | fits in 32 GB? |
|---|---|---|---|
| `fp16` | 61.0 GB | 67.1 GB | no |
| `q8` | 32.4 GB | 36.2 GB | no |
| `q6` | 25.2 GB | 28.4 GB | tight |
| `q5` | 21.0 GB | 23.8 GB | yes |
| **`q4`** | **18.3 GB** | **20.9 GB** | **yes, comfortably** |
| `q3` | 13.0 GB | 15.2 GB | yes |
| `q2` | 9.9 GB | 11.9 GB | yes, but why |

`q4_K_M` is the default for a reason: it is the last step down that costs almost
nothing in quality and the first that comfortably fits a 30B model in 32 GB
alongside a desktop session.

`RAM` above is `weights × 1.08 + KV cache + 0.7 GB`, covering allocator
overhead, activations and the framework itself. It is an estimate whose job is to
say *"this will not fit"* before a 30 GB download says it for you — not to
predict the last megabyte.

Ready-made quantisations:

```bash
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M    # Ollama does this for you
```

For `transformers`, set `llm.compression` to `4bit` or `8bit` (needs
`bitsandbytes`, and in practice a CUDA GPU). For vLLM, point `llm.model` at a
pre-quantised AWQ or GPTQ repo.

### The whole catalogue at q4, 8k context

| model | weights | KV cache | est. RAM |
|---|---|---|---|
| Qwen3-1.7B | 1.02 GB | 0.25 GB | 2.05 GB |
| Qwen3-4B-Instruct-2507 | 2.40 GB | 0.59 GB | 3.88 GB |
| Qwen3-8B | 4.92 GB | 1.21 GB | 7.22 GB |
| Qwen3-14B | 8.88 GB | 2.18 GB | 12.47 GB |
| **Qwen3-30B-A3B-Instruct-2507** | 18.30 GB | 0.49 GB | **20.95 GB** |
| Qwen3-32B | 19.68 GB | 4.84 GB | 26.79 GB |
| Qwen3-235B-A22B-Instruct-2507 | 141.0 GB | 3.24 GB | 156.22 GB |

---

## 4. Context length vs RAM

Context is not free. The KV cache grows **linearly with context** and
**proportionally to active parameters**, and it is separate from the weights.

JARVIS budgets roughly 0.018 GB per 1,000 tokens per billion active parameters
(fp16 cache, calibrated against Qwen3-8B's 36 layers / 8 KV heads / 128 head
dim). Accurate to about a factor of two across GQA models, which is all a budget
check needs.

`Qwen3-30B-A3B` (3.3B active), q4 weights = 18.3 GB:

| context | KV cache | est. total RAM |
|---|---|---|
| 4,096 | 0.24 GB | 20.71 GB |
| **8,192** (default) | **0.49 GB** | **20.95 GB** |
| 32,768 | 1.95 GB | 22.41 GB |
| 131,072 | 7.79 GB | 28.25 GB |
| 262,144 (native max) | 15.57 GB | 36.04 GB |

`Qwen3-32B` (32.8B active), q4 weights = 19.7 GB:

| context | KV cache | est. total RAM |
|---|---|---|
| 4,096 | 2.42 GB | 24.37 GB |
| 8,192 | 4.84 GB | 26.79 GB |
| 32,768 (native max) | 19.35 GB | 41.30 GB |

Two conclusions:

- **The MoE model can actually use its long context.** Even at 131k it fits in
  32 GB. The dense 32B blows past 32 GB at its own native maximum.
- **Do not set `context_tokens` to the model's maximum by reflex.** JARVIS
  defaults to `8192` and manages the window itself: `memory/context.py` keeps a
  rolling summary plus the recent turns, so a bigger raw window mostly buys
  memory pressure. Raise it when you actually feed it long documents.

```yaml
llm:
  context_tokens: 8192      # also sent to Ollama as num_ctx and to vLLM as --max-model-len
```

`llm/base.py:trim_to_context` enforces the limit before generation, dropping the
oldest non-system messages first, so exceeding it degrades rather than erroring.

---

## 5. Adding a model to the catalogue

**You usually do not need to.** `resolve()` synthesises a spec for any
well-formed repo id it does not recognise, inferring the parameter counts from
the name:

```python
>>> from jarvis.llm import models
>>> s = models.resolve("mistralai/Mistral-7B-Instruct-v0.3")
>>> s.params, s.family, s.quantised_size_gb
(7.0, 'mistral', 4.2)

>>> s = models.resolve("Qwen/Qwen3-Next-80B-A3B-Instruct")   # never heard of it
>>> s.params, s.active_params, s.quantised_size_gb
(80.0, 3.0, 48.0)
```

So `llm.model: mistralai/Mistral-7B-Instruct-v0.3` just works. Add a catalogue
entry when you want a short alias, accurate figures instead of inferred ones, or
a `gated` flag.

### The entry

Add a key to `KNOWN_MODELS` in `jarvis/llm/models.py`:

```python
    "mistral-7b": ModelSpec(
        id="mistralai/Mistral-7B-Instruct-v0.3",
        label="Mistral 7B Instruct v0.3",
        params=7.2,
        family="mistral",
        context=32768,
        quantised_size_gb=4.4,
        notes="Solid general model; weaker at tool calls than Qwen3.",
        backends=("ollama", "transformers", "vllm", "airllm"),
        ollama_tag="mistral:7b-instruct",
    ),
```

| field | required | meaning |
|---|---|---|
| `id` | yes | HF repo id `org/name`, or an Ollama tag |
| `label` | yes | human-readable name for menus and logs |
| `params` | yes | **total** parameters in billions |
| `family` | yes | architecture family, e.g. `"qwen3"`, `"llama3"` |
| `context` | yes | native context window in tokens |
| `quantised_size_gb` | yes | on-disk size at the usual 4-bit quantisation |
| `notes` | no | one line: when to pick this |
| `backends` | no | JARVIS backend names that can run it |
| `gated` | no | `True` if the repo needs a licence click-through |
| `exists` | no | `False` for announced-but-unpublished models |
| `ollama_tag` | no | equivalent Ollama tag, if published |
| `active_params` | no | **MoE only** — parameters per token |
| `alias` | no | filled in automatically from the dict key |

Rules that are easy to get wrong:

- **`active_params` is what makes it an MoE.** Leave it `None` for a dense
  model. Setting it equal to `params` does nothing useful; `is_moe` requires it
  to be strictly smaller.
- **`params` is the total, not the active count.** `Qwen3-30B-A3B` is
  `params=30.5, active_params=3.3`.
- **Sizes are 10⁹ bytes.** Read them off the model page's file listing.
- **The alias key is normalised**: lowercase, whitespace and underscores folded
  to dashes. `normalise_alias("Mistral_7B")` → `mistral-7b`.
- **Set `gated=True` for anything behind a licence.** It changes the message
  users get from `check_access()` and keeps `recommend()` from suggesting a
  model they cannot download.

Nothing else in the codebase needs touching. `_ID_INDEX` and `_TAG_INDEX` are
built at import time, and the `alias` field is back-filled from the dict key at
the bottom of the module.

### Verify it

```python
>>> from jarvis.llm import models
>>> models.describe(models.resolve("mistral-7b"))
>>> models.estimate_footprint("mistral-7b", "q4")
>>> models.check_access(models.resolve("mistral-7b"), models.hf_token())
```

`check_access` makes one cheap API call before you commit to the large one. It
returns `status` of exactly `ok`, `gated`, `needs_token`, `not_found` or
`offline`, and it never logs or returns your token — everything it echoes goes
through `models.redact()` first.

---

## 6. Running something that is not Qwen

Nothing in JARVIS is Qwen-specific. `llm.model` is the only place a model name
appears.

```yaml
llm:
  model: mistralai/Mistral-7B-Instruct-v0.3
```

```bash
JARVIS_LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct jarvis chat
jarvis --model microsoft/Phi-4 chat
```

For Ollama, set the tag rather than the repo:

```yaml
llm:
  backend: ollama
  ollama_model: mistral:7b-instruct
```

### What a non-Qwen model needs to work well

| requirement | why | if it is missing |
|---|---|---|
| A **chat template** in the tokeniser | `format_chat` uses `tokenizer.apply_chat_template` when present | falls back to a generic `role: content` rendering — usable, slightly worse |
| **Instruction tuning** | JARVIS sends a long system prompt with a tool catalogue | a base/completion model will ramble instead of answering |
| Willingness to **emit `<tool_call>` JSON** | that is the protocol `agent/protocol.py` parses | the model chats but never uses a tool |

The third is the one that bites. Qwen3 emits that form natively. Llama 3.1 and
Mistral will do it if instructed but are less reliable, and a model with no
tool-calling training will simply describe what it would do.

`agent/protocol.py` is deliberately forgiving — it accepts code fences around the
JSON, single quotes, a missing closing tag, and several calls in one block — so a
model that is merely *sloppy* about the format still works. A model that ignores
the format entirely does not.

### Architectures that need `trust_remote_code`

Some repos ship their own modelling code (a brand-new architecture that predates
its `transformers` release, or a custom attention implementation). Those need:

```yaml
llm:
  trust_remote_code: true
```

It is **off by default**, and that is a correctness default, not a rail: it
executes Python from the repo at load time, so leaving it off means a fresh
`transformers` fails loudly on an unsupported architecture rather than silently
running someone else's code. Every model in `KNOWN_MODELS` loads without it.

Turn it on when the load error says `trust_remote_code=True`, and only for repos
you would run a script from.

### Pinning a revision

`llm.model_revision` pins a commit SHA, branch or tag. Empty means "whatever
`main` is today", which is convenient until an upstream re-upload changes the
weights beneath a working deployment.

```yaml
llm:
  model: Qwen/Qwen3.6-27B
  model_revision: "a1b2c3d4e5f6..."     # a commit SHA from the repo's Files page
```

Pin it on the production machine. Leave it empty while experimenting.

---

## 7. Backends, briefly

Which backend to run a model on is a separate question from which model to run.
The comparison table with real measured throughput lives in the README under
[Serving and concurrency](../README.md#serving-and-concurrency). In one line
each:

| backend | run it when |
|---|---|
| `ollama` | default on a CPU laptop; easiest quantised weights |
| `vllm` | Linux, and you are running a tree of concurrent agents |
| `openai-compat` | llama.cpp, LM Studio, TGI, or a remote box |
| `transformers` | small model, in-process, or you have a real GPU |
| `airllm` | the model genuinely does not fit in RAM — and nothing else |
| `stub` | tests |

`backend: auto` probes `vllm → ollama → openai-compat → transformers → airllm`
and takes the first that answers. AirLLM is last on purpose: its availability
check is only an import test, so probing it earlier would select the disk-paged
backend — tens of seconds per token — while a perfectly good server was running.

---

## See also

- [README: Choosing and changing the model](../README.md#choosing-and-changing-the-model)
- [README: Hugging Face tokens and gated models](../README.md#hugging-face-tokens-and-gated-models)
- [OPERATIONS.md](OPERATIONS.md) — running and monitoring the assistant
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — when a model will not load
- `jarvis/llm/models.py` — the catalogue itself, with the reasoning in docstrings
