# Hardware & compatibility

JARVIS is meant to run on whatever machine it lands on: a RAM-only laptop, an
NVIDIA gaming rig, an AMD ROCm box, an Apple Silicon Mac, or a Google Cloud TPU
VM. This document explains how it decides what to do on each, using the real
detector (`jarvis/core/hardware.py`), the real advisory planner
(`jarvis/llm/planner.py`), and the real sizing functions
(`jarvis/llm/models.py`) — every number and every line of example output below
was produced by actually calling that code, not hand-calculated.

Reference machine for everything *else* in this repo's docs (README, MODELS.md)
is the owner's target laptop: i5-10210U, 4c/8t, 32 GB RAM, CPU-only. This page
additionally covers the hardware that laptop is *not*.

---

## Contents

- [What "auto" means](#what-auto-means)
- [Compatibility matrix](#compatibility-matrix)
- [Worked example: 32 GB RAM and 8 GB VRAM](#worked-example-32-gb-ram-and-8-gb-vram)
- [The ROCm/CUDA device-string fact](#the-rocmcuda-device-string-fact)
- [Google TPU: detected, not accelerated](#google-tpu-detected-not-accelerated)
- [Overriding auto-detection](#overriding-auto-detection)
- [`jarvis hardware` example output](#jarvis-hardware-example-output)
- [See also](#see-also)

---

## What "auto" means

`cfg.hardware` is a new section of the one JARVIS config
(`jarvis.core.config.HardwareConfig` — there is still only `config.yaml`, not a
second file):

```yaml
hardware:
  mode: auto            # auto | manual
  accelerator: auto     # auto | cuda | rocm | mps | tpu | cpu
  vram_gb: 0.0           # 0 = auto-detect (GPU/accelerator memory, in GB)
  ram_gb: 0.0            # 0 = auto-detect (system RAM, in GB)
  gpu_count: 0           # 0 = auto-detect (number of GPUs/accelerators)
```

Every field is also settable by environment variable, using the same generic
`JARVIS_<SECTION>_<FIELD>` mechanism every other config section uses — nothing
hardware-specific was added for this:

```bash
JARVIS_HARDWARE_MODE=manual
JARVIS_HARDWARE_ACCELERATOR=cuda
JARVIS_HARDWARE_VRAM_GB=8
JARVIS_HARDWARE_RAM_GB=32
JARVIS_HARDWARE_GPU_COUNT=1
```

**`mode: auto`** (the default) probes the real machine via
`jarvis.core.hardware.detect()` and lets `jarvis.llm.planner.plan()` turn that
into a recommendation. Every field above is `"auto"`/`0` out of the box, which
means nothing here overrides detection until you change it.

**`mode: manual`** trusts the fields you set instead of probing. In `manual`
mode, `plan()` still calls `detect()` to fill in *only* whichever fields you
left at their auto-detect sentinel — set `accelerator` and leave `vram_gb: 0`,
and the VRAM figure still gets detected for you. In `auto` mode the relationship
inverts: a manual field is used only as a *fallback hint* for whatever
detection itself could not determine (e.g. `nvidia-smi` absent and no
`torch` installed to fall back to) — it never overrides something detection
did successfully determine.

**This is an advisory layer, not a runtime switch.** `jarvis.llm.create_llm()`
— the function that actually starts a backend — is completely unchanged by any
of this: it still auto-selects from `AUTO_PROBE_ORDER` exactly as before this
feature existed. `plan()` is consulted by `jarvis hardware` and by the
installer to *report* a sensible starting point; it does not write to
`llm.model` or `llm.backend` for you. Concretely: the shipped default remains
`llm.model: Qwen/Qwen3.8-27B` regardless of what `jarvis hardware` recommends
for your machine — if the plan's recommendation looks better for your box, copy
its `model (interactive)` / `model (background)` value into `llm.model` /
`llm.ollama_model` yourself. (`jarvis.llm.models.recommend()`, which the planner
calls, also does not know that `qwen3.8-27b` is the configured default — it
ranks purely by size and purpose fit, which is why a 32 GB CPU-only plan
recommends the MoE `qwen3-30b-a3b` over the dense `qwen3.8-27b` for the
background slot: 30.5B total parameters beats 27B on the ranking, MoE-ness
aside. See [Worked example](#worked-example-32-gb-ram-and-8-gb-vram) and
[docs/MODELS.md §2](MODELS.md#2-dense-vs-mixture-of-experts) for why that
trade-off is usually still the right call on CPU.)

---

## Compatibility matrix

Every row below is real output from `jarvis.llm.planner.plan()` called against
a synthetic `jarvis.core.hardware.HardwareProfile` built for that exact RAM/VRAM
combination — real function, constructed input, not hand-calculated numbers.
(A genuinely detected profile from this project's actual dev box — CPU-only —
appears separately in [`jarvis hardware` example output](#jarvis-hardware-example-output).)
`q4` quantisation throughout, matching the planner's default.

| Scenario | Accelerator | Backend | Device | Model (interactive) | Model (background) | Caveat |
|---|---|---|---|---|---|---|
| CPU-only, 8 GB RAM | none | ollama | `cpu` | `qwen3:4b-instruct-2507-q4_K_M` | `qwen3:4b-instruct-2507-q4_K_M` | Small budget: interactive and background collapse to the same model. |
| CPU-only, 16 GB RAM | none | ollama | `cpu` | `qwen3:4b-instruct-2507-q4_K_M` | `qwen3:14b` | 14B background is "batch job", not conversation — see [MODELS.md](MODELS.md#2-dense-vs-mixture-of-experts). |
| CPU-only, 32 GB RAM (owner's laptop) | none | ollama | `cpu` | `qwen3:30b-a3b-instruct-2507-q4_K_M` | `qwen3:30b-a3b-instruct-2507-q4_K_M` | The MoE model's small active-parameter count makes it fast enough for both slots on CPU. |
| CPU-only, 64 GB+ RAM | none | ollama | `cpu` | `qwen3:30b-a3b-instruct-2507-q4_K_M` | `qwen3:32b` | Dense 32B affordable as the background model once RAM stops being the constraint; still ~1 tok/s on CPU. |
| NVIDIA, 4 GB VRAM (+16 GB RAM) | cuda | ollama | `cuda` | `qwen3:4b-instruct-2507-q4_K_M` | `qwen3:14b` | 4B (2.5 GB q4) fits with room to spare; background sizing is RAM-based, not VRAM-based (see below). |
| NVIDIA, 8 GB VRAM + 32 GB RAM (owner's example) | cuda | ollama | `cuda` | `qwen3:8b` | `qwen3:30b-a3b-instruct-2507-q4_K_M` | Interactive model fits entirely in VRAM; background MoE does not — partial GPU offload. Full detail [below](#worked-example-32-gb-ram-and-8-gb-vram). |
| NVIDIA, 12 GB VRAM (+32 GB RAM) | cuda | ollama | `cuda` | `qwen3:14b` | `qwen3:30b-a3b-instruct-2507-q4_K_M` | 14B (8.88 GB weights + 2.18 GB KV = 11.06 GB) just fits 12 GB. |
| NVIDIA, 24 GB+ VRAM (+64 GB RAM) | cuda | ollama | `cuda` | `qwen3:30b-a3b-instruct-2507-q4_K_M` | `qwen3:32b` | 30B-A3B (18.79 GB weights+KV) fits whole; plenty of headroom for real context. |
| AMD GPU on Linux (ROCm, VRAM known) | rocm | ollama | **`cuda`** | `qwen3:30b-a3b-instruct-2507-q4_K_M` | `qwen3:30b-a3b-instruct-2507-q4_K_M` | Device string is `cuda`, not `rocm` — see [the device-string fact](#the-rocmcuda-device-string-fact). Prefer Ollama's own ROCm build. |
| AMD GPU on Windows (ROCm presence only, VRAM unknown) | rocm | ollama | **`cuda`** | `qwen3:30b-a3b-instruct-2507-q4_K_M` | `qwen3:30b-a3b-instruct-2507-q4_K_M` | **Weaker support.** ROCm on Windows for consumer cards is patchy to absent; VRAM could not be read, so sizing falls back to system RAM only. |
| Apple Silicon (M-series, unified memory) | mps | ollama | `mps` | `qwen3:14b` | `qwen3:14b` | No separate VRAM pool — sized against system RAM, not a VRAM budget. |
| Google TPU | tpu | ollama | **`cpu`** | `qwen3:30b-a3b-instruct-2507-q4_K_M` | `qwen3:30b-a3b-instruct-2507-q4_K_M` | **Detected, not accelerated.** Falls back to CPU; see [below](#google-tpu-detected-not-accelerated). |

Notes that apply to the whole table:

- **`backend` is always `ollama`.** It is the one JARVIS-recommended backend
  that runs the same GGUF weights on CPU, NVIDIA CUDA, AMD's own ROCm build,
  and Apple Metal — the planner does not need a different recommendation per
  accelerator. It is advice, not a constraint: `jarvis.llm.create_llm()` still
  auto-selects from whatever is actually installed, independent of this.
- **Multiple GPUs**: the planner sizes against a *single* device's VRAM only.
  `HardwareProfile.vram_gb` for NVIDIA/AMD is the *sum* across all detected
  cards (so `jarvis hardware` on a 2×24 GB box reports "48 GB VRAM"), but
  `plan()` adds an explicit note when `gpu_count > 1` that multi-GPU
  tensor/pipeline parallelism is not modelled — treat the VRAM-fit columns
  above as true only for the single biggest card you actually have.
- **Background-model sizing never uses the VRAM figure** — only the
  interactive pick does (`recommend(..., vram_gb=...)`). The background model
  is deliberately "slower is fine", and Ollama's own partial GPU offload of
  MoE expert layers makes a background model that doesn't fit whole in VRAM a
  reasonable outcome rather than a failure — see the worked example.

---

## Worked example: 32 GB RAM and 8 GB VRAM

This is the exact scenario the owner asked about. Real numbers from
`jarvis.llm.models.estimate_footprint(spec, "q4", vram_gb=8.0)`:

| Model | Weights (q4) | KV cache @ 8k | Weights+KV | Fits in 8 GB VRAM? | Headroom |
|---|---|---|---|---|---|
| `qwen3-8b` | 4.92 GB | 1.21 GB | 6.13 GB | **Yes** | +1.87 GB |
| `qwen3-14b` | 8.88 GB | 2.18 GB | 11.06 GB | No | −3.06 GB |
| `qwen3-30b-a3b` | 18.30 GB | 0.49 GB | 18.79 GB | No | −10.79 GB |

- **What fits entirely in VRAM:** `Qwen3-8B` at Q4 — 4.92 GB of weights plus a
  modest 1.21 GB KV cache at the planner's 8k default context, with 1.87 GB of
  headroom left over. This is why `jarvis.llm.planner.plan()` picks
  `qwen3:8b` as `model_interactive` for this profile: it is the largest
  catalogue model whose weights *and* KV cache both fit whole on the card
  (`jarvis.llm.models.recommend(32.0, True, "chat", vram_gb=8.0)` returns
  `Qwen/Qwen3-8B`), so every token runs on the GPU with nothing offloaded.
  `Qwen3-14B` is 3.06 GB too large once its KV cache is included — it would
  need a reduced context window or partial CPU offload to run at all on this
  card.
- **What needs partial offload:** the background pick,
  `qwen3:30b-a3b-instruct-2507-q4_K_M`, does not fit — 18.79 GB of weights+KV
  against 8 GB of VRAM. This is expected and not a failure: `recommend()`
  deliberately does not pass `vram_gb` for the background slot (see
  `jarvis/llm/planner.py:_plan_impl`), because the background pick is sized
  against the 32 GB of *system* RAM instead (the same
  `recommend(32.0, has_gpu=True, "quality")` call CPU-only boxes get). Ollama
  offloads what it can of the MoE's expert layers to the 8 GB card and keeps
  the rest resident in the 32 GB of system RAM — a working, if not fully
  GPU-resident, way to run the larger model, and precisely the outcome the
  32 GB of system RAM is there for.
- **Why not recommend a smaller background model that fits VRAM too?** Because
  the background slot exists for subagents and batch work where "slower is
  fine" — see `docs/MODELS.md` §2 for the throughput argument. Trading
  capability for VRAM-residency only matters for the *interactive* slot, which
  is why only that one passes `vram_gb` into `recommend()`.

```pycon
>>> from jarvis.llm import models
>>> models.estimate_footprint(models.resolve("qwen3-8b"), "q4", vram_gb=8.0)
{'model': 'Qwen/Qwen3-8B', ..., 'weights_gb': 4.92, 'kv_cache_gb': 1.21,
 'ram_gb': 7.22, 'fits_vram': True, 'vram_headroom_gb': 1.87}
>>> models.estimate_footprint(models.resolve("qwen3-30b-a3b"), "q4", vram_gb=8.0)
{'model': 'Qwen/Qwen3-30B-A3B-Instruct-2507', ..., 'weights_gb': 18.3,
 'kv_cache_gb': 0.49, 'ram_gb': 20.95, 'fits_vram': False, 'vram_headroom_gb': -10.79}
>>> models.recommend(32.0, True, "chat", vram_gb=8.0).id
'Qwen/Qwen3-8B'
```

Real output, on this (CPU-only) dev box, forcing that exact profile with
`hardware.mode: manual`:

```bash
JARVIS_HARDWARE_MODE=manual JARVIS_HARDWARE_ACCELERATOR=cuda \
JARVIS_HARDWARE_VRAM_GB=8 JARVIS_HARDWARE_RAM_GB=32 jarvis hardware
```

```
Detected hardware
  CPU only, 8.43 GB RAM, 4 cores
    cpu_cores    4
    cpu_threads  8
    ram_gb       8.43
    accelerator  none
    gpu_vendor
    gpu_name
    gpu_count    0
    vram_gb      0.0
  Detection notes
    - No GPU/accelerator detected; running on CPU only.

Plan
  backend              ollama
  device               cuda
  accelerator          cuda
  model (interactive)  qwen3:8b
  model (background)   qwen3:30b-a3b-instruct-2507-q4_K_M
  quantisation         q4
  fits VRAM            yes

Notes
  - hardware.mode=manual: accelerator overridden to 'cuda'.
  - hardware.mode=manual: vram_gb overridden to 8.
  - hardware.mode=manual: ram_gb overridden to 32.
  - No GPU/accelerator detected; running on CPU only.

Manual overrides (hardware.mode = manual)
  accelerator  manual override -> cuda
  vram_gb      manual override -> 8.0
  ram_gb       manual override -> 32.0
  gpu_count    auto (not overridden)
```

Two things worth noticing here, because they are easy to misread:

- **"Detected hardware" always shows the real machine, never the override.**
  This dev box has no GPU, so that section reports `CPU only` regardless of
  `hardware.mode`. Only the **Plan** section (and the **Manual overrides**
  section at the bottom, present whenever `hardware.mode: manual`) reflects
  the 8 GB VRAM / 32 GB RAM being simulated.
- **The stray fourth note.** `gpu_count` was left at its `0` (auto) sentinel,
  so `jarvis.llm.planner.plan()` still runs real detection to fill it in
  (`accelerator`/`vram_gb`/`ram_gb` were already fully determined by the
  manual override and are not re-detected) — and that real detection's own
  note (`"No GPU/accelerator detected..."`) gets appended alongside the three
  manual-override notes, even though it describes the real machine, not the
  simulated one. Set `hardware.gpu_count: 1` explicitly (or accept it as
  harmless noise) if this reads confusingly on your box.

(`fits VRAM: yes` describes the *interactive* pick, `qwen3:8b` — the field the
CLI prints is `HardwarePlan.fits_vram`, which is only ever computed against
`model_interactive`.)

---

## The ROCm/CUDA device-string fact

Worth stating plainly, because getting it backwards silently breaks AMD GPU
usage: **a ROCm build of PyTorch places tensors with `torch.device("cuda")` —
the exact same API NVIDIA uses. There is no `torch.device("rocm")`.**

That means "AMD" is a *reporting and package-selection* distinction only, never
a device-string one:

- `HardwareProfile.accelerator` and `HardwarePlan.accelerator` report `"rocm"`
  for an AMD GPU — this is what `jarvis hardware` and this document's matrix
  show, and it is what should drive *which package* you install (Ollama's own
  ROCm build, not the CUDA one).
- `HardwarePlan.device` — the literal string a backend would hand to
  `torch.device(...)` — is **`"cuda"`** for a ROCm accelerator, never
  `"rocm"`. `jarvis/llm/planner.py:_device_and_notes()` is the one place this
  is enforced in code; `jarvis/core/hardware.py:detect_amd()`'s docstring
  states the same fact for the detection side.

Writing `torch.device(profile.accelerator)` naively when `accelerator ==
"rocm"` is a bug — that string is not a real torch device type and raises.
Every place in this codebase that turns a profile into a device already gets
this right; if you add a new one, use `.device`, never `.accelerator`, for the
literal runtime call.

**AMD support is real but weaker than NVIDIA's, and weaker still on Windows.**
ROCm PyTorch support is Linux-primary; consumer-card support on Windows is
patchy to absent as of this writing. `jarvis.core.hardware.detect_amd()`
recommends, and this document repeats: **prefer Ollama's own ROCm build over
this project's PyTorch-based `transformers`/`airllm` backends on AMD** — Ollama
ships ROCm support directly rather than depending on a separately installed,
version-matched ROCm + PyTorch stack, which is the thing that is fragile even
on Linux and often simply unavailable on Windows.

---

## Google TPU: detected, not accelerated

`jarvis.core.hardware.detect_tpu()` genuinely detects a TPU environment — the
GCP/Colab environment variables (`TPU_NAME`, `COLAB_TPU_ADDR`,
`TPU_WORKER_ID`), and a lazy import of `torch_xla` or a TPU-backed JAX. That
detection is honest and real: `jarvis hardware` will correctly say "a TPU was
detected" on a Cloud TPU VM or a Colab/Kaggle TPU runtime.

**But detecting a TPU is not the same as using one.** No backend in
`jarvis/llm/*_backend.py` places a single tensor on an XLA device today —
`OllamaBackend`, `TransformersBackend`, `AirLLMBackend`, `VLLMBackend` and
`OpenAICompatBackend` all either shell out to a server or call
`torch.cuda`/`torch.backends.mps`/plain CPU. None of them import `torch_xla`
or JAX. Accelerating on a TPU would need a new backend built on PyTorch-XLA or
JAX — a different code path this project does not have.

So the planner tells the truth about the gap instead of promising acceleration
it cannot deliver: `HardwarePlan.device` is always `"cpu"` for a TPU profile,
with a note attached (`jarvis/llm/planner.py:_device_and_notes()`) —

> A TPU was detected, but no backend in this codebase places tensors on an XLA
> device (jarvis/llm/\*\_backend.py has no PyTorch-XLA or JAX code path) —
> falling back to CPU rather than pretending acceleration will happen.

If you genuinely want TPU acceleration, the concrete work is: add
`jarvis/llm/xla_backend.py` (or `jax_backend.py`) implementing the same
`LLMBackend` contract as the existing backends, using PyTorch-XLA's
`xm.xla_device()` (or JAX's `pmap`/`jit` over `jax.devices("tpu")`) to actually
place weights and run generation on the TPU cores, then add it to
`jarvis.llm.BACKENDS` and `AUTO_PROBE_ORDER`. Nothing here does that yet.

A TPU environment is also, practically, almost always a **cloud** environment
— a GCP TPU VM or a Colab/Kaggle TPU runtime — essentially never a personal
machine, which is a second reason this gap has stayed low priority for a
project whose primary target is a laptop.

---

## Overriding auto-detection

Auto-detection is right most of the time, but three situations genuinely need
`hardware.mode: manual`:

1. **A VM with GPU passthrough `nvidia-smi` cannot see.** Some
   virtualisation/passthrough setups expose the device to `torch.cuda` (or
   nothing at all, if the guest driver stack is incomplete) without
   `nvidia-smi` itself being installed or working inside the guest. If
   `jarvis hardware` reports `none` on a box you know has a GPU, set
   `hardware.accelerator: cuda` (and `vram_gb` if you know it) rather than
   fighting the guest's driver installation.
2. **A container reporting the host's full RAM instead of its cgroup limit.**
   `jarvis.core.hardware.system_ram_gb()` reads `/proc/meminfo`
   (or `psutil`, which also reads host-wide figures) — inside a container with
   a memory limit lower than the host's total RAM, this over-reports what is
   actually available, and `recommend()` can pick a model that gets OOM-killed.
   Set `hardware.ram_gb` to the container's real cgroup limit.
3. **Testing a specific configuration deliberately** — CI, a support
   reproduction, or simply checking "what would JARVIS recommend on an 8 GB
   VRAM card" without owning one. `hardware.mode: manual` plus the four fields
   makes that reproducible without touching the actual machine.

In all three cases, only set the fields detection is getting wrong — anything
left at its `"auto"`/`0` sentinel in `manual` mode is still filled in from a
real `detect()` call (see [What "auto" means](#what-auto-means)).

---

## `jarvis hardware` example output

Real output, on this development machine (CPU-only, 4 cores / 8 threads,
~8.4 GB RAM as reported to this VM/container — not the owner's 32 GB target
laptop):

```
$ jarvis hardware

Detected hardware
  CPU only, 8.43 GB RAM, 4 cores
    cpu_cores    4
    cpu_threads  8
    ram_gb       8.43
    accelerator  none
    gpu_vendor
    gpu_name
    gpu_count    0
    vram_gb      0.0
  Detection notes
    - No GPU/accelerator detected; running on CPU only.

Plan
  backend              ollama
  device               cpu
  accelerator          none
  model (interactive)  qwen3:4b-instruct-2507-q4_K_M
  model (background)   qwen3:8b
  quantisation         q4
  fits VRAM            (unknown)

Notes
  - No GPU/accelerator detected; running on CPU only.
```

`jarvis hardware` itself always detects the real machine — there is no flag to
feed it a synthetic profile. The rest of this document's AMD/Apple/TPU rows
are instead produced by calling `jarvis.core.hardware.summary(profile)` and
`jarvis.llm.planner.plan(cfg, profile=profile)` **directly in Python** against
a hand-built `HardwareProfile`, formatted exactly as `cmd_hardware()` in
`jarvis/cli.py` formats its own output — real function output, a synthetic
input, clearly labelled as such. For example, with
`HardwareProfile(accelerator="rocm", gpu_vendor="amd", gpu_name="AMD Radeon RX
7900 XTX", vram_gb=20.0, ram_gb=32.0, gpu_count=1)`:

```
Detected hardware
  AMD Radeon RX 7900 XTX (20 GB VRAM), 32 GB RAM, 4 cores
    cpu_cores    4
    cpu_threads  8
    ram_gb       32.0
    accelerator  rocm
    gpu_vendor   amd
    gpu_name     AMD Radeon RX 7900 XTX
    gpu_count    1
    vram_gb      20.0
  Detection notes
    - AMD GPU detected via ROCm. A ROCm build of PyTorch places tensors with
      torch.device('cuda') -- the same API as NVIDIA -- so there is no
      torch.device('rocm'); any backend must map this to 'cuda'. Ollama's own
      ROCm build is the more reliable route than a hand-matched PyTorch/ROCm
      wheel, especially on Windows, where ROCm support for consumer cards is
      patchy to absent as of this writing.

Plan
  backend              ollama
  device               cuda
  accelerator          rocm
  model (interactive)  qwen3:30b-a3b-instruct-2507-q4_K_M
  model (background)   qwen3:30b-a3b-instruct-2507-q4_K_M
  quantisation         q4
  fits VRAM            yes

Notes
  - AMD GPU detected via ROCm. A ROCm build of PyTorch places tensors with
    torch.device('cuda') -- the same API as NVIDIA -- so there is no
    torch.device('rocm'); any backend must map this to 'cuda'. Ollama's own
    ROCm build is the more reliable route than a hand-matched PyTorch/ROCm
    wheel, especially on Windows, where ROCm support for consumer cards is
    patchy to absent as of this writing.
  - AMD ROCm builds of PyTorch place tensors with torch.device('cuda') --
    there is no torch.device('rocm'). 'rocm' is a reporting/package-selection
    label only; the runtime device string is 'cuda', same as NVIDIA.
  - ROCm PyTorch support is Linux-primary and patchy to absent for consumer
    cards on Windows. Ollama's own ROCm build is the more reliable path here:
    it ships ROCm support directly and does not depend on a working
    system-wide ROCm + PyTorch install.
```

(The `Detection notes` line under "Detected hardware" and the `Notes` list
under "Plan" overlap in content but come from two different places —
`HardwareProfile.notes` vs `HardwarePlan.notes` — which is why the ROCm
device-string caveat appears twice, worded slightly differently each time.
That duplication is real behaviour of the current code, not a copy-paste
error in this document.)

Note `accelerator: rocm` (the reporting label) alongside `device: cuda` (the
literal runtime string) in the same output — see
[the device-string fact](#the-rocmcuda-device-string-fact) for why that is
correct, not a typo.

`jarvis hardware` accepts the same global `-c/--config` flag as every other
subcommand, and reads `cfg.hardware` from whatever config file is in effect —
exactly like `jarvis config` does. (`--model`/`--backend` are also global
flags, but they override `cfg.llm`, which `jarvis.llm.planner.plan()` never
reads; they have no effect on this command's output.)

---

## See also

- [docs/MODELS.md](MODELS.md) — the model catalogue, dense vs MoE, quantisation
  cost, and `estimate_footprint`/`recommend` in full detail
- [docs/INSTALL.md](INSTALL.md) — the installer, including the hardware-detection
  report step it prints before choosing what to download
- `jarvis/core/hardware.py` — the detector itself, with the ROCm/TPU honesty
  rules spelled out in its module docstring
- `jarvis/llm/planner.py` — the advisory planner (`HardwarePlan`/`plan()`)
  consulted by `jarvis hardware` and the installer
- `jarvis/core/config.py` — `HardwareConfig`, the `hardware:` section itself
