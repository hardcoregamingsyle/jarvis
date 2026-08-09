"""The advisory layer that answers "what should this machine run".

This module never selects a runtime backend by itself — :func:`jarvis.llm.create_llm`
still does that, unchanged, via ``AUTO_PROBE_ORDER``. :func:`plan` is consulted
by the CLI (``jarvis hardware``) and the installer to *report* a sensible
starting point, given whatever hardware is present or configured.

Two inputs, composed:

* ``cfg.hardware`` (:class:`jarvis.core.config.HardwareConfig`) — the user's
  manual overrides, empty/`"auto"` by default.
* :func:`jarvis.core.hardware.detect` — a live probe of the machine, lazily
  imported so this module carries no hard dependency on it and still degrades
  to a CPU-only plan if that probe is missing, broken, or simply wrong (a
  headless VM whose passed-through GPU ``nvidia-smi`` cannot see, for
  instance — exactly the case ``HardwareConfig`` exists to override).

One fact worth stating up front because getting it wrong would silently break
AMD GPU usage: **a ROCm build of PyTorch places tensors with
``torch.device("cuda")`` — the same API NVIDIA uses.** There is no
``torch.device("rocm")``. So "accelerator" (``cuda`` / ``rocm`` / ``mps`` /
``tpu`` / ``none``) is a *reporting* distinction — what shows up in
``jarvis hardware`` and in :attr:`HardwarePlan.accelerator` — while
:attr:`HardwarePlan.device` is the literal string a backend hands to
``torch.device(...)`` or an equivalent runtime call, and for ROCm that string
is always ``"cuda"``, never ``"rocm"``.

Google TPU gets the same honesty treatment: detection can genuinely notice a
TPU is present (``torch_xla``, JAX with a TPU backend, or the GCP metadata
environment), but none of ``jarvis/llm/*_backend.py`` places a tensor on an
XLA device today — that needs PyTorch-XLA or JAX, a code path this project
does not have. A TPU plan therefore always reports ``device="cpu"`` with a
note explaining why, rather than promising acceleration nothing here delivers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from ..core.platform_utils import IS_WINDOWS
from .models import ModelSpec, estimate_footprint, recommend

logger = logging.getLogger(__name__)

# Internal accelerator vocabulary, shared with HardwareConfig.accelerator and
# (by contract) jarvis.core.hardware.HardwareProfile.accelerator. "cpu" means
# "no accelerator" here; HardwarePlan.accelerator below renders that as the
# more human-facing "none" instead.
_ACCELERATOR_VALUES = ("cuda", "rocm", "mps", "tpu", "cpu")

# Sensible defaults when nothing at all could be determined -- this is what
# lets `plan()` return a usable answer on a machine it knows nothing about.
_DEFAULT_RAM_GB = 32.0
_QUANTISATION = "q4"

# Ollama is the recommendation across every accelerator this module knows
# about: it runs the same GGUF weights on CPU, NVIDIA CUDA, AMD ROCm (its own
# build, not the community PyTorch/ROCm stack) and Apple Metal, which makes it
# the one backend whose recommendation does not need to change per platform.
# It is advice, not a constraint -- `jarvis.llm.create_llm` still auto-selects
# from whatever is actually installed.
_RECOMMENDED_BACKEND = "ollama"


@dataclass(frozen=True)
class HardwarePlan:
    """A recommended way to run JARVIS on this machine, advisory only.

    Attributes:
        backend: Recommended JARVIS backend name (``"ollama"``, ``"transformers"``,
            ``"airllm"`` or ``"vllm"``). Advisory: :func:`jarvis.llm.create_llm`
            still auto-selects from whatever is actually installed.
        device: The literal runtime device string a backend would use --
            ``"cuda"``, ``"mps"`` or ``"cpu"``. Never ``"rocm"``: an AMD ROCm
            PyTorch build is addressed as ``torch.device("cuda")``, the same
            API as NVIDIA. See the module docstring.
        accelerator: Human-facing accelerator label for display --
            ``"cuda"``, ``"rocm"``, ``"mps"``, ``"tpu"`` or ``"none"``. This is
            where ``"rocm"`` belongs; it never appears in ``device``.
        model_interactive: Ollama tag or HF repo id for the "answers
            immediately" pick -- small enough (or VRAM-resident enough) to
            feel conversational on this machine.
        model_background: Ollama tag or HF repo id for the more capable,
            slower-is-fine pick, for subagents and batch work.
        quantisation: Quantisation the two picks above were sized at (``"q4"``).
        fits_vram: Whether ``model_interactive`` fits entirely in the detected
            or configured VRAM at ``quantisation``. ``None`` when there is no
            VRAM figure to check against (no GPU, or the amount is unknown).
        notes: Every caveat that matters -- ROCm's Windows/Linux reality, the
            ROCm-vs-CUDA device-string distinction, "a TPU was detected but
            nothing here can use it yet", what was auto-detected vs. what came
            from a manual override, and any failure encountered along the way.
            Never silently dropped.
    """

    backend: str
    device: str
    accelerator: str
    model_interactive: str
    model_background: str
    quantisation: str
    fits_vram: Optional[bool]
    notes: tuple = ()


# --------------------------------------------------------------------------- #
#  Small, never-raising coercion helpers
# --------------------------------------------------------------------------- #
def _normalise_accelerator(value: Any) -> Optional[str]:
    """Fold a config/profile accelerator string to the internal vocabulary.

    Returns ``None`` when the value carries no information at all (``"auto"``,
    empty, unrecognised) -- the caller's cue to fall back to detection or to
    the final ``"cpu"`` default, rather than trusting a nonsense string.
    """
    text = str(value or "").strip().lower()
    if text in _ACCELERATOR_VALUES:
        return text
    if text == "none":
        return "cpu"
    return None


def _positive_float(value: Any) -> Optional[float]:
    """``float(value)`` if that is a genuine positive number, else ``None``."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _positive_int(value: Any) -> Optional[int]:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _model_ref(spec: ModelSpec) -> str:
    """The string a backend would actually be pointed at for this model.

    Ollama is the recommended backend everywhere in this module, so its tag
    is preferred when the catalogue has one; the HF repo id is the fallback
    for anything not published to Ollama.
    """
    return spec.ollama_tag or spec.id


def _safe_detect(notes: List[str]) -> Any:
    """Best-effort call into :func:`jarvis.core.hardware.detect`.

    Never raises. Import and call are both guarded: a missing module, a
    probe that throws, or any other surprise all degrade to "detection
    unavailable" rather than taking the caller down with them. This is a
    first-party module and normally fine to import at the top of a file, but
    :func:`plan` must keep working even on a build where it is absent or
    broken, so the import happens here instead.
    """
    try:
        from ..core import hardware  # first-party; see docstring above
    except Exception as exc:  # noqa: BLE001 - detection must never take the app down
        notes.append(
            f"jarvis.core.hardware could not be imported ({type(exc).__name__}: {exc}); "
            "planning for CPU-only."
        )
        return None
    try:
        return hardware.detect()
    except Exception as exc:  # noqa: BLE001
        notes.append(
            f"Hardware detection raised {type(exc).__name__}: {exc}; planning for CPU-only."
        )
        return None


def _device_and_notes(accelerator: str) -> Tuple[str, List[str]]:
    """The literal runtime device string for ``accelerator``, plus caveats.

    This is the one place the ROCm/TPU honesty rules from the module
    docstring get enforced in code.
    """
    notes: List[str] = []

    if accelerator == "cuda":
        return "cuda", notes

    if accelerator == "rocm":
        notes.append(
            "AMD ROCm builds of PyTorch place tensors with torch.device('cuda') -- "
            "there is no torch.device('rocm'). 'rocm' is a reporting/package-selection "
            "label only; the runtime device string is 'cuda', same as NVIDIA."
        )
        if IS_WINDOWS:
            notes.append(
                "ROCm PyTorch support is Linux-primary and patchy to absent for consumer "
                "cards on Windows. Ollama's own ROCm build is the more reliable path here: "
                "it ships ROCm support directly and does not depend on a working "
                "system-wide ROCm + PyTorch install."
            )
        else:
            notes.append(
                "Prefer Ollama's own ROCm build over the transformers backend on AMD: it "
                "ships ROCm support directly rather than depending on a separately "
                "installed, version-matched ROCm + PyTorch stack."
            )
        return "cuda", notes

    if accelerator == "mps":
        notes.append(
            "Apple Silicon shares system and GPU memory -- there is no separate VRAM "
            "pool, so sizing here is against system RAM rather than a VRAM budget."
        )
        return "mps", notes

    if accelerator == "tpu":
        notes.append(
            "A TPU was detected, but no backend in this codebase places tensors on an "
            "XLA device (jarvis/llm/*_backend.py has no PyTorch-XLA or JAX code path) -- "
            "falling back to CPU rather than pretending acceleration will happen."
        )
        return "cpu", notes

    return "cpu", notes


# --------------------------------------------------------------------------- #
#  The planner
# --------------------------------------------------------------------------- #
def plan(cfg: Any = None, *, profile: Any = None) -> HardwarePlan:
    """Recommend a backend, device and model pair for this machine.

    The default is fully automatic: with ``cfg.hardware.mode == "auto"`` (the
    default, and also what happens when ``cfg`` is ``None``), this calls
    :func:`jarvis.core.hardware.detect` -- unless ``profile`` is passed in,
    which is how tests and callers avoid re-probing real hardware -- and only
    falls back to a manual ``HardwareConfig`` field when detection could not
    determine that one specific value.

    ``cfg.hardware.mode == "manual"`` inverts that: every manual field is
    trusted as an override, and detection runs only to fill in whichever
    fields were themselves left at their auto-detect sentinel (``"auto"`` for
    ``accelerator``, ``0``/``0.0`` for the numeric fields).

    Never raises: any failure anywhere in detection or recommendation still
    returns a usable CPU-only plan, with the failure recorded in ``notes``.
    This holds even when the failure originates inside :func:`recommend`
    itself -- the recovery path below does not blindly re-call it unguarded,
    because doing so would just raise the same (or a new) exception a second
    time, this time with nothing left to catch it.
    """
    try:
        return _plan_impl(cfg, profile)
    except Exception as exc:  # noqa: BLE001 - this function must never raise
        logger.debug("hardware plan failed unexpectedly", exc_info=True)
        note = (
            f"Planning failed unexpectedly ({type(exc).__name__}: {exc}); "
            "falling back to a plain CPU-only plan.",
        )
        try:
            fallback = recommend(_DEFAULT_RAM_GB, False, "chat")
            background = recommend(_DEFAULT_RAM_GB, False, "quality")
            interactive_ref = _model_ref(fallback)
            background_ref = _model_ref(background)
        except Exception as exc2:  # noqa: BLE001 - the recovery path must never raise either
            logger.debug("hardware plan fallback recommendation also failed", exc_info=True)
            # A hardcoded literal, not a catalogue lookup: at this point both
            # detection and recommendation have failed, so this is the one
            # answer that cannot itself throw.
            interactive_ref = background_ref = "qwen3:4b-instruct-2507-q4_K_M"
            note = note + (
                f"Fallback model recommendation also failed "
                f"({type(exc2).__name__}: {exc2}); using a hardcoded default model.",
            )
        return HardwarePlan(
            backend=_RECOMMENDED_BACKEND,
            device="cpu",
            accelerator="none",
            model_interactive=interactive_ref,
            model_background=background_ref,
            quantisation=_QUANTISATION,
            fits_vram=None,
            notes=note,
        )


def _plan_impl(cfg: Any, profile: Any) -> HardwarePlan:
    notes: List[str] = []

    hw_cfg = getattr(cfg, "hardware", None) if cfg is not None else None
    mode = str(getattr(hw_cfg, "mode", "auto") or "auto").strip().lower()
    if mode not in ("auto", "manual"):
        notes.append(f"Unknown hardware.mode {mode!r}; treating as 'auto'.")
        mode = "auto"

    manual_accelerator = _normalise_accelerator(getattr(hw_cfg, "accelerator", "auto")) \
        if hw_cfg is not None else None
    manual_vram = _positive_float(getattr(hw_cfg, "vram_gb", 0.0)) if hw_cfg is not None else None
    manual_ram = _positive_float(getattr(hw_cfg, "ram_gb", 0.0)) if hw_cfg is not None else None
    manual_gpu_count = _positive_int(getattr(hw_cfg, "gpu_count", 0)) if hw_cfg is not None else None

    if mode == "manual" and hw_cfg is not None:
        accelerator = manual_accelerator
        vram_gb = manual_vram
        ram_gb = manual_ram
        gpu_count = manual_gpu_count
        if accelerator is not None:
            notes.append(f"hardware.mode=manual: accelerator overridden to {accelerator!r}.")
        if vram_gb is not None:
            notes.append(f"hardware.mode=manual: vram_gb overridden to {vram_gb:g}.")
        if ram_gb is not None:
            notes.append(f"hardware.mode=manual: ram_gb overridden to {ram_gb:g}.")
    else:
        accelerator = None
        vram_gb = None
        ram_gb = None
        gpu_count = None

    # Detect only what manual overrides (or auto mode) left undetermined.
    # `profile=` lets tests and callers supply hardware without re-probing.
    if accelerator is None or vram_gb is None or ram_gb is None or gpu_count is None:
        prof = profile if profile is not None else _safe_detect(notes)
        if prof is not None:
            if accelerator is None:
                accelerator = _normalise_accelerator(getattr(prof, "accelerator", None))
            if vram_gb is None:
                vram_gb = _positive_float(getattr(prof, "vram_gb", 0.0))
            if ram_gb is None:
                ram_gb = _positive_float(getattr(prof, "ram_gb", 0.0))
            if gpu_count is None:
                gpu_count = _positive_int(getattr(prof, "gpu_count", 0))
            prof_notes = getattr(prof, "notes", None)
            if isinstance(prof_notes, str) and prof_notes:
                notes.append(prof_notes)
            elif prof_notes:
                notes.extend(str(n) for n in prof_notes if n)

        # In auto mode a manual field is only ever a fallback *hint* for
        # whatever detection still could not pin down -- never an override
        # of something detection did determine.
        if mode == "auto" and hw_cfg is not None:
            if accelerator is None and manual_accelerator is not None:
                accelerator = manual_accelerator
                notes.append(
                    f"Accelerator not detected; using the configured hardware.accelerator "
                    f"hint ({manual_accelerator!r})."
                )
            if vram_gb is None and manual_vram is not None:
                vram_gb = manual_vram
                notes.append("VRAM not detected; using the configured hardware.vram_gb hint.")
            if ram_gb is None and manual_ram is not None:
                ram_gb = manual_ram
                notes.append("System RAM not detected; using the configured hardware.ram_gb hint.")
            if gpu_count is None and manual_gpu_count is not None:
                gpu_count = manual_gpu_count

    if accelerator is None:
        accelerator = "cpu"
        notes.append("No accelerator detected or configured; planning for CPU-only.")
    if ram_gb is None:
        ram_gb = _DEFAULT_RAM_GB
        notes.append(f"System RAM undetermined; assuming {_DEFAULT_RAM_GB:g} GB.")
    if gpu_count is None:
        gpu_count = 0

    device, device_notes = _device_and_notes(accelerator)
    notes.extend(device_notes)

    if accelerator in ("cuda", "rocm") and vram_gb is None:
        notes.append(
            "A GPU accelerator is present but its VRAM amount is unknown; sizing against "
            "system RAM only rather than guessing a VRAM budget."
        )
    if gpu_count > 1:
        notes.append(
            f"{gpu_count} GPUs detected; this planner sizes against a single device's VRAM "
            "only -- multi-GPU tensor/pipeline parallelism is not modelled here."
        )
    if accelerator == "cpu" and vram_gb is not None:
        # A manual override or a stray profile value set VRAM on a CPU-only
        # plan (e.g. TPU fallback, or hardware.accelerator="cpu" alongside a
        # leftover vram_gb) -- ignore it rather than let it leak into sizing.
        vram_gb = None

    # VRAM-aware sizing only makes sense for a discrete GPU with real VRAM
    # (cuda/rocm, both mapped to device="cuda"). Apple's unified memory is
    # sized against system RAM instead -- see _device_and_notes.
    #
    # Gated on `device`, NOT `accelerator`. A detected-but-unusable TPU has
    # accelerator="tpu" while device is correctly resolved to "cpu" above --
    # `accelerator != "cpu"` was True for that case, so recommend() thought a
    # real accelerator was present and sized a bigger, CPU-unfriendly model as
    # the *interactive* pick, directly contradicting the plan's own
    # device="cpu". `device` only ever takes the values "cuda"/"mps"/"cpu", so
    # this is the honest signal: "is there an accelerator this plan can
    # actually use", not "did detection see some kind of chip".
    has_gpu = device != "cpu"
    recommend_vram = vram_gb if device == "cuda" else None

    interactive_spec = recommend(ram_gb, has_gpu, "chat", vram_gb=recommend_vram)
    background_spec = recommend(ram_gb, has_gpu, "quality")

    fits_vram: Optional[bool] = None
    if recommend_vram:
        footprint = estimate_footprint(interactive_spec, _QUANTISATION, vram_gb=recommend_vram)
        fits_vram = footprint.get("fits_vram")

    display_accelerator = "none" if accelerator == "cpu" else accelerator

    return HardwarePlan(
        backend=_RECOMMENDED_BACKEND,
        device=device,
        accelerator=display_accelerator,
        model_interactive=_model_ref(interactive_spec),
        model_background=_model_ref(background_spec),
        quantisation=_QUANTISATION,
        fits_vram=fits_vram,
        notes=tuple(notes),
    )


__all__ = ["HardwarePlan", "plan"]
