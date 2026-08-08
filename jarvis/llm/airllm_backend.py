"""AirLLM backend — the layered-inference route the user asked for.

AirLLM (https://github.com/lyogavin/airllm) runs any Hugging Face causal LM
by paging one transformer layer at a time from disk into RAM/VRAM. That is
what lets a 30 B model "fit" in <8 GB of memory on a laptop.

Honest warning: on a CPU-only i5 the disk-streamed forward pass costs
somewhere around 10-50 SECONDS per generated token — roughly 0.02-0.1
tok/sec. It is perfectly usable for background subagents that can churn
overnight, but hopeless for live spoken conversation. When responsiveness
matters, prefer the Ollama backend — a 4-bit quantised Qwen3-30B-A3B MoE
(activated ~3 B params/token, ~19 GB on disk) fits comfortably in 32 GB RAM
and does ~4-8 tok/sec on the same box.

Two gotchas verified against airllm 3.1.0 that we work around here:
    * ``AutoModel.from_pretrained`` defaults ``device='cuda:0'`` and crashes
      on a CPU-only machine unless the caller passes ``device`` explicitly.
    * ``max_seq_len`` defaults to 512, silently capping the context window.
Both are always passed below.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Optional, Sequence

from ..core.contracts import GenerationConfig, LLMResult, Message
from ..core.config import LLMConfig
from .base import BaseLLM, apply_stop_strings, format_chat, strip_thinking


logger = logging.getLogger(__name__)


class AirLLMBackend(BaseLLM):
    """Disk-paged inference via AirLLM's ``AutoModel``."""

    name = "airllm"

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__(cfg)
        self._model: Any = None

    # -- availability ------------------------------------------------------- #
    def is_available(self) -> bool:
        return importlib.util.find_spec("airllm") is not None

    # -- device selection --------------------------------------------------- #
    def _resolve_device(self) -> str:
        want = (self.cfg.device or "auto").lower()
        if want == "cpu":
            return "cpu"
        if want in ("cuda", "cuda:0"):
            return "cuda:0"
        if want == "mps":
            return "mps"
        # "auto": use CUDA only when it is actually available; else CPU.
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                return "cuda:0"
        except ImportError:
            pass
        return "cpu"

    # -- load --------------------------------------------------------------- #
    def _do_load(self) -> None:
        try:
            from airllm import AutoModel  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "airllm backend requires the 'airllm' package. Install with "
                "`pip install airllm`."
            ) from exc

        device = self._resolve_device()
        base_kwargs = dict(
            device=device,
            max_seq_len=int(self.cfg.context_tokens),
        )
        shards_dir = self.cfg.layer_shards_dir or ""
        if shards_dir:
            base_kwargs["layer_shards_saving_path"] = shards_dir
        compression = (self.cfg.compression or "").strip().lower()
        if compression in ("4bit", "8bit"):
            base_kwargs["compression"] = compression

        attempts = [dict(base_kwargs)]
        if "compression" in base_kwargs:
            fb = dict(base_kwargs)
            fb.pop("compression", None)
            attempts.append(fb)
        if "layer_shards_saving_path" in attempts[-1]:
            fb = dict(attempts[-1])
            fb.pop("layer_shards_saving_path", None)
            attempts.append(fb)

        last_error: Optional[BaseException] = None
        model: Any = None
        for kwargs in attempts:
            try:
                model = AutoModel.from_pretrained(self.cfg.model, **kwargs)
                break
            except (TypeError, ImportError) as exc:
                # TypeError -> unknown kwarg for this airllm version.
                # ImportError -> compression="4bit"/"8bit" without bitsandbytes.
                last_error = exc
                logger.warning(
                    "AirLLM load failed with kwargs %r (%s); retrying with reduced kwargs",
                    kwargs,
                    exc,
                )
                continue
            except Exception as exc:
                last_error = exc
                break

        if model is None:
            raise RuntimeError(
                f"Could not initialise AirLLM for model {self.cfg.model!r}. "
                f"Last error: {last_error!r}. If this is a compression error, "
                f"install bitsandbytes (`pip install bitsandbytes`). If AirLLM "
                f"itself is missing, `pip install airllm`."
            ) from last_error

        self._model = model

    # -- prompt build ------------------------------------------------------- #
    def _build_prompt(self, messages: Sequence[Message]) -> str:
        tokenizer = getattr(self._model, "tokenizer", None)
        payload = [m.to_dict() for m in messages]
        if tokenizer is not None:
            try:
                template_fn = getattr(tokenizer, "apply_chat_template", None)
                if template_fn is not None:
                    return template_fn(payload, tokenize=False, add_generation_prompt=True)
            except Exception:
                logger.debug("apply_chat_template failed; using ChatML fallback", exc_info=True)
        return format_chat(messages, style="chatml")

    # -- generate ----------------------------------------------------------- #
    def generate(
        self,
        messages: Sequence[Message],
        config: Optional[GenerationConfig] = None,
    ) -> LLMResult:
        self.load()
        gen = self._gen_config(config)
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("AirLLM model has no attached tokenizer.")

        prompt = self._build_prompt(messages)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=False,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"]
        prompt_len = int(input_ids.shape[-1])

        output = self._model.generate(
            input_ids,
            max_new_tokens=int(gen.max_new_tokens),
            use_cache=True,
        )

        text = self._decode_completion(tokenizer, output, prompt, prompt_len)
        text = strip_thinking(text)
        text = apply_stop_strings(text, gen.stop)

        completion_tokens = 0
        try:
            completion_tokens = max(0, int(output.shape[-1]) - prompt_len)
        except AttributeError:
            completion_tokens = max(1, len(text) // 4)

        return LLMResult(
            text=text,
            prompt_tokens=prompt_len,
            completion_tokens=completion_tokens,
            finish_reason="stop",
        )

    def _decode_completion(
        self,
        tokenizer: Any,
        output: Any,
        prompt: str,
        prompt_len: int,
    ) -> str:
        """Return only the newly generated text, never the prompt echo."""
        # Preferred: slice the prompt off the output token tensor.
        try:
            shape = tuple(getattr(output, "shape", ()))
            if len(shape) == 2 and shape[-1] > prompt_len:
                new_ids = output[0][prompt_len:]
                return tokenizer.decode(new_ids, skip_special_tokens=True)
            if len(shape) == 1 and shape[0] > prompt_len:
                new_ids = output[prompt_len:]
                return tokenizer.decode(new_ids, skip_special_tokens=True)
        except (TypeError, IndexError, AttributeError):
            pass

        # Fallback: decode everything, then strip the decoded prompt prefix.
        try:
            full = tokenizer.decode(
                output[0] if hasattr(output, "shape") and len(output.shape) == 2 else output,
                skip_special_tokens=True,
            )
        except Exception:
            full = str(output)
        if full.startswith(prompt):
            return full[len(prompt):]
        # Last-ditch: strip a decoded-prompt prefix (special tokens removed
        # may not match a raw ChatML string exactly).
        try:
            reencoded = tokenizer.decode(
                tokenizer(prompt, add_special_tokens=False)["input_ids"],
                skip_special_tokens=True,
            )
            if full.startswith(reencoded):
                return full[len(reencoded):]
        except Exception:
            pass
        return full


__all__ = ["AirLLMBackend"]
