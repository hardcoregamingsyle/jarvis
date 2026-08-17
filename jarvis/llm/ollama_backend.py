"""HuggingFace Transformers backend.

Loads a HF model directly into memory. On the target i5-10210U / 32 GB laptop
this is only viable for smaller Qwen3 checkpoints (up to ~4B); larger ones
belong on the Ollama or AirLLM backends. Every heavy import is deferred until
first use so this module can be inspected on a bare Python.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from typing import Any, Iterator, List, Optional, Sequence

from ..core.contracts import GenerationConfig, LLMResult, Message
from ..core.config import LLMConfig
from .base import BaseLLM, apply_stop_strings, format_chat, strip_thinking


logger = logging.getLogger(__name__)


class TransformersBackend(BaseLLM):
    """A synchronous ``transformers.AutoModelForCausalLM`` wrapper."""

    name = "transformers"

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__(cfg)
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"
        self._torch: Any = None

    # -- availability ------------------------------------------------------- #
    def is_available(self) -> bool:
        if importlib.util.find_spec("transformers") is None:
            return False
        if importlib.util.find_spec("torch") is None:
            return False
        return True

    # -- device / dtype selection ------------------------------------------ #
    def _resolve_device(self, torch: Any) -> str:
        want = (self.cfg.device or "auto").lower()
        if want == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if want == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
            return "cpu"
        if want == "cpu":
            return "cpu"
        # "auto"
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self, torch: Any, device: str) -> Any:
        # bfloat16 halves memory versus float32 and is well supported on
        # modern CPUs (AVX-512 BF16) and every CUDA arch we would run.
        if device == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device == "mps":
            return torch.float16
        return torch.bfloat16

    # -- load --------------------------------------------------------------- #
    def _do_load(self) -> None:
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "transformers backend requires PyTorch. Install with "
                "`pip install torch`."
            ) from exc
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "transformers backend requires the 'transformers' package. "
                "Install with `pip install transformers`."
            ) from exc

        self._torch = torch
        self._device = self._resolve_device(torch)
        dtype = self._resolve_dtype(torch, self._device)

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.to(self._device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model

    # -- prompt build ------------------------------------------------------- #
    def _ensure_pad_token(self) -> None:
        # Qwen/Llama tokenizers frequently ship without a pad token; leaving
        # it as None makes .generate() warn every call and can corrupt padded
        # output entirely.  Alias to eos.
        tok = self._tokenizer
        if tok.pad_token_id is None:
            eos = tok.eos_token
            eos_id = tok.eos_token_id
            if eos is not None:
                tok.pad_token = eos
            if eos_id is not None:
                tok.pad_token_id = eos_id

    def _build_prompt(self, messages: Sequence[Message]) -> str:
        tok = self._tokenizer
        payload = [m.to_dict() for m in messages]
        try:
            template_fn = getattr(tok, "apply_chat_template", None)
            if template_fn is not None:
                return template_fn(payload, tokenize=False, add_generation_prompt=True)
        except Exception:
            logger.debug("apply_chat_template failed; falling back to format_chat", exc_info=True)
        return format_chat(messages, style="chatml")

    # -- generate ----------------------------------------------------------- #
    def generate(
        self,
        messages: Sequence[Message],
        config: Optional[GenerationConfig] = None,
    ) -> LLMResult:
        self.load()
        self._ensure_pad_token()
        torch = self._torch
        gen = self._gen_config(config)
        messages = self._fit(messages)

        prompt = self._build_prompt(messages)
        enc = self._tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        prompt_len = int(input_ids.shape[-1])

        gen_kwargs = dict(
            max_new_tokens=int(gen.max_new_tokens),
            do_sample=gen.temperature > 0,
            temperature=float(gen.temperature),
            top_p=float(gen.top_p),
            top_k=int(gen.top_k),
            pad_token_id=self._tokenizer.pad_token_id,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if gen.seed is not None:
            torch.manual_seed(int(gen.seed))

        with torch.no_grad():
            output_ids = self._model.generate(input_ids, **gen_kwargs)

        completion_ids = output_ids[0][prompt_len:]
        text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        text = strip_thinking(text)
        text = apply_stop_strings(text, gen.stop)

        return LLMResult(
            text=text,
            prompt_tokens=prompt_len,
            completion_tokens=int(output_ids.shape[-1]) - prompt_len,
            finish_reason="stop",
        )

    # -- stream ------------------------------------------------------------- #
    def stream(
        self,
        messages: Sequence[Message],
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        self.load()
        self._ensure_pad_token()
        torch = self._torch
        gen = self._gen_config(config)
        messages = self._fit(messages)

        try:
            from transformers import TextIteratorStreamer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Streaming requires transformers>=4.30 with TextIteratorStreamer."
            ) from exc

        prompt = self._build_prompt(messages)
        enc = self._tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        # A silent worker death must never leave the consumer blocked
        # forever, so give the streamer an explicit timeout and record any
        # exception on the worker thread so we can re-raise it here.
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=max(60.0, float(self.cfg.request_timeout)),
        )
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=int(gen.max_new_tokens),
            do_sample=gen.temperature > 0,
            temperature=float(gen.temperature),
            top_p=float(gen.top_p),
            top_k=int(gen.top_k),
            pad_token_id=self._tokenizer.pad_token_id,
            streamer=streamer,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if gen.seed is not None:
            torch.manual_seed(int(gen.seed))

        worker_error: List[BaseException] = []

        def _run() -> None:
            try:
                with torch.no_grad():
                    self._model.generate(**gen_kwargs)
            except BaseException as exc:  # noqa: BLE001
                worker_error.append(exc)
            finally:
                # Whatever happened, wake the consumer.
                try:
                    streamer.end()
                except Exception:  # pragma: no cover
                    pass

        thread = threading.Thread(target=_run, name="transformers-generate", daemon=True)
        thread.start()

        stops = tuple(gen.stop or ())
        buf = ""
        try:
            for piece in streamer:
                if not piece:
                    continue
                buf += piece
                cleaned = strip_thinking(buf)
                if stops:
                    truncated = apply_stop_strings(cleaned, stops)
                    if truncated != cleaned:
                        if truncated:
                            yield truncated
                        return
                yield piece
        finally:
            thread.join(timeout=1.0)

        if worker_error:
            raise RuntimeError(f"Transformers generate() failed: {worker_error[0]!r}") from worker_error[0]


__all__ = ["TransformersBackend"]
