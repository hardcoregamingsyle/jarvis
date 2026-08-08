"""Embedding backends for the memory store.

Three implementations, in decreasing quality but increasing availability:

* :class:`SentenceTransformerEmbedder` — high-quality sentence embeddings from
  the ``sentence-transformers`` package.  Lazy-imported so this module still
  loads on a bare stdlib Python.
* :class:`HashEmbedder` — a zero-dependency deterministic hashed n-gram
  embedder.  Uses ``hashlib.blake2b`` rather than Python's salted ``hash()``,
  so vectors are stable across processes and machines.
* :class:`NullEmbedder` — dim 0.  Selecting this collapses the store's search
  to keyword-only.

The :func:`create_embedder` factory reads ``cfg.embedder`` and picks the first
available option.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import List, Sequence

from ..core.contracts import Embedder

log = logging.getLogger(__name__)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two float sequences.

    Returns ``0.0`` when either side is empty or has zero magnitude — never
    raises and never returns ``nan``.
    """
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


class NullEmbedder(Embedder):
    """A no-op embedder.  Selecting this makes the store keyword-only."""

    name = "null"
    dim = 0

    def is_available(self) -> bool:
        return True

    def embed(self, texts: Sequence[str]) -> List[list]:
        return [[] for _ in texts]


_WORD_RE = re.compile(r"\w+", re.UNICODE)


class HashEmbedder(Embedder):
    """Deterministic hashed-n-gram embedder with no third-party dependencies.

    Character trigrams through 5-grams (case-folded) plus word unigrams are
    hashed with :func:`hashlib.blake2b` into ``dim`` signed buckets and the
    vector is L2-normalised.  Because blake2b is stable across processes and
    Python versions, two runs on different machines produce identical vectors
    for the same text — a property Python's built-in ``hash()`` does not have.
    """

    name = "hash"

    def __init__(self, dim: int = 384) -> None:
        self.dim = int(dim) if dim and dim > 0 else 384

    def is_available(self) -> bool:
        return True

    def _tokens(self, text: str):
        text = (text or "").lower()
        for size in (3, 4, 5):
            if len(text) >= size:
                for i in range(len(text) - size + 1):
                    yield text[i : i + size]
        for word in _WORD_RE.findall(text):
            yield "w:" + word

    def _bucket(self, token: str):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if (digest[4] & 1) else -1.0
        return idx, sign

    def embed(self, texts: Sequence[str]) -> List[list]:
        results: List[list] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in self._tokens(text):
                idx, sign = self._bucket(token)
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0.0:
                inv = 1.0 / norm
                vec = [v * inv for v in vec]
            results.append(vec)
        return results


class SentenceTransformerEmbedder(Embedder):
    """Semantic embedder backed by ``sentence-transformers`` (lazy-imported)."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self.dim = 0

    def is_available(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False
        except Exception:  # noqa: BLE001 - a partly broken install must not crash
            log.debug("sentence-transformers import raised", exc_info=True)
            return False
        return True

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed; "
                "fall back to HashEmbedder or install the package."
            ) from exc
        self._model = SentenceTransformer(self.model_name)
        try:
            self.dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:  # noqa: BLE001
            probe = self._model.encode(["_probe"], normalize_embeddings=True)
            self.dim = int(len(probe[0]))

    def embed(self, texts: Sequence[str]) -> List[list]:
        self._load()
        assert self._model is not None
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        out: List[list] = []
        for vec in vectors:
            try:
                out.append([float(x) for x in vec])
            except TypeError:
                out.append(list(vec))
        return out


def create_embedder(cfg) -> Embedder:
    """Factory: honour ``cfg.embedder``.

    ``"auto"`` prefers sentence-transformers if importable, otherwise
    :class:`HashEmbedder`.  ``"none"`` yields :class:`NullEmbedder`.
    """
    kind = str(getattr(cfg, "embedder", "auto") or "auto").strip().lower()
    dim = int(getattr(cfg, "embed_dim", 384) or 384)
    model_name = getattr(cfg, "embed_model", "sentence-transformers/all-MiniLM-L6-v2")

    if kind == "none":
        return NullEmbedder()
    if kind == "hash":
        return HashEmbedder(dim=dim)
    if kind in ("sentence-transformers", "st", "transformers"):
        emb = SentenceTransformerEmbedder(model_name)
        if emb.is_available():
            return emb
        log.warning(
            "sentence-transformers requested but not available; "
            "falling back to HashEmbedder(dim=%d)",
            dim,
        )
        return HashEmbedder(dim=dim)
    st = SentenceTransformerEmbedder(model_name)
    if st.is_available():
        return st
    return HashEmbedder(dim=dim)


__all__ = [
    "cosine",
    "NullEmbedder",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "create_embedder",
]
