"""Embedders: bge-small via fastembed, and a hashing embedder for offline tests.

Both return 384-dim normalised vectors. Rows are tagged with the embedder name so vectors
from different models are never compared.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Protocol

import numpy as np

from app.config import get_settings

log = logging.getLogger("aurelia.embed")
DIM = 384


class Embedder(Protocol):
    name: str
    dim: int
    min_score: float  # cosine below this = "the documents don't really cover this"
    semantic: bool  # False for the offline hash embedder, whose cosine can't tell topics apart

    def embed(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedEmbedder:
    name = "fastembed:bge-small-en-v1.5"
    dim = DIM
    semantic = True
    min_score = 0.55  # eval/retrieval_bench.py: in-domain questions score >= 0.60, unrelated ones 0.42-0.52

    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding("BAAI/bge-small-en-v1.5")

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self._model.embed(texts)), dtype=np.float32)
        return vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)


_STOP = frozenset(
    "the a an and or of to in on for with is are was were be been it this that at by as from your you my i me we our "
    "do does can how what when where which who will would should could may might have has had not no if than then so "
    "tell write about please explain need want know give show help like thing also much many any some get got let aurelia market".split()
)


def _stem(w: str) -> str:
    """Tiny suffix stripper: fees->fee, charges->charge, limits->limit, policies->policy."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("sses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    return w


def terms(text: str) -> list[str]:
    """Lower-cased, stop-word-free, lightly stemmed words (also used by BM25)."""
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]


def _tokens(text: str) -> list[str]:
    words = terms(text)
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


class LexicalEmbedder:
    name = "lexical-hash-v1"
    dim = DIM
    semantic = False
    min_score = 0.20

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in _tokens(text):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
                out[row, h % DIM] += 1.0 if (h >> 63) & 1 else -1.0  # signed hashing trick
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-9, None)


_cache: Embedder | None = None


def get_embedder() -> Embedder:
    global _cache
    if _cache is not None:
        return _cache
    choice = get_settings().embedder
    if choice == "lexical":
        _cache = LexicalEmbedder()
    elif choice == "fastembed":
        _cache = FastEmbedEmbedder()
    else:  # auto: real model if it can be loaded, otherwise the offline stand-in
        try:
            _cache = FastEmbedEmbedder()
        except Exception as exc:
            log.warning("fastembed unavailable (%s); using offline lexical embedder", type(exc).__name__)
            _cache = LexicalEmbedder()
    return _cache


def set_embedder(e: Embedder | None) -> None:
    global _cache
    _cache = e
