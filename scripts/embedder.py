#!/usr/bin/env python3
"""
Embedding backends
===================

A thin abstraction so the index builder (`embed.py`) and the searcher
(`search.py`) embed text the SAME way, whether that's:

  * LocalEmbedder  — a CPU SentenceTransformer model (default, offline, ungated)
  * APIEmbedder    — a remote OpenAI-compatible /embeddings endpoint
                     (e.g. internal `text-embedding-3-small`)

Why this matters
----------------
A query MUST be embedded with the exact same model/vector space as the indexed
documents, or similarity is meaningless. The build records the provider + model
+ api_base in the index `config.json`; the searcher reconstructs the matching
embedder from that config. Secrets (API tokens) are read from the environment
at runtime and are NEVER written to disk.

Selection (when not passed explicitly) comes from env vars:
  DOCSEARCH_EMBED_PROVIDER   local | api          (default: local)
  DOCSEARCH_EMBED_MODEL      model id / name
  DOCSEARCH_EMBED_API_BASE   embeddings base URL  (falls back to DOCSEARCH_API_BASE)
  DOCSEARCH_EMBED_API_KEY    bearer token         (falls back to DOCSEARCH_API_KEY)

Heavy third-party imports (sentence_transformers, openai) are deferred into the
backends that need them, so importing this module stays cheap and test-friendly.
"""

import os
import time
from typing import List, Optional, Sequence

import numpy as np

DEFAULT_LOCAL_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_API_MODEL = "text-embedding-3-small"
DEFAULT_API_BASE = "https://api.openai.com/v1"


class Embedder:
    """Common interface: encode text -> float32 ndarray of shape (n, dim)."""

    name: str = ""
    provider: str = ""

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError


class LocalEmbedder(Embedder):
    """CPU SentenceTransformer. Device is pinned to CPU on purpose: a CUDA
    build has been observed to crash the Windows GPU driver (dxgmms2.sys)."""

    provider = "local"

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer  # deferred import

        self.name = model_name
        print(f"Loading embedding model: {model_name} (device={device})")
        self._model = SentenceTransformer(model_name, device=device)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        emb = self._model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(emb, dtype="float32")

    @property
    def dimension(self) -> int:
        return self._dim


class APIEmbedder(Embedder):
    """Remote OpenAI-compatible embeddings endpoint.

    POST {api_base}/embeddings  {"model": ..., "input": [texts]}
      -> {"data": [{"embedding": [...]}, ...]}

    Offloads all embedding compute off the local machine.
    """

    provider = "api"

    def __init__(
        self,
        model_name: str = DEFAULT_API_MODEL,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: int = 64,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        from openai import OpenAI  # deferred import

        self.name = model_name
        self.api_base = (
            api_base
            or os.environ.get("DOCSEARCH_EMBED_API_BASE")
            or os.environ.get("DOCSEARCH_API_BASE")
            or DEFAULT_API_BASE
        )
        key = (
            api_key
            or os.environ.get("DOCSEARCH_EMBED_API_KEY")
            or os.environ.get("DOCSEARCH_API_KEY")
            or ""
        )
        if not key:
            raise RuntimeError(
                "No embedding API token found. Set DOCSEARCH_EMBED_API_KEY "
                "(or DOCSEARCH_API_KEY) before building or searching with provider=api."
            )

        self._client = OpenAI(api_key=key, base_url=self.api_base, timeout=timeout)
        self.batch_size = max(1, batch_size)
        self.max_retries = max(1, max_retries)
        self._dim = dimension  # may be supplied from index config to skip probing

    def _embed_batch(self, batch: List[str]) -> np.ndarray:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.embeddings.create(model=self.name, input=batch)
                vecs = [item.embedding for item in resp.data]
                return np.asarray(vecs, dtype="float32")
            except Exception as e:  # transient network/gateway errors -> backoff
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(
            f"Embedding API request failed after {self.max_retries} attempts "
            f"({self.api_base}, model={self.name}): {last_err}"
        )

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        out = []
        for start in range(0, len(texts), self.batch_size):
            out.append(self._embed_batch(texts[start:start + self.batch_size]))
        result = np.vstack(out)
        if self._dim is None:
            self._dim = result.shape[1]
        return result

    @property
    def dimension(self) -> int:
        if self._dim is None:
            # Lazy one-off probe so the FAISS index can be sized if needed.
            self._dim = self._embed_batch(["dimension probe"]).shape[1]
        return self._dim


def get_embedder(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    dimension: Optional[int] = None,
    device: str = "cpu",
) -> Embedder:
    """Build an embedder from explicit args, falling back to DOCSEARCH_EMBED_* env."""
    provider = (provider or os.environ.get("DOCSEARCH_EMBED_PROVIDER") or "local").lower()

    if provider == "api":
        model_name = model_name or os.environ.get("DOCSEARCH_EMBED_MODEL") or DEFAULT_API_MODEL
        return APIEmbedder(model_name, api_base=api_base, api_key=api_key, dimension=dimension)

    model_name = model_name or os.environ.get("DOCSEARCH_EMBED_MODEL") or DEFAULT_LOCAL_MODEL
    return LocalEmbedder(model_name, device=device)


def embedder_from_config(
    config: dict,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Embedder:
    """Reconstruct the embedder described by an index `config.json` (no secrets).

    `api_key`/`api_base` let the caller inject the credentials the user entered
    for the LLM (same API gateway), so the embedding API reuses that token instead
    of requiring a separate DOCSEARCH_EMBED_API_KEY env var. Falls back to env when
    not provided (see APIEmbedder).
    """
    return get_embedder(
        provider=config.get("embedding_provider", "local"),
        model_name=config.get("model_name"),
        api_base=api_base or config.get("embedding_api_base"),
        api_key=api_key,
        dimension=config.get("embedding_dim"),
    )


if __name__ == "__main__":
    # Connectivity / smoke test:
    #   DOCSEARCH_EMBED_PROVIDER=api DOCSEARCH_API_KEY=... python scripts/embedder.py "hello"
    import sys

    sample = sys.argv[1:] or ["hello world", "switch databases on ttc"]
    emb = get_embedder()
    vecs = emb.encode(sample, is_query=True)
    print(f"provider={emb.provider}  model={emb.name}  dim={emb.dimension}  shape={tuple(vecs.shape)}")
