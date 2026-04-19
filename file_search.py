"""
file_search.py — minimal RAG library for internal BU use.

Hybrid (semantic + BM25) retrieval over local files with automatic word-window
decomposition of long queries, Reciprocal Rank Fusion across sub-queries, and
grounded answer generation. Parquet persistence.

QUICKSTART
----------
    from file_search import VectorStore, QueryEngine

    store = VectorStore()
    store.add_file("docs/product.pdf")
    store.save("my_index.parquet")

    store = VectorStore.load("my_index.parquet")
    engine = QueryEngine(store, system_message="You are an AmEx expert.")
    hits = store.retrieve("annual fee")
    print(engine.synthesize("annual fee", hits)["response"])

DEPENDENCIES
------------
    pip install numpy pandas pyarrow rank-bm25 tenacity openai pypdf python-docx
"""
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt


__all__ = ["VectorStore", "QueryEngine"]


# ---- Tuning constants ------------------------------------------------------
LLM_MODEL = "gpt-4.1"
LLM_ENDPOINT_NAME = "gpt-41"
LLM_TOKEN_KEY = "gpt41"
EMBED_DIMS = 256
CHUNK_SIZE = 600
CHUNK_OVERLAP = 300
EMBED_BATCH_SIZE = 64
DEFAULT_SYSTEM = (
    "Answer using ONLY the context below. If the answer isn't in the context, "
    "say you don't know. Cite sources inline as [Source: <filename>]."
)


# ---- Burst retry -----------------------------------------------------------
def _burst_wait(retry_state):
    """5 rapid tries per burst (~0.2s apart), exp backoff between bursts
    (2/4/8/16s). 5 bursts max => 25 attempts total."""
    attempt = retry_state.attempt_number
    if attempt % 5 == 0:
        burst_idx = attempt // 5
        return 2 ** burst_idx
    return 0.2


_api_retry = retry(
    wait=_burst_wait,
    stop=stop_after_attempt(25),
    reraise=True,
    before_sleep=lambda rs: print(
        f" API error: {rs.outcome.exception()!r}. "
        f"Attempt {rs.attempt_number}/25. "
        f"Sleeping {rs.next_action.sleep:.1f}s..."
    ),
)


# ---- Private helpers -------------------------------------------------------
def _parse_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    ext = Path(path).suffix.lower()
    if ext in {".txt", ".md"}:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    if ext == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(path).paragraphs)
    raise ValueError(f"Unsupported file type: {ext}. Supported: .txt, .md, .pdf, .docx")


def _split_text(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        if chunk:
            chunks.append(chunk)
        if i + CHUNK_SIZE >= len(words):
            break
    return chunks


@_api_retry
def _call_embedding_api(batch_texts: list[str]):
    client = AzureOpenAI(
        azure_endpoint="https://eag-qa.aexp.com/genai/microsoft/v1/gcs_distribution_efficiency",
        api_key=get_token_from_env('ada-3-large'),  # noqa: F821
        api_version="2024-06-01",
    )
    return client.embeddings.create(
        model="text-embedding-3-large",
        input=batch_texts,
        dimensions=EMBED_DIMS,
    )


def _format_metadata_prefix(metadata: dict) -> str:
    if not metadata:
        return ""
    def clean(v):
        return str(v).replace("|", "/").replace("\n", " ").replace("\r", " ").strip()
    parts = [f"{k}: {clean(v)}" for k, v in metadata.items()]
    return "[" + " | ".join(parts) + "]"


def _embed_texts(texts: list[str]) -> np.ndarray:
    all_embs = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        resp = _call_embedding_api(batch)
        all_embs.extend(item.embedding for item in sorted(resp.data, key=lambda x: x.index))
    arr = np.asarray(all_embs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _build_chat_client() -> AzureOpenAI:
    """Build a fresh chat client per call (avoids stale-connection timeouts)."""
    return AzureOpenAI(
        azure_endpoint=f"https://eag-qa.aexp.com/genai/microsoft/v1/models/{LLM_ENDPOINT_NAME}/",
        api_key=get_token_from_env(LLM_TOKEN_KEY),  # noqa: F821
        api_version="2024-10-21",
    )


@_api_retry
def _chat_complete(messages: list[dict]) -> str:
    resp = _build_chat_client().chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


# ============================================================================
# QueryEngine
# ============================================================================
class QueryEngine:
    """Wraps a VectorStore with a bound system_message to generate grounded answers.

    Inspect/filter hits between store.retrieve() and engine.synthesize() when you
    need audit logging, quality checks, or multi-step flows.
    """

    def __init__(self, store: "VectorStore", system_message: Optional[str] = None):
        self.store = store
        self.system_message = system_message or DEFAULT_SYSTEM

    def synthesize(self, query: str, hits: list[dict]) -> dict:
        context = "\n\n---\n\n".join(
            f"[Source: {h['source']}]\n{h['text']}" for h in hits
        )
        full_system = self.system_message + f"\n\nCONTEXT:\n{context}"
        response = _chat_complete([
            {"role": "system", "content": full_system},
            {"role": "user", "content": query},
        ])
        return {
            "response": response,
            "sources": [h["source"] for h in hits],
            "chunks": hits,
        }


# ============================================================================
# VectorStore
# ============================================================================
class VectorStore:
    """Indexes files and retrieves relevant chunks with hybrid search."""

    def __init__(self):
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None

    # -------------------- Indexing --------------------
    def add_file(self, path: str, metadata: Optional[dict] = None) -> int:
        """Parse, chunk, embed, and index a single file. Returns chunk count added.

        `metadata` (if provided) is prefixed into every chunk's text — the LLM
        sees `[k: v | ...]` before each fact. No structural storage, no
        filtering. The file stem is stored separately on `self.sources` for
        citation; it is NOT auto-added to the in-text prefix.
        """
        return self.add_files([path], [metadata])

    def add_files(
        self,
        paths: list[str],
        metadatas: Optional[list[Optional[dict]]] = None,
    ) -> int:
        """Bulk-index N files with one BM25 rebuild at the end.

        `metadatas`, if provided, must align 1:1 with `paths`; pass `None` to
        index all files without a metadata prefix. Returns total chunk count
        added.
        """
        if metadatas is None:
            metadatas = [None] * len(paths)
        if len(metadatas) != len(paths):
            raise ValueError(
                f"len(metadatas)={len(metadatas)} must match len(paths)={len(paths)}"
            )
        total = 0
        for path, meta in zip(paths, metadatas):
            new_chunks = _split_text(_parse_file(path))
            if not new_chunks:
                continue
            source = Path(path).stem
            prefix = _format_metadata_prefix(meta or {})
            prefixed = [f"{prefix}\n{c}" for c in new_chunks] if prefix else new_chunks
            embs = _embed_texts(prefixed)
            self.chunks.extend(prefixed)
            self.sources.extend([source] * len(new_chunks))
            self.embeddings = embs if self.embeddings is None else np.vstack([self.embeddings, embs])
            total += len(new_chunks)
        if total:
            self.bm25 = BM25Okapi([c.lower().split() for c in self.chunks])
        return total

    # -------------------- Persistence --------------------
    def save(self, path: str) -> None:
        """Persist the index as a single Parquet file (inspectable via pandas)."""
        df = pd.DataFrame({
            "chunk": self.chunks,
            "source": self.sources,
            "embedding": [e.tolist() for e in self.embeddings] if self.embeddings is not None else [],
        })
        df.to_parquet(path, index=False)
        print(f"Saved {len(self.chunks)} chunks to {path}")

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Load a Parquet index written by save()."""
        df = pd.read_parquet(path)
        store = cls()
        if len(df):
            store.chunks = df["chunk"].tolist()
            store.sources = df["source"].tolist()
            store.embeddings = np.array(df["embedding"].tolist(), dtype=np.float32)
            store.bm25 = BM25Okapi([c.lower().split() for c in store.chunks])
        return store

    # -------------------- Retrieval primitives --------------------
    def _dense_retrieve(self, q_vec: np.ndarray, k: int) -> list[int]:
        sims = self.embeddings @ q_vec
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        return top[np.argsort(-sims[top])].tolist()

    def _sparse_retrieve(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(query.lower().split())
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    @staticmethod
    def _rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: -x[1])

    def _hybrid_search(self, query: str, q_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        dense = self._dense_retrieve(q_vec, k)
        sparse = self._sparse_retrieve(query, k)
        return self._rrf([dense, sparse])[:k]

    # -------------------- Retrieval --------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        top_n: int = 5,
    ) -> list[dict]:
        """Retrieve relevant chunks for a query.

        Long queries (more than CHUNK_SIZE words) are split internally into
        overlapping word-window sub-queries. Hybrid search runs per sub-query
        pulling top_k each; rankings are merged with Reciprocal Rank Fusion and
        the top top_n are returned. Short queries take the same path with one
        sub-query (degenerate RRF).

        All sub-query embeddings are computed in a single batched API call —
        one network round-trip regardless of N.

        top_n MAY exceed top_k: with N sub-queries, the RRF pool can hold up
        to N*top_k unique chunks. top_n is an upper bound on the returned
        list — when sub-queries overlap heavily, the unique pool may be smaller
        than top_n, in which case fewer hits are returned.
        """
        if not self.chunks:
            return []

        sub_queries = _split_text(query) or [query]
        q_vecs = _embed_texts(sub_queries)

        rankings = [
            [i for i, _ in self._hybrid_search(sub_q, q_vecs[i], top_k)]
            for i, sub_q in enumerate(sub_queries)
        ]
        fused = self._rrf(rankings)[:top_n]

        return [
            {
                "text": self.chunks[i],
                "source": self.sources[i],
                "score": float(score),
            }
            for i, score in fused
        ]
