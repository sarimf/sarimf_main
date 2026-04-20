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
MAX_SUB_QUERIES = 16
SUB_QUERY_CHUNK_SIZE = CHUNK_SIZE
SUB_QUERY_OVERLAP = CHUNK_OVERLAP
OUTER_COSINE_THRESHOLD = 0.3
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


def _split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
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

    # -------------------- Retrieval --------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        top_n: int = 5,
        sub_chunk_size: Optional[int] = None,
        sub_overlap: Optional[int] = None,
    ) -> list[dict]:
        """Retrieve relevant chunks for a query.

        Two-level fusion:
          - Inner (per sub-query): RRF over (dense top_k, BM25 top_k).
            Rank-based; cosine and BM25 scores aren't directly comparable.
          - Outer (across sub-queries): CombSUM over dense cosines.
            L2-normalized embeddings make cosines absolute-comparable across
            sub-queries, so a chunk's final score is the sum of its cosines
            to each sub-query that matched it in the inner top_k.
            Contributions below OUTER_COSINE_THRESHOLD are dropped so junk
            sub-queries (those whose inner top_k is populated by weak
            matches) can't accumulate nonzero votes.

        Long queries are split internally into overlapping word-window
        sub-queries via `_split_text`. Window size and overlap default to
        SUB_QUERY_CHUNK_SIZE / SUB_QUERY_OVERLAP; override per call via
        `sub_chunk_size` / `sub_overlap` — larger windows yield coarser
        probes and lower cost. The sub-query list is capped at
        MAX_SUB_QUERIES via evenly-spaced indices (preserving coverage
        across the prompt), so cost stays bounded regardless of query
        length.

        All sub-query embeddings are computed in a single batched API call,
        and dense similarities are computed in a single matmul.

        top_n MAY exceed top_k: the outer pool can hold up to N*top_k
        unique chunks. Heavy sub-query overlap may shrink the pool below
        top_n. The returned `score` is the outer CombSUM — sum of
        above-threshold dense cosines — not a rank-fusion score.
        """
        if not self.chunks:
            return []

        chunk_size = SUB_QUERY_CHUNK_SIZE if sub_chunk_size is None else sub_chunk_size
        overlap = SUB_QUERY_OVERLAP if sub_overlap is None else sub_overlap

        sub_queries = _split_text(query, chunk_size, overlap) or [query]
        if len(sub_queries) > MAX_SUB_QUERIES:
            idxs = np.linspace(0, len(sub_queries) - 1, MAX_SUB_QUERIES, dtype=int)
            sub_queries = [sub_queries[i] for i in idxs]

        q_vecs = _embed_texts(sub_queries)
        sims_all = self.embeddings @ q_vecs.T
        k_dense = min(top_k, sims_all.shape[0])

        outer_scores: dict[int, float] = {}
        for j, sub_q in enumerate(sub_queries):
            sims = sims_all[:, j]
            top = np.argpartition(-sims, k_dense - 1)[:k_dense]
            dense = top[np.argsort(-sims[top])].tolist()
            sparse = self._sparse_retrieve(sub_q, top_k)
            merged = self._rrf([dense, sparse])[:top_k]
            for chunk_id, _ in merged:
                cos_ij = float(sims[chunk_id])
                if cos_ij >= OUTER_COSINE_THRESHOLD:
                    outer_scores[chunk_id] = outer_scores.get(chunk_id, 0.0) + cos_ij

        fused = sorted(outer_scores.items(), key=lambda x: -x[1])[:top_n]

        return [
            {
                "text": self.chunks[i],
                "source": self.sources[i],
                "score": float(score),
            }
            for i, score in fused
        ]
