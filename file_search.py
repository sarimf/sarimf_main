"""
file_search.py — RAG library for internal BU use.

Hybrid (semantic + BM25) retrieval over local files, with deterministic
word-window query decomposition, metadata filtering, Parquet persistence,
and a universal burst-retry on every network call.

QUICKSTART
----------
    from file_search import VectorStore, QueryEngine, split_query

    store = VectorStore()
    store.add_file("docs/product.pdf", metadata={"product": "Platinum"})
    store.save("my_index.parquet")

    store = VectorStore.load("my_index.parquet")
    engine = QueryEngine(store, system_message="You are an AmEx expert.")
    sub_queries = split_query("a large messy customer prompt...")
    hits = store.retrieve(sub_queries, top_k=10, final_k=5)
    print(engine.synthesize("a large messy customer prompt...", hits)["answer"])

DEPENDENCIES
------------
    pip install numpy pandas pyarrow rank-bm25 tenacity openai pypdf python-docx
"""
import json
import os
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt


__all__ = ["VectorStore", "QueryEngine", "split_query"]


# ---- Tuning constants ------------------------------------------------------
LLM_MODEL = "gpt-4.1"
EMBED_DIMS = 256
CHUNK_SIZE = 600
CHUNK_OVERLAP = 300
SUBQUERY_SIZE = 60
SUBQUERY_OVERLAP = 30
MAX_QUERY_CHARS = 30000
EMBED_BATCH_SIZE = 64
TEMPERATURE = 0.2
MAX_TOKENS = 500
DEFAULT_SYSTEM = (
    "Answer using ONLY the context below. If the answer isn't in the context, "
    "say you don't know. Cite sources inline as [Source: <filename>]."
)


# ---- Retry: burst pattern --------------------------------------------------
def _burst_wait(retry_state):
    """5 rapid tries per burst (~0.2s each), exp backoff between bursts
    (2s, 4s, 8s, 16s). Max 5 bursts => 25 attempts total."""
    attempt = retry_state.attempt_number
    if attempt % 5 == 0:
        return 2 ** (attempt // 5)
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
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + size])
        if chunk:
            chunks.append(chunk)
        if i + size >= len(words):
            break
    return chunks


def _format_metadata_prefix(metadata: dict) -> str:
    if not metadata:
        return ""
    def clean(v):
        return str(v).replace("|", "/").replace("\n", " ").replace("\r", " ").strip()
    parts = [f"{k}: {clean(v)}" for k, v in metadata.items()]
    return "[" + " | ".join(parts) + "]"


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


def _embed_texts(texts: list[str]) -> np.ndarray:
    texts = [t[:MAX_QUERY_CHARS] if len(t) > MAX_QUERY_CHARS else t for t in texts]
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
        azure_endpoint=f"https://eag-qa.aexp.com/genai/microsoft/v1/models/{LLM_MODEL}/",
        api_key=get_token_from_env(LLM_MODEL),  # noqa: F821
        api_version="2024-10-21",
    )


@_api_retry
def _chat_complete(
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float = 0,
) -> str:
    resp = _build_chat_client().chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# ============================================================================
# Public module-level functions
# ============================================================================
def split_query(
    text: str,
    size: int = SUBQUERY_SIZE,
    overlap: int = SUBQUERY_OVERLAP,
) -> list[str]:
    """Split long raw input into fixed word-window sub-queries.

    Use when the caller has a big, messy prompt (call transcript, pasted
    email, concatenated chat history) and wants to feed it into multi-query
    retrieval without an LLM rewrite step. Pass the returned list to
    `VectorStore.retrieve(...)`.
    """
    return _split_text(text, size=size, overlap=overlap)


class QueryEngine:
    """Wraps a VectorStore with a bound system_message to generate grounded answers.

    Inspect/filter hits between store.retrieve() and engine.synthesize() when you
    need audit logging, quality checks, or multi-step flows.
    """

    def __init__(self, store: "VectorStore", system_message: Optional[str] = None):
        self.store = store
        self.system_message = system_message or DEFAULT_SYSTEM

    def synthesize(
        self,
        query: str,
        hits: list[dict],
        extra_context: Optional[str] = None,
    ) -> dict:
        context = "\n\n---\n\n".join(
            f"[Source: {h['source']}]\n{h['text']}" for h in hits
        )
        full_system = self.system_message + f"\n\nCONTEXT:\n{context}"
        user = f"{extra_context}\n\n{query}" if extra_context else query
        answer = _chat_complete(
            [
                {"role": "system", "content": full_system},
                {"role": "user", "content": user},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return {
            "answer": answer,
            "sources": [h["source"] for h in hits],
            "chunks": hits,
            "search_query_used": hits[0].get("search_query") if hits else None,
        }


# ============================================================================
# VectorStore
# ============================================================================
class VectorStore:
    """Indexes files and retrieves relevant chunks with hybrid search."""

    def __init__(self):
        self.chunks: list[str] = []
        self.metadata: list[dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None

    # -------------------- Indexing --------------------
    def add_file(self, path: str, metadata: Optional[dict] = None) -> int:
        """Parse, chunk, embed, and index a single file. Returns chunk count added.

        Metadata is prefixed into each chunk's text (so the LLM sees it) AND
        stored structurally (so code can filter on it). The 'source' key is
        added automatically from the file stem.
        """
        new_chunks = _split_text(_parse_file(path))
        if not new_chunks:
            return 0
        meta = {**(metadata or {}), "source": Path(path).stem}
        prefix = _format_metadata_prefix(meta)
        prefixed = [f"{prefix}\n{c}" for c in new_chunks] if prefix else new_chunks
        embs = _embed_texts(prefixed)
        self.chunks.extend(prefixed)
        self.metadata.extend([dict(meta) for _ in prefixed])
        self.embeddings = embs if self.embeddings is None else np.vstack([self.embeddings, embs])
        self.bm25 = BM25Okapi([c.lower().split() for c in self.chunks])
        return len(prefixed)

    # -------------------- Persistence --------------------
    def save(self, path: str) -> None:
        """Persist the index as a single Parquet file (inspectable via pandas)."""
        df = pd.DataFrame({
            "chunk": self.chunks,
            "metadata": [json.dumps(m) for m in self.metadata],
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
            store.metadata = [json.loads(m) for m in df["metadata"]]
            store.embeddings = np.array(df["embedding"].tolist(), dtype=np.float32)
            store.bm25 = BM25Okapi([c.lower().split() for c in store.chunks])
        return store

    # -------------------- Retrieval primitives --------------------
    def _dense_retrieve(self, query: str, k: int) -> list[int]:
        q = _embed_texts([query])[0]
        sims = self.embeddings @ q
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

    def _hybrid_search(
        self, query: str, k: int, allowed_idx: Optional[set] = None,
    ) -> list[tuple[int, float]]:
        pull_k = k * 3 if allowed_idx else k
        dense = self._dense_retrieve(query, pull_k)
        sparse = self._sparse_retrieve(query, pull_k)
        if allowed_idx is not None:
            dense = [i for i in dense if i in allowed_idx][:k]
            sparse = [i for i in sparse if i in allowed_idx][:k]
        return self._rrf([dense, sparse])[:k]

    def _llm_rerank(
        self, query: str, candidates: list[int], final_k: int,
    ) -> list[tuple[int, float]]:
        if not candidates:
            return []
        passages = "\n\n".join(
            f"[{i}] {self.chunks[idx][:800]}" for i, idx in enumerate(candidates)
        )
        content = _chat_complete(
            [
                {"role": "system", "content":
                    "Score each numbered passage's relevance to the query on a 0-10 scale. "
                    "Output ONLY a JSON object like: {\"0\": 8, \"1\": 3, \"2\": 9}. No other text."},
                {"role": "user", "content": f"QUERY: {query}\n\nPASSAGES:\n{passages}"},
            ],
            max_tokens=300,
            temperature=0,
        )
        try:
            match = re.search(r'\{[^}]+\}', content)
            scores_dict = json.loads(match.group()) if match else {}
            scored = [(candidates[i], float(scores_dict.get(str(i), 0))) for i in range(len(candidates))]
        except (json.JSONDecodeError, AttributeError, ValueError):
            scored = [(idx, len(candidates) - rank) for rank, idx in enumerate(candidates)]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:final_k]

    @staticmethod
    def _matches_filter(chunk_meta: dict, filter_spec: dict) -> bool:
        """AND across keys, OR within list values. String-equality comparison."""
        for key, expected in filter_spec.items():
            if key not in chunk_meta:
                return False
            actual = chunk_meta[key]
            if isinstance(expected, (list, tuple, set)):
                if str(actual) not in {str(v) for v in expected}:
                    return False
            elif str(actual) != str(expected):
                return False
        return True

    def _filter_indices(self, filters: Optional[dict]) -> Optional[set]:
        if not filters:
            return None
        return {
            i for i, m in enumerate(self.metadata)
            if self._matches_filter(m, filters)
        }

    # -------------------- Retrieval --------------------
    def retrieve(
        self,
        query: Union[str, list[str]],
        top_k: int = 10,
        final_k: int = 5,
        rerank: bool = False,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Retrieve relevant chunks for a single query or a list of queries.

        - Single query: hybrid search + optional LLM rerank.
        - List of queries: hybrid search per query, merged across queries
          with Reciprocal Rank Fusion (RRF). No LLM call on this path.

        If `filters` is set, retrieval is restricted to matching chunks
        (AND across keys; list values are OR'd).
        """
        if not self.chunks:
            return []
        if final_k > top_k:
            raise ValueError(f"final_k ({final_k}) must be <= top_k ({top_k})")

        allowed_idx = self._filter_indices(filters)
        if filters and not allowed_idx:
            return []

        def hit(i: int, score: float, search_query: str) -> dict:
            return {
                "text": self.chunks[i],
                "source": self.metadata[i].get("source", ""),
                "metadata": dict(self.metadata[i]),
                "score": float(score),
                "search_query": search_query,
            }

        if isinstance(query, (list, tuple)):
            rankings = [
                [i for i, _ in self._hybrid_search(q, top_k, allowed_idx)]
                for q in query
            ]
            fused = self._rrf(rankings)[:final_k]
            first_q: dict[int, str] = {}
            for q, ranking in zip(query, rankings):
                for i in ranking:
                    first_q.setdefault(i, q)
            return [hit(i, score, first_q[i]) for i, score in fused]

        fused = self._hybrid_search(query, top_k, allowed_idx)
        ranked = (
            self._llm_rerank(query, [i for i, _ in fused], final_k)
            if rerank else fused[:final_k]
        )
        return [hit(i, s, query) for i, s in ranked]
