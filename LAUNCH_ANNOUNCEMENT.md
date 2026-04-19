# `file_search` — minimal RAG Library for Our BU

## TL;DR

`file_search` is an internal Python library for grounded retrieval and
generation over local files. Index files, hybrid-retrieve relevant chunks
(semantic + BM25, fused via Reciprocal Rank Fusion), and synthesize answers
with a bound system message. Long queries are decomposed into overlapping
word-window sub-queries internally and merged with RRF — no extra API calls,
no new surface area.

The library is ~300 lines. Ships with a reference implementation (AmEx credit
card sales cue engine) other teams can copy.

---

## Public surface

```python
from file_search import VectorStore, QueryEngine
```

Two classes. That's it.

- `VectorStore` — `add_file`, `save`, `load`, `retrieve`.
- `QueryEngine` — `__init__(store, system_message=...)`, `synthesize(query, hits)`.

---

## Features

- Parses `.txt`, `.md`, `.pdf`, `.docx`.
- 600-word chunks with 300-word overlap (~800/400 tokens).
- API embeddings via `text-embedding-3-large`.
- Hybrid search: semantic (NumPy cosine) + keyword (BM25) via Reciprocal Rank
  Fusion.
- **Automatic query decomposition.** Long queries (more than `CHUNK_SIZE`
  words) are split internally into overlapping word-window sub-queries;
  per-sub-query hybrid rankings are RRF-merged. Short queries take the same
  path with one sub-query (degenerate RRF). No separate API — caller always
  passes a string.
- Separate `retrieve()` and `synthesize()` so production code can inspect,
  audit, or filter hits before generation.
- Uniform burst retry on every network call: 5 rapid attempts (~0.2 s apart)
  per burst, exponential backoff between bursts (2 / 4 / 8 / 16 s), up to 5
  bursts (25 attempts) — retries on any exception.
- Save/load via Parquet (inspectable with `pd.read_parquet(...)`).

---

## Design Choices & Rationale

### Why split retrieval from generation?

Every major RAG framework does this — LangChain has `Retriever` + `Chain`,
LlamaIndex has `Retriever` + `ResponseSynthesizer` + `QueryEngine`, Haystack
has pipeline nodes, and OpenAI exposes `vector_stores.search` separately from
`responses.create`. Production teams need to inspect chunks before generation
for audit, debugging, and compliance, and to reuse chunks across multiple LLM
calls. The class names follow LlamaIndex: `VectorStore` + `QueryEngine`, with
`retrieve()` + `synthesize()`.

### Why deterministic query splitting?

No LLM call means deterministic behavior, no latency, no cost, and no prompt
injection surface — raw customer input is never passed back to an LLM as
instructions. Reusing the same `CHUNK_SIZE`/`CHUNK_OVERLAP` as document
indexing keeps query and document granularity aligned, which is what hybrid
search expects.

### Why `final_k` can exceed `top_k`

With N sub-queries, the merged RRF pool can hold up to N × `top_k` unique
chunks. `final_k` is an upper bound on the returned list — when sub-queries
overlap heavily, the unique pool may be smaller than `final_k`, so fewer
hits are returned.

---

## Configurable Fields

### `QueryEngine` — answer-generation config

| Parameter | Default | Purpose |
|---|---|---|
| `system_message` | Grounded-RAG default | Your team's prompt (override) |

### `VectorStore.retrieve` — per-call retrieval config

| Parameter | Default | Purpose |
|---|---|---|
| `query` | — | A search string (long queries are split internally) |
| `top_k` | `10` | Candidates pulled from hybrid search per sub-query |
| `final_k` | `5` | Upper bound on the returned list (after RRF merge) |

Internal tuning knobs — `LLM_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`,
`EMBED_DIMS`, `EMBED_BATCH_SIZE` — are module-level constants in
`file_search.py`. Edit them there if needed.

---

## Dummy Usage Example

Q&A bot over internal policy documents:

```python
from file_search import VectorStore, QueryEngine

# 1. Index your files
store = VectorStore()
store.add_file("policies/remote_work.pdf")
store.add_file("policies/expense_policy.docx")
store.save("policy_index.parquet")

# 2. In your service — load once, reuse across queries
store = VectorStore.load("policy_index.parquet")
engine = QueryEngine(
    store,
    system_message=(
        "You are an HR policy assistant. Answer employee questions using ONLY "
        "the policy context below. Cite sources inline."
    ),
)

# 3. Per-query: split retrieval from generation for audit/inspection
def answer_question(question: str):
    hits = store.retrieve(question)
    return engine.synthesize(question, hits)

result = answer_question("Can I expense a taxi to a client dinner?")
print(result["answer"])
```

---

## Comparison with OpenAI's file_search

| Feature | OpenAI file_search | Our `file_search` |
|---|---|---|
| File upload & indexing | ✅ | ✅ |
| Chunking (800/400 tokens equiv) | ✅ | ✅ |
| Hybrid search (semantic + BM25) | ✅ | ✅ |
| Rank fusion (RRF) | ✅ | ✅ |
| Reranking | ✅ (built-in) | ❌ (removed) |
| Query rewriting | ✅ | ❌ (removed) |
| Query decomposition | ✅ (built-in) | ✅ (automatic, deterministic) |
| Metadata filtering | ✅ (attribute filters) | ❌ (removed) |
| Retrieval/generation split | ✅ | ✅ |
| Persistence | ✅ (cloud) | ✅ (local parquet) |
| Source citations | ✅ | ✅ |
| Customizable system prompt | ❌ (limited) | ✅ |
| Works behind our proxy | ❌ | ✅ |
| File format support | PDF, DOCX, TXT, MD, HTML, + | PDF, DOCX, TXT, MD |

### What's Still Missing vs. OpenAI

1. **OCR on scanned PDFs.** OpenAI handles scanned documents via OCR; we only
   extract embedded text. Workaround: run Tesseract upstream.
2. **Image parsing inside documents.** OpenAI parses charts and table images;
   we only extract text.
3. **Document-aware chunking.** OpenAI respects section boundaries; we chunk
   by word count.
4. **Parallel file ingestion.** OpenAI indexes in parallel server-side; ours
   is sequential.
5. **Structured file support (CSV, JSONL).** OpenAI parses with row awareness;
   we treat them as raw text.
6. **Cloud-managed vector store.** OpenAI is distributed; ours is a local
   parquet file. Fine for corpus sizes up to ~100K chunks.

---

## Getting Started

1. **Read the notebook** (`example_amex_card_assistant.ipynb`) — a real
   use-case end-to-end.
2. **Copy the pattern** for your own use case — mostly this means writing a
   good `system_message`.
3. **Ping us** with questions, bug reports, or feature requests.

Dependencies (most already available in our internal Python env):

```
pip install numpy pandas pyarrow rank-bm25 tenacity openai pypdf python-docx
```

---

## Roadmap

- OCR integration for scanned documents
- Async retrieval for higher-throughput services
- Swappable vector backend (FAISS) for larger corpora

If one of these is blocking your use case, let us know.

---

## Feedback

The library is small (~300 lines) and easy to modify. Keep the questions and
real-world integration requests coming.
