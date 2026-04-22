# CLAUDE.md — Project Context

## Project Overview

`sarimf_main` is a minimal, self-contained Python RAG (Retrieval-Augmented Generation) library (~368 lines, `file_search.py`). It is designed for regulated business use cases (e.g., American Express credit card product Q&A) where groundedness, auditability, and compliance are critical. It is a pattern template, not a production framework.

## Repository Structure

```
sarimf_main/
├── file_search.py                     # Core library — two public classes
├── example_amex_card_assistant.ipynb  # Reference implementation (AmEx cards)
├── LAUNCH_ANNOUNCEMENT.md             # Detailed design rationale and API docs
└── README.md                          # Stub
```

## Core Public API (`file_search.py`)

### `VectorStore`
Hybrid dense+sparse index over chunked documents.

```python
store = VectorStore()
store.add_file(path, metadata={...})          # Index one file
store.add_files(paths, metadatas=[...])       # Bulk index (one BM25 rebuild)
hits = store.retrieve(query, top_k=10, top_n=5)  # Hybrid retrieval
store.save("index.parquet")                   # Persist
store = VectorStore.load("index.parquet")     # Load
```

### `QueryEngine`
Wraps `VectorStore` with grounded generation via Azure OpenAI chat.

```python
engine = QueryEngine(store, system_message="You are a card sales assistant...")
result = engine.synthesize(query, hits)       # Returns answer string
```

## Retrieval Architecture

- **Chunking:** 600-word chunks with 300-word overlap
- **Metadata:** Prefixed into chunk text at index time as `[key: value | ...]`
- **Hybrid search:**
  - Dense: Azure OpenAI `text-embedding-3-large` embeddings (256 dims)
  - Sparse: BM25Okapi via `rank-bm25`
  - Inner fusion per sub-query: Reciprocal Rank Fusion (RRF)
  - Outer fusion across sub-queries: CombSUM over L2-normalized cosines
- **Query decomposition:** Long queries split into overlapping 600-word sub-queries (no LLM call — deterministic, no prompt injection risk), capped at `MAX_SUB_QUERIES = 16`
- **Retry logic:** Burst retry (5 rapid attempts ~0.2s apart), exponential backoff between bursts, 25 max attempts via `tenacity`

## Configuration Constants (top of `file_search.py`)

| Constant | Default | Description |
|---|---|---|
| `LLM_MODEL` | `"gpt-4.1"` | Chat model name |
| `LLM_ENDPOINT_NAME` | `"gpt-41"` | Azure endpoint segment |
| `LLM_TOKEN_KEY` | `"gpt41"` | Key passed to `get_token_from_env()` |
| `EMBED_DIMS` | `256` | Embedding dimensions |
| `CHUNK_SIZE` | `600` | Words per chunk |
| `CHUNK_OVERLAP` | `300` | Words overlapping between chunks |
| `EMBED_BATCH_SIZE` | `64` | Chunks per embedding API call |
| `MAX_SUB_QUERIES` | `16` | Cap on query decomposition sub-queries |
| `OUTER_COSINE_THRESHOLD` | `0.3` | Min cosine score to include a chunk |

## Environment / Authentication

The library uses **Azure OpenAI** endpoints (internal to American Express):

- **Embeddings endpoint:** `https://eag-qa.aexp.com/genai/microsoft/v1/gcs_distribution_efficiency`
- **LLM endpoint:** `https://eag-qa.aexp.com/genai/microsoft/v1/models/{LLM_ENDPOINT_NAME}/`

Authentication is provided by `get_token_from_env(key)` — an internal utility from `safechain.utils` (not in this repo, imported with `# noqa: F821`). It must be available in the calling environment:

- `get_token_from_env('ada-3-large')` — for embeddings
- `get_token_from_env('gpt41')` — for LLM chat

**No `.env` files or secrets are stored in this repo.**

## Dependencies

No `requirements.txt` or `pyproject.toml`. Install manually:

```bash
pip install numpy pandas pyarrow rank-bm25 tenacity openai pypdf python-docx
```

## Supported Document Types

`.txt`, `.md`, `.pdf` (via `pypdf`), `.docx` (via `python-docx`)

## Key Design Decisions

- **No LLM in retrieval path:** Query decomposition is deterministic word-window splitting. This prevents prompt injection and keeps costs bounded.
- **Metadata as text prefix:** Metadata is baked into chunk text at index time. This means it participates in both BM25 and dense search naturally, with no separate filter path.
- **CombSUM over RRF for outer fusion:** Dense cosine similarity is more semantically meaningful across sub-queries than rank; RRF is used only within a single sub-query's dense+sparse fusion.
- **Parquet persistence:** Index is stored as a pandas DataFrame in Parquet — inspectable with standard tools.
- **Single-file library:** Intentionally kept as one importable file so teams can copy-paste it without dependency on a package registry.

## Git Workflow

- Default branch: `main`
- Remote: `origin` (internal proxy at `127.0.0.1:31467`)
- Feature branches follow the pattern `claude/<description>-<id>`
- No CI/CD configured

## What This Repo Is NOT

- Not a production service (no HTTP server, no async support, no CLI)
- Not a general-purpose RAG framework (no LangChain/LlamaIndex compatibility)
- No OCR, image parsing, CSV/JSONL support, or cloud storage backends
- No parallel ingestion
