# Launching `file_search` v2 — RAG Library for Our BU

## TL;DR

`file_search` is an internal Python library that replicates OpenAI's Assistants file_search tool using our existing GenAI endpoints. v2 adds metadata-based filtering, split retrieval/generation, and query decomposition — bringing it to parity with the patterns used by LlamaIndex, LangChain, and Haystack.

Teams can index files, retrieve relevant chunks with strict isolation controls, and build grounded assistants without setting up their own RAG stack. Ships with a reference implementation (AmEx credit card sales cue engine) that other teams can copy as a pattern.

---

## What's New in v2

### 1. Arbitrary metadata on chunks

Attach any key-value metadata to files at index time:

```python
store.add_file("platinum_card.pdf", metadata={
    "product": "The Platinum Card from American Express",
    "short_name": "Platinum",
    "category": "Consumer Charge Card",
})
```

Metadata appears in two places:

- **Prefixed into chunk text** — every chunk starts with `[product: Platinum | category: ...]` so the LLM sees which product each fact belongs to
- **Stored structurally** — code can filter on it without parsing text

The `source` key is added automatically to every chunk for traceability.

### 2. Metadata-based retrieval filtering

Restrict retrieval to matching chunks:

```python
# Single card
hits = store.retrieve("annual fee", filters={"short_name": "Platinum"})

# Multiple cards (OR)
hits = store.retrieve("rewards", filters={"short_name": ["Platinum", "Gold"]})

# Multiple conditions (AND)
hits = store.retrieve(q, filters={"short_name": "Platinum", "category": "charge"})
```

This is critical for use cases where cross-contamination is a compliance risk (credit card products, regulated disclosures).

### 3. Split retrieval from generation

Following the standard pattern from LlamaIndex/LangChain/Haystack, retrieval and generation are separate operations:

```python
# Production pattern: inspect/filter chunks before the LLM call
hits = store.retrieve(query, filters={...})
audit_log.record(sources=[h["source"] for h in hits])
hits = [h for h in hits if h["score"] > 0.3]
result = engine.synthesize(query, hits)

# Retrieval only — use with any LLM, any prompt
hits = store.retrieve(query)
```

### 4. Query decomposition (injection-safe)

For long or multi-topic input (call transcripts, complex queries), break input into focused sub-queries:

```python
# Safe even when input contains adversarial content
sub_queries = rewrite_queries(
    data=long_transcript,
    rewrite_instructions="Extract customer concerns about AmEx products",
    max_queries=20,
)
# Returns: ["Platinum annual fee", "Gold groceries rewards", ...]

# Pass a list to retrieve() — results are ranked by vote count across queries
hits = store.retrieve(sub_queries, top_k=3, final_k=10)
```

The decomposer wraps input in `<DATA>` tags and explicitly instructs the LLM to treat content as inert — safe against prompt injection from user-generated content.

---

## Existing Features

- Parses `.txt`, `.md`, `.pdf`, `.docx`
- 600-word chunks with 300-word overlap (~800/400 tokens)
- API embeddings via `text-embedding-3-large` with retry/backoff
- Hybrid search: semantic (NumPy cosine) + keyword (BM25) via Reciprocal Rank Fusion
- Optional query rewriting (enabled by passing `rewrite_instructions`)
- Optional LLM-based reranking (batched)
- Save/load via Parquet (inspectable with `pd.read_parquet(...)`)

---

## Design Choices & Rationale

### Why split retrieval from generation?

Every major RAG framework does this — LangChain has `Retriever` + `Chain`, LlamaIndex has `Retriever` + `ResponseSynthesizer` + `QueryEngine`, Haystack has pipeline nodes, and OpenAI exposes `vector_stores.search` separately from `responses.create`. Production teams need to inspect chunks before generation for audit, debugging, compliance, and to reuse chunks across multiple LLM calls. The class names follow LlamaIndex: `VectorStore` + `QueryEngine`, with `retrieve()` + `synthesize()`.

### Why metadata in the chunk text AND in structured storage?

Both are needed. Structured storage enables code-level filtering before retrieval. Text prefix enables the LLM to attribute facts correctly at generation time. Either alone is insufficient for strict isolation — LLMs can still leak cross-product information even when retrieval is filtered, and filtering alone doesn't help the LLM organize the chunks it receives.

### Why auto-attach `source` to metadata?

Teams forget to add it, and then debugging "which file did this chunk come from?" becomes annoying. Having it always present makes `filters={"source": "..."}` work universally.

### Why injection-safe decomposition?

Raw user input (call transcripts, customer emails) can contain imperative language that an LLM interprets as instructions ("ignore previous instructions," "output X"). When we pass this to the decomposer, we need strong separation. Wrapping in `<DATA>` tags, explicit role declaration, and explicit immunity clauses cut this risk dramatically.

### Why all this if OpenAI's file_search just works?

OpenAI's file_search requires the `/files` upload endpoint, which our current proxy doesn't reliably support. Rather than block on proxy fixes, we built a local replica using the endpoints we know work (chat completions + embeddings).

---

## Configurable Fields

### `QueryEngine` — answer-generation config

| Parameter | Default | Purpose |
|---|---|---|
| `system_message` | Grounded-RAG default | Your team's prompt (override) |

### Per-call retrieval config

| Parameter | Default | Purpose |
|---|---|---|
| `query` | — | A single string, or a list of queries (list ⇒ vote-count ranking) |
| `top_k` | `10` | Candidates pulled from hybrid search (per query when list) |
| `final_k` | `5` | Returned after optional rerank / vote merge |
| `rerank` | `False` | Enable LLM reranking (+1 call, better quality; single-query only) |
| `rewrite_instructions` | `None` | If truthy, rewrite the query via LLM before searching (single-query only) |
| `filters` | `None` | Restrict to chunks whose metadata matches |

Internal tuning knobs (LLM model name, chunk size/overlap, embedding dims, batch size, generation temperature and max tokens) are module-level constants in `file_search.py`. Edit them there if needed — no real caller has wanted to change them per-instance.

---

## Dummy Usage Example

Q&A bot over internal policy documents with metadata filtering:

```python
from file_search import VectorStore, QueryEngine

# 1. Index your files with metadata
store = VectorStore()
store.add_file("policies/remote_work.pdf",
               metadata={"policy_type": "HR", "region": "US", "effective_year": "2024"})
store.add_file("policies/remote_work_india.pdf",
               metadata={"policy_type": "HR", "region": "India", "effective_year": "2024"})
store.add_file("policies/expense_policy.docx",
               metadata={"policy_type": "Finance", "region": "Global"})
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
def answer_question(question: str, user_region: str):
    # Restrict retrieval to policies relevant to the user's region
    hits = store.retrieve(
        question,
        filters={"region": [user_region, "Global"]},
    )

    # Log for audit (compliance trails, debugging)
    audit_log.record(
        question=question,
        region=user_region,
        sources=[h["source"] for h in hits],
    )

    # Generate the answer
    return engine.synthesize(question, hits)

result = answer_question("Can I expense a taxi to a client dinner?", user_region="US")
print(result["answer"])
```

---

## Comparison with OpenAI's file_search

| Feature | OpenAI file_search | Our `file_search` v2 |
|---|---|---|
| File upload & indexing | ✅ | ✅ |
| Chunking (800/400 tokens equiv) | ✅ | ✅ |
| Hybrid search (semantic + BM25) | ✅ | ✅ |
| Rank fusion (RRF) | ✅ | ✅ |
| Reranking | ✅ (built-in) | ✅ (LLM-based, batched, optional) |
| Query rewriting | ✅ | ✅ (opt-in, with custom extraction) |
| **Query decomposition** | ✅ (built-in) | ✅ **(new in v2)** |
| **Metadata filtering** | ✅ (attribute filters) | ✅ **(new in v2)** |
| **Retrieval/generation split** | ✅ (`vector_stores.search`) | ✅ **(new in v2)** |
| Persistence | ✅ (cloud) | ✅ (local parquet) |
| Source citations | ✅ | ✅ |
| Customizable system prompt | ❌ (limited) | ✅ |
| Works behind our proxy | ❌ | ✅ |
| File format support | PDF, DOCX, TXT, MD, HTML, + | PDF, DOCX, TXT, MD |

### What's Still Missing vs. OpenAI

Being honest about remaining gaps:

1. **OCR on scanned PDFs.** OpenAI handles scanned documents via OCR; we only extract embedded text. Workaround: run Tesseract upstream.
2. **Image parsing inside documents.** OpenAI parses charts and table images; we only extract text.
3. **Document-aware chunking.** OpenAI respects section boundaries; we chunk by word count.
4. **Operator-based metadata filters.** OpenAI supports `gt`, `lt`, `in`, `ne`; we support equality and membership only. For date ranges, post-filter in your code.
5. **Parallel file ingestion.** OpenAI indexes in parallel server-side; ours is sequential.
6. **Structured file support (CSV, JSONL).** OpenAI parses with row awareness; we treat them as raw text.
7. **Cloud-managed vector store.** OpenAI is distributed; ours is a local parquet file. Fine for corpus sizes up to ~100K chunks.

---

## Getting Started

1. **Read the notebook** (`example_amex_card_assistant.ipynb`) — real use case end-to-end showing metadata, filtering, split retrieval/generation, and decomposition
2. **Copy the pattern** for your own use case — mostly this means writing a good `system_message` and defining your metadata schema
3. **Ping us** with questions, bug reports, or feature requests

Dependencies (most already available in our internal Python env):

```
pip install numpy pandas pyarrow rank-bm25 tenacity openai pypdf python-docx
```

---

## Roadmap

Based on early team conversations, these are candidates for future releases:

- Operator-based metadata filters (`gt`, `lt`, `in`, `ne`)
- OCR integration for scanned documents
- Async retrieval for higher-throughput services
- Swappable vector backend (FAISS) for larger corpora

If one of these is blocking your use case, let us know.

---

## Feedback

This is v2. The library has been shaped heavily by team questions and real-world integration requests — keep them coming. The library is small (~500 lines) and easy to modify.
