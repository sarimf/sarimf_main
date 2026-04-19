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
hits = store.retrieve("annual fee", metadata_filter={"short_name": "Platinum"})

# Multiple cards (OR)
hits = store.retrieve("rewards", metadata_filter={"short_name": ["Platinum", "Gold"]})

# Multiple conditions (AND)
hits = store.retrieve(q, metadata_filter={"short_name": "Platinum", "category": "charge"})
```

This is critical for use cases where cross-contamination is a compliance risk (credit card products, regulated disclosures).

### 3. Split retrieval from generation

Following the standard pattern from LlamaIndex/LangChain/Haystack, retrieval and generation are now separate operations:

```python
# Pattern 1: Combined (simple Q&A)
result = assistant.ask(query)

# Pattern 2: Split (production — inspect/filter chunks before LLM)
hits = store.retrieve(query, metadata_filter={...})
audit_log.record(sources=[h["source"] for h in hits])
hits = [h for h in hits if h["score"] > 0.3]
result = assistant.synthesize(query, hits)

# Pattern 3: Retrieval only (custom generation)
hits = store.retrieve(query)
# use with any LLM, any prompt
```

### 4. Query decomposition (injection-safe)

For long or multi-topic input (call transcripts, complex queries), break input into focused sub-queries:

```python
# Safe even when input contains adversarial content
sub_queries = store.decompose_query(
    data=long_transcript,
    extraction_instructions="Extract customer concerns about AmEx products",
)
# Returns: ["Platinum annual fee", "Gold groceries rewards", ...]

# Run retrieval across all sub-queries and merge by vote count
hits = store.retrieve_multi(queries=sub_queries, per_query_k=3, final_k=10)
```

The decomposer wraps input in `<DATA>` tags and explicitly instructs the LLM to treat content as inert — safe against prompt injection from user-generated content.

### 5. Backward compatibility

v1 indexes (saved before v2) load seamlessly — the library fills in empty metadata dicts as a fallback. No migration required.

---

## Existing Features (unchanged from v1)

- Parses `.txt`, `.md`, `.pdf`, `.docx`
- 600-word chunks with 300-word overlap (~800/400 tokens)
- API embeddings via `text-embedding-3-large` with retry/backoff
- Hybrid search: semantic (NumPy cosine) + keyword (BM25) via Reciprocal Rank Fusion
- Optional query rewriting for long/messy user input
- Optional LLM-based reranking (batched)
- Save/load via pickle for index reuse

---

## Design Choices & Rationale

### Why split retrieval from generation?

Every major RAG framework does this — LangChain has `Retriever` + `Chain`, LlamaIndex has `Retriever` + `ResponseSynthesizer` + `QueryEngine`, Haystack has pipeline nodes, and OpenAI exposes `vector_stores.search` separately from `responses.create`. Production teams need to inspect chunks before generation for audit, debugging, compliance, and to reuse chunks across multiple LLM calls. v2 follows the LlamaIndex naming: `retrieve()` + `synthesize()` + `ask()` (convenience wrapper).

### Why metadata in the chunk text AND in structured storage?

Both are needed. Structured storage enables code-level filtering before retrieval. Text prefix enables the LLM to attribute facts correctly at generation time. Either alone is insufficient for strict isolation — LLMs can still leak cross-product information even when retrieval is filtered, and filtering alone doesn't help the LLM organize the chunks it receives.

### Why auto-attach `source` to metadata?

Teams forget to add it, and then debugging "which file did this chunk come from?" becomes annoying. Having it always present makes `metadata_filter={"source": "..."}` work universally.

### Why injection-safe decomposition?

Raw user input (call transcripts, customer emails) can contain imperative language that an LLM interprets as instructions ("ignore previous instructions," "output X"). When we pass this to the decomposer, we need strong separation. Wrapping in `<DATA>` tags, explicit role declaration, and explicit immunity clauses cut this risk dramatically.

### Why all this if OpenAI's file_search just works?

OpenAI's file_search requires the `/files` upload endpoint, which our current proxy doesn't reliably support. Rather than block on proxy fixes, we built a local replica using the endpoints we know work (chat completions + embeddings).

---

## Configurable Fields

### `FileSearch` — indexing-time config

| Parameter | Default | Purpose |
|---|---|---|
| `llm_model` | `"gpt-41"` | Model for rewrite/decompose/rerank calls |
| `embed_dims` | `256` | Embedding dimensions (256/1024/3072) |
| `chunk_size` | `600` words | Larger = more context per chunk |
| `chunk_overlap` | `300` words | Higher = better cross-boundary matches |
| `max_query_chars` | `30000` | Truncation guard against embedding API limits |
| `embed_batch_size` | `64` | Texts per embedding API call |

### `Assistant` — answer-generation config

| Parameter | Default | Purpose |
|---|---|---|
| `system_message` | Grounded-RAG default | Your team's prompt (override) |
| `context_template` | `"\n\nCONTEXT:\n{context}"` | How chunks are labeled in prompt |
| `temperature` | `0.2` | Lower for factual, higher for creative |
| `max_tokens` | `500` | Response length cap |
| `model` | `"gpt-41"` | Generation model |

### Per-call retrieval config

| Parameter | Default | Purpose |
|---|---|---|
| `top_k` | `10` | Candidates pulled from hybrid search |
| `final_k` | `5` | Returned after optional rerank |
| `rerank` | `False` | Enable LLM reranking (+1 call, better quality) |
| `rewrite` | `False` | Enable LLM query rewriting |
| `extraction_instructions` | `None` | Required if `rewrite=True` |
| `metadata_filter` | `None` | Restrict to chunks whose metadata matches |
| `extra_user_context` | `None` | (ask/synthesize only) Extra user message content |

---

## Dummy Usage Example

Q&A bot over internal policy documents with metadata filtering:

```python
from openai import AzureOpenAI
from file_search import FileSearch, Assistant

# 1. Build the LLM client
client = AzureOpenAI(
    azure_endpoint="https://eag-qa.aexp.com/genai/microsoft/v1/models/gpt-41/",
    api_key=get_token_from_env('gpt41'),
    api_version="2024-10-21",
)

# 2. Index your files with metadata
store = FileSearch(llm_client=client)
store.add_file("policies/remote_work.pdf",
               metadata={"policy_type": "HR", "region": "US", "effective_year": "2024"})
store.add_file("policies/remote_work_india.pdf",
               metadata={"policy_type": "HR", "region": "India", "effective_year": "2024"})
store.add_file("policies/expense_policy.docx",
               metadata={"policy_type": "Finance", "region": "Global"})
store.save("policy_index.pkl")

# 3. In your service — load once, reuse across queries
store = FileSearch.load("policy_index.pkl", llm_client=client)
assistant = Assistant(
    store=store,
    client=client,
    system_message=(
        "You are an HR policy assistant. Answer employee questions using ONLY "
        "the policy context below. Cite sources inline."
    ),
    temperature=0.1,
)

# 4. Per-query: split retrieval from generation for audit/inspection
def answer_question(question: str, user_region: str):
    # Restrict retrieval to policies relevant to the user's region
    hits = store.retrieve(
        question,
        metadata_filter={"region": [user_region, "Global"]},
    )
    
    # Log for audit (compliance trails, debugging)
    audit_log.record(
        question=question,
        region=user_region,
        sources=[h["source"] for h in hits],
    )
    
    # Generate the answer
    return assistant.synthesize(question, hits)

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
| Persistence | ✅ (cloud) | ✅ (local pickle) |
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
7. **Cloud-managed vector store.** OpenAI is distributed; ours is a local pickle file. Fine for corpus sizes up to ~100K chunks.

---

## Getting Started

1. **Read the notebook** (`example_amex_card_assistant.ipynb`) — real use case end-to-end showing metadata, filtering, split retrieval/generation, and decomposition
2. **Copy the pattern** for your own use case — mostly this means writing a good `system_message` and defining your metadata schema
3. **Ping us** with questions, bug reports, or feature requests

Dependencies (most already available in our internal Python env):

```
pip install numpy rank-bm25 tenacity openai pypdf python-docx
```

---

## Migrating from v1

Most v1 code works unchanged. The breaking considerations:

- **`add_file(path)` still works** — metadata is optional; auto-attaches `source`
- **`add_file(path, metadata={...})`** — new optional parameter
- **Chunks now have a metadata prefix in their text** — this affects embeddings (usually positively) and anywhere you inspect chunk text directly
- **v1 pickle indexes load in v2** — empty metadata dicts filled in automatically. No migration needed, but re-indexing gives you the benefits of the prefix.

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

This is v2. The library has been shaped heavily by team questions and real-world integration requests — keep them coming. The library is small (~700 lines) and easy to modify.
