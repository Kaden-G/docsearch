# docsearch

Semantic search over your own documents. Point it at a folder of PDFs, Word
docs, or plain text and query them in natural language — hybrid retrieval
(FAISS + BM25 + cross-encoder re-rank) plus optional LLM answer generation
when you want cited answers instead of just chunks.

The default install is **search-only** and runs entirely on your machine.
LLM answer generation is opt-in — plug in Ollama for a fully local RAG
stack, or an OpenAI-compatible / Anthropic API when you want a hosted
model.

---

## Features

- **Hybrid retrieval** — FAISS dense vectors + BM25 keyword scoring, fused
  via RRF, then re-ranked by a cross-encoder for precision.
- **Local by default** — `BAAI/bge-base-en-v1.5` embedder on CPU; no
  network calls needed to build the index or run a query.
- **Multi-format ingest** — PDF, DOCX, and plain text.
- **Query cache** — LRU cache for instant repeats.
- **Optional LLM answers** — Ollama, any OpenAI-compatible API, or
  Anthropic. Off unless you enable it.
- **Optional agentic pipeline** — plan → retrieve → assemble → answer →
  verify, with per-claim citation checking. Off unless you enable it.
- **Optional Smart Indexing** — build-time LLM enrichment (summaries +
  synthetic questions per chunk) for higher first-pass recall.
- **Localhost-only web UI** — Flask, binds to `127.0.0.1`. API tokens are
  kept in memory and never written to disk.

---

## Install

```bash
git clone https://github.com/Kaden-G/docsearch.git
cd docsearch
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick start (search-only, fully local)

```bash
# 1. Drop documents in
cp /path/to/your/*.pdf data/raw/

# 2. Build the index
python scripts/extract.py
python scripts/chunk.py
python scripts/embed.py

# 3. Serve
python app/server.py
```

Open <http://127.0.0.1:5000> and search. Or use the CLI:

```bash
python scripts/search.py
```

The first embed run downloads the SentenceTransformer (~440 MB) and the
cross-encoder (~80 MB) once, then everything runs offline.

---

## API

```bash
# Retrieval only
curl -X POST http://127.0.0.1:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how do I rotate the signing key?", "top_k": 5}'

# With LLM answer (requires a configured provider — see below)
curl -X POST http://127.0.0.1:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "top_k": 5, "generate_answer": true}'

# Stats
curl http://127.0.0.1:5000/api/stats
```

The response includes `results` (chunks with similarity + rerank scores,
doc name, page, section) and, if you asked for one, an `answer` string
with footnote citations back to the source chunks.

---

## Optional: enable LLM answers

Answer generation is off by default. To turn it on, pick a provider:

### Ollama (fully local)

```bash
# Once
ollama pull llama3.1:8b
```

In the web UI's **Settings** drawer, pick **Ollama** and save. Nothing
leaves your machine.

### OpenAI-compatible API (OpenAI, Azure, vLLM, LiteLLM, …)

```bash
export DOCSEARCH_API_KEY=sk-...
export DOCSEARCH_API_BASE=https://api.openai.com/v1   # or your endpoint
export DOCSEARCH_LLM_MODEL=gpt-4o-mini
```

Or paste them into **Settings** in the UI. Tokens are held in memory
only.

### Anthropic

```bash
export DOCSEARCH_LLM_PROVIDER=anthropic
export DOCSEARCH_API_KEY=sk-ant-...
export DOCSEARCH_LLM_MODEL=claude-sonnet-4-20250514
```

When AI Answer is on, the query and the top matching chunks are sent to
whichever provider you picked. It never runs without your explicit
opt-in.

---

## Optional: hosted embeddings

The default embedder is the local `BAAI/bge-base-en-v1.5` model. To use a
hosted OpenAI-compatible embedding endpoint instead:

```bash
export DOCSEARCH_EMBED_PROVIDER=api
export DOCSEARCH_EMBED_MODEL=text-embedding-3-small
# reuses DOCSEARCH_API_BASE / DOCSEARCH_API_KEY by default,
# or set DOCSEARCH_EMBED_API_BASE / DOCSEARCH_EMBED_API_KEY explicitly
```

The index records the provider + model in `data/index/config.json` so
queries are always embedded in the same vector space as the documents —
switching providers requires a rebuild.

---

## Optional: Smart Indexing

Runs an LLM once at build time to attach a summary and a handful of
synthetic questions to each chunk, then embeds the enriched text. This
lifts first-pass recall so the query-time retriever can stop after one
pass. Requires an OpenAI-compatible API. Kick it off from the UI's
**Build** panel (**Smart Indexing** button) after saving your provider
token.

---

## Optional: agentic pipeline

There's a five-stage pipeline (plan → retrieve → assemble → answer →
verify) that decomposes complex questions, verifies each claim against
its cited chunk, and gates low-confidence answers. It costs several LLM
calls per query. Flip the **Agentic** toggle in the UI, or call the API
with `"pipeline": "agentic"`.

---

## Project layout

```
docsearch/
├── app/
│   ├── server.py           # Flask API + web UI host
│   └── ui.html             # Single-file web UI
├── scripts/
│   ├── extract.py          # PDF / DOCX / TXT → text
│   ├── chunk.py            # text → chunks
│   ├── enrich.py           # (optional) Smart Indexing
│   ├── embedder.py         # local + API embedder backends
│   ├── embed.py            # build FAISS index
│   ├── search.py           # DocSearcher: hybrid retrieval + optional LLM answer
│   ├── pipeline.py         # Traditional + agentic orchestrators
│   ├── agents/             # Planner, Retriever, Assembler, Answerer, Verifier
│   ├── tools.py            # Shared retrieval + LLM toolbox
│   ├── schemas.py          # Typed result / usage dataclasses
│   ├── metrics.py          # SQLite query + feedback log
│   ├── evaluate.py         # Evaluation harness
│   ├── compare.py          # Traditional vs agentic comparison
│   └── run_pipeline.py     # One-command pipeline (supports --incremental)
├── tests/                  # pytest
├── data/
│   ├── raw/                # your source files (git-ignored)
│   ├── processed/          # extracted text + chunks (auto)
│   └── index/              # FAISS + metadata (auto)
└── requirements.txt
```

---

## Configuration

Almost everything can be set from the UI's Settings drawer. Env vars for
scripted / headless setups:

| var                            | purpose                                                    |
| ------------------------------ | ---------------------------------------------------------- |
| `DOCSEARCH_EMBED_PROVIDER`     | `local` (default) or `api`                                 |
| `DOCSEARCH_EMBED_MODEL`        | embedder model name                                        |
| `DOCSEARCH_EMBED_API_BASE`     | embedding endpoint (falls back to `DOCSEARCH_API_BASE`)    |
| `DOCSEARCH_EMBED_API_KEY`      | embedding token (falls back to `DOCSEARCH_API_KEY`)        |
| `DOCSEARCH_LLM_PROVIDER`       | `openai`, `anthropic`, or `ollama`                         |
| `DOCSEARCH_LLM_MODEL`          | LLM model name                                             |
| `DOCSEARCH_API_BASE`           | LLM endpoint base URL                                      |
| `DOCSEARCH_API_KEY`            | LLM token                                                  |
| `DOCSEARCH_ENRICH_WORKERS`     | parallel workers for Smart Indexing (default 6)            |

Chunking parameters (`chunk_size`, `overlap`, cache size, whether
re-ranking is on) are constructor args on `DocumentChunker`,
`DocSearcher`, etc.

---

## Rebuild / incremental index

```bash
# Full rebuild
python scripts/run_pipeline.py

# Only process new files
python scripts/run_pipeline.py --incremental
```

---

## Tests

```bash
pip install pytest
pytest tests/
```

---

## License

Reference implementation — adapt as needed.
