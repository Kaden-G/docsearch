#!/usr/bin/env python3
"""
DocSearch Flask Server
Provides web UI and API for searching documents.
"""

import time
import threading

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from pathlib import Path
from werkzeug.utils import secure_filename
import os
import sys

# Add scripts directory to path
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / 'scripts'))

from search import DocSearcher
from metrics import MetricsTracker
from extract import DocumentExtractor
from chunk import DocumentChunker
from embed import EmbeddingIndexer
from pipeline import AgenticPipeline

app = Flask(__name__)

# Initialize searcher and metrics tracker
index_dir = base_dir / 'data' / 'index'
raw_dir = base_dir / 'data' / 'raw'
searcher = None
metrics_tracker = None

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt'}

# Pipeline build state
_build_lock = threading.Lock()
_build_status = {'running': False, 'step': '', 'error': '', 'done': False, 'percent': 0}

# LLM/API credentials entered via the UI Settings drawer, kept in memory so the
# embedding API can reuse the SAME token, and so a build that runs before the
# searcher exists can still authenticate. Never written to disk.
_llm_settings = {}


def init_searcher():
    """Initialize the search engine and metrics tracker."""
    global searcher, metrics_tracker
    try:
        searcher = DocSearcher(str(index_dir))
        metrics_tracker = MetricsTracker()
        print(f"Search engine initialized with {len(searcher.metadata)} chunks")
        print("Metrics tracking enabled")
    except Exception as e:
        print(f"Error initializing searcher: {e}")
        print("Make sure to run extract.py, chunk.py, and embed.py first!")


@app.route('/')
def index():
    """Serve the main UI."""
    ui_file = Path(__file__).parent / 'ui.html'
    with open(ui_file, 'r') as f:
        return render_template_string(f.read())


def _agentic_response(query, result):
    """Map an agentic PipelineResult into the response shape the UI expects.

    Source cards reuse the same keys as the traditional path (doc_name, page,
    section, similarity, rank, text); we additionally expose `confidence`,
    `citations`, and a per-stage `trace` so the UI can render the badge and the
    'agent thinking' panel.
    """
    results = []
    for i, c in enumerate(result.chunks, 1):
        results.append({
            'rank': i,
            'similarity': c.similarity,
            'score': c.score,
            'chunk_id': c.chunk_id,
            'doc_name': c.doc_name,
            'page': c.page,
            'section': c.section,
            'text': c.text,
        })
    return {
        'query': query,
        'pipeline': 'agentic',
        'answer': result.answer,
        'confidence': result.confidence,
        'results': results,
        'count': len(results),
        'citations': [
            {'marker': c.marker, 'chunk_id': c.chunk_id,
             'doc_name': c.doc_name, 'page': c.page} for c in result.citations
        ],
        'trace': {
            'total_latency_ms': result.total_latency_ms,
            'total_llm_calls': result.usage.calls,
            'total_tokens': result.usage.total_tokens,
            'stages': [s.to_dict() for s in result.stages],
        },
    }


@app.route('/api/search', methods=['POST'])
def api_search():
    """Search API endpoint with metrics tracking."""
    if searcher is None:
        return jsonify({'error': 'Search engine not initialized'}), 500

    data = request.json
    if not data or not isinstance(data, dict):
        return jsonify({'error': 'Request body must be JSON'}), 400

    query = str(data.get('query', '')).strip()
    top_k = data.get('top_k', 5)
    doc_filter = data.get('document', None)
    generate_answer = data.get('generate_answer', False)
    pipeline_kind = str(data.get('pipeline', 'traditional')).lower().strip()

    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    if len(query) > 2000:
        return jsonify({'error': 'Query too long (max 2000 characters)'}), 400
    if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
        return jsonify({'error': 'top_k must be an integer between 1 and 100'}), 400

    try:
        start_time = time.time()

        # Use RAG pipeline if answer generation is requested
        if generate_answer and pipeline_kind == 'agentic':
            # Agentic v2: shares the same searcher (index, embeddings, LLM).
            print(f"[agentic] running query: {query!r}", flush=True)
            result = AgenticPipeline(searcher, top_k=top_k, verbose=True).run(query)
            print(f"[agentic] done in {result.total_latency_ms:.0f}ms, "
                  f"{result.usage.calls} LLM call(s), {result.usage.total_tokens} tokens", flush=True)
            response = _agentic_response(query, result)
            results = response['results']
        elif generate_answer:
            response = searcher.search_and_answer(query, top_k=top_k)
            results = response['results']
        else:
            # Otherwise, just return search results
            if doc_filter:
                results = searcher.search_by_document(query, doc_filter, top_k=top_k)
            else:
                results = searcher.search(query, top_k=top_k)

            response = {
                'query': query,
                'results': results,
                'count': len(results)
            }

        # Log metrics
        latency = time.time() - start_time
        if metrics_tracker:
            query_id = metrics_tracker.log_query(
                query=query,
                results=results,
                latency=latency,
                top_k=top_k,
                cache_hit=False,  # Could track this from searcher cache
                use_rerank=True,
                generate_answer=generate_answer
            )
            response['query_id'] = query_id  # For feedback collection

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents', methods=['GET'])
def api_documents():
    """Get list of indexed documents."""
    if searcher is None:
        return jsonify({'error': 'Search engine not initialized'}), 500

    try:
        docs = searcher.get_document_list()
        return jsonify({'documents': docs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Get index statistics."""
    if searcher is None:
        return jsonify({'error': 'Search engine not initialized'}), 500

    try:
        stats = searcher.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    status = 'ok' if searcher is not None else 'not_initialized'
    return jsonify({'status': status})


@app.route('/api/settings', methods=['GET'])
def api_settings_get():
    """Get current LLM settings."""
    if searcher is not None:
        return jsonify(searcher.get_llm_status())
    # No searcher yet — reflect any settings saved via POST before the build.
    if _llm_settings.get('provider'):
        return jsonify({
            'provider': _llm_settings.get('provider', 'none'),
            'model': _llm_settings.get('model', ''),
            'api_base': _llm_settings.get('api_base', ''),
            'has_key': bool(_llm_settings.get('api_key')),
        })
    return jsonify({'provider': 'none', 'model': '', 'api_base': '', 'has_key': False})


@app.route('/api/settings', methods=['POST'])
def api_settings_post():
    """Update LLM settings at runtime."""
    data = request.json
    if not data or not isinstance(data, dict):
        return jsonify({'error': 'Request body must be JSON'}), 400

    global _llm_settings, searcher, metrics_tracker

    provider = str(data.get('provider', '')).strip()
    api_key = str(data.get('apiKey', '')).strip()
    api_base = str(data.get('apiBase', '')).strip()
    model = str(data.get('model', '')).strip()
    ollama_model = str(data.get('ollamaModel', '')).strip()

    # Persist server-side so the embedding API reuses this token and a build that
    # runs before the index exists can authenticate. Kept in memory only.
    _llm_settings = {
        'provider': provider,
        'api_key': api_key,
        'api_base': api_base,
        'model': model,
        'ollama_model': ollama_model,
    }

    if searcher is not None:
        searcher.configure_llm(
            provider=provider,
            api_key=api_key,
            api_base=api_base,
            model=model,
            ollama_model=ollama_model
        )
        return jsonify({'status': 'ok', **searcher.get_llm_status()})

    # No live searcher yet. If an index already exists on disk, bring it up now
    # with the just-provided token: an API-embedding index needs a token even to
    # embed queries, so "start server -> paste token -> search" must work without
    # a rebuild or env vars.
    if (index_dir / 'faiss.index').exists():
        try:
            searcher = DocSearcher(str(index_dir), api_key=api_key, api_base=api_base)
            searcher.configure_llm(
                provider=provider,
                api_key=api_key,
                api_base=api_base,
                model=model,
                ollama_model=ollama_model,
            )
            if metrics_tracker is None:
                metrics_tracker = MetricsTracker()
            return jsonify({'status': 'ok', **searcher.get_llm_status()})
        except Exception as e:
            print(f'[settings] could not initialize searcher from existing index: {e}')
            return jsonify({
                'status': 'saved',
                'error': str(e),
                'provider': provider or 'none',
                'model': model,
                'api_base': api_base,
                'has_key': bool(api_key),
            })

    # No index on disk yet. Settings are saved and applied to both the LLM and
    # the embedding token when the index build runs.
    return jsonify({
        'status': 'saved',
        'provider': provider or 'none',
        'model': model,
        'api_base': api_base,
        'has_key': bool(api_key),
    })


@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """Collect user feedback for a query."""
    if metrics_tracker is None:
        return jsonify({'error': 'Metrics tracker not initialized'}), 500

    data = request.json
    if not data or not isinstance(data, dict):
        return jsonify({'error': 'Request body must be JSON'}), 400

    query_id = data.get('query_id')
    rating = data.get('rating')  # 1-5
    relevant_results = data.get('relevant_results', [])
    comments = str(data.get('comments', ''))[:1000]  # Cap comment length

    if not query_id or not isinstance(query_id, int):
        return jsonify({'error': 'query_id must be an integer'}), 400
    if not isinstance(rating, int) or rating < 1 or rating > 5:
        return jsonify({'error': 'rating must be an integer between 1 and 5'}), 400

    try:
        metrics_tracker.log_feedback(
            query_id=query_id,
            rating=rating,
            relevant_results=relevant_results,
            comments=comments
        )
        return jsonify({'status': 'success', 'message': 'Feedback recorded'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """Get system metrics."""
    if metrics_tracker is None:
        return jsonify({'error': 'Metrics tracker not initialized'}), 500

    try:
        days = min(max(int(request.args.get('days', 7)), 1), 365)
        metrics = {
            'query_stats': metrics_tracker.get_query_stats(days),
            'top_queries': metrics_tracker.get_top_queries(10),
            'feedback_stats': metrics_tracker.get_feedback_stats()
        }
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/library', methods=['GET'])
def api_library_list():
    """List files currently in the input directory (data/raw/)."""
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for f in sorted(raw_dir.iterdir()):
            if f.is_file() and not f.name.startswith('~$') and f.suffix.lower() in ALLOWED_EXTENSIONS:
                files.append({
                    'name': f.name,
                    'size': f.stat().st_size,
                    'ext': f.suffix.lower()
                })
        return jsonify({'files': files, 'count': len(files)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/library', methods=['POST'])
def api_library_upload():
    """Upload files to the input directory."""
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)

        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        uploaded = []
        skipped = []
        for f in request.files.getlist('files'):
            if not f.filename:
                continue
            fname = secure_filename(f.filename)
            ext = Path(fname).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                skipped.append({'name': f.filename, 'reason': f'Unsupported type ({ext}). Use .pdf, .docx, or .txt'})
                continue
            dest = raw_dir / fname
            f.save(str(dest))
            uploaded.append({'name': fname, 'size': dest.stat().st_size})

        return jsonify({
            'uploaded': uploaded,
            'skipped': skipped,
            'message': f'{len(uploaded)} file(s) added to the input directory'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/library/build', methods=['POST'])
def api_library_build():
    """Run the indexing pipeline and re-initialize the searcher."""
    global _build_status

    if _build_status['running']:
        return jsonify({'error': 'Build already in progress'}), 409

    data = request.get_json(silent=True) or {}
    smart = str(data.get('mode', '')).strip().lower() == 'smart'

    label = 'Smart Indexing' if smart else 'Starting'
    _build_status = {'running': True, 'step': f'{label}...', 'error': '', 'done': False, 'percent': 0}
    thread = threading.Thread(target=_run_pipeline_thread, args=(smart,), daemon=True)
    thread.start()
    return jsonify({'message': 'Build started', 'mode': 'smart' if smart else 'standard'})


@app.route('/api/library/build/status', methods=['GET'])
def api_library_build_status():
    """Get the current build status."""
    return jsonify(_build_status)


def _make_enrichment_llm():
    """Build an OpenAI-compatible chat caller from saved settings for build-time
    enrichment, or None if no usable provider/token is configured."""
    provider = (_llm_settings.get('provider') or '').strip().lower()
    api_key = _llm_settings.get('api_key') or os.environ.get('DOCSEARCH_API_KEY', '')
    api_base = _llm_settings.get('api_base') or os.environ.get('DOCSEARCH_API_BASE', '')
    model = _llm_settings.get('model') or os.environ.get('DOCSEARCH_LLM_MODEL', '')
    if provider not in ('openai',) or not api_key:
        return None
    try:
        from enrich import make_openai_llm
        return make_openai_llm(api_key, api_base, model)
    except Exception as e:
        print(f'[smart-index] enrichment LLM init failed: {e}')
        return None


def _run_pipeline_thread(smart=False):
    """Run the full indexing pipeline in a background thread.

    When smart=True, an LLM enrichment stage (summaries + synthetic questions,
    embedded as contextual text) runs between chunking and embedding — the
    "Smart Indexing" path: a heavier one-time build that lifts retrieval recall.
    """
    global searcher, metrics_tracker, _build_status
    import traceback

    processed_dir = base_dir / 'data' / 'processed'

    try:
        # Step 1: Extract
        _build_status['step'] = 'Extracting text from documents...'
        _build_status['percent'] = 5
        extractor = DocumentExtractor(str(raw_dir), str(processed_dir))
        results = extractor.extract_all()
        _build_status['step'] = f'Extracted {len(results)} documents. Chunking...'
        _build_status['percent'] = 10

        # Step 2: Chunk
        chunker = DocumentChunker(str(processed_dir), chunk_size=800, overlap=200)
        chunks = chunker.process_all()
        _build_status['percent'] = 15

        # Step 2.5 (Smart Indexing only): build-time LLM enrichment. Embedding
        # then maps into 60-90%; otherwise it takes the whole 15-90% band.
        embed_base, embed_range = 15, 75
        if smart:
            llm = _make_enrichment_llm()
            if llm is None:
                raise RuntimeError(
                    'Smart Indexing needs an OpenAI-compatible API token. Open '
                    'Settings, save your provider + token, then retry.'
                )
            from enrich import ChunkEnricher

            def on_enrich_progress(done, total):
                frac = (done / total) if total else 1.0
                _build_status['step'] = f'Smart Indexing: enriching {done}/{total} chunks...'
                _build_status['percent'] = 15 + int(45 * frac)

            _build_status['step'] = f'Smart Indexing: enriching {len(chunks)} chunks...'
            ChunkEnricher(str(processed_dir), llm).run(progress_callback=on_enrich_progress)
            embed_base, embed_range = 60, 30

        _build_status['step'] = f'{len(chunks)} chunks ready. Building embeddings...'

        # Step 3: Embed & Index (with progress callback). Reuse the token the
        # user entered for the LLM (same API gateway) for the embedding API.
        indexer = EmbeddingIndexer(
            str(processed_dir), str(index_dir),
            api_key=_llm_settings.get('api_key') or None,
            api_base=_llm_settings.get('api_base') or None,
        )

        def on_embed_progress(done, total):
            frac = (done / total) if total else 1.0
            _build_status['step'] = f'Embedding chunks: {done}/{total}...'
            _build_status['percent'] = embed_base + int(embed_range * frac)

        indexer.build_index(progress_callback=on_embed_progress)

        # Save source manifest
        current_files = {}
        for f in extractor.get_raw_files():
            stat = f.stat()
            current_files[f.name] = {'mtime': stat.st_mtime, 'size': stat.st_size}
        indexer.save_source_manifest(current_files)

        # Re-initialize searcher
        _build_status['step'] = 'Initializing search engine...'
        _build_status['percent'] = 95
        searcher = DocSearcher(str(index_dir))
        # Apply the user's saved settings so the LLM and the embedding API share
        # the same token without requiring a separate request.
        if _llm_settings.get('provider'):
            searcher.configure_llm(
                provider=_llm_settings.get('provider', ''),
                api_key=_llm_settings.get('api_key', ''),
                api_base=_llm_settings.get('api_base', ''),
                model=_llm_settings.get('model', ''),
                ollama_model=_llm_settings.get('ollama_model', ''),
            )
        if metrics_tracker is None:
            metrics_tracker = MetricsTracker()

        _build_status['step'] = 'Complete'
        _build_status['percent'] = 100
        _build_status['done'] = True
        _build_status['running'] = False
        print('Pipeline complete — searcher re-initialized.')

    except Exception as e:
        traceback.print_exc()
        _build_status['error'] = str(e)
        _build_status['step'] = f'Failed: {e}'
        _build_status['done'] = True
        _build_status['running'] = False
        print(f'Pipeline error: {e}')


@app.route('/api/library/<filename>', methods=['DELETE'])
def api_library_delete(filename):
    """Remove a file from the input directory."""
    try:
        fname = secure_filename(filename)
        target = raw_dir / fname
        if not target.exists():
            return jsonify({'error': 'File not found'}), 404
        target.unlink()
        return jsonify({'message': f'{fname} removed from the input directory'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    """Run the Flask server."""
    init_searcher()

    print("\n" + "="*60)
    print("DocSearch — Document Search")
    print("="*60)
    print(f"Access the UI at: http://127.0.0.1:5000")
    print(f"API endpoint: http://127.0.0.1:5000/api/search")
    print("="*60 + "\n")

    # Run server (localhost only for security)
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,  # Disable debug in production
        threaded=True
    )


if __name__ == '__main__':
    main()
