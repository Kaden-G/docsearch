#!/usr/bin/env python3
"""
Document Search Script
Semantic search over indexed chunks using FAISS.
"""

import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import CrossEncoder
import faiss
from rank_bm25 import BM25Okapi

from embedder import embedder_from_config

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic as _anthropic_mod
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class DocSearcher:
    """Semantic search over documents."""

    def __init__(
        self,
        index_dir: str,
        cache_size: int = 100,
        use_reranker: bool = True,
        ollama_model: str = 'llama3.1:8b',
        use_ollama: bool = True,
        llm_provider: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        llm_model: Optional[str] = None
    ):
        self.index_dir = Path(index_dir)

        # Load config
        config_file = self.index_dir / 'config.json'
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        # Load FAISS index
        index_file = self.index_dir / 'faiss.index'
        self.index = faiss.read_index(str(index_file))

        # Load metadata (JSON preferred, pickle fallback for old indexes)
        metadata_json = self.index_dir / 'metadata.json'
        metadata_pkl = self.index_dir / 'metadata.pkl'
        if metadata_json.exists():
            with open(metadata_json, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        elif metadata_pkl.exists():
            import pickle
            with open(metadata_pkl, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            raise FileNotFoundError('No metadata file found in index directory')

        # Build BM25 index for hybrid keyword search
        tokenized_corpus = [chunk['text'].lower().split() for chunk in self.metadata]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Embedding backend, reconstructed from the index config so queries are
        # embedded in the SAME space as the documents. This is either a local CPU
        # SentenceTransformer or a remote OpenAI-compatible API (token from env).
        self.embedder = embedder_from_config(self.config, api_key=api_key, api_base=api_base)

        # Load cross-encoder for re-ranking (optional but recommended)
        self.use_reranker = use_reranker
        if self.use_reranker:
            print("Loading cross-encoder for re-ranking...")
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', device='cpu')
        else:
            self.reranker = None

        # LLM configuration for answer generation
        # Priority: explicit llm_provider param > env vars > ollama fallback
        self.llm_provider = llm_provider or os.environ.get('DOCSEARCH_LLM_PROVIDER', '').lower()
        self._api_key = api_key or os.environ.get('DOCSEARCH_API_KEY', '')
        self._api_base = api_base or os.environ.get('DOCSEARCH_API_BASE', 'https://api.openai.com/v1')
        self._llm_model = llm_model or os.environ.get('DOCSEARCH_LLM_MODEL', '')
        self._openai_client = None
        self._anthropic_client = None

        if self.llm_provider == 'anthropic' and self._api_key:
            if not ANTHROPIC_AVAILABLE:
                print("Warning: anthropic provider requested but 'anthropic' package not installed.")
                print("  Install with: pip install anthropic")
                self.llm_provider = ''
            else:
                self._anthropic_client = _anthropic_mod.Anthropic(api_key=self._api_key)
                self._llm_model = self._llm_model or 'claude-sonnet-4-20250514'
                print(f"Anthropic LLM enabled (model: {self._llm_model})")
        elif self.llm_provider == 'openai' and self._api_key:
            if not OPENAI_AVAILABLE:
                print("Warning: openai provider requested but 'openai' package not installed.")
                print("  Install with: pip install openai")
                self.llm_provider = ''
            else:
                self._openai_client = OpenAI(
                    api_key=self._api_key,
                    base_url=self._api_base,
                    timeout=180.0,
                    max_retries=1
                )
                model_display = self._llm_model or 'default'
                print(f"OpenAI-compatible LLM enabled (base: {self._api_base}, model: {model_display})")
        elif self._api_key and not self.llm_provider:
            # Auto-detect: API key present but no explicit provider
            if OPENAI_AVAILABLE:
                self.llm_provider = 'openai'
                self._openai_client = OpenAI(
                    api_key=self._api_key,
                    base_url=self._api_base,
                    timeout=180.0,
                    max_retries=1
                )
                model_display = self._llm_model or 'default'
                print(f"OpenAI-compatible LLM auto-detected (model: {model_display})")

        # Ollama fallback
        self.use_ollama = use_ollama and OLLAMA_AVAILABLE and self.llm_provider not in ('openai', 'anthropic')
        self.ollama_model = ollama_model
        if self.use_ollama:
            self.llm_provider = self.llm_provider or 'ollama'
            print(f"Ollama integration enabled (model: {ollama_model})")
        elif not self.llm_provider:
            print("No LLM provider configured. Answer generation disabled.")
            print("  Options: set DOCSEARCH_API_KEY env var, or install ollama")

        # Initialize query cache for faster repeated queries (LRU, thread-safe)
        self.query_cache = OrderedDict()
        self.cache_size = cache_size
        self._cache_lock = threading.Lock()

        print(f"Loaded index with {len(self.metadata)} chunks")

    def configure_llm(self, provider: str, api_key: str = '', api_base: str = '', model: str = '', ollama_model: str = ''):
        """Reconfigure the LLM provider at runtime."""
        provider = (provider or '').lower().strip()

        # Reset clients
        self._openai_client = None
        self._anthropic_client = None
        self.use_ollama = False

        if provider == 'anthropic' and api_key:
            if not ANTHROPIC_AVAILABLE:
                print("Warning: anthropic provider requested but 'anthropic' package not installed.")
                return
            self.llm_provider = 'anthropic'
            self._api_key = api_key
            self._llm_model = model or 'claude-sonnet-4-20250514'
            self._anthropic_client = _anthropic_mod.Anthropic(api_key=self._api_key)
            print(f"LLM reconfigured: anthropic (model: {self._llm_model})")
        elif provider in ('openai',) and api_key:
            if not OPENAI_AVAILABLE:
                print("Warning: openai provider requested but 'openai' package not installed.")
                return
            self.llm_provider = provider  # Keep 'openai' or 'openai' for display
            self._api_key = api_key
            self._api_base = api_base or 'https://api.openai.com/v1'
            self._llm_model = model or ''
            self._openai_client = OpenAI(
                api_key=self._api_key,
                base_url=self._api_base,
                timeout=180.0,
                max_retries=1
            )
            self._refresh_api_embedder()
            print(f"LLM reconfigured: {provider} (base: {self._api_base}, model: {self._llm_model or 'default'})")
        elif provider == 'ollama':
            self.llm_provider = 'ollama'
            self._api_key = ''
            self.use_ollama = OLLAMA_AVAILABLE
            if ollama_model:
                self.ollama_model = ollama_model
            print(f"LLM reconfigured: ollama (model: {self.ollama_model})")
        else:
            self.llm_provider = ''
            self._api_key = ''
            print("LLM disabled")

    def _refresh_api_embedder(self):
        """Point the API embedder at the current LLM token.

        Embeddings and the LLM share the same API gateway, so when the user sets
        their token in Settings we reuse it for the embedding API rather than a
        separate env var. No-op for a local (offline) embedder.
        """
        if getattr(self, 'config', {}).get('embedding_provider') == 'api':
            self.embedder = embedder_from_config(self.config, api_key=self._api_key)

    def get_llm_status(self) -> dict:
        """Return current LLM configuration status."""
        if self.llm_provider == 'anthropic':
            return {'provider': 'anthropic', 'model': self._llm_model, 'api_base': '', 'has_key': True}
        elif self.llm_provider in ('openai',):
            return {'provider': self.llm_provider, 'model': self._llm_model, 'api_base': self._api_base, 'has_key': bool(self._api_key)}
        elif self.llm_provider == 'ollama':
            return {'provider': 'ollama', 'model': self.ollama_model, 'api_base': '', 'has_key': False}
        return {'provider': 'none', 'model': '', 'api_base': '', 'has_key': False}

    def _hyde_expand(self, query: str) -> Optional[str]:
        """Generate a hypothetical document snippet for query expansion (HyDE).
        Returns a short hypothetical answer, or None if no LLM is available."""
        if not self.llm_provider:
            return None

        prompt = (
            "Write a short, factual paragraph (3-5 sentences) that would answer this question "
            "as if it were an excerpt from a reference document. Do not include citations or headings. "
            f"Just write the content.\n\nQuestion: {query}"
        )

        try:
            if self.llm_provider in ('openai',) and self._openai_client:
                resp = self._openai_client.chat.completions.create(
                    model=self._llm_model or 'gpt-4o-mini',
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=200, temperature=0.0
                )
                return resp.choices[0].message.content
            elif self.llm_provider == 'anthropic' and self._anthropic_client:
                resp = self._anthropic_client.messages.create(
                    model=self._llm_model, max_tokens=200,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.0
                )
                return resp.content[0].text
            elif self.llm_provider == 'ollama' and self.use_ollama:
                resp = ollama.chat(
                    model=self.ollama_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.0, 'num_predict': 200}
                )
                return resp['message']['content']
        except Exception as e:
            print(f"HyDE expansion failed (non-fatal): {e}")
        return None

    def search(self, query: str, top_k: int = 5, rerank: bool = True,
               use_hyde: bool = True) -> List[Dict]:
        """
        Search for relevant chunks.

        Args:
            query: Natural language query
            top_k: Number of results to return
            rerank: Whether to use cross-encoder re-ranking (default: True)
            use_hyde: Whether to expand the query with a hypothetical answer via
                an LLM call (default: True). The agentic Retriever passes False
                because it does its own query reformulation — so query-time HyDE
                would be a redundant LLM round-trip per search.

        Returns:
            List of matching chunks with metadata
        """
        # Check cache first (LRU: promote to most-recently-used on hit)
        cache_key = f"{query}|{top_k}|{rerank}|{use_hyde}"
        with self._cache_lock:
            if cache_key in self.query_cache:
                self.query_cache.move_to_end(cache_key)
                return self.query_cache[cache_key]

        # HyDE: if enabled and an LLM is available, expand query with a hypothetical answer
        hyde_text = self._hyde_expand(query) if use_hyde else None
        if hyde_text:
            # Embed both original query and hypothetical answer, then average
            embeddings = self.embedder.encode([query, hyde_text], is_query=True)
            query_embedding = np.mean(embeddings, axis=0, keepdims=True)
        else:
            query_embedding = self.embedder.encode([query], is_query=True)

        # Normalize query embedding for cosine similarity
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)

        # --- Semantic retrieval (FAISS) ---
        retrieve_k = min(top_k * 4, len(self.metadata)) if (rerank and self.use_reranker) else top_k
        distances, indices = self.index.search(query_embedding, retrieve_k)

        faiss_ranks = {}  # idx -> rank (1-based)
        for rank, idx in enumerate(indices[0], 1):
            if idx != -1:
                faiss_ranks[int(idx)] = rank

        # --- Keyword retrieval (BM25) ---
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top = np.argsort(bm25_scores)[::-1][:retrieve_k]
        bm25_ranks = {}  # idx -> rank (1-based)
        for rank, idx in enumerate(bm25_top, 1):
            if bm25_scores[idx] > 0:
                bm25_ranks[int(idx)] = rank

        # --- Reciprocal Rank Fusion (RRF) ---
        k_rrf = 60  # Standard RRF constant
        all_indices = set(faiss_ranks.keys()) | set(bm25_ranks.keys())
        rrf_scores = {}
        for idx in all_indices:
            score = 0.0
            if idx in faiss_ranks:
                score += 1.0 / (k_rrf + faiss_ranks[idx])
            if idx in bm25_ranks:
                score += 1.0 / (k_rrf + bm25_ranks[idx])
            rrf_scores[idx] = score

        # Sort by RRF score and take top candidates
        fused_top = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:retrieve_k]

        # Prepare results
        results = []
        for rank, idx in enumerate(fused_top, 1):
            chunk = self.metadata[idx]
            # Use FAISS distance for similarity display if available, else estimate
            faiss_dist = float(distances[0][faiss_ranks[idx] - 1]) if idx in faiss_ranks else 0.0
            result = {
                'rank': rank,
                'score': rrf_scores[idx],
                'similarity': self._distance_to_similarity(faiss_dist),
                'chunk_id': chunk['chunk_id'],
                'doc_name': chunk['doc_name'],
                'page': chunk['page'],
                'section': chunk['section'],
                'text': chunk['text']
            }
            results.append(result)

        # Re-rank with cross-encoder if enabled
        if rerank and self.use_reranker and len(results) > 0:
            results = self._rerank_results(query, results)[:top_k]
        else:
            results = results[:top_k]

        # Update ranks after re-ranking
        for i, result in enumerate(results):
            result['rank'] = i + 1

        # Add to cache (LRU eviction: remove least-recently-used)
        with self._cache_lock:
            if len(self.query_cache) >= self.cache_size:
                self.query_cache.popitem(last=False)
            self.query_cache[cache_key] = results

        return results

    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Re-rank results using cross-encoder for better relevance.

        Args:
            query: User query
            results: Initial retrieval results

        Returns:
            Re-ranked results
        """
        # Prepare pairs for cross-encoder
        pairs = [(query, result['text']) for result in results]

        # Get cross-encoder scores
        rerank_scores = self.reranker.predict(pairs)

        # Add rerank scores to results
        for result, score in zip(results, rerank_scores):
            result['rerank_score'] = float(score)

        # Sort by rerank score (descending)
        reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)

        return reranked

    def _distance_to_similarity(self, score: float) -> float:
        """Convert cosine similarity score (0-1)."""
        # For cosine similarity (IndexFlatIP), score is already similarity (0-1)
        # Higher is better, 1 is perfect match
        return max(0.0, min(1.0, score))

    def search_by_document(self, query: str, doc_name: str, top_k: int = 5) -> List[Dict]:
        """Search within a specific document."""
        # Calculate how many chunks belong to this doc to set a smart retrieval budget
        total_chunks = len(self.metadata)
        doc_chunks = sum(1 for c in self.metadata if c['doc_name'] == doc_name)
        if doc_chunks == 0:
            return []

        # Retrieve enough from FAISS to likely include top_k from target doc
        # Ratio-based: if doc is 10% of corpus, retrieve 10x to compensate
        ratio = doc_chunks / total_chunks if total_chunks > 0 else 1
        retrieve_k = min(total_chunks, max(top_k * 3, int(top_k / ratio)))

        results = self.search(query, top_k=retrieve_k)

        # Filter by document and re-rank
        filtered = [r for r in results if r['doc_name'] == doc_name]
        for i, result in enumerate(filtered):
            result['rank'] = i + 1

        return filtered[:top_k]

    def get_document_list(self) -> List[str]:
        """Get list of all indexed documents."""
        docs = set(chunk['doc_name'] for chunk in self.metadata)
        return sorted(list(docs))

    def get_stats(self) -> Dict:
        """Get index statistics."""
        docs = self.get_document_list()
        return {
            'total_chunks': len(self.metadata),
            'total_documents': len(docs),
            'documents': docs,
            'embedding_dimension': self.config['embedding_dim'],
            'index_type': self.config['index_type']
        }

    def _build_prompt(self, query: str, results: List[Dict], max_context_chunks: int = 8):
        """Build system and user prompts for answer generation."""
        context_chunks = results[:max_context_chunks]
        context = "\n\n".join([
            f"[Source: {chunk['doc_name']}, Page {chunk['page']}, Section: {chunk.get('section', 'N/A')}]\n{chunk['text']}"
            for chunk in context_chunks
        ])

        system_prompt = """You answer questions using ONLY the provided document excerpts.

Formatting requirements:
- Respond in **Markdown** format
- Use headings (##), bold, numbered lists, and bullet points for clarity
- For procedural questions, provide complete **step-by-step instructions** with every detail needed to follow along
- Include ALL relevant details from the source material — do not summarize away important specifics

Citation requirements:
- Use numbered footnotes for citations: place a superscript marker like [^1] in the text after the relevant claim or instruction
- At the END of your response, include a "References" section with each footnote, e.g.:
  [^1]: handbook, Page 27
  [^2]: quickstart, Page 3
- Reuse the same footnote number when citing the same source page again
- Every factual claim or fact-derived step MUST have at least one footnote

Other guidelines:
- If the answer is not in the context, say so clearly
- Preserve exact values, commands, paths, and configuration details when present in the source"""

        user_prompt = f"""Context from documents:
{context}

Question: {query}

Answer the question thoroughly with citations to the source documents."""

        return system_prompt, user_prompt

    def _generate_anthropic(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Generate answer via Anthropic API."""
        try:
            response = self._anthropic_client.messages.create(
                model=self._llm_model,
                max_tokens=2000,
                system=system_prompt,
                messages=[
                    {'role': 'user', 'content': user_prompt}
                ],
                temperature=0.3
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error generating answer with Anthropic: {e}")
            return f"[LLM Error] {e}"

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Generate answer via OpenAI-compatible API."""
        try:
            kwargs = {
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                'temperature': 0.3,
                'max_tokens': 2000
            }
            if self._llm_model:
                kwargs['model'] = self._llm_model
            else:
                kwargs['model'] = 'gpt-4o-mini'

            response = self._openai_client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating answer with OpenAI API: {e}")
            print(f"  → base_url: {self._api_base}")
            print(f"  → model: {self._llm_model}")
            print(f"  → Full target: {self._api_base}/chat/completions")
            return f"[LLM Error] {e}"

    def _generate_ollama(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Generate answer via local Ollama."""
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'temperature': 0.3,
                    'num_predict': 2000
                }
            )
            return response['message']['content']
        except Exception as e:
            print(f"Error generating answer with Ollama: {e}")
            return f"[LLM Error] {e}"

    def generate_answer(
        self,
        query: str,
        results: List[Dict],
        max_context_chunks: int = 5
    ) -> Optional[str]:
        """
        Generate an answer using the configured LLM provider.

        Args:
            query: User question
            results: Search results (chunks)
            max_context_chunks: Maximum number of chunks to use as context

        Returns:
            Generated answer or None if no LLM provider is available
        """
        if not self.llm_provider:
            return None

        system_prompt, user_prompt = self._build_prompt(query, results, max_context_chunks)

        if self.llm_provider == 'anthropic' and self._anthropic_client:
            return self._generate_anthropic(system_prompt, user_prompt)
        elif self.llm_provider in ('openai',) and self._openai_client:
            return self._generate_openai(system_prompt, user_prompt)
        elif self.llm_provider == 'ollama' and self.use_ollama:
            return self._generate_ollama(system_prompt, user_prompt)

        return None

    def search_and_answer(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = True
    ) -> Dict:
        """
        Complete RAG pipeline: search + answer generation.

        Args:
            query: User question
            top_k: Number of chunks to retrieve
            rerank: Whether to use re-ranking

        Returns:
            Dictionary with results and generated answer
        """
        # Search for relevant chunks
        results = self.search(query, top_k=top_k, rerank=rerank)

        # Generate answer if any LLM provider is available
        answer = None
        if self.llm_provider and len(results) > 0:
            answer = self.generate_answer(query, results)

        return {
            'query': query,
            'results': results,
            'answer': answer,
            'count': len(results)
        }


def main():
    """Interactive search demo."""
    base_dir = Path(__file__).parent.parent
    index_dir = base_dir / 'data' / 'index'

    searcher = DocSearcher(index_dir)

    # Show stats
    stats = searcher.get_stats()
    print("=== DocSearch Ready ===")
    print(f"Indexed documents: {stats['total_documents']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"\nDocuments:")
    for doc in stats['documents']:
        print(f"  - {doc}")
    print("\n")

    # Interactive search
    while True:
        query = input("Enter search query (or 'quit' to exit): ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            break

        if not query:
            continue

        results = searcher.search(query, top_k=5)

        print(f"\n{'='*80}")
        print(f"Results for: '{query}'")
        print(f"{'='*80}\n")

        if not results:
            print("No results found.")
            continue

        for result in results:
            print(f"[{result['rank']}] {result['doc_name']} (Page {result['page']})")
            print(f"    Section: {result['section']}")
            print(f"    Similarity: {result['similarity']:.2%}")
            print(f"    Text: {result['text'][:200]}...")
            print()


if __name__ == '__main__':
    main()
