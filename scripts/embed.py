#!/usr/bin/env python3
"""
document Embedding Script
Creates vector embeddings and builds FAISS index for semantic search.
"""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss

from embedder import get_embedder


class EmbeddingIndexer:
    """Create and manage vector embeddings for chunks."""

    def __init__(
        self,
        processed_dir: str,
        index_dir: str,
        model_name: str = None,
        provider: str = None,
        api_key: str = None,
        api_base: str = None
    ):
        self.processed_dir = Path(processed_dir)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Resolve the embedding backend (local CPU model or remote OpenAI-compatible
        # API), honoring explicit args first, then DOCSEARCH_EMBED_* environment vars.
        # api_key/api_base let the build reuse the token the user entered for the LLM.
        self.embedder = get_embedder(
            provider=provider, model_name=model_name, api_key=api_key, api_base=api_base
        )
        self.model_name = self.embedder.name
        self.provider = self.embedder.provider
        print(f"Embedder ready: provider={self.provider} model={self.model_name}")

    def load_chunks(self) -> List[Dict]:
        """Load processed chunks from JSON."""
        chunks_file = self.processed_dir / 'chunks.json'

        if not chunks_file.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {chunks_file}\n"
                "Run chunk.py first to create chunks."
            )

        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        print(f"Loaded {len(chunks)} chunks")
        return chunks

    def create_embeddings(self, chunks: List[Dict], progress_callback=None) -> np.ndarray:
        """Generate embeddings for all chunks in batches.
        
        Args:
            chunks: List of chunk dicts with 'text' key.
            progress_callback: Optional callable(done: int, total: int) for progress updates.
        """
        print(f"Generating embeddings for {len(chunks)} chunks...")

        # Smart Indexing stores a contextual 'embedding_text' (Doc > Section +
        # summary + synthetic questions + text); embed it when present, else the
        # raw text. Display/answering always use the original 'text'.
        texts = [chunk.get('embedding_text') or chunk['text'] for chunk in chunks]
        batch_size = 32
        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            batch_emb = self.embedder.encode(batch)
            all_embeddings.append(batch_emb)

            done = min(start + batch_size, len(texts))
            print(f"  Embedded {done}/{len(texts)} chunks")
            if progress_callback:
                progress_callback(done, len(texts))

        embeddings = np.concatenate(all_embeddings, axis=0)
        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index for fast similarity search."""
        print("Building FAISS index...")

        dimension = embeddings.shape[1]

        # Use IndexFlatIP for cosine similarity (better for semantic search)
        # Normalize embeddings for cosine similarity
        index = faiss.IndexFlatIP(dimension)

        # Normalize embeddings before adding (required for cosine similarity)
        embeddings_normalized = embeddings.astype('float32')
        faiss.normalize_L2(embeddings_normalized)

        # Add vectors to index
        index.add(embeddings_normalized)

        print(f"Index built with {index.ntotal} vectors (cosine similarity)")
        return index

    def save_index(
        self,
        index: faiss.Index,
        chunks: List[Dict],
        embeddings: np.ndarray
    ):
        """Save FAISS index and metadata."""
        # Save FAISS index
        index_file = self.index_dir / 'faiss.index'
        faiss.write_index(index, str(index_file))
        print(f"FAISS index saved: {index_file}")

        # Save chunk metadata (without text to save space)
        metadata = [
            {
                'chunk_id': c['chunk_id'],
                'doc_name': c['doc_name'],
                'page': c['page'],
                'section': c['section'],
                'text': c['text'],  # Keep text for display
                'summary': c.get('summary', ''),
                'questions': c.get('questions', []),
            }
            for c in chunks
        ]

        metadata_file = self.index_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False)
        print(f"Metadata saved: {metadata_file}")

        # Save index config
        config = {
            'num_chunks': len(chunks),
            'embedding_dim': embeddings.shape[1],
            'model_name': self.model_name,
            'embedding_provider': self.provider,
            'embedding_api_base': getattr(self.embedder, 'api_base', None),
            'index_type': 'IndexFlatIP',
            'similarity_metric': 'cosine'
        }

        config_file = self.index_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved: {config_file}")

    def load_existing_index(self):
        """Load existing FAISS index and metadata. Returns (index, metadata) or (None, None)."""
        index_file = self.index_dir / 'faiss.index'
        metadata_json = self.index_dir / 'metadata.json'
        metadata_pkl = self.index_dir / 'metadata.pkl'  # Backwards compat

        if not index_file.exists():
            return None, None

        index = faiss.read_index(str(index_file))

        if metadata_json.exists():
            with open(metadata_json, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        elif metadata_pkl.exists():
            import pickle
            with open(metadata_pkl, 'rb') as f:
                metadata = pickle.load(f)
        else:
            return None, None

        return index, metadata

    def get_source_manifest(self) -> Dict:
        """Load the source file manifest (tracks what's been indexed)."""
        manifest_file = self.index_dir / 'source_manifest.json'
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                return json.load(f)
        return {}

    def save_source_manifest(self, manifest: Dict):
        """Save the source file manifest."""
        manifest_file = self.index_dir / 'source_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

    def detect_changes(self, raw_dir: str) -> Dict:
        """
        Compare raw files against the stored manifest.
        Returns {'new': [paths], 'modified': [paths], 'deleted': [names]}.
        """
        from pathlib import Path as P
        raw_path = P(raw_dir)
        manifest = self.get_source_manifest()

        current_files = {}
        for ext in ('*.pdf', '*.docx', '*.txt'):
            for f in raw_path.glob(ext):
                if not f.name.startswith('~$'):
                    stat = f.stat()
                    current_files[f.name] = {
                        'path': str(f),
                        'mtime': stat.st_mtime,
                        'size': stat.st_size
                    }

        new_files = []
        modified_files = []
        deleted_names = []

        # Find new and modified
        for fname, info in current_files.items():
            if fname not in manifest:
                new_files.append(P(info['path']))
            elif info['mtime'] != manifest[fname]['mtime'] or info['size'] != manifest[fname]['size']:
                modified_files.append(P(info['path']))

        # Find deleted
        for fname in manifest:
            if fname not in current_files:
                deleted_names.append(fname)

        return {
            'new': new_files,
            'modified': modified_files,
            'deleted': deleted_names,
            'current_files': current_files
        }

    def build_index(self, progress_callback=None):
        """Full pipeline: load chunks, embed, build index.
        
        Args:
            progress_callback: Optional callable(done: int, total: int) for embedding progress.
        """
        # Load chunks
        chunks = self.load_chunks()

        # Create embeddings
        embeddings = self.create_embeddings(chunks, progress_callback=progress_callback)

        # Build FAISS index
        index = self.build_faiss_index(embeddings)

        # Save everything
        self.save_index(index, chunks, embeddings)

        print("\n=== Index Build Complete ===")
        print(f"Total chunks indexed: {len(chunks)}")
        print(f"Index directory: {self.index_dir}")

    def incremental_build(self, new_doc_names: List[str]):
        """
        Append new documents to existing index without re-embedding unchanged docs.

        Args:
            new_doc_names: List of doc_name stems that are new (already extracted & chunked)
        """
        existing_index, existing_metadata = self.load_existing_index()
        if existing_index is None:
            print("No existing index found, running full build...")
            return self.build_index()

        # Load all chunks and filter to just the new ones
        all_chunks = self.load_chunks()
        new_chunks = [c for c in all_chunks if c['doc_name'] in new_doc_names]

        if not new_chunks:
            print("No new chunks to index.")
            return

        print(f"Incrementally indexing {len(new_chunks)} new chunks from {len(new_doc_names)} document(s)...")

        # Embed only the new chunks
        new_embeddings = self.create_embeddings(new_chunks)

        # Normalize and append to existing FAISS index
        new_embeddings_norm = new_embeddings.astype('float32')
        faiss.normalize_L2(new_embeddings_norm)
        existing_index.add(new_embeddings_norm)

        # Merge metadata
        new_metadata = [
            {
                'chunk_id': c['chunk_id'],
                'doc_name': c['doc_name'],
                'page': c['page'],
                'section': c['section'],
                'text': c['text'],
                'summary': c.get('summary', ''),
                'questions': c.get('questions', []),
            }
            for c in new_chunks
        ]
        merged_metadata = existing_metadata + new_metadata

        # Save updated index
        faiss.write_index(existing_index, str(self.index_dir / 'faiss.index'))
        with open(self.index_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(merged_metadata, f, ensure_ascii=False)

        # Update config
        config_file = self.index_dir / 'config.json'
        with open(config_file, 'r') as f:
            config = json.load(f)
        config['num_chunks'] = len(merged_metadata)
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n=== Incremental Index Update Complete ===")
        print(f"Added: {len(new_chunks)} chunks")
        print(f"Total indexed: {len(merged_metadata)} chunks")


def main():
    """Build the embedding index."""
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / 'data' / 'processed'
    index_dir = base_dir / 'data' / 'index'

    indexer = EmbeddingIndexer(processed_dir, index_dir)
    indexer.build_index()


if __name__ == '__main__':
    main()
