#!/usr/bin/env python3
"""
Complete DocSearch Pipeline Runner
Runs extract, chunk, and embed in sequence.
Supports --incremental flag to only process new documents.
"""

import sys
from pathlib import Path

# Add scripts directory to path
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / 'scripts'))

from extract import DocumentExtractor
from chunk import DocumentChunker
from embed import EmbeddingIndexer


def run_full_pipeline():
    """Run the complete indexing pipeline."""

    print("="*80)
    print("DocSearch - Full Pipeline")
    print("="*80)
    print()

    # Define directories
    raw_dir = base_dir / 'data' / 'raw'
    processed_dir = base_dir / 'data' / 'processed'
    index_dir = base_dir / 'data' / 'index'

    # Step 1: Extract
    print("STEP 1/3: Extracting text from documents...")
    print("-"*80)
    try:
        extractor = DocumentExtractor(raw_dir, processed_dir)
        results = extractor.extract_all()
        print(f"✓ Extraction complete: {len(results)} documents processed")
    except Exception as e:
        print(f"✗ Error during extraction: {e}")
        return False

    print()

    # Step 2: Chunk
    print("STEP 2/3: Chunking documents...")
    print("-"*80)
    try:
        chunker = DocumentChunker(processed_dir, chunk_size=800, overlap=100)
        chunks = chunker.process_all()
        print(f"✓ Chunking complete: {len(chunks)} chunks created")
    except Exception as e:
        print(f"✗ Error during chunking: {e}")
        return False

    print()

    # Step 3: Embed & Index
    print("STEP 3/3: Building search index...")
    print("-"*80)
    print("Note: First run will download the embedding model (~130MB) and re-ranker (~80MB)")
    try:
        indexer = EmbeddingIndexer(processed_dir, index_dir)
        indexer.build_index()
        # Save source manifest for future incremental runs
        current_files = {}
        for f in extractor.get_raw_files():
            stat = f.stat()
            current_files[f.name] = {'mtime': stat.st_mtime, 'size': stat.st_size}
        indexer.save_source_manifest(current_files)
        print(f"✓ Index build complete")
    except Exception as e:
        print(f"✗ Error during indexing: {e}")
        return False

    print()
    print("="*80)
    print("✓ PIPELINE COMPLETE!")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Run the search server: python app/server.py")
    print("  2. Open browser to: http://127.0.0.1:5000")
    print("  3. Or use CLI search: python scripts/search.py")
    print()

    return True


def run_incremental_pipeline():
    """Run an incremental pipeline — only process new documents."""

    print("="*80)
    print("DocSearch - Incremental Pipeline")
    print("="*80)
    print()

    raw_dir = base_dir / 'data' / 'raw'
    processed_dir = base_dir / 'data' / 'processed'
    index_dir = base_dir / 'data' / 'index'

    # Detect changes
    indexer = EmbeddingIndexer(processed_dir, index_dir)
    changes = indexer.detect_changes(str(raw_dir))

    new_files = changes['new']
    modified_files = changes['modified']
    deleted_names = changes['deleted']

    print(f"  New documents:      {len(new_files)}")
    print(f"  Modified documents: {len(modified_files)}")
    print(f"  Deleted documents:  {len(deleted_names)}")
    print()

    # If docs were modified or deleted, fall back to full rebuild
    if modified_files or deleted_names:
        print("Modified or deleted documents detected — running full rebuild...")
        print()
        return run_full_pipeline()

    # No changes at all
    if not new_files:
        print("No changes detected. Index is up to date.")
        return True

    # Incremental path: only process new files
    print(f"STEP 1/3: Extracting {len(new_files)} new document(s)...")
    print("-"*80)
    try:
        extractor = DocumentExtractor(raw_dir, processed_dir)
        results = extractor.extract_files(new_files)
        print(f"✓ Extraction complete: {len(results)} documents processed")
    except Exception as e:
        print(f"✗ Error during extraction: {e}")
        return False

    print()

    print("STEP 2/3: Re-chunking all documents...")
    print("-"*80)
    try:
        chunker = DocumentChunker(processed_dir, chunk_size=800, overlap=100)
        chunks = chunker.process_all()
        print(f"✓ Chunking complete: {len(chunks)} chunks created")
    except Exception as e:
        print(f"✗ Error during chunking: {e}")
        return False

    print()

    print("STEP 3/3: Incrementally updating search index...")
    print("-"*80)
    try:
        new_doc_names = [f.stem for f in new_files]
        indexer.incremental_build(new_doc_names)
        # Update manifest with all current files
        indexer.save_source_manifest(changes['current_files'])
        print(f"✓ Incremental index update complete")
    except Exception as e:
        print(f"✗ Error during indexing: {e}")
        return False

    print()
    print("="*80)
    print("✓ INCREMENTAL PIPELINE COMPLETE!")
    print("="*80)
    print()

    return True


if __name__ == '__main__':
    incremental = '--incremental' in sys.argv or '-i' in sys.argv

    if incremental:
        success = run_incremental_pipeline()
    else:
        success = run_full_pipeline()

    sys.exit(0 if success else 1)
