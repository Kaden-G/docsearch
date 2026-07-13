#!/usr/bin/env python3
"""
document Chunking Script
Splits extracted text into semantic chunks while preserving metadata.
"""

import json
import re
from pathlib import Path
from typing import List, Dict


class DocumentChunker:
    """Split document text into retrievable chunks."""

    def __init__(
        self,
        processed_dir: str,
        chunk_size: int = 800,
        overlap: int = 200
    ):
        self.processed_dir = Path(processed_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def detect_section_headers(self, text: str) -> List[tuple]:
        """Detect section headers in text. Returns (char_offset, header_text) tuples."""
        # Common document header patterns
        patterns = [
            r'^(\d+\.\d*\.?\d*)\s+([A-Z][^\n]+)',  # 1.2.3 HEADER
            r'^([A-Z][A-Z\s]{3,}[A-Z])$',          # ALL CAPS HEADER
            r'^(STEP\s+\d+[:\.])\s*(.+)',          # STEP 1: Description
            r'^(Procedure\s+\d+[:\.])\s*(.+)',     # Procedure 1: Description
        ]

        headers = []
        char_offset = 0
        for line in text.split('\n'):
            for pattern in patterns:
                match = re.match(pattern, line.strip())
                if match:
                    headers.append((char_offset, line.strip()))
                    break
            char_offset += len(line) + 1  # +1 for the newline
        return headers

    def _split_sentences(self, text: str) -> List[tuple]:
        """Split text into sentences, returning (char_start, char_end, sentence_text) tuples."""
        # Split on sentence-ending punctuation followed by whitespace or end-of-string,
        # also split on newlines (common in documents for numbered steps)
        sentence_pattern = re.compile(
            r'(?<=[.!?])\s+|\n\s*\n|\n(?=\d+[.\)]\s)|\n(?=[A-Z])'
        )

        sentences = []
        last_end = 0
        for match in sentence_pattern.finditer(text):
            start = last_end
            end = match.start()
            sent = text[start:end].strip()
            if sent:
                sentences.append((start, end, sent))
            last_end = match.end()

        # Remaining text
        remaining = text[last_end:].strip()
        if remaining:
            sentences.append((last_end, len(text), remaining))

        # Fallback: if regex produced nothing useful, split on newlines
        if not sentences and text.strip():
            sentences.append((0, len(text), text.strip()))

        return sentences

    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text into overlapping chunks using sentence boundaries."""
        chunks = []

        # Detect sections for better chunking (returns char offsets)
        sections = self.detect_section_headers(text)

        # Split into sentences
        sentences = self._split_sentences(text)
        if not sentences:
            return chunks

        # Group sentences into chunks respecting size limits
        current_sents = []   # indices into sentences list
        current_len = 0

        def flush_chunk(sent_indices):
            """Create a chunk from the given sentence indices."""
            if not sent_indices:
                return
            first = sentences[sent_indices[0]]
            last = sentences[sent_indices[-1]]
            char_start = first[0]
            char_end = last[1]
            chunk_text = ' '.join(sentences[si][2] for si in sent_indices)

            # Find closest section header
            section_header = "Unknown Section"
            for sec_offset, sec_name in sections:
                if sec_offset <= char_start:
                    section_header = sec_name

            chunks.append({
                'chunk_id': f"{metadata['doc_name']}_p{metadata['page']}_c{len(chunks)}",
                'doc_name': metadata['doc_name'],
                'page': metadata['page'],
                'section': section_header,
                'text': chunk_text,
                'char_start': char_start,
                'char_end': char_end
            })

        i = 0
        while i < len(sentences):
            sent_len = len(sentences[i][2])

            if current_len + sent_len + 1 <= self.chunk_size or not current_sents:
                current_sents.append(i)
                current_len += sent_len + 1
                i += 1
            else:
                flush_chunk(current_sents)

                # Calculate overlap: walk backwards from end to find sentences that fit
                overlap_sents = []
                overlap_len = 0
                for j in reversed(current_sents):
                    s_len = len(sentences[j][2])
                    if overlap_len + s_len + 1 > self.overlap:
                        break
                    overlap_sents.insert(0, j)
                    overlap_len += s_len + 1

                # Guarantee forward progress. If the next sentence still won't fit
                # alongside the retained overlap, drop the overlap so it starts a
                # fresh chunk. Without this, any sentence longer than
                # (chunk_size - overlap) loops forever — re-flushing the same
                # overlap and growing the chunk list without bound.
                if overlap_sents and overlap_len + sent_len + 1 > self.chunk_size:
                    overlap_sents = []
                    overlap_len = 0

                current_sents = overlap_sents
                current_len = overlap_len

        # Flush remaining
        if current_sents:
            flush_chunk(current_sents)

        return chunks

    def process_all(self) -> List[Dict]:
        """Process all extracted documents into chunks."""
        all_chunks = []

        # Process each JSON file in processed directory. Skip our own outputs
        # (manifest.json, chunks.json) so re-running a build never re-chunks them.
        for json_file in self.processed_dir.glob('*.json'):
            if json_file.name in ('manifest.json', 'chunks.json'):
                continue

            print(f"Chunking: {json_file.name}")

            with open(json_file, 'r', encoding='utf-8') as f:
                pages = json.load(f)

            for page in pages:
                chunks = self.chunk_text(page['text'], page)
                all_chunks.extend(chunks)

        # Save all chunks
        chunks_file = self.processed_dir / 'chunks.json'
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        print(f"\nChunking complete: {len(all_chunks)} chunks created")
        print(f"Saved to: {chunks_file}")

        return all_chunks


def main():
    """Run chunking on all processed documents."""
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / 'data' / 'processed'

    chunker = DocumentChunker(processed_dir, chunk_size=800, overlap=200)
    chunks = chunker.process_all()

    print(f"\nTotal chunks: {len(chunks)}")


if __name__ == '__main__':
    main()
