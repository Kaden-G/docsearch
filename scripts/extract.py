#!/usr/bin/env python3
"""
document Document Extraction Script
Extracts text from PDF and Word documents for indexing.
"""

import os
import json
from pathlib import Path
from typing import Dict, List
import pypdf
from docx import Document


class DocumentExtractor:
    """Extract text from documents with metadata preservation."""

    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def extract_pdf(self, pdf_path: Path) -> List[Dict]:
        """Extract text from PDF with page numbers."""
        chunks = []
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        chunks.append({
                            'doc_name': pdf_path.stem,
                            'page': page_num,
                            'text': text.strip(),
                            'doc_type': 'pdf'
                        })
        except Exception as e:
            print(f"Error extracting {pdf_path.name}: {e}")
        return chunks

    def extract_docx(self, docx_path: Path, chars_per_page: int = 3000) -> List[Dict]:
        """Extract text from Word documents with virtual page numbering."""
        chunks = []
        try:
            doc = Document(docx_path)

            # Accumulate paragraphs into virtual pages (~3000 chars each)
            current_page_text = []
            current_char_count = 0
            page_num = 1

            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue

                current_page_text.append(text)
                current_char_count += len(text) + 1  # +1 for newline

                if current_char_count >= chars_per_page:
                    chunks.append({
                        'doc_name': docx_path.stem,
                        'page': page_num,
                        'text': '\n'.join(current_page_text),
                        'doc_type': 'docx'
                    })
                    current_page_text = []
                    current_char_count = 0
                    page_num += 1

            # Flush remaining text
            if current_page_text:
                chunks.append({
                    'doc_name': docx_path.stem,
                    'page': page_num,
                    'text': '\n'.join(current_page_text),
                    'doc_type': 'docx'
                })

        except Exception as e:
            print(f"Error extracting {docx_path.name}: {e}")
        return chunks

    def extract_txt(self, txt_path: Path) -> List[Dict]:
        """Extract text from plain text files."""
        chunks = []
        try:
            # Try common encodings in order. latin-1 maps every byte, so the
            # chain always succeeds — a non-UTF8 .txt (e.g. Windows cp1252 from
            # Word/Notepad) must not be silently dropped from the index.
            text = None
            for enc in ('utf-8', 'utf-8-sig', 'cp1252', 'latin-1'):
                try:
                    with open(txt_path, 'r', encoding=enc) as f:
                        text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            if text is None:
                with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()

            if text.strip():
                chunks.append({
                    'doc_name': txt_path.stem,
                    'page': 1,
                    'text': text.strip(),
                    'doc_type': 'txt'
                })
        except Exception as e:
            print(f"Error extracting {txt_path.name}: {e}")
        return chunks

    def get_raw_files(self) -> List[Path]:
        """Get all supported document files in the raw directory."""
        files = []
        for ext in ('*.pdf', '*.docx', '*.txt'):
            for f in self.raw_dir.glob(ext):
                if not f.name.startswith('~$'):  # Skip temp files
                    files.append(f)
        return sorted(files)

    def extract_files(self, file_paths: List[Path]) -> Dict[str, List[Dict]]:
        """Extract only the specified files. Returns {doc_name: [pages]}."""
        all_extracts = {}
        for fpath in file_paths:
            print(f"Extracting: {fpath.name}")
            if fpath.suffix.lower() == '.pdf':
                extracts = self.extract_pdf(fpath)
            elif fpath.suffix.lower() == '.docx':
                extracts = self.extract_docx(fpath)
            elif fpath.suffix.lower() == '.txt':
                extracts = self.extract_txt(fpath)
            else:
                continue

            all_extracts[fpath.stem] = extracts

            output_file = self.processed_dir / f"{fpath.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extracts, f, indent=2, ensure_ascii=False)

        return all_extracts

    def extract_all(self) -> Dict[str, List[Dict]]:
        """Process all documents in raw directory."""
        all_extracts = {}
        failed = []

        all_files = self.get_raw_files()
        total = len(all_files)
        for i, fpath in enumerate(all_files, 1):
            print(f"Extracting ({i}/{total}): {fpath.name}")
            try:
                if fpath.suffix.lower() == '.pdf':
                    extracts = self.extract_pdf(fpath)
                elif fpath.suffix.lower() == '.docx':
                    extracts = self.extract_docx(fpath)
                elif fpath.suffix.lower() == '.txt':
                    extracts = self.extract_txt(fpath)
                else:
                    continue

                if extracts:
                    all_extracts[fpath.stem] = extracts
                    output_file = self.processed_dir / f"{fpath.stem}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(extracts, f, indent=2, ensure_ascii=False)
                else:
                    print(f"  Warning: No text extracted from {fpath.name}")
            except Exception as e:
                print(f"  FAILED to extract {fpath.name}: {e}")
                failed.append(fpath.name)

        # Save combined manifest
        manifest_file = self.processed_dir / "manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_docs': len(all_extracts),
                'documents': list(all_extracts.keys()),
                'failed': failed
            }, f, indent=2)

        print(f"\nExtraction complete: {len(all_extracts)} documents processed")
        if failed:
            print(f"  Failed: {len(failed)} files: {', '.join(failed)}")
        return all_extracts


def main():
    """Run extraction on all documents."""
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / 'data' / 'raw'
    processed_dir = base_dir / 'data' / 'processed'

    extractor = DocumentExtractor(raw_dir, processed_dir)
    results = extractor.extract_all()

    print(f"\nResults saved to: {processed_dir}")
    print(f"Total documents: {len(results)}")


if __name__ == '__main__':
    main()
