"""
Diagnostic: run section-aware chunking on a single PDF and report detected
headers around a target phrase (e.g., "Sequence of Play").

Usage:
  venv\Scripts\python.exe temp_tests\test_chunking_headers.py \
      --pdf data\ASLSK4_Rules_September_2021.pdf \
      --phrase "Sequence of Play"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List


def load_pages(pdf_path: str):
    try:
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(pdf_path).load()
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF pages: {e}")


def run_split(pages):
    # Import from local src
    try:
        from src.populate_database import split_documents
    except Exception:
        import sys
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from src.populate_database import split_documents  # type: ignore
    return split_documents(pages)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--phrase", required=True)
    ap.add_argument("--max", type=int, default=20)
    args = ap.parse_args()

    pdf_path = os.path.normpath(args.pdf)
    print(f"Loading: {pdf_path}")
    pages = load_pages(pdf_path)
    print(f"Loaded {len(pages)} pages")

    # Show any page(s) that contain the phrase before chunking
    lowered_phrase = args.phrase.lower()
    print("\nRaw PDF scan for phrase:")
    raw_hits = []
    for d in pages:
        txt = (d.page_content or "").strip()
        if lowered_phrase in txt.lower():
            raw_hits.append((d.metadata.get("page"), txt))
    for page_no, txt in raw_hits[: args.max]:
        print("-" * 80)
        print(f"Page {page_no}")
        # Print a small preview
        idx = txt.lower().find(lowered_phrase)
        start = max(0, idx - 150)
        end = min(len(txt), idx + len(args.phrase) + 150)
        print(txt[start:end])

    # Now split into section-aware chunks
    print("\nRunning section-aware split...\n")
    chunks = run_split(pages)
    print(f"Total chunks: {len(chunks)}")

    # Gather 3.x sections and any chunk containing the phrase
    three_x_sections: List[str] = []
    phrase_chunks = []
    for c in chunks:
        sec = (c.metadata.get("section") or "").strip()
        if sec.startswith("3."):
            three_x_sections.append(sec)
        if lowered_phrase in (c.page_content or "").lower():
            phrase_chunks.append(c)

    # Show unique 3.x section labels
    uniq = sorted({s for s in three_x_sections})
    print("\nDetected 3.x section labels (sample up to 50):")
    for s in uniq[:50]:
        print(f"  - {s}")
    if not any(s.startswith("3.0") for s in uniq):
        print("\nWARNING: No '3.0' section label detected in chunking output")

    # Show chunks that contain the phrase
    print("\nChunks containing the phrase:")
    if not phrase_chunks:
        print("  (none)")
    else:
        for c in phrase_chunks[: args.max]:
            meta = c.metadata
            print("-" * 80)
            print(f"Page: {meta.get('page')}  Section: {meta.get('section')}  Section#:{meta.get('section_number')}")
            preview = (c.page_content or "").splitlines()[:8]
            for line in preview:
                print(line)


if __name__ == "__main__":
    main()




