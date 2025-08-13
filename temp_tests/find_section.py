"""
Quick inspector for locating sections in the Chroma store.

Usage (from repo root):
    venv\Scripts\python.exe temp_tests\find_section.py --pdf ASLSK4_Rules_September_2021.pdf --section 3.0

Options:
    --pdf       PDF filename to filter to (case-insensitive, basename match)
    --section   Section prefix to search for (e.g., 3.0 or 3.5)
    --contains  Optional substring to search for in chunk text
    --max       Max rows to print (default 50)
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List

import chromadb
from langchain_chroma import Chroma

# Support running from repo root without installing as a package
try:
    from src.config import CHROMA_PATH, get_chromadb_settings, suppress_chromadb_telemetry
    from src.embedding_function import get_embedding_function
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.config import CHROMA_PATH, get_chromadb_settings, suppress_chromadb_telemetry
    from src.embedding_function import get_embedding_function


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="PDF filename to filter, e.g. ASLSK4_Rules_September_2021.pdf")
    ap.add_argument("--section", required=False, help="Section prefix to match, e.g. 3.0 or 3.5")
    ap.add_argument("--contains", required=False, help="Substring to search for in chunk text")
    ap.add_argument("--max", type=int, default=50)
    args = ap.parse_args()

    pdf_name = os.path.basename(args.pdf).lower()
    section_prefix = (args.section or "").strip()
    text_contains = (args.contains or "").strip().lower()

    with suppress_chromadb_telemetry():
        client = chromadb.PersistentClient(path=CHROMA_PATH, settings=get_chromadb_settings())
        db = Chroma(client=client, embedding_function=get_embedding_function())

    data = db.get()
    docs: List[str] = data.get("documents", [])
    metas: List[dict] = data.get("metadatas", [])
    ids: List[str] = data.get("ids", [])

    # Gather candidates
    matches = []
    for doc, meta, cid in zip(docs, metas, ids):
        if not isinstance(meta, dict):
            continue
        src = str(meta.get("source", ""))
        if not src.lower().endswith(pdf_name):
            continue
        section_label = (meta.get("section") or "").strip()
        if section_prefix and not section_label.startswith(section_prefix):
            continue
        if text_contains and text_contains not in (doc or "").lower():
            continue
        matches.append((cid, meta.get("page"), section_label, (doc or "").strip()))


    print(f"PDF: {pdf_name}")
    if section_prefix:
        print(f"Looking for section starting with: {section_prefix}")
    if text_contains:
        print(f"Containing text: {text_contains!r}")
    print(f"Found {len(matches)} matching chunks\n")

    for i, (cid, page, sec, text) in enumerate(matches[: args.max], 1):
        preview = text.replace("\n", " ")[:160]
        print(f"{i:>3}. {cid} | page {page} | section: {sec}")
        print(f"     {preview}")
        print("-" * 120)

    # Removed fixed diagnostic of 3.x sections; this script now only prints matches


if __name__ == "__main__":
    main()


