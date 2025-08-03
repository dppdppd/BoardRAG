#!/usr/bin/env python3
"""inspect_chunks.py – quick command-line viewer for BoardRAG chunks

Run with:
    python inspect_chunks.py [--max 50] [--no-content] [--filter game_name]

It connects to your local Chroma vector store and prints a table-like view of
chunk IDs, page numbers, section headers, and (optionally) the first few lines
of text. No Argilla or extra services required.
"""

from __future__ import annotations

import argparse
import textwrap
from typing import List, Tuple

import chromadb
from langchain_chroma import Chroma

# Project helpers (imported from src)
from src.config import CHROMA_PATH, get_chromadb_settings, suppress_chromadb_telemetry
from src.embedding_function import get_embedding_function


def fetch_chunks() -> Tuple[List[str], List[dict], List[str]]:
    """Fetch all documents, metadatas and ids from the Chroma DB."""

    with suppress_chromadb_telemetry():
        client = chromadb.PersistentClient(
            path=CHROMA_PATH, settings=get_chromadb_settings()
        )
        db = Chroma(client=client, embedding_function=get_embedding_function())

    data = db.get()  # pulls the entire collection
    return data["documents"], data["metadatas"], data["ids"]


def print_chunks(max_rows: int, show_content: bool, filter_str: str | None):
    documents, metadatas, ids = fetch_chunks()

    # Optionally filter by substring in the chunk id (useful for game names)
    if filter_str:
        filtered = [
            (doc, meta, cid)
            for doc, meta, cid in zip(documents, metadatas, ids)
            if filter_str.lower() in cid.lower()
        ]
    else:
        filtered = list(zip(documents, metadatas, ids))

    total = len(filtered)
    print(f"\n📚 Chunks matching filter: {total}")
    print(f"🔎 Showing the first {min(max_rows, total)}\n")

    wrap_width = 110

    for idx, (chunk, meta, cid) in enumerate(filtered[:max_rows], 1):
        page = meta.get("page")
        section = meta.get("section", "—")
        print(f"{idx:>3}. {cid} | page {page} | section: {section}")
        if show_content:
            print(textwrap.fill(chunk.strip(), wrap_width))
        print("-" * 120)



def main():
    parser = argparse.ArgumentParser(description="Inspect stored BoardRAG chunks")
    parser.add_argument("--max", type=int, default=50, help="How many chunks to show")
    parser.add_argument(
        "--no-content",
        action="store_true",
        help="Only print id, page and section (omit chunk text)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Substring to filter chunk IDs (e.g. 'monopoly' or 'data/monopoly.pdf')",
    )

    args = parser.parse_args()

    print_chunks(
        max_rows=args.max,
        show_content=not args.no_content,
        filter_str=args.filter,
    )


if __name__ == "__main__":
    main()
