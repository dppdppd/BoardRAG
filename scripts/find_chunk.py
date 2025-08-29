#!/usr/bin/env python
from __future__ import annotations

import argparse
from typing import Optional
from pathlib import Path
import sys

# Ensure repository root is on sys.path so `import src` works when executed as a script
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.vector_store import get_chunk_by_page, search_chunks


def main() -> int:
    ap = argparse.ArgumentParser(description="Find a chunk by page or text search")
    ap.add_argument("--pdf", type=str, help="PDF basename (e.g., catan_base.pdf)")
    ap.add_argument("--page", type=int, help="1-based page number", default=None)
    ap.add_argument("--query", type=str, help="search text", default=None)
    args = ap.parse_args()

    if args.page is not None and args.pdf:
        pdf_base = args.pdf.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        res = get_chunk_by_page(pdf_base.replace(".pdf", ""), args.page)
        if not res:
            print("not found")
            return 1
        doc, score = res
        print(doc.page_content)
        print("-- meta --")
        print(doc.metadata)
        return 0

    if args.query:
        from src.vector_store import search_section_chunks
        results = search_section_chunks(args.query, pdf=args.pdf)
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i:>2}. score={score:.4f} {doc.metadata.get('source')}#p{doc.metadata.get('page')}")
            print(doc.page_content[:400].replace("\n", " ") + ("…" if len(doc.page_content) > 400 else ""))
            print()
        return 0

    print("Provide either --pdf and --page, or --query [--pdf]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


