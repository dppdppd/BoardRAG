"""
List all detected section headers for a given PDF from the vector DB, and
optionally scan the PDF text to list headers found directly in the PDF. Useful
for comparing chunking vs source.

Usage (from repo root):
  venv\Scripts\python.exe temp_tests\list_sections.py --pdf ASLSK4_Rules_September_2021.pdf

Options:
  --pdf         Basename or path to the PDF (case-insensitive, basename match)
  --prefix      Optional numeric prefix filter (e.g., 3. to show only 3.x)
  --scan-pdf    Also scan the PDF itself to list headers from raw text
  --max         Max rows per list (default 500)
  --sort        Sort mode: alpha | page | numeric (default numeric)
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


def _numeric_key(label: str) -> Tuple:
    m = re.match(r"^\s*(\d+(?:\.\d+)*)\b", label)
    if not m:
        return ((10**9,), label.lower())
    parts = tuple(int(p) for p in m.group(1).split("."))
    # Return a uniform tuple that sorts cleanly: (parts..., 0, label)
    return (*parts, 0, label.lower())


def _connect_db():
    # Lazy import to avoid hard dependency during pure PDF scans
    import chromadb
    from langchain_chroma import Chroma
    try:
        from src.config import CHROMA_PATH, get_chromadb_settings, suppress_chromadb_telemetry
        from src.embedding_function import get_embedding_function
    except ModuleNotFoundError:
        import sys
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from src.config import CHROMA_PATH, get_chromadb_settings, suppress_chromadb_telemetry  # type: ignore
        from src.embedding_function import get_embedding_function  # type: ignore

    with suppress_chromadb_telemetry():
        client = chromadb.PersistentClient(path=CHROMA_PATH, settings=get_chromadb_settings())
        db = Chroma(client=client, embedding_function=get_embedding_function())
    return db


def list_db_sections(pdf_basename: str, prefix: str | None, sort_mode: str, max_rows: int):
    db = _connect_db()
    data = db.get()
    docs: List[str] = data.get("documents", [])
    metas: List[dict] = data.get("metadatas", [])
    ids: List[str] = data.get("ids", [])

    target = pdf_basename.lower()
    seen: Dict[str, Dict] = {}
    for doc, meta, cid in zip(docs, metas, ids):
        if not isinstance(meta, dict):
            continue
        src = str(meta.get("source", ""))
        if not src.lower().endswith(target):
            continue
        sec = (meta.get("section") or "").strip()
        if not sec:
            continue
        if prefix and not sec.startswith(prefix):
            continue
        page = meta.get("page")
        num = (meta.get("section_number") or "").strip()
        rec = seen.get(sec)
        if rec is None:
            seen[sec] = {"count": 1, "first_page": page, "section_number": num}
        else:
            rec["count"] += 1
            if isinstance(page, int) and (rec["first_page"] is None or page < rec["first_page"]):
                rec["first_page"] = page

    if sort_mode == "alpha":
        ordering = sorted(seen.items(), key=lambda kv: kv[0].lower())
    elif sort_mode == "page":
        ordering = sorted(seen.items(), key=lambda kv: (kv[1]["first_page"] if kv[1]["first_page"] is not None else 1e9, kv[0].lower()))
    else:  # numeric
        ordering = sorted(seen.items(), key=lambda kv: _numeric_key(kv[0]))

    print("DB sections:")
    for i, (label, meta) in enumerate(ordering[:max_rows], 1):
        p = meta["first_page"]
        cnt = meta["count"]
        num = meta.get("section_number")
        print(f"{i:>3}. page={p} count={cnt} num={num or '-'} | {label}")


def scan_pdf_sections(pdf_path: Path, prefix: str | None, sort_mode: str, max_rows: int):
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        from PyPDF2 import PdfReader  # type: ignore

    # Regexes aligned with populate_database.split_documents()
    NUMERIC_TITLE_RE = re.compile(r"^\s*((?:\d+(?:\.\d+)*))\s+(.+?)\s*$")
    NUMERIC_ONLY_HEADER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:\.)?\s*:?\s*$")

    reader = PdfReader(str(pdf_path))
    found: Dict[str, Dict] = {}
    for page_index, page in enumerate(getattr(reader, "pages", [])):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            m1 = NUMERIC_TITLE_RE.match(line)
            if m1:
                num = m1.group(1)
                title = m1.group(2).split(":", 1)[0].strip()
                label = f"{num} {title}"
                if (prefix is None) or label.startswith(prefix):
                    rec = found.get(label)
                    if rec is None:
                        found[label] = {"first_page": page_index, "count": 1}
                    else:
                        rec["count"] += 1
                i += 1
                continue
            m2 = NUMERIC_ONLY_HEADER_RE.match(line)
            if m2:
                # Try to grab the next non-empty line as title
                j = i + 1
                title = None
                while j < len(lines):
                    if lines[j].strip():
                        title = lines[j].strip().split(":", 1)[0].strip()
                        break
                    j += 1
                num = m2.group(1)
                if title:
                    label = f"{num} {title}"
                    if (prefix is None) or label.startswith(prefix):
                        rec = found.get(label)
                        if rec is None:
                            found[label] = {"first_page": page_index, "count": 1}
                        else:
                            rec["count"] += 1
                i = j + 1 if title else i + 1
                continue
            i += 1

    if sort_mode == "alpha":
        ordering = sorted(found.items(), key=lambda kv: kv[0].lower())
    elif sort_mode == "page":
        ordering = sorted(found.items(), key=lambda kv: (kv[1]["first_page"], kv[0].lower()))
    else:
        ordering = sorted(found.items(), key=lambda kv: _numeric_key(kv[0]))

    print("\nPDF-detected sections:")
    for i, (label, meta) in enumerate(ordering[:max_rows], 1):
        print(f"{i:>3}. page={meta['first_page']} count={meta['count']} | {label}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--scan-pdf", action="store_true")
    ap.add_argument("--max", type=int, default=500)
    ap.add_argument("--sort", choices=["alpha", "page", "numeric"], default="numeric")
    args = ap.parse_args()

    pdf_basename = os.path.basename(args.pdf).lower()
    prefix = args.prefix or None

    list_db_sections(pdf_basename, prefix, args.sort, args.max)
    if args.scan_pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            # Try resolving under data/
            pdf_path = Path("data") / pdf_basename
        scan_pdf_sections(pdf_path, prefix, args.sort, args.max)


if __name__ == "__main__":
    main()


