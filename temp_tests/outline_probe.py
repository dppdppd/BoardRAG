from __future__ import annotations

"""
outline_probe.py — Debug raw LLM response for PDF outline extraction.

Run from repo root (Windows examples):
  venv\Scripts\python.exe temp_tests\outline_probe.py --pdf "data\HF4 Core Rules.pdf" --max-lines 400
"""

import argparse
import re
import json
from typing import List, Tuple


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe LLM outline raw response")
    ap.add_argument("--pdf", required=True, help="Path to PDF")
    ap.add_argument("--max-lines", type=int, default=400, help="Max candidate header lines")
    args = ap.parse_args()

    # Import helpers from outline module with robust path handling
    try:
        from src.llm_outline import _load_pdf_pages, _make_llm, _SYS, _USR_TEMPLATE  # type: ignore
        from langchain.schema import HumanMessage, SystemMessage  # type: ignore
    except Exception:
        import sys
        from pathlib import Path
        ROOT = Path(__file__).resolve().parents[1]
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.llm_outline import _load_pdf_pages, _make_llm, _SYS, _USR_TEMPLATE  # type: ignore
        from langchain.schema import HumanMessage, SystemMessage  # type: ignore

    print(f"[probe] Loading PDF: {args.pdf}", flush=True)
    pages = _load_pdf_pages(args.pdf)
    print(f"[probe] Loaded {len(pages)} pages", flush=True)
    # LLM-first mode: skip regex candidate lines; send page texts directly
    print("[probe] Skipping regex candidate lines (LLM-first mode)", flush=True)

    # Build pages block directly (LLM finds candidates itself now)
    MAX_PAGES_PER_BATCH = 8
    MAX_CHARS_PER_PAGE = 1800
    blocks = []
    cur: List[Tuple[int, str]] = []
    for i, text in enumerate(pages, start=1):
        t = (text or "").strip()
        if len(t) > MAX_CHARS_PER_PAGE:
            t = t[:MAX_CHARS_PER_PAGE]
        cur.append((i, t))
        if len(cur) >= MAX_PAGES_PER_BATCH:
            blocks.append("\n\n".join([f"=== Page {p} ===\n{tt}" for (p, tt) in cur]))
            cur = []
    if cur:
        blocks.append("\n\n".join([f"=== Page {p} ===\n{tt}" for (p, tt) in cur]))

    print("[probe] Initializing LLM …", flush=True)
    try:
        llm = _make_llm()
        print(f"[probe] LLM: {type(llm).__name__}", flush=True)
    except Exception as e:
        print(f"[probe] LLM init error: {e}")
        return
    # Use replace instead of format to avoid interpreting braces in JSON schema
    usr = _USR_TEMPLATE.replace("{pages}", blocks[0])
    print("\n--- SYSTEM ---\n" + _SYS + "\n", flush=True)
    print("--- USER (first 800 chars) ---\n" + usr[:800] + ("..." if len(usr) > 800 else "") + "\n", flush=True)

    print("[probe] Calling LLM …", flush=True)
    try:
        out = llm.invoke([SystemMessage(content=_SYS), HumanMessage(content=usr)])
        raw = str(getattr(out, "content", "") or "").strip()
    except Exception as e:
        print(f"[probe] LLM invoke error: {e}")
        return
    print("--- RAW OUTPUT (first 2000 chars) ---\n" + raw[:2000] + ("..." if len(raw) > 2000 else ""), flush=True)

    # Try to extract JSON object(s) using robust brace-aware parser
    def _strip_code_fences(text: str) -> str:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if m:
            return (m.group(1) or "").strip()
        return text

    def _find_json_objects(text: str) -> List[str]:
        objs: List[str] = []
        n = len(text)
        i = 0
        while i < n:
            if text[i] == '{':
                start = i
                depth = 0
                in_str = False
                esc = False
                while i < n:
                    ch = text[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == '\\':
                            esc = True
                        elif ch == '"':
                            in_str = False
                    else:
                        if ch == '"':
                            in_str = True
                        elif ch == '{':
                            depth += 1
                        elif ch == '}':
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                objs.append(text[start:end])
                                break
                    i += 1
            else:
                i += 1
        return objs

    raw_clean = _strip_code_fences(raw)
    matches: List[str] = _find_json_objects(raw_clean)
    print(f"\n[probe] Found {len(matches)} JSON-like object(s)", flush=True)
    for i, js in enumerate(matches[:3], 1):
        print(f"\n--- JSON CANDIDATE #{i} (first 800 chars) ---\n" + js[:800] + ("..." if len(js) > 800 else ""), flush=True)
        try:
            data = json.loads(js)
            print(f"  parsed: keys={list(data.keys())}", flush=True)
        except Exception as e:
            print(f"  parse error: {e}", flush=True)


if __name__ == "__main__":
    main()


