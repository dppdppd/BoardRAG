from __future__ import annotations

"""
llm_extract_sections.py — Use an LLM to extract section codes/titles for a whole PDF.

Run from repo root inside venv:
  venv\Scripts\python.exe temp_tests\llm_extract_sections.py --pdf "data/HF4 Core Rules.pdf"

Features:
- Reads the entire PDF and gathers candidate header lines (by regex) with page numbers
- Sends candidates in batches to the configured LLM (OpenAI/Anthropic/Ollama)
- The LLM returns a JSON array of sections: [{code, title, first_page}]
- Merges batches and prints a final JSON summary and a short table

Notes:
- Temperature is fixed to 0 for determinism
- We enforce a strict JSON output in the prompt
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass


# Lazy provider setup mirroring src/query.py
try:
    from src import config as cfg  # type: ignore
except Exception:  # pragma: no cover
    class _CfgShim:
        LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
        GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-4o-mini")
        OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    cfg = _CfgShim()  # type: ignore


def _load_pdf_pages(pdf_path: str) -> List[str]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception as e:  # pragma: no cover
            raise SystemExit(f"Failed to import pypdf/PyPDF2: {e}")

    reader = PdfReader(pdf_path)
    out: List[str] = []
    for page in getattr(reader, "pages", []):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        # normalize NBSP and spacing
        if text:
            text = text.replace("\u00a0", " ")
        out.append(text)
    return out


HEADER_PATTERNS: List[re.Pattern[str]] = [
    # Numeric dotted: 3.0, 3.2.1
    re.compile(r"^\s*(\d+(?:\.[A-Za-z0-9]+)+)\b"),
    # Letter-first alphanum: F4, A10.2, I4a, F3.1b
    re.compile(r"^\s*([A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?)\b"),
    # Digit-first alphanum: 1B6, 1A5a, 2D2
    re.compile(r"^\s*(\d+[A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?)\b"),
]


def _gather_candidate_lines(pages: List[str]) -> List[Tuple[int, str]]:
    candidates: List[Tuple[int, str]] = []
    for idx, text in enumerate(pages, start=1):  # 1-based page numbers
        if not text:
            continue
        for raw in text.splitlines():
            s = raw.strip()
            if len(s) < 2:
                continue
            for pat in HEADER_PATTERNS:
                if pat.match(s):
                    # Keep the whole line; the model will extract code/title
                    candidates.append((idx, s))
                    break
    return candidates


def _make_llm() -> Any:
    provider = (cfg.LLM_PROVIDER or "").lower()
    model_name = getattr(cfg, "GENERATOR_MODEL", None) or "gpt-4o-mini"
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # type: ignore
        return ChatOpenAI(model=model_name, temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # type: ignore
        return ChatAnthropic(model=model_name, temperature=0, max_tokens=2048)
    if provider == "ollama":
        from langchain_community.llms.ollama import Ollama  # type: ignore
        return Ollama(model=model_name, base_url=getattr(cfg, "OLLAMA_URL", "http://localhost:11434"))
    raise SystemExit(f"Unsupported LLM_PROVIDER: {cfg.LLM_PROVIDER}")


SYSTEM_MSG = (
    "You are an expert in parsing boardgame rulebooks. "
    "Given candidate header lines with page numbers from a single PDF, "
    "extract section codes and titles."
)

USER_INSTRUCTIONS = (
    "Extract sections as JSON array. Rules:\n"
    "- Only include lines that contain a valid code: numeric dotted (e.g., 3.5, 3.5.1), letter-first alphanum (e.g., F4, F4.a, A10.2b), or digit-first alphanum (e.g., 1B6, 1B6b).\n"
    "- Normalize code to the exact visible code, including trailing letters if present.\n"
    "- Group duplicates by code; set first_page to the smallest page number where the code appears.\n"
    "- Title is the human-readable label after the code on the same line (if present); omit trailing punctuation like ':' or '.'.\n"
    "- STRICT OUTPUT: Return ONLY a JSON array of objects with fields: code (string), title (string), first_page (integer). No extra text.\n"
)


def _prompt_for_batch(batch: List[Tuple[int, str]]) -> str:
    lines = [f"{p}: {t}" for (p, t) in batch]
    joined = "\n".join(lines)
    example = (
        "Example input lines:\n"
        "12: I4. Boost Operation\n"
        "16: F3 Wet Mass Adjustment\n"
        "19: 1B6. Freighter Promotion & Mobile Factories\n\n"
        "Example output JSON:\n"
        "[\n  {\"code\": \"I4\", \"title\": \"Boost Operation\", \"first_page\": 12},\n"
        "  {\"code\": \"F3\", \"title\": \"Wet Mass Adjustment\", \"first_page\": 16},\n"
        "  {\"code\": \"1B6\", \"title\": \"Freighter Promotion & Mobile Factories\", \"first_page\": 19}\n]"
    )
    return f"{USER_INSTRUCTIONS}\n\nCANDIDATE LINES (page: text)\n{joined}\n\n{example}"


def _run_llm_extract(llm: Any, batch: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
    from langchain.schema import HumanMessage, SystemMessage  # type: ignore
    prompt = _prompt_for_batch(batch)
    msgs = [SystemMessage(content=SYSTEM_MSG), HumanMessage(content=prompt)]
    out = llm.invoke(msgs)
    text = getattr(out, "content", None) or str(out)
    # Expect strict JSON; if not, try to find the first JSON array substring
    s = text.strip()
    if not (s.startswith("[") and s.endswith("]")):
        m = re.search(r"\[.*\]", s, flags=re.DOTALL)
        if m:
            s = m.group(0)
    try:
        data = json.loads(s)
        if isinstance(data, list):
            # Basic validation and cleaning
            cleaned: List[Dict[str, Any]] = []
            for it in data:
                try:
                    code = str(it.get("code") or "").strip()
                    title = str(it.get("title") or "").strip()
                    first_page = int(it.get("first_page"))
                except Exception:
                    continue
                if not code or not isinstance(first_page, int):
                    continue
                cleaned.append({"code": code, "title": title, "first_page": first_page})
            return cleaned
    except Exception:
        pass
    return []


def _merge_sections(batches: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for arr in batches:
        for it in arr:
            code = str(it.get("code") or "").strip()
            if not code:
                continue
            title = str(it.get("title") or "").strip()
            page = int(it.get("first_page") or 0)
            prev = merged.get(code)
            if prev is None:
                merged[code] = {"code": code, "title": title, "first_page": page}
            else:
                # Keep earliest page; prefer non-empty title
                if isinstance(page, int) and (not isinstance(prev.get("first_page"), int) or page < prev["first_page"]):
                    prev["first_page"] = page
                if title and not prev.get("title"):
                    prev["title"] = title
    # Sort by logical order: numeric path if possible otherwise lexicographic
    def _key(it: Dict[str, Any]):
        c = str(it.get("code") or "")
        m_num = re.match(r"^(\d+(?:\.[\dA-Za-z]+)*)$", c)
        if m_num:
            parts = tuple((int(p) if p.isdigit() else p) for p in c.split("."))
            return (0, parts, c)
        return (1, c)
    return sorted(merged.values(), key=_key)


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-extract section codes/titles from a PDF")
    ap.add_argument("--pdf", required=True, help="Path to PDF file")
    ap.add_argument("--max-lines-per-batch", type=int, default=250)
    ap.add_argument("--json", action="store_true", help="Print only JSON")
    args = ap.parse_args()

    pdf_path = os.path.normpath(args.pdf)
    print(f"Loading PDF: {pdf_path}")
    pages = _load_pdf_pages(pdf_path)
    print(f"Loaded {len(pages)} pages")

    print("\nScanning for candidate header lines …")
    cands = _gather_candidate_lines(pages)
    print(f"Found {len(cands)} candidate lines")
    if not cands:
        print("No candidates found; aborting")
        return

    # Batch candidates to fit prompt window
    batch_size = max(20, int(args.max_lines_per_batch))
    batches: List[List[Tuple[int, str]]] = [cands[i:i+batch_size] for i in range(0, len(cands), batch_size)]
    print(f"Processing {len(batches)} batch(es) with provider={cfg.LLM_PROVIDER}, model={getattr(cfg, 'GENERATOR_MODEL', '')}")

    llm = _make_llm()
    results_per_batch: List[List[Dict[str, Any]]] = []
    for bi, batch in enumerate(batches, 1):
        print(f"  • Batch {bi}/{len(batches)} with {len(batch)} lines …")
        out = _run_llm_extract(llm, batch)
        print(f"    → extracted {len(out)} sections")
        results_per_batch.append(out)

    merged = _merge_sections(results_per_batch)
    if args.json:
        print(json.dumps(merged, ensure_ascii=False, indent=2))
        return

    print("\nExtracted sections (merged):")
    for i, it in enumerate(merged, 1):
        code = it.get("code")
        title = it.get("title")
        page = it.get("first_page")
        print(f"{i:>3}. p{page:<3} {code:<10} | {title}")

    print(f"\nTotal unique sections: {len(merged)}")


if __name__ == "__main__":
    main()


