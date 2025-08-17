from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFLoader

from .. import config as cfg
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


def normalize_game_title(title: str) -> str:
    stripped_title = title.strip()
    if stripped_title.lower().startswith("the "):
        return stripped_title[4:] + ", The"
    if stripped_title.lower().startswith("a "):
        return stripped_title[2:] + ", A"
    return stripped_title


def extract_game_name_from_filename(filename: str, debug: bool = False) -> str:
    pdf_context = ""
    try:
        from ..config import DATA_PATH

        pdf_path = Path(DATA_PATH) / filename
        if pdf_path.exists():
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            pages_limit = int(os.getenv("NAME_EXTRACTION_PAGES", 10))
            pages_to_read = min(pages_limit, len(documents))
            page_texts: List[str] = []
            for i in range(pages_to_read):
                page_text = documents[i].page_content.strip()
                if page_text:
                    page_texts.append(page_text[:500])
            if page_texts:
                pdf_context = "\n\n".join(page_texts)
    except Exception as e:
        print(f"⚠️ Could not read PDF content from {filename}: {e}")
        pdf_context = ""

    max_retries = 3
    last_raw_response = None
    for attempt in range(max_retries):
        try:
            if cfg.LLM_PROVIDER.lower() == "anthropic":
                model = ChatAnthropic(model=cfg.GENERATOR_MODEL, temperature=0)
            elif cfg.LLM_PROVIDER.lower() == "ollama":
                from langchain_community.llms.ollama import Ollama

                model = Ollama(model=cfg.GENERATOR_MODEL, base_url=cfg.OLLAMA_URL)
            else:
                # use default temperature=1 for o3
                temp = 1 if str(cfg.GENERATOR_MODEL).lower().startswith("o3") else 0
                model = ChatOpenAI(model=cfg.GENERATOR_MODEL, temperature=temp, timeout=60)

            context_section = f"\n\nCONTENT FROM FIRST PAGES:\n{pdf_context}\n\n" if pdf_context else ""
            prompt = (
                f""" Extract the proper board game name from this filename: "{filename}"{context_section}

Guidelines:
- Return ONLY the official published game name with no preamble or formatting.
- Remove file-related words: "rules", "manual", "rulebook", "complete", "rework", "bw", "color", "v1", "v2"
- If an acronym, find a possible name from the PDF content
- Use proper capitalization for official game titles
- If you see the actual game title in the PDF content, prefer that over filename guessing

Filename: {filename}
Official game name:"""
            )
            response = model.invoke(prompt)
            last_raw_response = getattr(response, "content", str(response))
            game_name = str(last_raw_response).strip().strip("\"'")
            if "\n" in game_name:
                game_name = game_name.split("\n")[0].strip()
            if game_name and len(game_name) <= 50:
                normalized_name = normalize_game_title(game_name)
                print(f"Successfully extracted game name: '{normalized_name}' from '{filename}'")
                return normalized_name
            else:
                raise ValueError("Invalid response")
        except Exception as e:
            if attempt < max_retries - 1 and any(s in str(e).lower() for s in ["overload", "rate limit", "529", "503"]):
                import random, time

                delay = (2 ** attempt) + random.uniform(0, 1)
                print(f"API overloaded (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            print(f"LLM extraction failed for {filename}: {e}")
            break

    fallback_name = filename.replace(".pdf", "").replace("-", " ").replace("_", " ").title()
    normalized_fallback = normalize_game_title(fallback_name)
    print(f"Using fallback name: '{normalized_fallback}' for '{filename}'")
    return normalized_fallback


def improve_fallback_name(filename: str) -> str:
    return filename.replace(".pdf", "").replace("-", " ").replace("_", " ").title()


def store_game_name(filename: str, game_name: str) -> None:
    # Catalog-only storage in DB-less mode
    try:
        from ..catalog import load_catalog, save_catalog, _now_iso  # type: ignore
        cat = load_catalog()
        key = Path(filename).name
        entry = cat.get(key) or {}
        entry["game_name"] = game_name
        entry["updated_at"] = _now_iso()
        cat[key] = entry
        save_catalog(cat)
        print(f"✅ Catalog: stored game name '{game_name}' for '{key}'")
    except Exception:
        pass


def get_stored_game_names() -> Dict[str, str]:
    # Catalog-based mapping only
    try:
        from ..catalog import load_catalog  # type: ignore
        cat = load_catalog()
        out: Dict[str, str] = {}
        for fname, meta in cat.items():
            g = (meta or {}).get("game_name")
            if isinstance(g, str) and g.strip():
                out[Path(fname).name] = g
        return out
    except Exception:
        return {}


def extract_and_store_game_name(filename: str) -> str:
    stored_names = get_stored_game_names()
    if filename in stored_names:
        return stored_names[filename]
    game_name = extract_game_name_from_filename(filename)
    store_game_name(filename, game_name)
    return game_name


def get_available_games() -> List[str]:
    # Catalog-only listing
    try:
        from ..catalog import list_games_from_catalog  # type: ignore
        return list_games_from_catalog()
    except Exception:
        return []


