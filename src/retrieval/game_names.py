from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFLoader

from ..config import CHROMA_PATH, get_chromadb_settings, suppress_chromadb_telemetry
from ..embedding_function import get_embedding_function
from .. import config as cfg
from langchain_chroma import Chroma
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
    try:
        with suppress_chromadb_telemetry():
            persistent_client = chromadb.PersistentClient(path=CHROMA_PATH, settings=get_chromadb_settings())
            game_names_collection = persistent_client.get_or_create_collection(
                name="game_names",
                metadata={"description": "Stores extracted game names from PDF filenames"},
            )
            game_names_collection.upsert(
                ids=[filename],
                documents=[game_name],
                metadatas=[{"filename": filename, "game_name": game_name}],
            )
            print(f"✅ Stored game name: '{game_name}' for '{filename}'")
    except Exception as e:
        print(f"❌ Error storing game name for {filename}: {e}")


def get_stored_game_names() -> Dict[str, str]:
    try:
        with suppress_chromadb_telemetry():
            persistent_client = chromadb.PersistentClient(path=CHROMA_PATH, settings=get_chromadb_settings())
            try:
                game_names_collection = persistent_client.get_collection("game_names")
                results = game_names_collection.get()
                filename_to_game: Dict[str, str] = {}
                for filename, game_name in zip(results["ids"], results["documents"]):
                    filename_to_game[filename] = game_name
                return filename_to_game
            except Exception:
                return {}
    except Exception as e:
        print(f"❌ Error retrieving stored game names: {e}")
        return {}


def extract_and_store_game_name(filename: str) -> str:
    stored_names = get_stored_game_names()
    if filename in stored_names:
        return stored_names[filename]
    game_name = extract_game_name_from_filename(filename)
    store_game_name(filename, game_name)
    return game_name


def get_available_games() -> List[str]:
    try:
        embedding_function = get_embedding_function()
        with suppress_chromadb_telemetry():
            persistent_client = chromadb.PersistentClient(path=CHROMA_PATH, settings=get_chromadb_settings())
            db = Chroma(client=persistent_client, embedding_function=embedding_function)
        all_docs = db.get()
        if not hasattr(get_available_games, '_last_count') or len(all_docs['ids']) == 0:
            print(f"[DEBUG] get_available_games: Main collection has {len(all_docs['ids'])} documents")
            if len(all_docs['ids']) == 0:
                print(f"[DEBUG] get_available_games: Main collection is EMPTY - this is the problem!")
            get_available_games._last_count = len(all_docs['ids'])  # type: ignore[attr-defined]

        filenames = set()
        for doc_id in all_docs["ids"]:
            if ":" in doc_id:
                source_path = doc_id.split(":")[0]
                if "/" in source_path or "\\" in source_path:
                    filename = os.path.basename(source_path)
                    if filename.endswith(".pdf"):
                        filenames.add(filename)

        if not filenames:
            try:
                data_root = Path(CHROMA_PATH).parent
                disk_files: List[str] = []
                try:
                    from .. import config as _cfg
                    data_root = Path(_cfg.DATA_PATH)
                except Exception:
                    pass
                if data_root.exists():
                    disk_files = [p.name for p in data_root.rglob("*.pdf")]
                stored_names = get_stored_game_names()
                for fname in disk_files or stored_names.keys():
                    base = os.path.basename(fname)
                    if base.endswith(".pdf"):
                        filenames.add(base)
            except Exception:
                pass

        stored_names = get_stored_game_names()
        games: List[str] = []
        game_to_files: Dict[str, List[str]] = {}
        for filename in sorted(filenames):
            if filename in stored_names:
                proper_name = stored_names[filename]
            else:
                try:
                    proper_name = extract_and_store_game_name(filename)
                except Exception:
                    proper_name = improve_fallback_name(filename)
            simple_name = filename.replace(".pdf", "").lower().replace(" ", "_")
            if proper_name not in games:
                games.append(proper_name)
            game_to_files.setdefault(proper_name, []).append(simple_name)

        get_available_games._filename_mapping = game_to_files  # type: ignore[attr-defined]
        return sorted(games)
    except Exception as e:
        print(f"Error getting available games: {e}")
        return []


