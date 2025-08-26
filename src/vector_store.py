from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json
import os
import logging


_DB_CACHE = None


def _get_chroma(path: Optional[str] = None):
    global _DB_CACHE
    if _DB_CACHE is not None:
        return _DB_CACHE
    # Ensure telemetry is disabled BEFORE importing chromadb
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
    os.environ.setdefault("CHROMA_TELEMETRY_ANONYMIZED", "false")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
    import chromadb  # type: ignore
    from langchain_chroma import Chroma  # type: ignore
    from .embeddings import embed_texts
    from . import config as cfg  # type: ignore
    # Disable OpenTelemetry-based spew in Chroma (redundant safeguard)
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    try:
        logging.getLogger("chromadb").setLevel(logging.ERROR)
        logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
    except Exception:
        pass
    try:
        import chromadb.telemetry as _telemetry  # type: ignore
        def _noop(*args, **kwargs):
            return None
        # Patch common capture sites
        try:
            if hasattr(_telemetry, "capture"):
                _telemetry.capture = _noop  # type: ignore
        except Exception:
            pass
        try:
            if hasattr(_telemetry, "telemetry") and hasattr(_telemetry.telemetry, "capture"):
                _telemetry.telemetry.capture = _noop  # type: ignore
        except Exception:
            pass
        try:
            if hasattr(_telemetry, "posthog"):
                ph = _telemetry.posthog
                try:
                    setattr(ph, "capture", _noop)  # type: ignore
                except Exception:
                    class _PH:
                        def capture(self, *args, **kwargs):
                            return None
                        def flush(self, *args, **kwargs):
                            return None
                    _telemetry.posthog = _PH()  # type: ignore
        except Exception:
            pass
    except Exception:
        pass

    chroma_path = path or getattr(cfg, "CHROMA_PATH", "chroma")
    try:
        from chromadb.config import Settings  # type: ignore
        client = chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))
    except Exception:
        client = chromadb.PersistentClient(path=chroma_path)

    # Ensure HNSW search_ef is high enough to support larger k queries
    try:
        coll = client.get_or_create_collection(name="boardrag_pages")
        try:
            _ef = int(os.getenv("CHROMA_SEARCH_EF", "64"))
        except Exception:
            _ef = 64
        try:
            coll.modify(metadata={"hnsw:search_ef": _ef})
        except Exception:
            # Some versions may not allow modifying ef at runtime; ignore
            pass
    except Exception:
        pass

    # Minimal adapter to LangChain Chroma: provide embedding_function with proper interface
    class _EmbeddingFn:
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return embed_texts(texts)
        def embed_query(self, text: str) -> List[float]:
            return embed_texts([text])[0]

    db = Chroma(client=client, collection_name="boardrag_pages", embedding_function=_EmbeddingFn())
    _DB_CACHE = db
    return _DB_CACHE


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for k, v in (metadata or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            sanitized[k] = v
        else:
            # Chroma expects scalars; store complex types as JSON strings
            try:
                sanitized[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                sanitized[k] = str(v)
    return sanitized


def upsert_page_chunk(doc_id: str, text: str, metadata: Dict[str, Any]) -> None:
    db = _get_chroma()
    md = _sanitize_metadata(metadata)
    db.add_texts([text], metadatas=[md], ids=[doc_id])


def get_chunk_by_page(pdf_basename: str, page_1based: int) -> Optional[Tuple[Any, float]]:
    db = _get_chroma()
    doc_id = f"{pdf_basename}#p{page_1based}"
    try:
        got = db.get(ids=[doc_id])
        # LangChain Chroma.get returns a dict of lists
        ids = (got or {}).get("ids") or []
        docs = (got or {}).get("documents") or []
        metas = (got or {}).get("metadatas") or []
        if not ids:
            return None
        from langchain.schema import Document  # type: ignore
        d = Document(page_content=docs[0] or "", metadata=metas[0] or {})
        return (d, 0.0)
    except Exception:
        return None


def search_chunks(query: str, *, pdf: Optional[str] = None, k: int = 8) -> List[Tuple[Any, float]]:
    db = _get_chroma()
    # Try to bias server-side by filtering on search hints if possible; fall back to plain search
    filt = {"source": pdf} if pdf else None
    try:
        # Use server-side metadata filter when available
        results = db.similarity_search_with_score(query, k=k, filter=filt) if filt else db.similarity_search_with_score(query, k=k)
    except Exception:
        # Some langchain-chroma versions do not support 'filter' – do NOT fall back to global mixing.
        # Instead, over-fetch and filter client-side by exact 'source'.
        raw = db.similarity_search_with_score(query, k=max(k, 24))
        if pdf:
            target = str(pdf).strip().lower()
            def _is_match(doc) -> bool:
                try:
                    meta = getattr(doc, 'metadata', {}) or {}
                    src = str(meta.get('source') or '').strip().lower()
                    return src == target
                except Exception:
                    return False
            results = [(d, s) for (d, s) in raw if _is_match(d)]
        else:
            results = raw
    # Client-side re-rank: boost chunks whose metadata search hints contain query terms
    try:
        import re as _re
        q = (query or "").lower()
        tokens = [t for t in _re.split(r"[^a-z0-9]+", q) if t]
        def _boost(doc, score):
            meta = getattr(doc, 'metadata', {}) or {}
            # Field-specific weights (summary and headers strongest)
            field_weights = {
                # Strongly prioritize dense summary and curated vocab
                "embedding_prefix_summary": 8.0,
                "search_keywords": 7.0,
                "search_synonyms": 5.0,
                # Human-readable headers and section names (from LLM extraction)
                # These directly capture phrases like "INFILTRATION & CLOSE COMBAT"
                "section_start_by_code": 9.0,
                # New: per-section title/summary extracted by LLM
                "section_titles": 9.5,
                "section_summaries": 8.5,
                # Namespaced, normalized header ids also include short slugs like
                # "20-infiltration-close-xxxx" which are highly indicative
                "primary_section_id2": 7.0,
                "section_id2_by_code": 6.0,
                # Raw structured sections JSON contains header lines as provided by the LLM
                "sections_raw": 6.0,
                # Keep numeric header codes meaningful but secondary
                "embedding_prefix_headers": 2.0,
                # Helpful, but lighter influence
                "embedding_prefix_search_hints": 2.0,
                "search_rules": 1.5,
                "search_questions": 0.5,
            }
            # Build lowercase blobs per field
            blobs: dict[str, str] = {}
            for key in field_weights.keys():
                v = meta.get(key)
                if isinstance(v, str) and v:
                    blobs[key] = v.lower()
            # Stopwords to ignore in token matches
            stop = {
                "the","a","an","and","or","of","to","in","on","for","with",
                "what","how","when","where","why","who","explain","tell","me","about","please",
                # Additional function words and auxiliaries to reduce noise
                "is","are","was","were","be","been","being","do","does","did",
                "can","could","should","would","may","might","must","will","shall",
                # Frequently unhelpful in intent questions
                "use","uses","using"
            }
            toks = [t for t in tokens if t and (t not in stop)]
            # Build simple bigram/trigram phrases from filtered tokens for phrase-level boosting
            phrases: set[str] = set()
            for i in range(len(toks) - 1):
                p = f"{toks[i]} {toks[i+1]}".strip()
                if len(p) >= 5:
                    phrases.add(p)
            for i in range(len(toks) - 2):
                p = f"{toks[i]} {toks[i+1]} {toks[i+2]}".strip()
                if len(p) >= 7:
                    phrases.add(p)
            # Basic plural/singular normalization: test token as-is and common variants
            def _variants(tok: str):
                t = tok
                out = {t}
                # plural → singular
                if t.endswith("ies") and len(t) > 3:
                    out.add(t[:-3] + "y")
                if t.endswith("es") and len(t) > 2:
                    out.add(t[:-2])
                if t.endswith("s") and len(t) > 1:
                    out.add(t[:-1])
                # singular → plural
                if t.endswith("y") and len(t) > 1 and t[-2] not in "aeiou":
                    out.add(t[:-1] + "ies")
                elif t.endswith(("s","x","z","ch","sh")):
                    out.add(t + "es")
                else:
                    out.add(t + "s")
                return out
            weighted_hits = 0.0
            for key, blob in blobs.items():
                w = field_weights.get(key, 1.0)
                matched_any = False
                for t in toks:
                    vars = _variants(t)
                    if any(v in blob for v in vars):
                        matched_any = True
                phrase_hits = 0
                if phrases:
                    for ph in phrases:
                        if ph in blob:
                            phrase_hits += 1
                if matched_any:
                    weighted_hits += w
                if phrase_hits:
                    # Give additional credit for phrase matches in stronger fields
                    weighted_hits += min(phrase_hits, 3) * (w * 0.6)
            # Lower score is better in LC-Chroma; apply a stronger negative offset per field hit
            return score - (0.20 * weighted_hits)
        results = [(d, _boost(d, s)) for (d, s) in results]
        results.sort(key=lambda pair: pair[1])
    except Exception:
        pass
    return results


def count_processed_pages(pdf_stem: str, total_pages: int) -> int:
    """Count how many page chunk ids exist for a given PDF stem.

    Uses Chroma .get(ids=[...]) in a simple loop for reliability.
    """
    db = _get_chroma()
    present = 0
    # Use current schema version as filter for "processed"
    try:
        from .chunk_schema import SCHEMA_VERSION  # type: ignore
        required_version = str(SCHEMA_VERSION)
    except Exception:
        required_version = "v1"
    for i in range(max(0, int(total_pages))):
        # 1-based page ids
        doc_id = f"{pdf_stem}#p{i+1}"
        try:
            got = db.get(ids=[doc_id])
            ids = (got or {}).get("ids") or []
            metas = (got or {}).get("metadatas") or []
            if not ids:
                continue
            md = metas[0] if metas else {}
            ver = str((md or {}).get("version") or "")
            if ver == required_version:
                present += 1
        except Exception:
            continue
    return present


def _get_native_collection(path: Optional[str] = None):
    # Ensure telemetry disabled at import time for native client
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
    os.environ.setdefault("CHROMA_TELEMETRY_ANONYMIZED", "false")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
    import chromadb  # type: ignore
    try:
        from chromadb.config import Settings  # type: ignore
        from . import config as cfg  # type: ignore
        chroma_path = path or getattr(cfg, "CHROMA_PATH", "chroma")
        client = chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))
    except Exception:
        from . import config as cfg  # type: ignore
        chroma_path = path or getattr(cfg, "CHROMA_PATH", "chroma")
        client = chromadb.PersistentClient(path=chroma_path)
    # Use fixed collection name
    coll = client.get_or_create_collection(name="boardrag_pages")
    # Attempt to raise search_ef for better recall and to support larger k
    try:
        try:
            _ef = int(os.getenv("CHROMA_SEARCH_EF", "64"))
        except Exception:
            _ef = 64
        coll.modify(metadata={"hnsw:search_ef": _ef})
    except Exception:
        pass
    return coll


def clear_pdf_chunks(pdf_filename: str) -> int:
    """Delete all chunks for a given PDF filename (metadata source exact match)."""
    coll = _get_native_collection()
    got = coll.get(where={"source": pdf_filename})  # type: ignore[arg-type]
    ids = (got or {}).get("ids") or []
    if not ids:
        return 0
    coll.delete(ids=ids)
    return len(ids)


def get_metadata_for_page(pdf_stem: str, page_1based: int) -> Optional[Dict[str, Any]]:
    """Return metadata dict for a given page id or None if missing."""
    coll = _get_native_collection()
    doc_id = f"{pdf_stem}#p{page_1based}"
    got = coll.get(ids=[doc_id])
    ids = (got or {}).get("ids") or []
    metas = (got or {}).get("metadatas") or []
    if not ids:
        return None
    return metas[0] if metas else {}


def is_current_page_chunk(pdf_stem: str, page_1based: int, required_version: str, page_pdf_sha256: str) -> bool:
    """Return True if this page chunk exists with matching version and page hash."""
    md = get_metadata_for_page(pdf_stem, page_1based)
    if not md:
        return False
    try:
        ver = str((md or {}).get("version") or "")
        ph = str((md or {}).get("page_pdf_sha256") or "")
        return (ver == str(required_version)) and (ph == str(page_pdf_sha256))
    except Exception:
        return False


