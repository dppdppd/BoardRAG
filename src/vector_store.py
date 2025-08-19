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
    if pdf:
        results = db.similarity_search_with_score(query, k=k, filter={"source": pdf})
    else:
        results = db.similarity_search_with_score(query, k=k)
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
    return client.get_or_create_collection(name="boardrag_pages")


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


