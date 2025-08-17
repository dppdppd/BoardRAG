"""
Python file that populates the database with the documents from the data folder, along with their embeddings.
Embeddings are calculated using either OpenAI or a local Ollama model, depending
on the `LLM_PROVIDER` environment variable.
The documents are split into chunks and added to the database, once the embedding has been calculated.
"""

import argparse
import os
import shutil
import warnings
import re
from typing import List, Tuple
import concurrent.futures

import chromadb
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF

# Handle both direct execution and module import
try:
    # When run as a module (python -m src.populate_database)
    from .config import (
        CHROMA_PATH,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        DATA_PATH,
        disable_chromadb_telemetry,
        get_chromadb_settings,
        suppress_chromadb_telemetry,
        validate_config,
    )
    from .embedding_function import get_embedding_function
    from .pdf_utils import optimize_with_raster_fallback_if_large
    # Best-effort .env loader for API keys when using LLM splitter
    try:
        from dotenv import load_dotenv, find_dotenv  # type: ignore
        load_dotenv(find_dotenv(), override=False)
    except Exception:
        pass
except ImportError:
    # When run directly (python src/populate_database.py)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.config import (
        CHROMA_PATH,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        DATA_PATH,
        disable_chromadb_telemetry,
        get_chromadb_settings,
        suppress_chromadb_telemetry,
        validate_config,
    )
    from src.embedding_function import get_embedding_function
    from src.pdf_utils import optimize_with_raster_fallback_if_large
    try:
        from dotenv import load_dotenv, find_dotenv  # type: ignore
        load_dotenv(find_dotenv(), override=False)
    except Exception:
        pass

# Ignore deprecation warnings.
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Disable ChromaDB telemetry and validate configuration after imports
disable_chromadb_telemetry()
validate_config()


def load_documents(target_paths: List[str] | None = None):
    """Load PDF documents from the specified *target_paths*.

    If *target_paths* is ``None`` or empty, the entire ``DATA_PATH`` directory is
    scanned (current behaviour). Otherwise, *target_paths* can contain either
    filenames or sub-directories **relative to** ``DATA_PATH`` or absolute
    paths. This allows callers to load **only the newly-added game rulebooks**,
    dramatically reducing the work that needs to be done when expanding the
    library.

    Args:
        target_paths (List[str] | None): Files or directories to load. If not
            provided, the whole ``DATA_PATH`` tree is loaded.

    Returns:
        List[Document]: The loaded documents.
    """

    # Default: load everything (original behaviour)
    if not target_paths:
        # Scan for PDF files first to show progress
        from pathlib import Path
        
        pdf_files = list(Path(DATA_PATH).rglob("*.pdf"))
        print(f"📁 Found {len(pdf_files)} PDF files to process...")
        
        documents: List[Document] = []
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"📖 Loading PDF {i}/{len(pdf_files)}: {pdf_file.name}")
            try:
                # Optionally optimize large PDFs before loading
                try:
                    from . import config as _cfg  # type: ignore
                except Exception:
                    import src.config as _cfg  # type: ignore
                if getattr(_cfg, "ENABLE_PDF_OPTIMIZATION", False):
                    try:
                        replaced, orig, opt, msg = optimize_with_raster_fallback_if_large(
                            pdf_file,
                            min_size_mb=getattr(_cfg, "PDF_OPTIMIZE_MIN_SIZE_MB", 25.0),
                            linearize=getattr(_cfg, "PDF_LINEARIZE", True),
                            garbage_level=getattr(_cfg, "PDF_GARBAGE_LEVEL", 3),
                            enable_raster_fallback=getattr(_cfg, "PDF_ENABLE_RASTER_FALLBACK", False),
                            raster_dpi=getattr(_cfg, "PDF_RASTER_DPI", 150),
                            jpeg_quality=getattr(_cfg, "PDF_JPEG_QUALITY", 70),
                        )
                        if replaced or opt < orig:
                            print(f"   🛠 Optimized {pdf_file.name}: {msg}")
                    except Exception as _e:
                        print(f"   ⚠️ Optimization failed for {pdf_file.name}: {_e}")
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                print(f"   ✅ Loaded {len(docs)} pages from {pdf_file.name}")
                documents.extend(docs)
            except Exception as e:
                print(f"   ❌ Failed to load {pdf_file.name}: {e}")
        
        print(f"📚 Total documents loaded: {len(documents)} pages from {len(pdf_files)} PDFs")
        return documents

    documents: List[Document] = []

    for path in target_paths:
        
        # Normalise path for safe comparisons on any OS
        norm_path = os.path.normpath(path)

        # If the caller already passed a full path (absolute **or** already under DATA_PATH)
        # we leave it untouched. Otherwise we treat it as relative to DATA_PATH.
        if os.path.isabs(norm_path) or norm_path.startswith(os.path.normpath(DATA_PATH)):
            full_path = norm_path
        else:
            full_path = os.path.join(DATA_PATH, norm_path)

        if os.path.isdir(full_path):
            # Optional optimization for each PDF in directory
            try:
                from . import config as _cfg  # type: ignore
            except Exception:
                import src.config as _cfg  # type: ignore
            if getattr(_cfg, "ENABLE_PDF_OPTIMIZATION", False):
                from pathlib import Path as _P
                for _p in _P(full_path).rglob("*.pdf"):
                    try:
                        optimize_with_raster_fallback_if_large(
                            _p,
                            min_size_mb=getattr(_cfg, "PDF_OPTIMIZE_MIN_SIZE_MB", 25.0),
                            linearize=getattr(_cfg, "PDF_LINEARIZE", True),
                            garbage_level=getattr(_cfg, "PDF_GARBAGE_LEVEL", 3),
                            enable_raster_fallback=getattr(_cfg, "PDF_ENABLE_RASTER_FALLBACK", False),
                            raster_dpi=getattr(_cfg, "PDF_RASTER_DPI", 150),
                            jpeg_quality=getattr(_cfg, "PDF_JPEG_QUALITY", 70),
                        )
                    except Exception:
                        pass
            documents.extend(PyPDFDirectoryLoader(full_path).load())
        elif os.path.isfile(full_path):
            # Optional optimization for single file
            try:
                from . import config as _cfg  # type: ignore
            except Exception:
                import src.config as _cfg  # type: ignore
            if getattr(_cfg, "ENABLE_PDF_OPTIMIZATION", False) and full_path.lower().endswith(".pdf"):
                try:
                    optimize_with_raster_fallback_if_large(
                        full_path,
                        min_size_mb=getattr(_cfg, "PDF_OPTIMIZE_MIN_SIZE_MB", 25.0),
                        linearize=getattr(_cfg, "PDF_LINEARIZE", True),
                        garbage_level=getattr(_cfg, "PDF_GARBAGE_LEVEL", 3),
                        enable_raster_fallback=getattr(_cfg, "PDF_ENABLE_RASTER_FALLBACK", False),
                        raster_dpi=getattr(_cfg, "PDF_RASTER_DPI", 150),
                        jpeg_quality=getattr(_cfg, "PDF_JPEG_QUALITY", 70),
                    )
                except Exception:
                    pass
            documents.extend(PyPDFLoader(full_path).load())
        else:
            print(f"⚠️  Path not found – skipping: {full_path}")

    return documents


def _resolve_target_pdfs(target_paths: List[str] | None) -> List[str]:
    """Return a list of absolute PDF paths to process based on target_paths.

    - If target_paths is None/empty: scan entire DATA_PATH for PDFs
    - If entries are directories: include all PDFs under them
    - If entries are files: include them if they exist and are PDFs
    """
    from pathlib import Path
    pdfs: List[str] = []
    if not target_paths:
        pdfs = [str(p) for p in Path(DATA_PATH).rglob("*.pdf")]
        return pdfs
    for path in target_paths:
        norm_path = os.path.normpath(path)
        if os.path.isabs(norm_path) or norm_path.startswith(os.path.normpath(DATA_PATH)):
            full_path = norm_path
        else:
            full_path = os.path.join(DATA_PATH, norm_path)
        if os.path.isdir(full_path):
            pdfs.extend(str(p) for p in Path(full_path).rglob("*.pdf"))
        elif os.path.isfile(full_path) and full_path.lower().endswith(".pdf"):
            pdfs.append(full_path)
        else:
            print(f"⚠️  Path not found or not a PDF – skipping: {full_path}")
    return pdfs


def split_documents_llm(documents: List[Document]) -> List[Document]:
    """
    LLM-first splitting: use an LLM to extract section codes/titles per PDF, then
    segment pages by those codes. Falls back to regex splitter on failure.

    Implementation outline:
    1) Group docs by source PDF
    2) For each PDF, extract candidate header lines and call the same logic as temp_tests/llm_extract_sections.py
    3) Build a code → first_page map and a sorted sequence to segment pages
    4) Emit sections as Documents with metadata.section and metadata.section_number
    """
    try:
        from src.llm_outline import extract_pdf_outline  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"⚠️ LLM splitter unavailable ({e}); falling back to regex split.")
        return split_documents(documents)

    # Group by source
    by_src: dict[str, List[Document]] = {}
    for d in documents:
        try:
            src = str(d.metadata.get("source") or "")
            if not src:
                continue
            by_src.setdefault(src, []).append(d)
        except Exception:
            continue

    out_docs: List[Document] = []
    for src, pages_docs in by_src.items():
        print(f"🧠 LLM splitting for {os.path.basename(src)} ({len(pages_docs)} pages)…")
        try:
            outline = extract_pdf_outline(src)
            merged = outline.get("sections") or []
            alias_map = outline.get("alias_map") or {}
            fine_objs = outline.get("objects") or []
            game_name = str(outline.get("game_name") or "").strip()
            # Build an ordered list of (first_page, code, title, kind)
            ordered = sorted([(int(s.get("first_page", 0)), str(s.get("code", "")), str(s.get("title", "")), str(s.get("section_kind", ""))) for s in merged if s.get("first_page")], key=lambda t: t[0])
            if not ordered:
                print("   ⚠️ LLM returned no sections; keeping unsplit pages for this PDF")
                # Keep pages as single section chunks to avoid losing data
                for d in pages_docs:
                    out_docs.append(d)
                continue
            # Create page index → accumulated lines, flushing when we hit the next section
            # Map page index (0-based) to raw text from the original Document sequence
            # Note: PyPDFLoader assigns 0-based 'page' in metadata
            def _emit(chunk_lines: List[str], header_label: str, base_meta: dict, section_kind: str):
                content = "\n".join(chunk_lines).strip()
                if not content:
                    return
                meta = dict(base_meta)
                meta["section"] = header_label
                # Derive section_number from code
                m = re.match(r"^\s*([A-Za-z]?\d+(?:\.[A-Za-z0-9]+)*[a-z]?)\b", header_label)
                if m:
                    meta["section_number"] = m.group(1)
                if section_kind:
                    meta["section_kind"] = section_kind
                if alias_map:
                    meta["alias_map"] = alias_map
                if game_name:
                    meta["game_name"] = game_name
                # Attach fine-grained objects belonging to this parent section
                try:
                    parent = str(meta.get("section_number") or "")
                    objs = [o for o in fine_objs if str(o.get("parent_code") or "") == parent]
                    if objs:
                        meta["objects"] = objs
                except Exception:
                    pass
                out_docs.append(Document(page_content=content, metadata=meta))

            # Build a quick page→doc map
            idx_map: dict[int, Document] = {}
            for d in pages_docs:
                try:
                    pg = int(d.metadata.get("page"))
                    idx_map[pg] = d
                except Exception:
                    continue
            # Iterate pages in order; assign current section based on ordered boundaries
            ordered_iter = iter(ordered)
            cur = next(ordered_iter)
            cur_page0 = max(0, int(cur[0]) - 1)
            cur_code = cur[1]
            cur_title = cur[2]
            cur_kind = ordered[0][3]
            cur_header = f"{cur_code} {cur_title}".strip()
            buf: List[str] = []
            last_meta: dict = {}
            max_page = max(idx_map.keys()) if idx_map else -1
            def _advance_to(page1_based: int) -> None:
                nonlocal cur, cur_page0, cur_code, cur_title, cur_header, buf, last_meta
                # If the next section starts at this page, flush current and start new
                while True:
                    try:
                        nxt = next(ordered_iter)
                    except StopIteration:
                        return
                    nxt_page0 = max(0, int(nxt[0]) - 1)
                    if page1_based - 1 >= nxt_page0:
                        # Flush previous if any buffer
                        _emit(buf, cur_header, last_meta, cur_kind)
                        buf = []
                        cur = nxt
                        cur_page0 = nxt_page0
                        cur_code = nxt[1]
                        cur_title = nxt[2]
                        cur_kind = nxt[3]
                        cur_header = f"{cur_code} {cur_title}".strip()
                    else:
                        # Put back by rewinding one step in iterator is hard; instead store state and break
                        # We'll re-use cur and the iterator remains on nxt for future checks
                        # To simulate lookahead, we keep current cur; the emission will occur when crossing the boundary
                        # No action here
                        # Actually, to handle lookahead correctly, we rely on checking on each page and comparing with nxt_page0
                        # Since we can't easily push-back, we preserve nxt as global by re-creating iterator
                        break

            # Process pages in ascending order
            for pg in range(0, max_page + 1):
                d = idx_map.get(pg)
                if not d:
                    continue
                last_meta = dict(d.metadata)
                # On boundary pages, if this page is >= next section start, rotate header
                # Simple approach: check if any future start equals this page+1; re-compute on each step
                for start_page1, code, title, kind in ordered:
                    start0 = max(0, int(start_page1) - 1)
                    if pg == start0 and (code != cur_code):
                        _emit(buf, cur_header, last_meta, cur_kind)
                        buf = []
                        cur_code = code
                        cur_title = title
                        cur_kind = kind
                        cur_header = f"{cur_code} {cur_title}".strip()
                buf.append(d.page_content or "")
            # Flush remainder
            _emit(buf, cur_header, last_meta, cur_kind)
        except Exception as e:
            print(f"   ⚠️ LLM split failed for {os.path.basename(src)}: {e}; keeping unsplit pages.")
            out_docs.extend(pages_docs)

    print(f"✅ LLM splitting complete: {len(documents)} pages → {len(out_docs)} sections")
    # Optionally further split oversized sections character-level
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " "],
        length_function=len,
        is_separator_regex=False,
    )
    final: List[Document] = []
    for sec in out_docs:
        if len(sec.page_content) <= CHUNK_SIZE:
            final.append(sec)
        else:
            final.extend(char_splitter.split_documents([sec]))
    print(f"✅ Phase 2 complete (LLM): {len(out_docs)} sections → {len(final)} final chunks")
    return final

# LLM-only policy: alias legacy splitter name to LLM splitter
split_documents = split_documents_llm


def add_to_chroma(chunks: List[Document]) -> bool:
    """
    Adds chunks passed as argument to the Chroma vector database.

    Args:
        chunks (List[Document]): The chunks to be added to the database.

    Returns:
        bool: True if the chunks were added successfully, False otherwise.
    """
    # Load the existing database with PersistentClient for proper persistence
    with suppress_chromadb_telemetry():
        persistent_client = chromadb.PersistentClient(
            path=CHROMA_PATH, settings=get_chromadb_settings()
        )
        db = Chroma(
            client=persistent_client, embedding_function=get_embedding_function()
        )

    # Calculate Page IDs.
    chunks_with_ids = calc_chunk_ids(chunks)

    # Sanitize metadata (Chroma supports only scalar values). Serialize complex values.
    import json as _json
    def _sanitize_metadata(meta: dict) -> dict:
        safe: dict = {}
        for k, v in (meta or {}).items():
            if isinstance(v, (str, int, float, bool)):
                safe[k] = v
            elif v is None:
                continue
            else:
                try:
                    s = _json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                    if len(s) > 8000:
                        s = s[:8000] + "…"
                    safe[k] = s
                except Exception:
                    sv = str(v)
                    safe[k] = sv[:8000] + ("…" if len(sv) > 8000 else "")
        return safe
    try:
        for ch in chunks_with_ids:
            ch.metadata = _sanitize_metadata(getattr(ch, "metadata", {}) or {})
    except Exception:
        pass

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        from itertools import islice

        def batched(it, n=100):
            """Yield *n*-sized batches from *it*."""
            it = iter(it)
            while True:
                batch = list(islice(it, n))
                if not batch:
                    break
                yield batch

        try:
            total_new = len(new_chunks)
            batches = batched(new_chunks, 100)
            for idx, batch in enumerate(batches, start=1):
                batch_ids = [chunk.metadata["id"] for chunk in batch]
                db.add_documents(batch, ids=batch_ids)
                print(f"   ✅ Added batch {idx}: {len(batch)} chunks")

            print(f"✅ Documents added successfully – {total_new} new chunks")

            # With PersistentClient, persistence is automatic
            print("📝 Using PersistentClient - auto-persistence enabled")

            # Verify documents were actually added
            verification = db.get()
            print(
                f"📊 Verification: Database now contains {len(verification['ids'])} documents"
            )
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            import traceback

            traceback.print_exc()
            return False
    else:
        print("✅ No new documents to add")

    print("Data added successfully.")
    return True


def calc_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    This function will create IDs like "data/monopoly.pdf:6:2", following this pattern:
    `Page Source : Page Number : Chunk Index`. It will add these IDs to the chunks and return them.

    Args:
        chunks (List[Document]): The chunks to be processed.

    Returns:
        List[Document]: The chunks with the IDs added.
    """

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

        # Derive a simple game identifier from the PDF filename (e.g. "dixit")
        if source and isinstance(source, str):
            import os
            pdf_name = os.path.basename(source)
            # Store helper metadata fields
            chunk.metadata["pdf_filename"] = pdf_name.lower()  # for fast server-side filtering

    return chunks


# Rect highlighting computation removed per request — no rect metadata will be stored


def reorder_documents_by_columns(documents: List[Document]) -> List[Document]:
	"""
	Reorder per-page text using data-driven 2-column detection.

	- Extract blocks via PyMuPDF
	- Identify full-width blocks (>= 60% of page width) → neutral
	- Cluster remaining blocks by center-x into two columns (1D k-means)
	- Split threshold s = (mu_left + mu_right)/2 with small padding
	- Assign ambiguous blocks within padding to the same column as the previous
	  block in reading order when possible; otherwise to the nearest centroid
	- Emit left column blocks (top-down), then right column blocks (top-down)
	  and finally neutral blocks in reading order
	"""
	# Group documents by source path with page index
	by_pdf: dict[str, List[Tuple[int, Document]]] = {}
	for idx, doc in enumerate(documents):
		try:
			src = str(doc.metadata.get("source") or "")
			pg = doc.metadata.get("page")
			if not src or pg is None:
				continue
			by_pdf.setdefault(src, []).append((int(pg), doc))
		except Exception:
			continue

	for src, items in by_pdf.items():
		try:
			pdf = fitz.open(src)
		except Exception:
			continue
		try:
			for page_index, doc in items:
				try:
					page = pdf.load_page(page_index)
				except Exception:
					continue
				try:
					blocks = page.get_text("blocks")  # list of (x0,y0,x1,y1, text, ...)
				except Exception:
					blocks = []
				if not blocks:
					# Fallback: keep original text
					continue

				page_w = float(page.rect.width or 1.0)
				full_width_thresh = 0.60 * page_w

				# Build precise block list
				raw_blocks: List[Tuple[int, float, float, float, float, float, float, str]] = []
				for bi, b in enumerate(blocks):
					try:
						x0, y0, x1, y1, text = float(b[0]), float(b[1]), float(b[2]), float(b[3]), str(b[4])
					except Exception:
						x0 = float(b[0]); y0 = float(b[1]); x1 = float(b[2]); y1 = float(b[3]); text = str(b[4])
					if not text.strip():
						continue
					w = float(x1 - x0)
					cx = (x0 + x1) / 2.0
					raw_blocks.append((bi, x0, y0, x1, y1, cx, w, text))

				normals: List[Tuple[float, float, float, str, float, int]] = []
				neutral: List[Tuple[float, float, str, int]] = []
				for bi, x0, y0, x1, y1, cx, w, text in raw_blocks:
					if w >= full_width_thresh:
						neutral.append((y0, x0, text, bi))
					else:
						normals.append((cx, y0, x0, text, w, bi))

				# Fallback: if not enough normals, keep reading order
				if len(normals) < 3:
					all_blocks = [(y0, x0, text) for (_cx, y0, x0, text, _w, _bi) in normals] + [(y0, x0, text) for (y0, x0, text, _bi) in neutral]
					all_blocks.sort(key=lambda t: (t[0], t[1]))
					combined = "\n\n".join(s[2].rstrip() for s in all_blocks if s[2])
					try:
						doc.page_content = combined
					except Exception:
						try:
							new_meta = dict(doc.metadata)
							new_doc = Document(page_content=combined, metadata=new_meta)
							documents[documents.index(doc)] = new_doc
						except Exception:
							pass
					continue

				# 1D k-means (k=2)
				cxs = [cx for (cx, _y0, _x0, _text, _w, _bi) in normals]
				mu_left, mu_right = min(cxs), max(cxs)
				for _ in range(10):
					left, right = [], []
					for tup in normals:
						cx, y0, x0, text, w, bi = tup
						if abs(cx - mu_left) <= abs(cx - mu_right):
							left.append(tup)
						else:
							right.append(tup)
					if not left or not right:
						break
					new_mu_left = sum(t[0] for t in left) / len(left)
					new_mu_right = sum(t[0] for t in right) / len(right)
					if abs(new_mu_left - mu_left) < 0.5 and abs(new_mu_right - mu_right) < 0.5:
						mu_left, mu_right = new_mu_left, new_mu_right
						break
					mu_left, mu_right = new_mu_left, new_mu_right
				if mu_left > mu_right:
					mu_left, mu_right = mu_right, mu_left

				split = (mu_left + mu_right) / 2.0
				pad = max(6.0, 0.02 * page_w)

				reading = sorted(normals, key=lambda t: (t[1], t[2]))
				left_seq: List[Tuple[float, float, str]] = []
				right_seq: List[Tuple[float, float, str]] = []
				prev_col = None
				prev_x0 = prev_w = 0.0
				for cx, y0, x0, text, w, bi in reading:
					if cx < split - pad:
						col = 0
					elif cx > split + pad:
						col = 1
					else:
						# Ambiguous: keep same column when overlapping previous
						if prev_col is not None and (x0 < (prev_x0 + 0.5 * max(prev_w, 1.0))):
							col = prev_col
						else:
							col = 0 if abs(cx - mu_left) <= abs(cx - mu_right) else 1
					if col == 0:
						left_seq.append((y0, x0, text))
					else:
						right_seq.append((y0, x0, text))
					prev_col, prev_x0, prev_w = col, x0, w

				left_seq.sort(key=lambda t: (t[0], t[1]))
				right_seq.sort(key=lambda t: (t[0], t[1]))
				neutral.sort(key=lambda t: (t[0], t[1]))
				combined = "\n\n".join([s[2].rstrip() for s in left_seq] + [s[2].rstrip() for s in right_seq] + [s[2].rstrip() for s in neutral])

				# Replace page content
				try:
					doc.page_content = combined
				except Exception:
					try:
						new_meta = dict(doc.metadata)
						new_doc = Document(page_content=combined, metadata=new_meta)
						documents[documents.index(doc)] = new_doc
					except Exception:
						pass
		finally:
			try:
				pdf.close()
			except Exception:
				pass
	return documents


def clear_database():
    """
    Clears the Chroma vector database.
    Note: This function is for standalone use only. 
    UI should use ChromaDB reset() method to avoid file locks.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def main() -> None:
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset", action="store_true", help="Reset the database.")
    parser.add_argument(
        "--skip-name-extraction",
        action="store_true",
        help="Skip extracting and storing game names for PDFs.",
    )
    parser.add_argument(
        "--splitter",
        choices=["regex", "llm"],
        default="llm",
        help="Section splitter: 'regex' or 'llm' (default: llm)",
    )
    # Streamed processing is always ON (no batch mode)
    # LLM feature toggles (default ON)
    parser.add_argument("--llm-enrich", dest="llm_enrich", action="store_true", help="LLM: add summaries/keywords/anchors/cross-refs to chunk metadata (default ON)")
    parser.add_argument("--llm-semantic-split", dest="llm_semantic_split", action="store_true", help="LLM: split oversized sections at semantic breakpoints (default ON)")
    parser.add_argument("--llm-aliases", dest="llm_aliases", action="store_true", help="LLM: build alias map (synonyms→codes) per PDF and store in metadata (default ON)")
    parser.add_argument("--llm-validate-missing", dest="llm_validate_missing", action="store_true", help="LLM: compare detected sections to TOC and report missing (default ON)")
    parser.set_defaults(llm_enrich=True, llm_semantic_split=True, llm_aliases=True, llm_validate_missing=True)

    # Optional positional paths (files or directories) to process. If omitted,
    # the entire DATA_PATH will be processed (original behaviour).
    parser.add_argument(
        "paths",
        nargs="*",
        help="Specific PDF files or directories to (re)process. "
        "Relative paths are resolved against DATA_PATH.",
    )
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Helper: build enrichment LLM once if needed
    enrich_llm = None
    if args.llm_enrich or args.llm_aliases or args.llm_validate_missing or args.llm_semantic_split:
        try:
            from temp_tests.llm_extract_sections import _make_llm  # type: ignore
            enrich_llm = _make_llm()
        except Exception as e:
            print(f"⚠️ LLM init failed for enrichment: {e}")
            enrich_llm = None

    def _invoke_with_timeout(llm, messages, timeout_s: float = 30.0) -> str:
        def _call():
            out_msg = llm.invoke(messages)
            return str(getattr(out_msg, "content", "") or "").strip()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_call)
            return fut.result(timeout=timeout_s)

    def _apply_enrichment(chs: List[Document], debug_dir: str | None = None) -> List[Document]:
        if not (enrich_llm and (args.llm_enrich or args.llm_aliases or args.llm_semantic_split)):
            return chs
        # Enrichment helpers (scoped)
        def _write_debug(filename: str, payload: dict) -> None:
            if not debug_dir:
                return
            try:
                from pathlib import Path as _P
                import json as _json
                p = _P(debug_dir) / "enrichment"
                p.mkdir(parents=True, exist_ok=True)
                (p / filename).write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
        def _try_llm_enrich(chs0: List[Document]) -> List[Document]:
            if not (enrich_llm and args.llm_enrich):
                return chs0
            print("✨ LLM Enrich: summaries/keywords/anchors/cross-refs …")
            out_local: List[Document] = []
            total_local = len(chs0)
            for idx_local, d in enumerate(chs0, 1):
                try:
                    text = d.page_content or ""
                    meta = dict(d.metadata)
                    from langchain.schema import HumanMessage, SystemMessage  # type: ignore
                    sys = (
                        "You extract metadata for a boardgame rules chunk. Return STRICT JSON with keys:\n"
                        "- summary: <=160 chars\n"
                        "- keywords: array of <=8 lowercase tokens\n"
                        "- anchors: array of <=2 short anchor phrases\n"
                        "- cross_refs: array of section codes referenced in text (e.g., I4, 1B6b)\n"
                        "- section_kind: one of {setup, operation, movement, hazard, scoring, endgame, glossary, component, reference, politics, economy, glossary-example, example, note, other}.\n"
                        "Choose the most specific that applies; use 'glossary' for term definitions. OUTPUT JSON ONLY."
                    )
                    usr = f"TEXT (truncated):\n{text[:1800]}\n\nReturn only a JSON object with those keys."
                    if idx_local % 25 == 1 or total_local <= 10:
                        print(f"   • Enriching chunk {idx_local}/{total_local}")
                    try:
                        s = _invoke_with_timeout(enrich_llm, [SystemMessage(content=sys), HumanMessage(content=usr)], timeout_s=30.0)
                    except concurrent.futures.TimeoutError:
                        print(f"   ⚠️ Enrich timeout for chunk {idx_local}; skipping")
                        _write_debug(f"enrich_{idx_local:04d}_timeout.json", {
                            "meta": meta,
                            "system": sys,
                            "user": usr,
                            "error": "timeout",
                        })
                        out_local.append(d)
                        continue
                    m = re.search(r"\{[\s\S]*\}$", s)
                    data = {}
                    if m:
                        import json as _json
                        try:
                            data = _json.loads(m.group(0))
                        except Exception:
                            data = {}
                    _write_debug(f"enrich_{idx_local:04d}.json", {
                        "meta": meta,
                        "system": sys,
                        "user": usr,
                        "raw": s,
                        "parsed": data,
                    })
                    if isinstance(data, dict):
                        for k in ("summary", "keywords", "anchors", "cross_refs"):
                            if k in data:
                                meta[k] = data[k]
                        sk = str(data.get("section_kind") or "").strip().lower()
                        if sk:
                            meta["section_kind"] = sk
                        else:
                            try:
                                code = str(meta.get("section_number") or "")
                                if re.match(r"^[Ll]\d", code):
                                    meta["section_kind"] = "glossary"
                            except Exception:
                                pass
                    out_local.append(Document(page_content=text, metadata=meta))
                except Exception as e:
                    _write_debug(f"enrich_{idx_local:04d}_error.json", {
                        "meta": dict(d.metadata),
                        "error": str(e),
                    })
                    out_local.append(d)
            return out_local

        def _try_llm_aliases(chs0: List[Document]) -> List[Document]:
            if not (enrich_llm and args.llm_aliases):
                return chs0
            print("✨ LLM Aliases: per-PDF alias map …")
            by_pdf: dict[str, dict] = {}
            for d in chs0:
                src = str(d.metadata.get("source") or "")
                if src and src not in by_pdf:
                    by_pdf[src] = {"codes": set(), "alias": {}}
                try:
                    code = str(d.metadata.get("section_number") or "").strip()
                    if src and code:
                        by_pdf[src]["codes"].add(code)
                except Exception:
                    pass
            from langchain.schema import HumanMessage, SystemMessage  # type: ignore
            for src, bag in by_pdf.items():
                codes = sorted(list(bag["codes"]))
                if not codes:
                    continue
                sys = "You map aliases to canonical section codes. Return strict JSON object: { alias(string): code(string) }. Only use provided codes."
                usr = f"Codes: {', '.join(codes)}\nFind common alias phrases for these sections (e.g., 'sunspot cycle phase' -> C8)."
                try:
                    s = _invoke_with_timeout(enrich_llm, [SystemMessage(content=sys), HumanMessage(content=usr)], timeout_s=45.0)
                    import json as _json
                    m = re.search(r"\{[\s\S]*\}$", s)
                    alias_map = _json.loads(m.group(0)) if m else {}
                    _write_debug("aliases.json", {
                        "source": src,
                        "system": sys,
                        "user": usr,
                        "raw": s,
                        "parsed": alias_map,
                    })
                except concurrent.futures.TimeoutError:
                    print("   ⚠️ Alias mapping timeout; skipping")
                    _write_debug("aliases_timeout.json", {
                        "source": src,
                        "system": sys,
                        "user": usr,
                        "error": "timeout",
                    })
                    alias_map = {}
                except Exception as e:
                    _write_debug("aliases_error.json", {
                        "source": src,
                        "system": sys,
                        "user": usr,
                        "error": str(e),
                    })
                    alias_map = {}
                for d in chs0:
                    if str(d.metadata.get("source") or "") == src:
                        meta = dict(d.metadata)
                        if alias_map:
                            meta["alias_map"] = alias_map
                        d = Document(page_content=d.page_content, metadata=meta)
            return chs0

        def _try_llm_semantic_split(chs0: List[Document]) -> List[Document]:
            if not (enrich_llm and args.llm_semantic_split):
                return chs0
            print("✨ LLM Semantic Split: oversized sections …")
            out_local: List[Document] = []
            total_local = len(chs0)
            for idx_local, d in enumerate(chs0, 1):
                if len(d.page_content) <= CHUNK_SIZE * 1.5:
                    out_local.append(d)
                    continue
                try:
                    text = d.page_content
                    from langchain.schema import HumanMessage, SystemMessage  # type: ignore
                    sys = "You find semantic breakpoints that preserve list/step integrity. Return JSON: {breaks: [char_index,...]} where indices split the text."
                    usr = f"TEXT (truncate to 4k):\n{text[:4000]}\n\nReturn JSON only."
                    if idx_local % 25 == 1 or total_local <= 10:
                        print(f"   • Semantic split {idx_local}/{total_local}")
                    try:
                        s = _invoke_with_timeout(enrich_llm, [SystemMessage(content=sys), HumanMessage(content=usr)], timeout_s=30.0)
                    except concurrent.futures.TimeoutError:
                        print(f"   ⚠️ Semantic split timeout for chunk {idx_local}; keeping unsplit")
                        _write_debug(f"semantic_{idx_local:04d}_timeout.json", {
                            "meta": dict(d.metadata),
                            "system": sys,
                            "user": usr,
                            "error": "timeout",
                        })
                        out_local.append(d)
                        continue
                    import json as _json
                    data = _json.loads(re.search(r"\{[\s\S]*\}$", s).group(0)) if re.search(r"\{[\s\S]*\}$", s) else {}
                    _write_debug(f"semantic_{idx_local:04d}.json", {
                        "meta": dict(d.metadata),
                        "system": sys,
                        "user": usr,
                        "raw": s,
                        "parsed": data,
                    })
                    breaks = [int(x) for x in (data.get("breaks") or []) if isinstance(x, (int, float))]
                    if not breaks:
                        out_local.append(d)
                        continue
                    last = 0
                    for b in sorted(set([i for i in breaks if 32 <= i < len(text)-32])):
                        seg = text[last:b].strip()
                        if seg:
                            out_local.append(Document(page_content=seg, metadata=dict(d.metadata)))
                        last = b
                    tail = text[last:].strip()
                    if tail:
                        out_local.append(Document(page_content=tail, metadata=dict(d.metadata)))
                except Exception as e:
                    _write_debug(f"semantic_{idx_local:04d}_error.json", {
                        "meta": dict(d.metadata),
                        "error": str(e),
                    })
                    out_local.append(d)
            return out_local

        chs_out = chs
        if args.llm_semantic_split:
            chs_out = _try_llm_semantic_split(chs_out)
        if args.llm_enrich:
            chs_out = _try_llm_enrich(chs_out)
        if args.llm_aliases:
            chs_out = _try_llm_aliases(chs_out)
        return chs_out

    # Streamed per-PDF processing (always on)
    pdfs = _resolve_target_pdfs(args.paths)
    total = len(pdfs)
    print(f"📁 Found {total} PDF file(s) to process (streamed)")
    for i, pdf_path in enumerate(pdfs, 1):
        try:
            print(f"\n📖 [{i}/{total}] {os.path.basename(pdf_path)}")
            # Optional optimization
            try:
                from . import config as _cfg  # type: ignore
            except Exception:
                import src.config as _cfg  # type: ignore
            if getattr(_cfg, "ENABLE_PDF_OPTIMIZATION", False):
                try:
                    replaced, orig, opt, msg = optimize_with_raster_fallback_if_large(
                        pdf_path,
                        min_size_mb=getattr(_cfg, "PDF_OPTIMIZE_MIN_SIZE_MB", 25.0),
                        linearize=getattr(_cfg, "PDF_LINEARIZE", True),
                        garbage_level=getattr(_cfg, "PDF_GARBAGE_LEVEL", 3),
                        enable_raster_fallback=getattr(_cfg, "PDF_ENABLE_RASTER_FALLBACK", False),
                        raster_dpi=getattr(_cfg, "PDF_RASTER_DPI", 150),
                        jpeg_quality=getattr(_cfg, "PDF_JPEG_QUALITY", 70),
                    )
                    if replaced or opt < orig:
                        print(f"   🛠 Optimized {os.path.basename(pdf_path)}: {msg}")
                except Exception as _e:
                    print(f"   ⚠️ Optimization failed for {os.path.basename(pdf_path)}: {_e}")

            # Load one PDF
            docs = PyPDFLoader(str(pdf_path)).load()
            print(f"   ✅ Loaded {len(docs)} pages")
            # Column reorder per PDF
            docs = reorder_documents_by_columns(docs)
            # Split
            if args.splitter == "llm":
                # Enable debug capture for outline extraction
                try:
                    from src.llm_outline import extract_pdf_outline  # type: ignore
                    from pathlib import Path as _P
                    dbg_dir = _P("debug") / _P(pdf_path).stem
                    outline = extract_pdf_outline(pdf_path, debug_dir=str(dbg_dir))
                    # Reuse downstream normalization by building docs from outline
                    # Convert pages_docs-style input from outline into split docs by calling the same normalizer
                    # Simplest: reconstruct via split_documents_llm’s expected input
                    chunks = split_documents_llm(docs)
                except Exception:
                    chunks = split_documents_llm(docs)
            else:
                chunks = split_documents(docs)
            # Enrichment per PDF
            try:
                from pathlib import Path as _P
                dbg_dir = _P("debug") / _P(pdf_path).stem
                chunks = _apply_enrichment(chunks, debug_dir=str(dbg_dir))
            except Exception:
                chunks = _apply_enrichment(chunks)
            # Add to DB
            add_to_chroma(chunks)
            # Store names unless skipped
            if not args.skip_name_extraction:
                try:
                    from src.query import store_game_name  # type: ignore
                    stored = set()
                    for ch in chunks:
                        meta = getattr(ch, 'metadata', {}) or {}
                        fn = os.path.basename(str(meta.get('source') or ''))
                        gn = str(meta.get('game_name') or '').strip()
                        if fn and gn and (fn, gn) not in stored:
                            store_game_name(fn, gn)
                            stored.add((fn, gn))
                    if stored:
                        print(f"   🔤 Stored LLM-extracted game name: {', '.join(sorted({gn for (_fn, gn) in stored}))}")
                except Exception as e:
                    print(f"   ⚠️  Failed to store game names: {e}")
            else:
                print("   ⚠️  Skipped storing LLM-extracted game names as requested")
        except Exception as e:
            print(f"   ❌ Failed processing {os.path.basename(pdf_path)}: {e}")
    return


if __name__ == "__main__":
    main()
