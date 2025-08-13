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
from typing import List

import chromadb
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
            documents.extend(PyPDFDirectoryLoader(full_path).load())
        elif os.path.isfile(full_path):
            documents.extend(PyPDFLoader(full_path).load())
        else:
            print(f"⚠️  Path not found – skipping: {full_path}")

    return documents


def split_documents(documents: List[Document]):
    """
    Perform section-aware chunking of PDF pages.

    1. Each PDF *page* is scanned for rule-style headers (e.g. "1. SETUP").
    2. Pages are first split on those headers into smaller *section* documents.
    3. Any resulting section that is still larger than CHUNK_SIZE is further
       divided by `RecursiveCharacterTextSplitter` while preserving paragraph
       boundaries ("\n\n", "\n", " ").

    This keeps logical rule sections together and dramatically improves RAG
    retrieval quality compared with naive fixed-width splitting.
    """

    # Prefer numeric headers (e.g., "3.6 Interdiction", "2.3.4 Morale Check").
    # Separate regexes make precedence explicit and reduce false-positives.
    # Dotted numeric headers only (e.g., "3.0 Sequence of Play", "3.2.1 Morale Check").
    # This avoids mistaking page numbers like "7" or years like "2021" as headers.
    NUMERIC_TITLE_RE = re.compile(r"^\s*(((?:\d+\.)+\d+))\s+(.+?)\s*$")
    # Numeric-only line that represents a header must contain a dot (e.g., "3.0", "3.2.1").
    NUMERIC_ONLY_HEADER_RE = re.compile(r"^\s*((?:\d+\.)+\d+)\s*:?\s*$")

    # As a last resort, allow non-numeric headers, but guard against short tokens
    # like "RtPh:" or codes like "X3Z1" by requiring length and at least two words.
    # Generic headers (very strict):
    #  - Either ALL-CAPS style headings with length
    #  - Or Title Case headings that END WITH a colon
    GENERIC_HEADER_RE = re.compile(
        r"^\s*(?:"
        r"[A-Z][A-Z0-9 &'\-/()]{3,}"  # ALL CAPS style, at least 4 chars
        r"|"
        r"(?:[A-Z][A-Za-z0-9&'\-/()]+(?:\s+[A-Z][A-Za-z0-9&'\-/()]+)+):"  # Two+ TitleCased words ending with ':'
        r")\s*$"
    )

    # Detect Table of Contents entries with dot leaders and trailing page numbers
    TOC_ENTRY_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+.+?\.{2,}\s*\d+\s*$")

    # Track section state across pages to handle continuations
    last_section_on_previous_page = "intro"
    
    def _split_on_headers(page_doc: Document, inherited_section: str) -> tuple[List[Document], str]:
        """Split a single PDF page into sections based on detected headers.
        
        Returns: (chunks, last_section_on_this_page)
        """
        lines = page_doc.page_content.splitlines()
        # TOC heuristic: if a page contains multiple TOC-like entries, treat it as TOC
        toc_like_count = sum(1 for ln in lines if TOC_ENTRY_RE.match(ln))
        is_toc_page = toc_like_count >= 3
        current_header = inherited_section  # Start with inherited section, not always "intro"
        pending_header_text: str | None = None  # header line to prepend to the next section's content
        buffer: list[str] = []
        out: List[Document] = []

        def _normalize_section_number(raw: str) -> str:
            """Normalize numeric section like '3' → '3.0' for chapter headings.
            Leaves dotted forms unchanged (e.g., '3.2' stays '3.2').
            """
            try:
                raw = raw.strip()
                if not raw:
                    return raw
                # If it already contains a dot, keep as-is
                if "." in raw:
                    return raw
                # Otherwise map top-level chapter numbers to N.0
                return f"{raw}.0"
            except Exception:
                return raw

        def _flush(buf: list[str], header: str):
            if not buf:
                return
            content = "\n".join(buf).strip()
            if content:
                meta = dict(page_doc.metadata)
                # Store rich section metadata to improve downstream matching
                meta["section"] = header
                meta["section_full"] = header
                try:
                    m_num = re.match(r"^\s*(\d+(?:\.\d+)*)\b", header)
                    if m_num:
                        sec_num = m_num.group(1)
                        meta["section_number"] = _normalize_section_number(sec_num)
                except Exception:
                    pass
                out.append(Document(page_content=content, metadata=meta))

        i = 0
        while i < len(lines):
            line = lines[i]

            # Safety: if we have a pending header and the next line is also detected as a header,
            # emit a minimal chunk containing just the pending header to avoid losing it.
            if pending_header_text is not None and not buffer:
                if (
                    NUMERIC_TITLE_RE.match(line)
                    or NUMERIC_ONLY_HEADER_RE.match(line)
                    or GENERIC_HEADER_RE.match(line)
                ):
                    buffer.append(pending_header_text)
                    pending_header_text = None
                    _flush(buffer, current_header)
                    buffer = []
            # Skip header detection on Table of Contents pages
            if is_toc_page:
                if pending_header_text is not None and not buffer:
                    buffer.append(pending_header_text)
                    pending_header_text = None
                buffer.append(line)
                i += 1
                continue

            # Case A: Numeric header with title on the same line (e.g., "3.3 Movement Phase (MPh)")
            m_num_title = NUMERIC_TITLE_RE.match(line)
            if m_num_title:
                _flush(buffer, current_header)
                buffer = []
                section_number_raw = m_num_title.group(1)
                section_number = _normalize_section_number(section_number_raw)
                title_text = m_num_title.group(3)
                # Remove trailing subtitle after a colon in the same line to make a stable label
                clean_title = title_text.split(':', 1)[0].strip()
                combined_header = f"{section_number} {clean_title}"
                current_header = combined_header
                pending_header_text = combined_header
                i += 1
                continue

            # Case B: Numeric-only line e.g., "3.0" with title on the next non-empty line
            m_num = NUMERIC_ONLY_HEADER_RE.match(line)
            if m_num:
                j = i + 1
                next_title = None
                while j < len(lines):
                    if lines[j].strip():
                        next_title = lines[j].strip()
                        break
                    j += 1
                if next_title:
                    _flush(buffer, current_header)
                    buffer = []
                    section_number = _normalize_section_number(m_num.group(1))
                    combined_header = f"{section_number} {next_title.split(':',1)[0].strip()}"
                    current_header = combined_header
                    pending_header_text = f"{combined_header}"
                    i = j + 1
                    continue

            # Case C: Fallback generic header (non-numeric). Guard against short codes.
            if GENERIC_HEADER_RE.match(line):
                text = line.strip()
                _flush(buffer, current_header)
                buffer = []
                header_text = text
                # Trim trailing colon in display label
                if header_text.endswith(":"):
                    header_text = header_text[:-1].strip()
                current_header = header_text
                pending_header_text = header_text
                i += 1
                continue

            # Default: accumulate content
            if pending_header_text is not None and not buffer:
                buffer.append(pending_header_text)
                pending_header_text = None
            buffer.append(line)
            i += 1

        # If the page ended right after a header, still emit a minimal chunk with just the header
        if pending_header_text is not None and not buffer:
            buffer.append(pending_header_text)
            pending_header_text = None

        _flush(buffer, current_header)

        # If no headers were found, return the original page so that downstream
        # logic still sees it, but preserve the section state
        final_chunks = out or [page_doc]
        if not out and page_doc:
            # Update the original page's section metadata to inherited section
            page_doc.metadata["section"] = inherited_section
        
        return final_chunks, current_header

    print(f"🔍 Phase 1: Section-aware splitting of {len(documents)} pages...")
    
    # 1️⃣ First pass – header splitting with cross-page section tracking
    section_docs: List[Document] = []
    for i, page in enumerate(documents):
        if (i + 1) % 50 == 0 or i == len(documents) - 1:  # Progress every 50 pages
            print(f"   📄 Processed {i + 1}/{len(documents)} pages...")
        page_chunks, last_section_on_previous_page = _split_on_headers(page, last_section_on_previous_page)
        section_docs.extend(page_chunks)

    print(f"✅ Phase 1 complete: {len(documents)} pages → {len(section_docs)} sections")
    print(f"🔍 Phase 2: Character-level splitting for oversized sections...")

    # 2️⃣ Second pass – character splitting only when needed
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " "],
        length_function=len,
        is_separator_regex=False,
    )

    final_chunks: List[Document] = []
    oversized_count = 0
    for i, sec in enumerate(section_docs):
        if (i + 1) % 100 == 0:  # Progress every 100 sections
            print(f"   📝 Processed {i + 1}/{len(section_docs)} sections...")
        
        if len(sec.page_content) <= CHUNK_SIZE:
            final_chunks.append(sec)
        else:
            oversized_count += 1
            final_chunks.extend(char_splitter.split_documents([sec]))

    print(f"✅ Phase 2 complete: {len(section_docs)} sections → {len(final_chunks)} final chunks")
    print(f"📊 Split {oversized_count} oversized sections into smaller chunks")

    return final_chunks


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

    # Create (or update) the data store. If the user supplied specific paths we
    # only process those, which speeds up incremental updates (e.g. when adding
    # a single new game).
    documents = load_documents(args.paths)
    chunks = split_documents(documents)
    add_to_chroma(chunks)

    # ---------------------------------------------
    # Optional: Extract & store official game names
    # ---------------------------------------------
    if not args.skip_name_extraction:
        try:
            from src.query import extract_and_store_game_name  # local import to avoid circular deps
        except ImportError:
            from query import extract_and_store_game_name  # fallback when running as module

        import os
        filenames = {
            os.path.basename(doc.metadata.get("source", ""))
            for doc in documents
        }
        print(f"🔤 Extracting game names for {len(filenames)} PDFs …")
        for fname in filenames:
            if fname:
                extract_and_store_game_name(fname)
    else:
        print("⚠️  Skipped game-name extraction as requested")


if __name__ == "__main__":
    main()
