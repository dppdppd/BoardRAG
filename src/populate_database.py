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
        import os
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

    HEADER_RE = re.compile(r"^\s*(?:[A-Z0-9]+(?:\.[A-Z0-9]+)*\.?)?\s*[A-Z][A-Z0-9 &'\-/]{2,}(?::\s*.*)?$")

    # Track section state across pages to handle continuations
    last_section_on_previous_page = "intro"
    
    def _split_on_headers(page_doc: Document, inherited_section: str) -> tuple[List[Document], str]:
        """Split a single PDF page into sections based on detected headers.
        
        Returns: (chunks, last_section_on_this_page)
        """
        lines = page_doc.page_content.splitlines()
        current_header = inherited_section  # Start with inherited section, not always "intro"
        buffer: list[str] = []
        out: List[Document] = []

        def _flush(buf: list[str], header: str):
            if not buf:
                return
            content = "\n".join(buf).strip()
            if content:
                meta = dict(page_doc.metadata)
                meta["section"] = header
                out.append(Document(page_content=content, metadata=meta))

        for line in lines:
            if HEADER_RE.match(line):
                _flush(buffer, current_header)
                buffer = []
                # Extract just the section title (before any colon)
                header_text = line.strip()
                if ':' in header_text:
                    current_header = header_text.split(':', 1)[0].strip()
                else:
                    current_header = header_text
            else:
                buffer.append(line)
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
