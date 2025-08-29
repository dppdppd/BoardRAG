"""Storage monitoring utilities for BoardRAG application."""

from pathlib import Path
import json
from . import config


def get_pdf_storage_usage():
    """Calculate total disk space used by top-level PDF files only.

    Only counts PDFs located directly under `DATA_PATH` and excludes any PDFs in
    subdirectories (e.g., per-page PDFs under `<PDF_STEM>/1_pdf_pages`).
    """
    try:
        pdf_dir = Path(config.DATA_PATH)
        if not pdf_dir.exists():
            return 0, 0, "PDF directory not found"
        
        total_size = 0
        file_count = 0
        
        # Only count PDFs that are direct children of the data directory
        for pdf_file in pdf_dir.glob("*.pdf"):
            if pdf_file.is_file():
                total_size += pdf_file.stat().st_size
                file_count += 1
        
        # Convert to MB for readability
        size_mb = total_size / (1024 * 1024)
        return size_mb, file_count, None
    except Exception as e:
        return 0, 0, f"Error calculating PDF storage: {str(e)}"


def get_chat_storage_usage():
    """Calculate disk space used by chat history storage."""
    try:
        conversations_file = Path("conversations.json")
        if not conversations_file.exists():
            return 0, 0, "No chat history file found"
        
        file_size = conversations_file.stat().st_size
        size_mb = file_size / (1024 * 1024)
        
        # Count conversations
        try:
            with open(conversations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            total_conversations = 0
            for user_data in data.values():
                total_conversations += len(user_data)
            
            return size_mb, total_conversations, None
        except Exception:
            return size_mb, 0, "Could not parse conversation count"
            
    except Exception as e:
        return 0, 0, f"Error calculating chat storage: {str(e)}"


def format_storage_info():
    """Generate formatted storage information for display.

    Chat history is no longer tracked server-side; only PDF storage is reported.
    """
    pdf_size, pdf_count, pdf_error = get_pdf_storage_usage()

    info = "## 💾 Storage Usage\n\n"

    # PDF Storage
    info += "### 📄 PDF Documents\n"
    if pdf_error:
        info += f"❌ **Error:** {pdf_error}\n"
    else:
        info += f"**Size:** {pdf_size:.2f} MB\n"
        info += f"**Files:** {pdf_count} PDFs\n"

    # Append configuration details mirrored from startup prints
    try:
        info += "\n\n## ⚙️ Configuration\n\n"
        # Core provider/models
        info += f"Provider: {config.LLM_PROVIDER}\n"
        info += f"Generator Model: {config.GENERATOR_MODEL}\n"
        info += f"Embedder Model: {config.EMBEDDER_MODEL}\n"
        # Chunking
        info += f"Chunk Size: {config.CHUNK_SIZE}\n"
        info += f"Chunk Overlap: {config.CHUNK_OVERLAP}\n"
        # Retrieval/processing flags
        info += f"ENABLE_STREAM_VALIDATION: {getattr(config, 'ENABLE_STREAM_VALIDATION', False)}\n"
        # Paths and templates
        info += f"DATA_PATH: {config.DATA_PATH}\n"
        # Web search
        info += f"ENABLE_WEB_SEARCH: {config.ENABLE_WEB_SEARCH}\n"
        info += f"WEB_SEARCH_RESULTS: {getattr(config, 'WEB_SEARCH_RESULTS', 0)}\n"
        # API key presence (do not print actual values)
        openai_status = "✅ Set" if getattr(config, 'OPENAI_API_KEY', None) else "❌ Missing"
        anthropic_status = "✅ Set" if getattr(config, 'ANTHROPIC_API_KEY', None) else "❌ Missing"
        info += f"OpenAI API Key: {openai_status}\n"
        info += f"Anthropic API Key: {anthropic_status}\n"
    except Exception:
        # Keep storage info usable even if config rendering fails
        pass

    return info