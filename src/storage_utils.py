"""Storage monitoring utilities for BoardRAG application."""

from pathlib import Path
import json
from . import config


def get_pdf_storage_usage():
    """Calculate total disk space used by PDF files."""
    try:
        pdf_dir = Path(config.DATA_PATH)
        if not pdf_dir.exists():
            return 0, 0, "PDF directory not found"
        
        total_size = 0
        file_count = 0
        
        for pdf_file in pdf_dir.rglob("*.pdf"):
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
    """Generate formatted storage information for display."""
    pdf_size, pdf_count, pdf_error = get_pdf_storage_usage()
    chat_size, chat_count, chat_error = get_chat_storage_usage()
    
    info = "## 💾 Storage Usage\n\n"
    
    # PDF Storage
    info += "### 📄 PDF Documents\n"
    if pdf_error:
        info += f"❌ **Error:** {pdf_error}\n"
    else:
        info += f"**Size:** {pdf_size:.2f} MB\n"
        info += f"**Files:** {pdf_count} PDFs\n"
    
    info += "\n"
    
    # Chat History Storage
    info += "### 💬 Chat Histories\n"
    if chat_error:
        info += f"❌ **Error:** {chat_error}\n"
    else:
        info += f"**Size:** {chat_size:.2f} MB\n"
        info += f"**Conversations:** {chat_count} saved conversations\n"
    
    info += "\n"
    
    # Total
    if not pdf_error and not chat_error:
        total_size = pdf_size + chat_size
        info += f"### 📊 Total Storage\n"
        info += f"**Combined Size:** {total_size:.2f} MB\n"
    
    return info 