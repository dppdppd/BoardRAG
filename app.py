"""
Gradio app for running the RAG model with an interface.
"""

# We avoid complex typing annotations that confuse Gradio's schema generation.

import os
import shutil
from pathlib import Path
from typing import List

import chromadb
import gradio as gr
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from embedding_function import get_embedding_function
from query import get_available_games, query_rag, get_stored_game_names

# Disable ChromaDB telemetry after imports
config.disable_chromadb_telemetry()

INTRO_STRING = """
# 🎲 BoardRAG
"""

theme_css = ""

# -----------------------------------------------------------------------------
# Access control passwords (set in environment)
# -----------------------------------------------------------------------------
USER_PW = os.getenv("USER_PW")
ADMIN_PW = os.getenv("ADMIN_PW")


# Utility helpers -------------------------------------------------------------


def list_pdfs() -> List[str]:
    """Return names of all PDFs in data directory."""
    if not os.path.isdir(config.DATA_PATH):
        return []
    return sorted(
        [f for f in os.listdir(config.DATA_PATH) if f.lower().endswith(".pdf")]
    )


def refresh_game_lists() -> List[str]:
    """Wrapper around get_available_games with error swallow."""
    try:
        return get_available_games()
    except Exception:
        return []


def get_config_info():
    """Format configuration information for display."""
    # Show API key status without revealing keys
    openai_status = "✅ Set" if config.OPENAI_API_KEY else "❌ Missing"
    anthropic_status = "✅ Set" if config.ANTHROPIC_API_KEY else "❌ Missing"

    config_text = f"""
**🔧 Current Configuration:**

**Provider & Models:**
- Provider: `{config.LLM_PROVIDER}`
- Generator: `{config.GENERATOR_MODEL}`
- Embedder: `{config.EMBEDDER_MODEL}`
- Evaluator: `{config.EVALUATOR_MODEL}`

**Database & Processing:**
- Database Path: `{config.CHROMA_PATH}`
- Chunk Size: `{config.CHUNK_SIZE}` chars (~{config.CHUNK_SIZE//4} tokens)
- Chunk Overlap: `{config.CHUNK_OVERLAP}` chars
- Data Path: `{config.DATA_PATH}`

**Templates:**
- Query Template: `{config.JINJA_TEMPLATE_PATH}`
- Eval Template: `{config.EVAL_TEMPLATE_PATH}`

**API Keys:**
- OpenAI API Key: {openai_status}
- Anthropic API Key: {anthropic_status}

**Password Configuration:**
- USER_PW: {"✅ Set" if USER_PW else "❌ Missing"}
- ADMIN_PW: {"✅ Set" if ADMIN_PW else "❌ Missing"}

**Optional Services:**
- Ollama URL: `{config.OLLAMA_URL}`
- Argilla API URL: `{config.ARGILLA_API_URL or "Not configured"}`
"""
    return config_text


def process_single_pdf(pdf_path: str) -> List[Document]:
    """Process a single PDF file and return chunks."""
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)

    # Calculate chunk IDs (same logic as populate_database.py)
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks


def add_chunks_to_database(chunks: List[Document]) -> bool:
    """Add chunks to the ChromaDB database."""
    try:
        with config.suppress_chromadb_telemetry():
            persistent_client = chromadb.PersistentClient(
                path=config.CHROMA_PATH, settings=config.get_chromadb_settings()
            )
            db = Chroma(
                client=persistent_client, embedding_function=get_embedding_function()
            )

        # Get existing documents
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])

        # Only add new chunks
        new_chunks = [
            chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids
        ]

        if new_chunks:
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)

            # Extract and store game names for new chunks
            extract_and_store_game_names_for_chunks(new_chunks)

            return True
        return False

    except Exception as e:
        print(f"Error adding to database: {e}")
        return False


def extract_and_store_game_names_for_chunks(chunks: List[Document]):
    """
    Extract and store game names for the PDFs represented in the chunks.

    Args:
        chunks (List[Document]): The chunks to extract game names for.
    """
    try:
        # Import here to avoid circular imports
        from query import extract_and_store_game_name

        # Get unique filenames from the chunks
        filenames = set()
        for chunk in chunks:
            source = chunk.metadata.get("source", "")
            if source:
                # Handle both Windows (\) and Unix (/) path separators
                filename = os.path.basename(source)
                if filename.endswith(".pdf"):
                    filenames.add(filename)

        # Extract and store game names for each unique filename
        for filename in sorted(filenames):
            extract_and_store_game_name(filename)

    except Exception as e:
        print(f"⚠️ Error extracting game names: {e}")


def upload_pdf_handler(pdf_file):
    """Handle PDF upload and database update."""
    if pdf_file is None:
        return "❌ No file uploaded", gr.update()

    try:
        # Save the uploaded file to the data directory
        filename = os.path.basename(pdf_file.name)
        destination_path = os.path.join(config.DATA_PATH, filename)

        # Create data directory if it doesn't exist
        os.makedirs(config.DATA_PATH, exist_ok=True)

        # Copy the uploaded file
        shutil.copy2(pdf_file.name, destination_path)

        # Process the PDF
        chunks = process_single_pdf(destination_path)

        # Add to database
        success = add_chunks_to_database(chunks)

        if success:
            # Refresh the game list
            available_games = get_available_games()
            status = f"✅ Successfully added '{filename}' to the database! ({len(chunks)} chunks processed)"
            return status, gr.update(choices=available_games)
        else:
            status = f"⚠️ '{filename}' was uploaded but no new content was added (might already exist in database)"
            available_games = get_available_games()
            return status, gr.update(choices=available_games)

    except Exception as e:
        return f"❌ Error processing '{filename}': {str(e)}", gr.update()


def query_interface(message, selected_game, chat_history):
    """
    Queries the RAG model with the given query and returns the response.

    Args:
        message (str): The query to be passed to the RAG model.
        selected_game (str): The selected game to filter results by.
        chat_history (str): The chat history.

    Returns:
        str: The response from the RAG model.
        List: The chat history, with the new message and response appended.
    """

    # Check if a game is selected
    if not message.strip():
        return "", chat_history

    if not selected_game:
        error_message = (
            "⚠️ Please select a specific game from the dropdown before asking questions."
        )
        chat_history.append((message, error_message))
        return "", chat_history

    # Get the simple filename mapping for filtering
    mapping = getattr(get_available_games, "_filename_mapping", {})
    game_filter = mapping.get(selected_game, selected_game.lower())

    resp = query_rag(message, game_filter)

    # Get fresh available games list to ensure we have the latest data
    available_games = get_available_games()

    # Determine proper game name - use selected game if available, otherwise extract from sources
    if selected_game and selected_game != "All Games":
        game_name = selected_game
    elif resp["sources"] and resp["sources"][0]:
        # Try to find matching game from available games based on source
        source_file = Path(resp["sources"][0]).name.lower()
        game_name = "Game"  # fallback
        for available_game in available_games:
            # Check if any part of the available game name is in the source filename
            if any(word.lower() in source_file for word in available_game.split()):
                game_name = available_game
                break
        # If no match found, use first word capitalized as fallback
        if game_name == "Game":
            game_name = Path(resp["sources"][0]).name.split()[0].capitalize()
    else:
        game_name = "Game"

    # Create clean source citations with clickable PDF links
    if resp["sources"]:
        # Extract file and page info from sources (format: "data/filename.pdf:page:chunk")
        source_info = {}
        for source in resp["sources"]:
            if ":" in source:
                parts = source.split(":")
                if len(parts) >= 2:
                    filepath = parts[0]
                    page_num = parts[1]
                    filename = Path(filepath).name

                    if filename not in source_info:
                        source_info[filename] = set()
                    source_info[filename].add(int(page_num))

        # Create clean citations with original filenames
        clean_citations = []
        for filename, pages in list(source_info.items())[:2]:  # Limit to 2 files
            # Keep the original filename with .pdf extension
            clean_name = filename

            # Sort page numbers and create page reference
            sorted_pages = sorted(pages)
            if len(sorted_pages) == 1:
                page_ref = f"p. {sorted_pages[0]}"
            elif len(sorted_pages) <= 3:
                page_ref = f"pp. {', '.join(map(str, sorted_pages))}"
            elif len(sorted_pages) <= 8:
                # Show individual pages if 8 or fewer
                page_ref = f"pp. {', '.join(map(str, sorted_pages))}"
            else:
                # Show first few pages + "..." + last few pages for large ranges
                first_pages = sorted_pages[:3]
                last_pages = sorted_pages[-2:]
                if first_pages[-1] + 1 < last_pages[0]:  # Check if there's a gap
                    page_ref = f"pp. {', '.join(map(str, first_pages))}, ..., {', '.join(map(str, last_pages))}"
                else:
                    # No gap, just show the range
                    page_ref = f"pp. {sorted_pages[0]}-{sorted_pages[-1]}"

            # Create citation with original filename
            clean_citations.append(f"{clean_name} ({page_ref})")

        sources_str = ", ".join(clean_citations)
        if len(source_info) > 2:
            sources_str += f" (+ {len(source_info) - 2} more)"
    else:
        sources_str = "N/A"

    # Just provide the direct answer with source citation
    bot_message = f"{resp['response_text']}\n\n**Source:** {sources_str}"
    chat_history.append((message, bot_message))
    return "", chat_history


def refresh_games_handler():
    """Refresh the list of available games from the database and process any new PDFs."""
    try:
        # First, check for new PDFs in the data directory and process them
        from pathlib import Path

        status_messages = []

        # Get list of PDFs in data directory
        data_path = Path(config.DATA_PATH)
        pdf_files = list(data_path.glob("*.pdf"))

        if pdf_files:
            # Connect to database to check existing documents
            with config.suppress_chromadb_telemetry():
                persistent_client = chromadb.PersistentClient(
                    path=config.CHROMA_PATH, settings=config.get_chromadb_settings()
                )
                db = Chroma(
                    client=persistent_client,
                    embedding_function=get_embedding_function(),
                )

            # Get existing document IDs to check what's already processed
            existing_items = db.get(include=[])
            existing_sources = set()
            for doc_id in existing_items["ids"]:
                if ":" in doc_id:
                    source_path = doc_id.split(":")[0]
                    if "/" in source_path:
                        filename = source_path.split("/")[-1]
                        existing_sources.add(filename)

            # Find new PDFs that haven't been processed
            new_pdfs = []
            for pdf_file in pdf_files:
                if pdf_file.name not in existing_sources:
                    new_pdfs.append(pdf_file)

            if new_pdfs:
                status_messages.append(
                    f"📄 Found {len(new_pdfs)} new PDFs to process..."
                )

                # Process each new PDF individually to avoid token limits
                for pdf_file in new_pdfs:
                    try:
                        # Process the PDF
                        chunks = process_single_pdf(str(pdf_file))

                        # Add to database in smaller batches
                        batch_size = 50  # Process in batches of 50 chunks
                        total_added = 0

                        for i in range(0, len(chunks), batch_size):
                            batch_chunks = chunks[i : i + batch_size]
                            success = add_chunks_to_database(batch_chunks)
                            if success:
                                total_added += len(batch_chunks)

                        if total_added > 0:
                            status_messages.append(
                                f"✅ Added {pdf_file.name} ({total_added} chunks)"
                            )
                        else:
                            status_messages.append(
                                f"⚠️ {pdf_file.name} - no new content added"
                            )

                    except Exception as e:
                        status_messages.append(
                            f"❌ Error processing {pdf_file.name}: {str(e)}"
                        )
            else:
                status_messages.append("📄 No new PDFs found in data directory")
        else:
            status_messages.append("📄 No PDFs found in data directory")

        # Now refresh the games list
        available_games = get_available_games()
        if available_games:
            status_messages.append(
                f"✅ Refreshed! Found {len(available_games)} games: {', '.join(available_games[:3])}{'...' if len(available_games) > 3 else ''}"
            )
        else:
            status_messages.append("⚠️ No games found in database")

        final_status = "\n".join(status_messages)
        return final_status, gr.update(choices=available_games)

    except Exception as e:
        return f"❌ Error rebuilding library: {str(e)}", gr.update()


def rebuild_library_handler():
    """Rebuild the entire library by processing all PDFs in the data directory."""
    try:
        from pathlib import Path

        status_messages = []
        status_messages.append("🔄 Starting library rebuild...")

        # Get list of PDFs in data directory
        data_path = Path(config.DATA_PATH)
        pdf_files = list(data_path.glob("*.pdf"))

        if not pdf_files:
            return "📄 No PDFs found in data directory", gr.update()

        status_messages.append(f"📄 Found {len(pdf_files)} PDFs to process...")

        # Process each PDF individually to avoid token limits
        total_processed = 0
        for pdf_file in pdf_files:
            try:
                # Process the PDF
                chunks = process_single_pdf(str(pdf_file))

                # Add to database in smaller batches
                batch_size = 50  # Process in batches of 50 chunks
                total_added = 0

                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i : i + batch_size]
                    success = add_chunks_to_database(batch_chunks)
                    if success:
                        total_added += len(batch_chunks)

                if total_added > 0:
                    status_messages.append(
                        f"✅ Processed {pdf_file.name} ({total_added} chunks)"
                    )
                    total_processed += 1
                else:
                    status_messages.append(f"⚠️ {pdf_file.name} - no new content added")

            except Exception as e:
                status_messages.append(f"❌ Error processing {pdf_file.name}: {str(e)}")

        # Refresh the games list
        available_games = get_available_games()
        status_messages.append(
            f"🎯 Library rebuild complete! {total_processed} PDFs processed."
        )
        status_messages.append(f"📚 Total games available: {len(available_games)}")

        final_status = "\n".join(status_messages)
        return final_status, gr.update(choices=available_games)

    except Exception as e:
        return f"❌ Error rebuilding library: {str(e)}", gr.update()


# Create the interface
with gr.Blocks(theme=gr.themes.Glass(), css=theme_css) as demo:
    # State holding current access level: "none", "user", "admin"
    access_state = gr.State(value="none")

    gr.Markdown(INTRO_STRING)

    # Get available games for the dropdown (with proper names)
    available_games = get_available_games()
    game_choices = available_games

    # Mobile-first responsive layout
    with gr.Row(elem_classes=["main-content"]):
        with gr.Column(scale=3, elem_classes=["chat-column"]):
            chatbot = gr.Chatbot(
                height="80vh",
                show_copy_button=True,
                elem_classes=["custom-chatbot"],
                render_markdown=True,
            )

            # Input section - move here for better mobile UX
            with gr.Row(elem_classes=["input-row"]):
                msg = gr.Textbox(
                    placeholder="First select a game above, then ask your question...",
                    scale=9,
                    container=False,
                )
                clear = gr.ClearButton([msg, chatbot], scale=1, size="sm")

        with gr.Column(scale=1, elem_classes=["sidebar"]):
            # Password field at top of sidebar
            password_tb = gr.Textbox(
                type="password", placeholder="🔐 Enter Password", label="Access"
            )
            access_msg = gr.Markdown("", visible=False)

            # Game selection (hidden until unlocked)
            game_dropdown = gr.Dropdown(
                choices=game_choices,
                value=None,
                label="Select Game (Required)",
                info="Choose a game to get answers from its rulebook",
                visible=False,
            )

            # Add PDF upload section in expanding panel
            with gr.Accordion(
                "📤 Add New Game", open=False, visible=False
            ) as upload_accordion:
                upload_file = gr.File(
                    file_types=[".pdf"],
                    label="Upload PDF Rulebook",
                    file_count="single",
                )
                upload_status = gr.Textbox(
                    label="Upload Status", interactive=False, visible=False
                )

            # Admin-only panels ------------------------------------------------

            with gr.Accordion(
                "🗑️ Delete PDF", open=False, visible=False
            ) as delete_accordion:
                delete_dropdown = gr.Dropdown(choices=list_pdfs(), label="Select PDF")
                delete_button = gr.Button("Delete", variant="stop")
                delete_status = gr.Textbox(interactive=False)

            with gr.Accordion(
                "✏️ Rename Game", open=False, visible=False
            ) as rename_accordion:
                rename_game_dropdown = gr.Dropdown(
                    choices=game_choices, label="Select Game"
                )
                new_name_tb = gr.Textbox(label="New Name")
                rename_button = gr.Button("Rename", variant="primary")
                rename_status = gr.Textbox(interactive=False)

            # Technical info (admin only)
            with gr.Accordion(
                "⚙️ Technical Info", open=False, visible=False
            ) as tech_accordion:
                gr.Markdown(get_config_info())

                # Add rebuild library button
                with gr.Row():
                    rebuild_button = gr.Button(
                        "🔄 Rebuild Library", variant="secondary"
                    )
                    refresh_button = gr.Button(
                        "🔄 Process New PDFs", variant="secondary"
                    )

                rebuild_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    visible=False,
                    lines=6,
                    placeholder="Click 'Rebuild Library' to process all PDFs or 'Process New PDFs' to add only new ones",
                )

    msg.submit(query_interface, [msg, game_dropdown, chatbot], [msg, chatbot])

    # Connect rebuild library button
    rebuild_button.click(
        rebuild_library_handler, inputs=[], outputs=[rebuild_status, game_dropdown]
    ).then(lambda: gr.update(visible=True), outputs=[rebuild_status])

    # Connect process new PDFs button
    refresh_button.click(
        refresh_games_handler, inputs=[], outputs=[rebuild_status, game_dropdown]
    ).then(lambda: gr.update(visible=True), outputs=[rebuild_status])

    # Connect upload button
    def upload_with_status_update(pdf_file):
        status, dropdown_update = upload_pdf_handler(pdf_file)
        # Refresh PDF list for delete dropdown
        new_pdfs = list_pdfs()
        return (
            gr.update(value=status, visible=True),
            dropdown_update,
            gr.update(value=None),  # reset file input
            gr.update(choices=new_pdfs),
        )

    upload_file.upload(
        upload_with_status_update,
        inputs=[upload_file],
        outputs=[
            upload_status,
            game_dropdown,
            upload_file,
            delete_dropdown,
        ],
    )

    # ------------------------------------------------------------------
    # ACCESS CONTROL HANDLER
    # ------------------------------------------------------------------

    def unlock_handler(pw):
        if ADMIN_PW and pw == ADMIN_PW:
            level = "admin"
            msg = "### ✅ Admin access granted"
        elif USER_PW and pw == USER_PW:
            level = "user"
            msg = "### ✅ User access granted"
        else:
            level = "none"
            msg = "### ❌ Invalid password"

        # Determine visibilities
        show_user = level in {"user", "admin"}
        show_admin = level == "admin"

        # Refresh game lists when logging in
        updated_games = refresh_game_lists() if show_user else []

        # Updates
        return (
            level,
            gr.update(value=msg, visible=True),
            gr.update(choices=updated_games, visible=show_user),  # game_dropdown
            gr.update(visible=show_user),  # upload_accordion
            gr.update(visible=show_admin),  # delete_accordion
            gr.update(visible=show_admin),  # rename_accordion
            gr.update(visible=show_admin),  # tech_accordion
        )

    password_tb.change(
        unlock_handler,
        inputs=[password_tb],
        outputs=[
            access_state,
            access_msg,
            game_dropdown,
            upload_accordion,
            delete_accordion,
            rename_accordion,
            tech_accordion,
        ],
    ).then(
        # Refresh admin dropdown contents when panels become visible
        lambda level: (
            gr.update(choices=list_pdfs()) if level == "admin" else gr.update(),
            gr.update(choices=refresh_game_lists())
            if level == "admin"
            else gr.update(),
        ),
        inputs=[access_state],
        outputs=[delete_dropdown, rename_game_dropdown],
    )

    # ------------------------------------------------------------------
    # DELETE PDF HANDLER
    # ------------------------------------------------------------------

    def delete_pdf_handler(pdf_name):
        if not pdf_name:
            return "❌ No PDF selected", gr.update(), gr.update(), gr.update()
        try:
            # Remove file from disk
            file_path = os.path.join(config.DATA_PATH, pdf_name)
            if os.path.exists(file_path):
                os.remove(file_path)

            # Remove chunks from DB
            with config.suppress_chromadb_telemetry():
                client = chromadb.PersistentClient(
                    path=config.CHROMA_PATH, settings=config.get_chromadb_settings()
                )
                db = Chroma(client=client, embedding_function=get_embedding_function())
                ids_to_del = [i for i in db.get(include=[])["ids"] if pdf_name in i]
                if ids_to_del:
                    db.delete(ids=ids_to_del)

                # Remove stored game name
                try:
                    names_col = client.get_collection("game_names")
                    names_col.delete(ids=[pdf_name])
                except Exception:
                    pass

            # Refresh dropdowns
            new_pdfs = list_pdfs()
            new_games = refresh_game_lists()
            return (
                f"✅ Deleted {pdf_name}",
                gr.update(choices=new_pdfs, value=None),
                gr.update(choices=new_games, value=None),
                gr.update(choices=new_games, value=None),
            )
        except Exception as e:
            return f"❌ Error: {e}", gr.update(), gr.update(), gr.update()

    delete_button.click(
        delete_pdf_handler,
        inputs=[delete_dropdown],
        outputs=[delete_status, delete_dropdown, game_dropdown, rename_game_dropdown],
    )

    # ------------------------------------------------------------------
    # RENAME GAME HANDLER
    # ------------------------------------------------------------------

    def rename_game_handler(old_name, new_name):
        if not old_name or not new_name:
            return "❌ Provide both fields", gr.update(), gr.update()
        try:
            stored = get_stored_game_names()
            filename = None
            for fn, gn in stored.items():
                if gn == old_name:
                    filename = fn
                    break
            if not filename:
                return "❌ Game not found", gr.update(), gr.update()

            with config.suppress_chromadb_telemetry():
                client = chromadb.PersistentClient(
                    path=config.CHROMA_PATH, settings=config.get_chromadb_settings()
                )
                col = client.get_or_create_collection(
                    name="game_names",
                    metadata={
                        "description": "Stores extracted game names from PDF filenames"
                    },
                )
                col.update(ids=[filename], documents=[new_name])

            games = refresh_game_lists()
            return (
                f"✅ Renamed to {new_name}",
                gr.update(choices=games, value=None),
                gr.update(choices=games, value=None),
            )
        except Exception as e:
            return f"❌ Error: {e}", gr.update(), gr.update()

    rename_button.click(
        rename_game_handler,
        inputs=[rename_game_dropdown, new_name_tb],
        outputs=[rename_status, game_dropdown, rename_game_dropdown],
    )


if __name__ == "__main__":
    import os

    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get("PORT", 7860))

    demo.queue(max_size=50)  # Allow up to 50 users in queue
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        allowed_paths=[config.DATA_PATH],
        share=False,
    )
