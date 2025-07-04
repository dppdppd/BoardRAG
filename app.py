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
from query import get_available_games, query_rag

# Disable ChromaDB telemetry after imports
config.disable_chromadb_telemetry()

INTRO_STRING = """
# 🎲 BoardRAG
"""

# CSS for dark/light mode toggle
theme_css = """
/* Light mode (default) */
.gradio-container {
    transition: background-color 0.3s ease, color 0.3s ease;
}

/* Dark mode overrides */
.dark-mode .gradio-container {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
}

.dark-mode .gradio-container .block,
.dark-mode .gradio-container [data-testid="block"] {
    background-color: #2d2d2d !important;
    border-color: #404040 !important;
    color: #ffffff !important;
}

.dark-mode .gradio-container input,
.dark-mode .gradio-container textarea,
.dark-mode .gradio-container select {
    background-color: #333333 !important;
    border-color: #555555 !important;
    color: #ffffff !important;
}

.dark-mode .gradio-container button {
    background-color: #404040 !important;
    color: #ffffff !important;
    border-color: #555555 !important;
}

.dark-mode .gradio-container button:hover {
    background-color: #555555 !important;
}

.dark-mode .gradio-container .chatbot,
.dark-mode .gradio-container [data-testid="chatbot"] {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
}

.dark-mode .gradio-container p,
.dark-mode .gradio-container h1,
.dark-mode .gradio-container h2,
.dark-mode .gradio-container h3,
.dark-mode .gradio-container h4,
.dark-mode .gradio-container h5,
.dark-mode .gradio-container h6,
.dark-mode .gradio-container span,
.dark-mode .gradio-container div,
.dark-mode .gradio-container label {
    color: #ffffff !important;
}

/* Theme toggle button */
#theme-toggle {
    margin-bottom: 10px;
}
"""


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
    # State to track current theme
    current_theme = gr.State(value="light")

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
            # Theme toggle button
            theme_toggle = gr.Button(
                "🌙 Dark Mode", elem_id="theme-toggle", variant="secondary", size="sm"
            )

            game_dropdown = gr.Dropdown(
                choices=game_choices,
                value=None,
                label="Select Game (Required)",
                info="Choose a game to get answers from its rulebook",
            )

            # Add PDF upload section in expanding panel
            with gr.Accordion("📤 Add New Game", open=False):
                upload_file = gr.File(
                    file_types=[".pdf"],
                    label="Upload PDF Rulebook",
                    file_count="single",
                )
                upload_button = gr.Button("Process PDF", variant="primary")
                upload_status = gr.Textbox(
                    label="Upload Status", interactive=False, visible=False
                )

            # Add collapsible config section
            with gr.Accordion("⚙️ Technical Info", open=False):
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
        return (
            status,
            dropdown_update,
            gr.update(visible=True),
            gr.update(value=None),
        )

    upload_button.click(
        upload_with_status_update,
        inputs=[upload_file],
        outputs=[
            upload_status,
            game_dropdown,
            upload_status,
            upload_file,
        ],
    )

    # Theme toggle functionality
    def toggle_theme_handler(current_theme_state):
        """Toggle between light and dark themes."""
        new_theme = "dark" if current_theme_state == "light" else "light"
        button_text = "☀️ Light Mode" if new_theme == "dark" else "🌙 Dark Mode"

        return new_theme, gr.update(value=button_text)

    theme_toggle.click(
        fn=toggle_theme_handler,
        inputs=[current_theme],
        outputs=[current_theme, theme_toggle],
        js="""
        function(current_theme_state) {
            const body = document.body;
            const isDarkMode = body.classList.contains('dark-mode');
            
            if (isDarkMode) {
                body.classList.remove('dark-mode');
                localStorage.setItem('theme', 'light');
            } else {
                body.classList.add('dark-mode');
                localStorage.setItem('theme', 'dark');
            }
            
            return current_theme_state;
        }
        """,
    )

    # Initialize theme on page load
    demo.load(
        fn=None,
        js="""
        function() {
            const savedTheme = localStorage.getItem('theme');
            const body = document.body;
            
            if (savedTheme === 'dark') {
                body.classList.add('dark-mode');
                
                setTimeout(() => {
                    const themeButton = document.querySelector('#theme-toggle button');
                    if (themeButton) {
                        themeButton.textContent = '☀️ Light Mode';
                    }
                }, 100);
            }
        }
        """,
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
