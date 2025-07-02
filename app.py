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


def load_css():
    """Load CSS from external file for easy hot-reloading."""
    try:
        with open("styles.css", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Fallback CSS if file doesn't exist
        return """
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * { font-family: 'Inter', sans-serif !important; }
        .gradio-container { height: 100vh !important; }
        .main { height: calc(100vh - 60px) !important; }
        """


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
            return True
        return False

    except Exception as e:
        print(f"Error adding to database: {e}")
        return False


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


def download_pdf_handler(selected_game):
    """Handle PDF download request."""
    if not selected_game:
        return gr.update(visible=False)

    try:
        # Get the filename mapping
        mapping = getattr(get_available_games, "_filename_mapping", {})
        simple_name = mapping.get(selected_game, selected_game.lower())

        # Find the PDF file
        data_path = Path(config.DATA_PATH)
        pdf_files = list(data_path.glob("*.pdf"))

        for pdf_file in pdf_files:
            if simple_name in pdf_file.name.lower():
                return gr.update(value=str(pdf_file), visible=True)

        # If no exact match, try partial match
        for pdf_file in pdf_files:
            if any(
                word.lower() in pdf_file.name.lower() for word in selected_game.split()
            ):
                return gr.update(value=str(pdf_file), visible=True)

        return gr.update(visible=False)

    except Exception as e:
        print(f"Error finding PDF for {selected_game}: {e}")
        return gr.update(visible=False)


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

        # Create clean citations with clickable links
        clean_citations = []
        for filename, pages in list(source_info.items())[:2]:  # Limit to 2 files
            clean_name = (
                filename.replace(".pdf", "").replace("_", " ").replace("-", " ")
            )
            clean_name = " ".join(word.capitalize() for word in clean_name.split())

            # Sort page numbers and create page reference
            sorted_pages = sorted(pages)
            if len(sorted_pages) == 1:
                page_ref = f"p. {sorted_pages[0]}"
            elif len(sorted_pages) <= 3:
                page_ref = f"pp. {', '.join(map(str, sorted_pages))}"
            else:
                page_ref = f"pp. {sorted_pages[0]}-{sorted_pages[-1]}"

            # Create citation without clickable link for now
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


with gr.Blocks(css=load_css()) as demo:
    gr.Markdown(INTRO_STRING)

    # Get available games for the dropdown (with proper names)
    available_games = get_available_games()
    game_choices = available_games

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height="70vh", show_copy_button=True, elem_classes=["custom-chatbot"]
            )
        with gr.Column(scale=1):
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

            # Add PDF viewer section
            with gr.Accordion("📖 Download PDFs", open=False):
                pdf_dropdown = gr.Dropdown(
                    choices=[f for f in available_games],
                    label="Select PDF to Download",
                    value=None,
                )
                download_button = gr.Button("Download PDF", variant="secondary")
                pdf_file_output = gr.File(label="PDF Download", visible=False)

            # Add collapsible config section
            with gr.Accordion("⚙️ Technical Info", open=False):
                gr.Markdown(get_config_info())

    with gr.Row():
        msg = gr.Textbox(
            placeholder="First select a game above, then ask your question...", scale=9
        )
        clear = gr.ClearButton([msg, chatbot], scale=1, size="sm")

    msg.submit(query_interface, [msg, game_dropdown, chatbot], [msg, chatbot])

    # Connect upload button
    def upload_with_status_update(pdf_file):
        status, dropdown_update = upload_pdf_handler(pdf_file)
        available_games = get_available_games()
        return (
            status,
            dropdown_update,
            gr.update(visible=True),
            gr.update(value=None),
            gr.update(choices=available_games),  # Update PDF dropdown too
        )

    upload_button.click(
        upload_with_status_update,
        inputs=[upload_file],
        outputs=[
            upload_status,
            game_dropdown,
            upload_status,
            upload_file,
            pdf_dropdown,
        ],
    )

    # Connect download button
    download_button.click(
        download_pdf_handler, inputs=[pdf_dropdown], outputs=[pdf_file_output]
    )

if __name__ == "__main__":
    demo.queue(max_size=50)  # Allow up to 50 users in queue
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[config.DATA_PATH],
        share=False,  # Set to True only for temporary sharing
    )
