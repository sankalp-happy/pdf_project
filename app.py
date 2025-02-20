# app.py
import streamlit as st
import os
import time
import shutil
import pandas as pd
from datetime import datetime
from typing import List, Dict
from search_logic import DocumentProcessor
import PyPDF2  # For basic PDF text extraction (fallback)
import logging
from pathlib import Path

# Configuration
DATA_DIR = "data/"
PROCESSED_DIR = "processed/"
EMBEDDINGS_DIR = "embeddings/"

# Ensure directories exist
Path(DATA_DIR).mkdir(exist_ok=True)
Path(PROCESSED_DIR).mkdir(exist_ok=True)
Path(EMBEDDINGS_DIR).mkdir(exist_ok=True)


if 'processor' not in st.session_state:
    st.session_state.processor = DocumentProcessor(
        data_dir=DATA_DIR,
        processed_dir=PROCESSED_DIR,
        embeddings_dir=EMBEDDINGS_DIR
    )

# Type definitions
SearchResult = Dict[str, str]

def main():
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
    # Page setup
    st.set_page_config(
        page_title="Hybrid Semantic Search",
        page_icon="🔍",
        layout="centered"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
        .badge {
            background-color: #4f8bf9;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin-left: 10px;
        }
        .progress-bar {
            height: 8px;
            background: #eee;
            border-radius: 4px;
            margin: 5px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4f8bf9 0%, #2ecc71 100%);
            border-radius: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🔍 Hybrid Semantic Search")
    st.markdown("Upload PDFs and search using semantic understanding + keyword matching")

    with st.sidebar:
        st.subheader("System Status")
        docs_processed = os.path.exists(os.path.join("processed", "documents.csv"))
        status_icon = "🟢" if docs_processed else "🔴"
        st.markdown(f"{status_icon} Documents Processed")
        
        if docs_processed:
            df = pd.read_csv(os.path.join("processed", "documents.csv"))
            st.caption(f"• {len(df)} text chunks\n• Last processed: {datetime.fromtimestamp(os.path.getmtime('processed/documents.csv')).strftime('%Y-%m-%d %H:%M')}")
        if st.checkbox("Show debug info"):
            st.write("Files in data directory:", os.listdir(DATA_DIR))
            st.write("Files in processed directory:", os.listdir(PROCESSED_DIR))  # Now using the d
    # Layout columns for responsive design
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        # Search interface
        search_query = st.text_input(
            "Search documents...",
            placeholder="Enter your query (e.g., 'AI ethics in healthcare')"
        )
        
        # Search mode selector
        search_mode = st.radio(
            "Search Mode:",
            ["Semantic", "Keyword", "Hybrid"],
            horizontal=True,
            index=2  # Default to Hybrid
        )

    with col2:
        # PDF Upload system
        uploaded_files = st.file_uploader(
            "Drag PDFs here",
            type="pdf",
            accept_multiple_files=True,
            help="Upload multiple research papers or documents"
        )

    # Handle file uploads
    if uploaded_files:
        handle_file_upload(uploaded_files)

    # Display results if search query is provided
    if search_query:
        if not docs_processed:
            st.error("No documents processed! Upload PDFs first.")
            return
        
        try:
            with st.spinner(f"Searching {len(df)} chunks..."):
                start_time = time.time()
                
                # Perform search
                results = st.session_state.processor.hybrid_search(
                    query=search_query,
                    top_k=10
                )
                
                # Display performance
                st.caption(f"Found {len(results)} results in {time.time()-start_time:.2f}s")
                display_results(results, search_query)
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            logging.error(f"Search error: {str(e)}")


def handle_file_upload(uploaded_files: List) -> None:
    """Process and store uploaded PDF files"""
    try:
        if not uploaded_files:
            return

        # Get current files without resetting directory
        current_files = {f.name: f.stat().st_size for f in Path(DATA_DIR).glob("*") if f.is_file()}
        new_files = {f.name: f.size for f in uploaded_files}

        # Check if files are identical to previous upload
        if st.session_state.get('processed_files') == new_files:
            st.info("Using previously processed files")
            return

        # Only remove files that aren't in the new upload
        for existing_file in Path(DATA_DIR).glob("*"):
            if existing_file.name not in new_files:
                existing_file.unlink()

        # Save new files without wiping directory
        for file in uploaded_files:
            file_path = Path(DATA_DIR) / file.name
            if not file_path.exists() or file_path.stat().st_size != file.size:
                with file_path.open("wb") as f:
                    f.write(file.getbuffer())

        # Continue with processing only if files changed
        with st.status("Processing documents...", expanded=True) as status:
            try:
                df = st.session_state.processor.process_documents()
                st.session_state.processor.build_indices()
                st.session_state.processed_files = new_files
                st.success(f"Processed {len(df)} chunks from {len(uploaded_files)} files")
                
            except Exception as e:
                status.update(label="Processing failed!", state="error")
                st.error(f"Error: {str(e)}")
                logging.exception("Processing crash:")

    except Exception as e:
        st.error(f"Upload failed: {str(e)}")
        logging.exception("Upload error:")


def display_results(results: List[SearchResult], search_query: str) -> None:
    """Render search results with styling"""
    for result in results:
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                # Title with metadata
                st.markdown(f"**{result['title']}**")
                st.caption(f"Author: {result['author']} | Year: {result['year']}")
                
                # Score breakdown
                st.markdown(f"""
                <div style="font-size:0.8em; color:#666;">
                    Semantic: {result['semantic_score']:.2f} | 
                    Keyword: {result['keyword_score']:.2f} | 
                    Combined: {result['score']:.2f}
                </div>
                """, unsafe_allow_html=True)
                
                # Snippet with highlight
                st.markdown(f"`...{highlight_query_terms(result['snippet'], search_query)}...`")
            with col2:
                # Visual score indicator
                score = result['score'] * 100  # Convert 0-1 scale to percentage
                st.markdown(f"""
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {score}%"></div>
                </div>
                <div style="text-align: center; font-size: 0.8em;">
                    {score:.1f}%
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()

def highlight_query_terms(text: str, query: str) -> str:
    """Highlight matching terms in results"""
    terms = set(query.lower().split())
    highlighted = []
    for word in text.split():
        if word.lower() in terms:
            highlighted.append(f"<mark>{word}</mark>")
        else:
            highlighted.append(word)
    return " ".join(highlighted)

if __name__ == "__main__":
    main()
