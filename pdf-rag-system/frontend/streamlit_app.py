import streamlit as st
import requests
import os
from pathlib import Path
import time

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="PDF RAG System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG System")
st.markdown("Upload PDFs and ask questions with citation support")

# Initialize session state
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "messages" not in st.session_state:
    st.session_state.messages = []


def load_documents():
    """Load list of uploaded documents."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/documents")
        if response.status_code == 200:
            return response.json().get("documents", [])
    except Exception as e:
        st.error(f"Error loading documents: {e}")
    return []


def upload_pdf(file, use_vlm=False):
    """Upload a PDF file."""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        params = {"use_vlm": "true"} if use_vlm else None
        # VLM extraction can be slow; use a longer timeout when enabled
        timeout = 300 if use_vlm else 60
        response = requests.post(
            f"{API_BASE_URL}/api/upload",
            files=files,
            params=params,
            timeout=timeout
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            try:
                detail = response.json().get("detail", "Upload failed")
            except Exception:
                detail = f"Upload failed (HTTP {response.status_code})"
            return {"error": detail}
    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}


def query_rag(question, top_k=5, pdf_filter=None, use_multimodal=True):
    """Query the RAG system."""
    try:
        payload = {
            "question": question,
            "top_k": top_k,
            "pdf_filter": pdf_filter,
            "use_multimodal": use_multimodal
        }
        response = requests.post(f"{API_BASE_URL}/api/query", json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Query failed")}
    except Exception as e:
        return {"error": str(e)}


# Sidebar for document management
with st.sidebar:
    st.header("📄 Document Management")
    
    # Upload section
    st.subheader("Upload PDF")
    use_vlm = st.checkbox(
        "Use vision extraction (slower)",
        value=False,
        help="Enable VLM extraction during upload for precise product crops. This is slower."
    )
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF document to add to the knowledge base"
    )
    
    if uploaded_file is not None:
        if st.button("Upload", type="primary"):
            with st.spinner("Uploading and processing PDF..."):
                result = upload_pdf(uploaded_file, use_vlm=use_vlm)
                
                if "error" in result:
                    st.error(f"Upload failed: {result['error']}")
                else:
                    st.success(f"✅ {result['filename']} uploaded successfully!")
                    st.info(f"📄 {result['total_pages']} pages, {result['chunks_added']} chunks processed")
                    st.session_state.uploaded_docs = load_documents()
                    time.sleep(1)
                    st.rerun()
    
    st.divider()
    
    # List uploaded documents
    st.subheader("Uploaded Documents")
    documents = load_documents()
    
    if documents:
        for doc in documents:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(doc)
            with col2:
                if st.button("🗑️", key=f"delete_{doc}", help="Delete document"):
                    try:
                        response = requests.delete(f"{API_BASE_URL}/api/documents/{doc}")
                        if response.status_code == 200:
                            st.success(f"Deleted {doc}")
                            st.session_state.uploaded_docs = load_documents()
                            time.sleep(1)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("No documents uploaded yet")
    
    st.divider()
    
    # PDF filter
    st.subheader("Filter by Document")
    pdf_filter = st.selectbox(
        "Select a document to filter queries",
        options=[None] + documents,
        format_func=lambda x: "All Documents" if x is None else x
    )
    
    st.session_state.pdf_filter = pdf_filter

    st.divider()
    st.subheader("Query Options")
    use_multimodal = st.checkbox(
        "Use multimodal search (slower)",
        value=True,
        help="Uses VLM/CLIP for product crops and visual matching. Turn off for faster queries."
    )
    st.session_state.use_multimodal = use_multimodal


# Main content area
tab1, tab2 = st.tabs(["💬 Chat", "📊 About"])

with tab1:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show citations if available
            if message["role"] == "assistant" and "citations" in message:
                with st.expander("📎 Citations", expanded=False):
                    for i, citation in enumerate(message["citations"], 1):
                        st.markdown(f"**Source {i}** - Page {citation['page_number']} of *{citation['pdf_name']}*")
                        st.markdown(f"*Relevance: {citation.get('relevance_score', 'N/A'):.2f}*" if citation.get('relevance_score') else "")
                        st.markdown(f"*Snippet: {citation['text_snippet']}*")
                        
                        # Display cropped citation image first (product-level), then full page
                        page_num = citation.get('page_number')
                        image_displayed = False
                        
                        # Always try to show CROPPED image first (product-level),
                        # then provide full page as an optional expander.
                        crop_url = citation.get("crop_image_url")
                        full_url = citation.get("image_url")

                        # Prefer local file paths when available (more reliable than HTTP)
                        crop_path = citation.get("crop_image_path")
                        full_path = citation.get("image_path")

                        # Try to display cropped image first
                        if crop_path and os.path.exists(crop_path):
                            try:
                                st.image(
                                    crop_path,
                                    caption=f"🍪 Cropped Product View - Page {page_num}",
                                    use_container_width=True,
                                )
                                image_displayed = True
                            except Exception:
                                image_displayed = False
                        elif crop_url:
                            try:
                                import requests
                                from PIL import Image
                                import io
                                
                                # Fetch and display cropped image
                                img_response = requests.get(f"{API_BASE_URL}{crop_url}", timeout=5)
                                if img_response.status_code == 200:
                                    img = Image.open(io.BytesIO(img_response.content))
                                    st.image(
                                        img,
                                        caption=f"🍪 Cropped Product View - Page {page_num}",
                                        use_container_width=True,
                                    )
                                    image_displayed = True
                            except Exception as e:
                                # Try direct URL as fallback
                                try:
                                    st.image(
                                        f"{API_BASE_URL}{crop_url}",
                                        caption=f"🍪 Cropped Product View - Page {page_num}",
                                        use_container_width=True,
                                    )
                                    image_displayed = True
                                except:
                                    image_displayed = False

                        # If cropped image failed, try full page
                        if not image_displayed and full_path and os.path.exists(full_path):
                            try:
                                st.image(
                                    full_path,
                                    caption=f"📄 Full Page {page_num}",
                                    use_container_width=True,
                                )
                                image_displayed = True
                            except Exception:
                                image_displayed = False
                        elif not image_displayed and full_url:
                            try:
                                import requests
                                from PIL import Image
                                import io
                                
                                img_response = requests.get(f"{API_BASE_URL}{full_url}", timeout=5)
                                if img_response.status_code == 200:
                                    img = Image.open(io.BytesIO(img_response.content))
                                    st.image(
                                        img,
                                        caption=f"📄 Full Page {page_num}",
                                        use_container_width=True,
                                    )
                                    image_displayed = True
                            except Exception:
                                try:
                                    st.image(
                                        f"{API_BASE_URL}{full_url}",
                                        caption=f"📄 Full Page {page_num}",
                                        use_container_width=True,
                                    )
                                    image_displayed = True
                                except:
                                    image_displayed = False

                        # Show full page as additional option if we have both
                        if image_displayed and full_url and crop_url:
                            if st.button(f"📄 Show Full Page {page_num}", key=f"fullpage_{i}_{page_num}"):
                                if full_path and os.path.exists(full_path):
                                    st.image(full_path, caption=f"Full Page {page_num}", use_container_width=True)
                                else:
                                    st.image(f"{API_BASE_URL}{full_url}", caption=f"Full Page {page_num}", use_container_width=True)

                        # Last resort: show links only if images completely failed
                        if not image_displayed:
                            if crop_url:
                                st.markdown(f"🧩 **Cropped image:** [Open]({API_BASE_URL}{crop_url})")
                            if full_url:
                                st.markdown(f"📄 **Full page image:** [Open]({API_BASE_URL}{full_url})")

# Chat input (must be outside tabs)
if prompt := st.chat_input("Ask a question about your PDFs..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get response
    with st.spinner("Searching documents and generating answer..."):
        result = query_rag(
            prompt,
            top_k=5,
            pdf_filter=st.session_state.get("pdf_filter"),
            use_multimodal=st.session_state.get("use_multimodal", True)
        )
        
        if "error" in result:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {result['error']}"
            })
        else:
            # Save to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "citations": result.get("citations", [])
            })
    
    # Rerun to show new messages
    st.rerun()

with tab2:
    st.header("About PDF RAG System")
    st.markdown("""
    ### Features
    
    - **PDF Upload**: Upload PDF documents to build your knowledge base
    - **Semantic Search**: Ask questions and get relevant answers
    - **Citations**: See exactly where answers come from with page references
    - **Visual Citations**: View images of PDF pages where answers are found
    - **Document Filtering**: Filter queries to specific documents
    
    ### How It Works
    
    1. **Upload**: PDFs are processed and text is extracted with page-level tracking
    2. **Embedding**: Text chunks are converted to vector embeddings
    3. **Storage**: Embeddings are stored in a vector database (ChromaDB)
    4. **Retrieval**: When you ask a question, relevant chunks are retrieved
    5. **Generation**: An LLM generates an answer based on retrieved context
    6. **Citation**: You see the source pages and can view page images
    
    ### Technology Stack
    
    - **Backend**: FastAPI
    - **Vector Store**: ChromaDB
    - **Embeddings**: Sentence Transformers
    - **LLM**: Ollama (local) or OpenAI
    - **Frontend**: Streamlit
    - **PDF Processing**: pdfplumber, pdf2image
    """)


if __name__ == "__main__":
    pass

