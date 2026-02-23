import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
import shutil
from pathlib import Path

from .pdf_processor import PDFProcessor
from .vector_store import VectorStore
from .rag_service import RAGService

load_dotenv()

app = FastAPI(title="PDF RAG System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
citation_images_dir = os.getenv("CITATION_IMAGES_DIR", "./static/citations")
vector_store_path = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

pdf_processor = PDFProcessor(upload_dir, citation_images_dir)
vector_store = VectorStore(embedding_model=embedding_model, persist_dir=vector_store_path)

llm_provider = os.getenv("LLM_PROVIDER", "ollama")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
openai_api_key = os.getenv("OPENAI_API_KEY")

rag_service = RAGService(
    vector_store=vector_store,
    pdf_processor=pdf_processor,
    llm_provider=llm_provider,
    ollama_base_url=ollama_base_url,
    ollama_model=ollama_model,
    openai_api_key=openai_api_key
)

# Mount static files for citation images
app.mount("/static", StaticFiles(directory=citation_images_dir), name="static")


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    pdf_filter: Optional[str] = None
    use_multimodal: Optional[bool] = True  # Enable hybrid search by default


@app.get("/")
async def root():
    return {"message": "PDF RAG System API", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...), use_vlm: bool = False):
    """
    Upload a PDF file and process it.
    
    Args:
        file: PDF file to upload
        use_vlm: If True, use VLM to extract structured product data
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Save uploaded file
    print(f"[UPLOAD] Start: {file.filename} (use_vlm={use_vlm})")
    file_path = Path(upload_dir) / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"[UPLOAD] Saved: {file_path}")
    
    try:
        # Process PDF (with optional VLM extraction)
        print(f"[UPLOAD] Processing PDF: {file.filename}")
        processed_data = pdf_processor.process_pdf(str(file_path), use_vlm=use_vlm)
        print(f"[UPLOAD] Processed PDF: pages={processed_data['total_pages']}, chunks={len(processed_data['chunks'])}, products={len(processed_data.get('structured_products', []))}")
        
        # Update chunks with absolute path for citation image generation
        absolute_pdf_path = str(file_path.resolve())
        for chunk in processed_data["chunks"]:
            chunk["pdf_path"] = absolute_pdf_path
        
        # Add text chunks to vector store
        print(f"[UPLOAD] Adding chunks to vector store: {file.filename}")
        num_chunks = vector_store.add_documents(
            processed_data["chunks"],
            pdf_name=file.filename
        )
        print(f"[UPLOAD] Chunks added: {num_chunks}")
        
        # Add structured products if VLM was used
        num_products = 0
        if use_vlm and processed_data.get("structured_products"):
            try:
                print(f"[UPLOAD] Adding structured products to vector store: {file.filename}")
                num_products = vector_store.add_multimodal_documents(
                    processed_data["structured_products"],
                    pdf_name=file.filename
                )
                print(f"[UPLOAD] Products added: {num_products}")
            except Exception as e:
                print(f"⚠️  Failed to add multimodal products: {e}")
        
        return {
            "message": "PDF uploaded and processed successfully",
            "filename": file.filename,
            "total_pages": processed_data["total_pages"],
            "chunks_added": num_chunks,
            "products_added": num_products,
            "vlm_enabled": use_vlm,
            "file_path": str(file_path)
        }
    except Exception as e:
        # Clean up on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/api/query")
async def query_documents(request: QueryRequest):
    """Query the RAG system."""
    try:
        result = rag_service.query(
            question=request.question,
            top_k=request.top_k or 5,
            pdf_filter=request.pdf_filter,
            use_multimodal=request.use_multimodal if request.use_multimodal is not None else True
        )
        
        # Convert image paths to URLs
        for citation in result["citations"]:
            if citation.get("image_path") and os.path.exists(citation["image_path"]):
                # Use just the filename for the URL
                filename = os.path.basename(citation["image_path"])
                citation["image_url"] = f"/static/{filename}"
            if citation.get("crop_image_path") and os.path.exists(citation["crop_image_path"]):
                crop_filename = os.path.basename(citation["crop_image_path"])
                citation["crop_image_url"] = f"/static/{crop_filename}"
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/api/documents")
async def list_documents():
    """List all uploaded PDF documents."""
    pdfs = vector_store.get_all_pdfs()
    return {"documents": pdfs, "count": len(pdfs)}


@app.delete("/api/documents/{pdf_name}")
async def delete_document(pdf_name: str):
    """Delete a PDF document and its chunks."""
    try:
        # Delete from vector store
        deleted_count = vector_store.delete_pdf(pdf_name)
        
        # Delete file if exists
        file_path = Path(upload_dir) / pdf_name
        if file_path.exists():
            file_path.unlink()
        
        return {
            "message": f"Document {pdf_name} deleted",
            "chunks_deleted": deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@app.get("/api/citation-image/{image_name}")
async def get_citation_image(image_name: str):
    """Get a citation image."""
    image_path = Path(citation_images_dir) / image_name
    if image_path.exists():
        return FileResponse(image_path)
    raise HTTPException(status_code=404, detail="Image not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

