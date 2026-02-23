# Quick Start Guide

## Step-by-Step Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Install Poppler (for PDF to Image conversion)

**macOS:**
```bash
brew install poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**Windows:**
- Download from: https://github.com/oschwartz10612/poppler-windows/releases/
- Extract and add `bin` folder to your PATH

### 3. Setup LLM

**Option A: Ollama (Local - Recommended)**
```bash
# Install Ollama from https://ollama.ai
# Then pull a model:
ollama pull llama3.1:8b
```

**Option B: OpenAI**
- Get API key from https://platform.openai.com
- Add to `.env` file

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 5. Start the Application

**Terminal 1 - Backend:**
```bash
cd pdf-rag-system
source venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
cd pdf-rag-system
source venv/bin/activate
streamlit run frontend/streamlit_app.py --server.port 8501
```

### 6. Use the System

1. Open http://localhost:8501 in your browser
2. Upload a PDF using the sidebar
3. Ask questions in the chat interface
4. View citations with page images

## Testing the API

```bash
# Health check
curl http://localhost:8000/health

# List documents
curl http://localhost:8000/api/documents

# Upload a PDF
curl -X POST http://localhost:8000/api/upload \
  -F "file=@your_document.pdf"

# Query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

## Troubleshooting

### "Poppler not found" error
- Ensure poppler is installed and in PATH
- On macOS: `brew install poppler`
- Verify: `pdftoppm -h` should work

### Ollama connection error
- Check Ollama is running: `ollama list`
- Verify model is available: `ollama pull llama3.1:8b`
- Check `.env` has correct `OLLAMA_BASE_URL`

### Import errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

