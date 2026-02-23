#!/bin/bash

# PDF RAG System Startup Script

echo "🚀 Starting PDF RAG System..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "📝 Please edit .env with your configuration"
    elif [ -f "env.example" ]; then
        cp env.example .env
        echo "📝 Please edit .env with your configuration"
    else
        echo "⚠️  No env.example found. Creating minimal .env..."
        echo "# PDF RAG - add your settings" > .env
        echo "LLM_PROVIDER=ollama" >> .env
        echo "OLLAMA_BASE_URL=http://localhost:11434" >> .env
        echo "📝 Edit .env or copy env.example if available"
    fi
fi

# Create necessary directories
mkdir -p uploads static/citations chroma_db

echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "  Terminal 1: uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload"
echo "  Terminal 2: streamlit run frontend/streamlit_app.py --server.port 8501"
echo ""
echo "Or run both in background:"
echo "  ./start_backend.sh &"
echo "  ./start_frontend.sh &"

