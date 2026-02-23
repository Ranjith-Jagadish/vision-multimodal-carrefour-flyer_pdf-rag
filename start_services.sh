#!/bin/bash

# Start PDF RAG System Services
# This script starts both the FastAPI backend and Streamlit frontend

cd "$(dirname "$0")"
source venv/bin/activate

# Kill any existing processes on these ports
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:8501 | xargs kill -9 2>/dev/null

echo "🚀 Starting PDF RAG System..."
echo ""

# Start FastAPI backend
echo "📡 Starting FastAPI backend on http://localhost:8000"
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
sleep 3

# Check if backend started successfully
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend is running (PID: $BACKEND_PID)"
else
    echo "❌ Backend failed to start. Check /tmp/backend.log"
    exit 1
fi

# Start Streamlit frontend
echo "🎨 Starting Streamlit frontend on http://localhost:8501"
streamlit run frontend/streamlit_app.py --server.port 8501 --server.address localhost > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!
sleep 3

# Check if frontend started successfully
if curl -s --max-time 2 http://localhost:8501 > /dev/null; then
    echo "✅ Frontend is running (PID: $FRONTEND_PID)"
else
    echo "⚠️  Frontend may still be starting..."
fi

echo ""
echo "✨ Services started!"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:8501"
echo ""
echo "To stop services, run:"
echo "   kill $BACKEND_PID $FRONTEND_PID"
echo "   or: lsof -ti:8000,8501 | xargs kill -9"
