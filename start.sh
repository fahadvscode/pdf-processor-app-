#!/bin/bash

# PDF Processor - Web Interface Launcher

echo "🎨 Starting PDF Processor Web Interface..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not installed!"
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

echo "🚀 Launching web interface..."
echo "📍 The app will open in your browser at: http://localhost:8501"
echo ""
echo "💡 To stop: Press Ctrl+C"
echo ""
cd "$(dirname "$0")"
streamlit run app.py