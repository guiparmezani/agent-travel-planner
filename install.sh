#!/bin/bash

# Agent Travel Planner Installation Script
# This script sets up the application environment

set -e  # Exit on any error

echo "🚀 Setting up Agent Travel Planner..."
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found. Please run this script from the project root directory."
    exit 1
fi

# Function to run commands with error handling
run_command() {
    local description="$1"
    local command="$2"
    
    echo "🔄 $description..."
    if eval "$command"; then
        echo "✅ $description completed successfully"
    else
        echo "❌ $description failed"
        exit 1
    fi
}

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    run_command "Creating virtual environment" "python3 -m venv venv"
else
    echo "📁 Virtual environment already exists"
fi

# Activate virtual environment and install requirements
if [ -f "requirements.txt" ]; then
    run_command "Installing requirements" "source venv/bin/activate && pip install -r requirements.txt"
else
    echo "❌ requirements.txt not found"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# Agent Travel Planner Environment Variables
# Copy this file and fill in your actual values

# Google Gemini API Key
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI API Key (fallback for Gemini)
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Optional: LangSmith configuration for tracing
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_langsmith_api_key_here
# LANGCHAIN_PROJECT=agent-travel-planner
EOF
    echo "✅ Created .env file with placeholder values"
    echo "📝 Please edit .env file and add your actual API keys"
else
    echo "📄 .env file already exists"
fi

echo "=================================================="
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file and add your GEMINI_API_KEY and OPENAI_API_KEY"
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo "3. Run the application:"
echo "   python main.py"
