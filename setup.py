#!/usr/bin/env python3
"""
Setup script for Agent Travel Planner
This script helps users set up the application environment.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_virtual_environment():
    """Create a virtual environment if it doesn't exist."""
    venv_path = Path("venv")
    if venv_path.exists():
        print("📁 Virtual environment already exists")
        return True
    
    print("🐍 Creating virtual environment...")
    return run_command(f"{sys.executable} -m venv venv", "Virtual environment creation")

def install_requirements():
    """Install requirements from requirements.txt."""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    # Determine the correct pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = "venv\\Scripts\\pip"
    else:  # Unix-like systems
        pip_path = "venv/bin/pip"
    
    return run_command(f"{pip_path} install -r requirements.txt", "Requirements installation")

def create_env_file():
    """Create .env file with placeholder values if it doesn't exist."""
    env_path = Path(".env")
    if env_path.exists():
        print("📄 .env file already exists")
        return True
    
    env_content = """# Agent Travel Planner Environment Variables
# Copy this file and fill in your actual values

# Google Gemini API Key
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: LangSmith configuration for tracing
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_langsmith_api_key_here
# LANGCHAIN_PROJECT=agent-travel-planner
"""
    
    try:
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env file with placeholder values")
        print("📝 Please edit .env file and add your actual API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Agent Travel Planner...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ main.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    success = True
    
    # Create virtual environment
    if not create_virtual_environment():
        success = False
    
    # Install requirements
    if not install_requirements():
        success = False
    
    # Create .env file
    if not create_env_file():
        success = False
    
    print("=" * 50)
    if success:
        print("🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Edit .env file and add your GEMINI_API_KEY")
        print("2. Activate the virtual environment:")
        if os.name == 'nt':  # Windows
            print("   venv\\Scripts\\activate")
        else:  # Unix-like systems
            print("   source venv/bin/activate")
        print("3. Run the application:")
        print("   python main.py")
    else:
        print("❌ Setup completed with errors. Please check the messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
