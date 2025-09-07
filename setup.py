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
        # Verify the virtual environment is working
        if verify_virtual_environment():
            return True
        else:
            print("⚠️  Virtual environment exists but appears corrupted. Recreating...")
            import shutil
            try:
                shutil.rmtree(venv_path)
                print("🗑️  Removed corrupted virtual environment")
            except Exception as e:
                print(f"⚠️  Could not remove corrupted venv: {e}")
                print("💡 Please manually delete the 'venv' folder and run setup again")
                return False
    
    print("🐍 Creating virtual environment...")
    # Use system Python, not the potentially corrupted venv Python
    return run_command(f"{sys.executable} -m venv venv", "Virtual environment creation")

def verify_virtual_environment():
    """Verify that the virtual environment is working properly."""
    try:
        # Check if pip exists and works
        if os.name == 'nt':  # Windows
            pip_path = "venv\\Scripts\\pip"
        else:  # Unix-like systems
            pip_path = "venv/bin/pip"
        
        result = subprocess.run(f"{pip_path} --version", shell=True, capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

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
    
    # Check if pip exists before trying to use it
    if not Path(pip_path).exists():
        print(f"❌ pip not found at {pip_path}. Virtual environment may be corrupted.")
        return False
    
    return run_command(f"{pip_path} install -r requirements.txt", "Requirements installation")

def create_or_update_env_file():
    """Create .env file with placeholder values or update existing one."""
    env_path = Path(".env")
    
    env_content = """# Agent Travel Planner Environment Variables
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
"""
    
    if env_path.exists():
        # Check if the file needs updating
        try:
            with open(".env", "r") as f:
                existing_content = f.read()
            
            # Check if OPENAI_API_KEY is missing
            if "OPENAI_API_KEY" not in existing_content:
                print("📄 .env file exists but missing OPENAI_API_KEY. Updating...")
                try:
                    with open(".env", "w") as f:
                        f.write(env_content)
                    print("✅ Updated .env file with new template")
                    print("📝 Please edit .env file and add your actual API keys")
                    return True
                except Exception as e:
                    print(f"❌ Failed to update .env file: {e}")
                    return False
            else:
                print("📄 .env file already exists and up to date")
                return True
        except Exception as e:
            print(f"⚠️  Could not read existing .env file: {e}")
            print("📄 Creating new .env file...")
    else:
        print("📄 Creating new .env file...")
    
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
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required. Please upgrade Python.")
        sys.exit(1)
    
    # Check if virtual environment exists
    venv_exists = Path("venv").exists()
    
    if venv_exists:
        print("📁 Virtual environment found. Running update mode...")
        print("=" * 50)
        
        # Just update what's needed
        success = True
        
        # Verify virtual environment is working
        if not verify_virtual_environment():
            print("⚠️  Virtual environment appears corrupted. Recreating...")
            if not create_virtual_environment():
                success = False
        
        # Install/update requirements
        if success and not install_requirements():
            success = False
        
        # Create or update .env file
        if success and not create_or_update_env_file():
            success = False
        
        print("=" * 50)
        if success:
            print("🎉 Update completed successfully!")
            print("\n📋 Next steps:")
            print("1. Edit .env file and add your GEMINI_API_KEY and OPENAI_API_KEY")
            print("2. Activate the virtual environment:")
            if os.name == 'nt':  # Windows
                print("   venv\\Scripts\\activate")
            else:  # Unix-like systems
                print("   source venv/bin/activate")
            print("3. Run the application:")
            print("   python main.py")
        else:
            print("❌ Update completed with errors. Please check the messages above.")
            sys.exit(1)
    else:
        print("📁 No virtual environment found. Running full setup...")
        print("=" * 50)
        
        success = True
        
        # Create virtual environment
        if not create_virtual_environment():
            success = False
        
        # Install requirements
        if success and not install_requirements():
            success = False
        
        # Create or update .env file
        if success and not create_or_update_env_file():
            success = False
        
        print("=" * 50)
        if success:
            print("🎉 Setup completed successfully!")
            print("\n📋 Next steps:")
            print("1. Edit .env file and add your GEMINI_API_KEY and OPENAI_API_KEY")
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
