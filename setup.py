#!/usr/bin/env python3
"""
Setup script for Enhanced Math Quest LaTeX RAG Pipeline
Installs dependencies and configures the environment for 90%+ accuracy
"""

import os
import sys
import subprocess
import platform

def run_command(command, check=True):
    """Run a shell command and handle errors"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor} detected")
    return True

def install_python_dependencies():
    """Install Python backend dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists("venv"):
        if not run_command(f"{sys.executable} -m venv venv"):
            return False
    
    # Determine the correct pip path based on OS
    if platform.system() == "Windows":
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    # Upgrade pip
    run_command(f"{pip_cmd} install --upgrade pip")
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r server/requirements.txt"):
        return False
    
    print("✅ Python dependencies installed successfully")
    return True

def install_node_dependencies():
    """Install Node.js frontend dependencies"""
    print("\n📦 Installing Node.js dependencies...")
    
    if not run_command("npm --version", check=False):
        print("❌ Node.js/npm not found. Please install Node.js first.")
        return False
    
    if not run_command("npm install"):
        return False
    
    print("✅ Node.js dependencies installed successfully")
    return True

def create_env_file():
    """Create environment configuration file"""
    print("\n⚙️  Creating environment configuration...")
    
    env_content = """# Enhanced Math Quest LaTeX Configuration
# OpenAI API Key (required for production)
OPENAI_API_KEY=your_openai_api_key_here

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True

# Logging
LOG_LEVEL=INFO

# PDF Processing
MAX_PDF_SIZE_MB=100
CHUNK_SIZE=3000
CONFIDENCE_THRESHOLD=0.6

# Performance Settings
MAX_PAGES_TO_PROCESS=10
ENABLE_CACHING=True
"""
    
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env file (please add your OpenAI API key)")
    else:
        print("✅ .env file already exists")
    
    return True

def create_startup_scripts():
    """Create convenient startup scripts"""
    print("\n📝 Creating startup scripts...")
    
    # Windows batch file
    batch_content = """@echo off
echo Starting Enhanced Math Quest LaTeX Pipeline...
echo.

echo Starting Python Backend...
start "Backend" cmd /c "venv\\Scripts\\python server\\app.py"

echo Waiting for backend to initialize...
timeout /t 5 /nobreak > nul

echo Starting Frontend...
start "Frontend" cmd /c "npm run dev"

echo.
echo Both servers started! Access the app at: http://localhost:5173
echo Backend API available at: http://localhost:5000
echo.
pause
"""
    
    # Unix shell script
    shell_content = """#!/bin/bash
echo "Starting Enhanced Math Quest LaTeX Pipeline..."
echo

echo "Starting Python Backend..."
source venv/bin/activate
python server/app.py &
BACKEND_PID=$!

echo "Waiting for backend to initialize..."
sleep 5

echo "Starting Frontend..."
npm run dev &
FRONTEND_PID=$!

echo
echo "Both servers started! Access the app at: http://localhost:5173"
echo "Backend API available at: http://localhost:5000"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo

# Trap to kill both processes on script exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT

wait
"""
    
    with open("start.bat", "w") as f:
        f.write(batch_content)
    
    with open("start.sh", "w") as f:
        f.write(shell_content)
    
    # Make shell script executable on Unix systems
    if platform.system() != "Windows":
        run_command("chmod +x start.sh", check=False)
    
    print("✅ Startup scripts created (start.bat for Windows, start.sh for Unix)")
    return True

def verify_installation():
    """Verify that everything is installed correctly"""
    print("\n🔍 Verifying installation...")
    
    # Check Python packages
    venv_python = "venv\\Scripts\\python" if platform.system() == "Windows" else "venv/bin/python"
    
    packages_to_check = ["flask", "openai", "PyMuPDF", "scikit-learn", "numpy"]
    for package in packages_to_check:
        if not run_command(f"{venv_python} -c \"import {package.replace('-', '_').lower()}\"", check=False):
            print(f"❌ Package {package} not properly installed")
            return False
    
    print("✅ All Python packages verified")
    
    # Check Node.js packages
    if not os.path.exists("node_modules"):
        print("❌ Node modules not found")
        return False
    
    print("✅ Node.js packages verified")
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Add your OpenAI API key to the .env file")
    print("   - Open .env in a text editor")
    print("   - Replace 'your_openai_api_key_here' with your actual API key")
    print("   - Get an API key from: https://platform.openai.com/api-keys")
    print()
    print("2. Start the application:")
    if platform.system() == "Windows":
        print("   - Double-click start.bat, or")
        print("   - Run: start.bat")
    else:
        print("   - Run: ./start.sh")
    print()
    print("3. Access the application:")
    print("   - Frontend: http://localhost:5173")
    print("   - Backend API: http://localhost:5000")
    print()
    print("4. Test with a chapter (e.g., '30.3' or 'Integration')")
    print()
    print("🎯 Target: 90%+ accuracy with enhanced RAG pipeline!")
    print("="*60)

def main():
    """Main setup function"""
    print("🚀 Enhanced Math Quest LaTeX Setup")
    print("Setting up the RAG pipeline for 90%+ accuracy")
    print("=" * 50)
    
    if not check_python_version():
        sys.exit(1)
    
    success = True
    success &= install_python_dependencies()
    success &= install_node_dependencies()
    success &= create_env_file()
    success &= create_startup_scripts()
    success &= verify_installation()
    
    if success:
        print_next_steps()
    else:
        print("\n❌ Setup failed. Please check the errors above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
