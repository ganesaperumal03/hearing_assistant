#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "==================================================="
echo "  Hearing Assistant - Linux/macOS Environment Setup"
echo "==================================================="

# Resolve script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Redirect pip downloads cache locally
export PIP_CACHE_DIR="$SCRIPT_DIR/.pip_cache"
echo "Project working directory: $SCRIPT_DIR"
echo "Local pip cache directory: $PIP_CACHE_DIR"
echo

# ── 1. PYTHON VERSION VALIDATION ─────────────────────
echo "[1/4] Checking Python installation..."

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed or not in PATH."
    echo "Please install Python 3.10 or 3.11 before running this script."
    exit 1
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

echo "Detected Python: $PY_VER"

if [ "$PY_MAJOR" -ne 3 ]; then
    echo "[ERROR] Python 3 is required."
    exit 1
fi

if [ "$PY_MINOR" -eq 10 ] || [ "$PY_MINOR" -eq 11 ]; then
    echo "[OK] Python version $PY_VER is fully compatible."
else
    echo "[WARNING] Detected Python version is $PY_VER."
    echo "We strongly recommend Python 3.10 or 3.11 for deep learning compatibility (PyTorch, MediaPipe)."
    read -p "Do you want to proceed anyway? (y/n): " proceed
    if [[ ! "$proceed" =~ ^[Yy]$ ]]; then
        echo "Installation aborted."
        exit 1
    fi
fi
echo

# ── 2. CREATE VIRTUAL ENVIRONMENT ────────────────────
echo "[2/4] Initializing clean virtual environment 'venv'..."
if [ -d "venv" ]; then
    echo "Virtual environment 'venv' already exists."
else
    python3 -m venv venv
    echo "[OK] Virtual environment created successfully."
fi

# Activate venv
source venv/bin/activate
echo "[OK] Virtual environment activated."
echo

# Upgrade base packaging packages
echo "Upgrading pip, setuptools, and wheel..."
python3 -m pip install --upgrade pip setuptools wheel
echo

# ── 3. DETECT CUDA AND INSTALL PYTORCH ───────────────
echo "[3/4] Checking for CUDA capability..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing CUDA-optimized PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No NVIDIA GPU found. Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio
fi
echo

# Install other requirements
echo "Installing remaining modular dependencies from requirements.txt..."
pip install -r requirements.txt
echo

# ── 4. GENERATE .ENV TEMPLATE & CONFIG ───────────────
echo "[4/4] Configuring environment variables (.env)..."
if [ ! -f ".env" ]; then
    echo "Creating default .env configuration file..."
    echo "# Hearing Assistant API Keys" > .env
    echo "GROQ_API_KEY=your_groq_api_key_here" >> .env
    echo "GEMINI_API_KEY=your_gemini_api_key_here" >> .env
    echo "[OK] Created .env template."
    
    read -p "Do you want to enter your API keys now? (y/n): " setup_keys
    if [[ "$setup_keys" =~ ^[Yy]$ ]]; then
        read -p "Enter your GROQ_API_KEY: " groq_key
        read -p "Enter your GEMINI_API_KEY: " gemini_key
        
        echo "# Hearing Assistant API Keys" > .env
        echo "GROQ_API_KEY=$groq_key" >> .env
        echo "GEMINI_API_KEY=$gemini_key" >> .env
        echo "[OK] Configuration saved to .env."
    else
        echo "Please remember to edit .env and replace placeholder keys before launching."
    fi
else
    echo ".env configuration file already exists. Skipping template creation."
fi
echo

echo "==================================================="
echo "  Setup Complete!"
echo "==================================================="
echo "To run your backend:"
echo "  source venv/bin/activate"
echo "  uvicorn main:app --reload"
echo
echo "To run your frontend:"
echo "  source venv/bin/activate"
echo "  streamlit run ui/streamlit_app.py"
echo "==================================================="
