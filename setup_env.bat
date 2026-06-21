@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   Hearing Assistant - Advanced Environment Setup
echo ===================================================

cd /d "%~dp0"

:: Set local pip cache directory to save space on C: drive
set "PIP_CACHE_DIR=%CD%\.pip_cache"
echo Project working directory: %CD%
echo Local pip cache directory: %PIP_CACHE_DIR%
echo.

:: ── 1. PYTHON VERSION VALIDATION ─────────────────────
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in system PATH.
    echo Please install Python 3.10 or 3.11 from python.org before running this script.
    exit /b 1
)

:: Retrieve major.minor version
for /f "tokens=2 delims= " %%i in ('python --version') do (
    set "PY_VER=%%i"
)
for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)

echo Detected Python: !PY_MAJOR!.!PY_MINOR! (!PY_VER!)

if not "!PY_MAJOR!"=="3" (
    echo [ERROR] Python 3 is required.
    exit /b 1
)

if "!PY_MINOR!"=="10" (
    echo [OK] Python version 3.10 is fully compatible.
) else if "!PY_MINOR!"=="11" (
    echo [OK] Python version 3.11 is fully compatible.
) else (
    echo [WARNING] Detected Python version is !PY_MAJOR!.!PY_MINOR!.
    echo We strongly recommend Python 3.10 or 3.11 for deep learning compatibility (PyTorch, MediaPipe).
    set /p "proceed=Do you want to proceed anyway? (Y/N): "
    if /i not "!proceed!"=="Y" (
        echo Installation aborted.
        exit /b 1
    )
)
echo.

:: ── 2. CREATE VIRTUAL ENVIRONMENT ────────────────────
echo [2/4] Initializing clean virtual environment 'venv'...
if exist "venv" (
    echo Virtual environment 'venv' already exists.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
    echo [OK] Virtual environment created successfully.
)

:: Activate the venv
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated.
echo.

:: Upgrade base packaging packages
echo Upgrading pip, setuptools, and wheel to prevent build compilation issues...
python -m pip install --upgrade pip setuptools wheel
echo.

:: ── 3. DETECT CUDA AND INSTALL PYTORCH ───────────────
echo [3/4] Checking for CUDA capability...
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo No NVIDIA GPU tools found. Installing PyTorch CPU version...
    pip install torch torchvision torchaudio
) else (
    echo NVIDIA GPU detected via nvidia-smi. Installing CUDA-optimized PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)
echo.

:: Install rest of requirements
echo Installing remaining modular dependencies from requirements.txt...
pip install -r requirements.txt
echo.

:: ── 4. GENERATE .ENV TEMPLATE & CONFIG ───────────────
echo [4/4] Configuring environment variables (.env)...
if not exist ".env" (
    echo Creating default .env configuration file...
    echo # Hearing Assistant API Keys > .env
    echo GROQ_API_KEY=your_groq_api_key_here >> .env
    echo GEMINI_API_KEY=your_gemini_api_key_here >> .env
    echo [OK] Created .env template.
    
    set /p "setup_keys=Do you want to enter your API keys now? (Y/N): "
    if /i "!setup_keys!"=="Y" (
        set /p "groq_key=Enter your GROQ_API_KEY: "
        set /p "gemini_key=Enter your GEMINI_API_KEY: "
        
        echo # Hearing Assistant API Keys > .env
        echo GROQ_API_KEY=!groq_key! >> .env
        echo GEMINI_API_KEY=!gemini_key! >> .env
        echo [OK] Configuration saved to .env.
    ) else (
        echo Please remember to edit .env and replace placeholder keys before launching.
    )
) else (
    echo .env configuration file already exists. Skipping template creation.
)
echo.

echo ===================================================
echo   Setup Complete!
echo ===================================================
echo To run your backend:
echo   venv\Scripts\activate.bat
echo   uvicorn main:app --reload
echo.
echo To run your frontend:
echo   venv\Scripts\activate.bat
echo   streamlit run ui/streamlit_app.py
echo ===================================================
pause
