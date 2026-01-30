@echo off
REM ============================================================================
REM CNN Image Classification Setup Script for Windows
REM ============================================================================

echo.
echo ============================================================================
echo CNN Image Classification - Setup Script
echo ============================================================================
echo.

REM Check if Visual C++ Redistributable is needed
echo [1/4] Checking for Microsoft Visual C++ Redistributable...
echo.
echo NOTE: PyTorch requires Microsoft Visual C++ Redistributable
echo To fix the DLL error, please download and install from:
echo https://aka.ms/vs/17/release/vc_redist.x64.exe
echo.
echo Press any key after installing Visual C++ Redistributable...
pause

REM Create virtual environment if not exists
if not exist "env" (
    echo [2/4] Creating Python virtual environment...
    python -m venv env
) else (
    echo [2/4] Virtual environment already exists, skipping...
)

REM Activate virtual environment
echo [3/4] Activating virtual environment...
call env\Scripts\activate.bat

REM Install requirements
echo [4/4] Installing Python packages...
echo Upgrading pip and installing pinned PyTorch wheels (this may take a few minutes)...
pip install --upgrade pip && pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt

echo.
echo ============================================================================
echo Setup Complete!
echo ============================================================================
echo.
echo To run the training:
echo   1. Open PowerShell or Command Prompt in this directory
echo   2. Run: .\env\Scripts\activate.ps1  (PowerShell)
echo      or: env\Scripts\activate.bat     (Command Prompt)
echo   3. Run: python train.py
echo.
pause
