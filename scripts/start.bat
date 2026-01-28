@echo off
echo ===============================
echo Project Setup and Start
echo ===============================

:: Move to project root
cd /d %~dp0\..

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed.
    echo Install Python 3.9+ and try again.
    pause
    exit /b
)

:: Create virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Run the project
echo Starting application...
python app.py

pause