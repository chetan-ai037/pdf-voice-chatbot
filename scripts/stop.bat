@echo off
echo ===============================
echo Stopping and Cleaning Project
echo ===============================

:: Move to project root
cd /d %~dp0\..

:: Stop Python processes (best-effort)
taskkill /F /IM python.exe >nul 2>&1

:: Remove virtual environment
if exist venv (
    echo Removing virtual environment...
    rmdir /s /q venv
)

:: Remove Python cache
for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"

echo Project cleaned successfully.
pause
