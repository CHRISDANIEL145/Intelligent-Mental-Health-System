@echo off
title DC Well Being - AI System Launcher
echo ===================================================
echo     STARTING DC WELL BEING AI SYSTEM
echo ===================================================

:: 1. check for python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b
)

:: 2. Start Ollama (in new window if possible, or background)
echo [INFO] Checking Ollama...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [INFO] Ollama is already running.
) else (
    echo [INFO] Starting Ollama...
    start "Ollama Server" ollama serve
    timeout /t 5 /nobreak
)

:: 3. Start Flask App
tasklist /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq DC Well Being App*" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [INFO] App is already running.
    timeout /t 3
    exit
) else (
    echo [INFO] Starting Flask Application...
    echo [INFO] Access the app at: http://localhost:5000
    echo.
    title DC Well Being App
    python app.py
    pause
)
