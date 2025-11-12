@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PYTHON_EXE=%SCRIPT_DIR%venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo ERROR: Virtual environment python.exe not found at "%PYTHON_EXE%"
    echo Please create the venv by running: python -m venv venv
    exit /b 1
)

set "TRAINER_CLI=%SCRIPT_DIR%trainer_cli.py"
if not exist "%TRAINER_CLI%" (
    echo ERROR: trainer_cli.py not found at "%TRAINER_CLI%"
    exit /b 1
)

"%PYTHON_EXE%" "%TRAINER_CLI%" --tenant PRODUCTION --type similar_customers

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo Training script failed with exit code %EXIT_CODE%.
)

exit /b %EXIT_CODE%

