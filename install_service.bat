@echo off
REM Install FMCG AI Trainer as a Windows Service
REM This script must be run as Administrator

REM Keep window open and show all output
setlocal enabledelayedexpansion

echo ========================================
echo FMCG AI Trainer Service Installer
echo ========================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script must be run as Administrator!
    echo Right-click and select "Run as administrator"
    pause
    exit /b 1
)

REM Get the script directory
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Check for virtual environment first
set "PYTHON_EXE=python"
set VENV_CREATED=0

if exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
    set "PYTHON_EXE=%SCRIPT_DIR%venv\Scripts\python.exe"
    echo Found virtual environment, using venv Python
    echo.
) else if exist "%SCRIPT_DIR%venv\bin\python.exe" (
    set "PYTHON_EXE=%SCRIPT_DIR%venv\bin\python.exe"
    echo Found virtual environment, using venv Python
    echo.
) else (
    REM Check if system Python is available
    where python >nul 2>&1
    if %errorLevel% neq 0 (
        echo ERROR: Python not found in PATH
        echo Please ensure Python is installed and added to PATH
        pause
        exit /b 1
    )
    
    REM Virtual environment doesn't exist, create it
    echo Virtual environment not found.
    echo Creating virtual environment...
    echo.
    python -m venv "%SCRIPT_DIR%venv"
    if %errorLevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        echo Please ensure Python venv module is available.
        pause
        exit /b 1
    )
    
    REM Use the newly created virtual environment
    if exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
        set "PYTHON_EXE=%SCRIPT_DIR%venv\Scripts\python.exe"
        set VENV_CREATED=1
        echo Virtual environment created successfully!
        echo Using virtual environment Python
        echo.
    ) else (
        echo ERROR: Virtual environment was created but python.exe not found.
        pause
        exit /b 1
    )
)

REM If we just created a new virtual environment, upgrade pip first
if %VENV_CREATED% equ 1 (
    echo Upgrading pip in virtual environment...
    "%PYTHON_EXE%" -m pip install --upgrade pip >nul 2>&1
    echo.
)

REM Check if required dependencies are installed
echo Checking for required dependencies...
"%PYTHON_EXE%" -c "import uvicorn" 2>nul
set UVICORN_CHECK=%errorlevel%
if %UVICORN_CHECK% neq 0 (
    echo.
    echo Required dependencies are not installed.
    echo Installing dependencies from requirements.txt...
    echo.
    if exist "%SCRIPT_DIR%requirements.txt" (
        "%PYTHON_EXE%" -m pip install -r "%SCRIPT_DIR%requirements.txt"
        if errorlevel 1 (
            echo.
            echo ERROR: Failed to install dependencies.
            echo Please install manually: "%PYTHON_EXE%" -m pip install -r requirements.txt
            pause
            exit /b 1
        )
        echo.
        echo Dependencies installed successfully!
        echo.
    ) else (
        echo ERROR: requirements.txt not found in %SCRIPT_DIR%
        echo Please ensure you're running this from the project root directory.
        pause
        exit /b 1
    )
) else (
    echo Required dependencies are installed.
    echo.
)

echo Checking for pywin32...
REM Try to import pywin32 - if it fails, we'll install it
"%PYTHON_EXE%" -c "import win32serviceutil" 2>nul
set PYWIN32_CHECK=%errorlevel%
if %PYWIN32_CHECK% neq 0 (
    echo pywin32 is not installed. Installing...
    echo.
    REM Install pywin32 (show output for user feedback)
    "%PYTHON_EXE%" -m pip install pywin32
    if errorlevel 1 (
        echo.
        echo WARNING: pip install returned an error, but continuing...
        echo.
    )
    echo.
    REM Try to run post-install script (may not exist in all pywin32 versions)
    echo Running pywin32 post-install script (if available)...
    "%PYTHON_EXE%" -m pywin32_postinstall -install 2>nul
    if not errorlevel 1 (
        echo pywin32 post-install completed.
    ) else (
        echo pywin32_postinstall module not found (this is OK for some pywin32 versions).
    )
    echo.
    REM Now verify installation by trying to import
    echo Verifying pywin32 installation...
    "%PYTHON_EXE%" -c "import win32serviceutil" 2>nul
    set PYWIN32_VERIFY=%errorlevel%
    if %PYWIN32_VERIFY% neq 0 (
        echo.
        echo ERROR: pywin32 installation verification failed.
        echo The package was installed but cannot be imported.
        echo.
        echo Please try the following steps manually:
        echo   1. "%PYTHON_EXE%" -m pip install pywin32
        echo   2. "%PYTHON_EXE%" -m pywin32_postinstall -install
        echo   3. Verify: "%PYTHON_EXE%" -c "import win32serviceutil"
        echo.
        pause
        exit /b 1
    )
    echo pywin32 installed and verified successfully!
    echo.
) else (
    echo pywin32 is already installed.
    echo.
)

echo.
echo ========================================
echo Installing Windows Service...
echo ========================================
echo Python executable: %PYTHON_EXE%
echo Script: %SCRIPT_DIR%windows_service.py
echo.
echo Please wait, this may take a moment...
echo.

REM Install the service
"%PYTHON_EXE%" windows_service.py install

if not errorlevel 1 (
    echo.
    echo ========================================
    echo Service installed successfully!
    echo ========================================
    echo.
    echo Service Name: FMCGAITrainer
    echo Display Name: FMCG AI Trainer API Service
    echo.
    echo Next steps:
    echo   1. Start the service: net start FMCGAITrainer
    echo   2. Or use Services Manager (services.msc)
    echo   3. Test the API: http://localhost:8003/docs
    echo.
    echo To verify installation, run:
    echo   sc query FMCGAITrainer
    echo.
) else (
    echo.
    echo ========================================
    echo Service installation FAILED!
    echo ========================================
    echo.
    echo Error Code: %errorLevel%
    echo.
    echo Common issues:
    echo   1. Not running as Administrator - Right-click and select "Run as administrator"
    echo   2. pywin32 not installed - Check if it was installed correctly
    echo   3. Dependencies missing - Run: "%PYTHON_EXE%" -m pip install -r requirements.txt
    echo   4. Python path issues - Make sure virtual environment is set up correctly
    echo.
    echo To troubleshoot:
    echo   1. Check Windows Event Viewer for detailed errors
    echo   2. Try installing manually:
    echo      cd %SCRIPT_DIR%
    echo      "%PYTHON_EXE%" windows_service.py install
    echo.
)


echo.
echo ========================================
echo Installation process completed.
echo ========================================
echo.
echo IMPORTANT: Please review the output above for any errors.
echo.
echo If installation was successful, you can now:
echo   1. Verify: sc query FMCGAITrainer
echo   2. Start: net start FMCGAITrainer
echo.
echo Press any key to close this window...
pause >nul

