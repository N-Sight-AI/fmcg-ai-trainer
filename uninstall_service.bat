@echo off
REM Uninstall FMCG AI Trainer Windows Service
REM This script must be run as Administrator

echo ========================================
echo FMCG AI Trainer Service Uninstaller
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

REM Get the Python executable path
set PYTHON_EXE=python
where python >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Python not found in PATH
    echo Please ensure Python is installed and added to PATH
    pause
    exit /b 1
)

REM Get the script directory
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo Stopping service (if running)...
net stop FMCGAITrainer >nul 2>&1

echo Uninstalling service...
echo.

REM Uninstall the service
%PYTHON_EXE% windows_service.py remove

if %errorLevel% equ 0 (
    echo.
    echo ========================================
    echo Service uninstalled successfully!
    echo ========================================
    echo.
) else (
    echo.
    echo ========================================
    echo Service uninstallation failed!
    echo ========================================
    echo.
    echo Make sure:
    echo 1. You are running as Administrator
    echo 2. The service is stopped
    echo.
)

pause

