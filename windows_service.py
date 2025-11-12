#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows Service wrapper for FMCG AI Trainer API.

This script allows the FastAPI application to run as a Windows service.
"""

import os
import sys
import time
import logging
import threading
from pathlib import Path
from copy import deepcopy

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Ensure the virtual environment site-packages (including PyWin32 modules) are available
venv_dir = project_root / "venv"
site_packages_paths = [
    venv_dir / "Lib" / "site-packages",
    venv_dir / "Lib" / "site-packages" / "win32",
    venv_dir / "Lib" / "site-packages" / "win32" / "lib",
    venv_dir / "Lib" / "site-packages" / "Pythonwin",
]

for path in site_packages_paths:
    if path.exists():
        str_path = str(path)
        if str_path not in sys.path:
            sys.path.insert(0, str_path)

# Ensure PyWin32 DLL directory is available on PATH
pywin32_system32 = venv_dir / "Lib" / "site-packages" / "pywin32_system32"
if pywin32_system32.exists():
    os.environ["PATH"] = f"{str(pywin32_system32)};{os.environ.get('PATH', '')}"

# Debug: record sys.path for troubleshooting service startup issues
try:
    debug_path_file = project_root / "service_sys_path.txt"
    with open(debug_path_file, "w") as f:
        f.write("Sys.path at service import time:\n")
        for entry in sys.path:
            f.write(f"{entry}\n")
        f.write("\nPATH environment variable:\n")
        f.write(os.environ.get("PATH", ""))
except Exception:
    pass

try:
    import win32serviceutil
    import win32service
    import win32event
    import servicemanager
except ImportError:
    if len(sys.argv) > 1 and sys.argv[1] in ['install', 'remove', 'start', 'stop', 'restart']:
        print("ERROR: pywin32 is required for Windows service support.")
        print("Install it with: pip install pywin32")
        print("After installation, you may need to run: python -m pywin32_postinstall -install")
        sys.exit(1)
    else:
        # If running as service, we need these imports
        raise

import uvicorn
from uvicorn.config import LOGGING_CONFIG as UVICORN_DEFAULT_LOGGING_CONFIG

# Disable colorized logging to avoid sys.stdout.isatty() calls when running as a service
UVICORN_LOGGING_CONFIG = deepcopy(UVICORN_DEFAULT_LOGGING_CONFIG)
try:
    UVICORN_LOGGING_CONFIG["formatters"]["default"]["use_colors"] = False
    if "access" in UVICORN_LOGGING_CONFIG.get("formatters", {}):
        UVICORN_LOGGING_CONFIG["formatters"]["access"]["use_colors"] = False
except Exception:
    # If the structure changes unexpectedly, fall back to original config
    UVICORN_LOGGING_CONFIG = UVICORN_DEFAULT_LOGGING_CONFIG


class FMCGAITrainerService(win32serviceutil.ServiceFramework):
    """Windows Service class for FMCG AI Trainer API."""
    
    _svc_name_ = "FMCGAITrainer"
    _svc_display_name_ = "FMCG AI Trainer API Service"
    _svc_description_ = "AI Training API for FMCG recommendations and similar customer analysis"
    _svc_reg_class_ = "windows_service.FMCGAITrainerService"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.server_thread = None
        self.server = None
        
        # Setup logging early - use a simple file write first in case logging fails
        log_dir = project_root / "logs"
        try:
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "service.log"
            
            # Write initial startup message directly to file
            with open(log_file, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Service starting at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Python: {sys.executable}\n")
                f.write(f"Working dir: {os.getcwd()}\n")
                f.write(f"Project root: {project_root}\n")
                f.write(f"{'='*60}\n")
        except Exception as e:
            # If we can't even write to log, write to a temp file
            try:
                with open(project_root / "service_error.txt", 'w') as f:
                    f.write(f"Failed to setup logging: {e}\n")
            except:
                pass
        
        # Now setup proper logging
        try:
            log_file = log_dir / "service.log"
            logging.basicConfig(
                filename=str(log_file),
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self.logger = logging.getLogger(__name__)
            self.logger.info("Service logger initialized")
        except Exception as e:
            # Fallback: create a basic logger
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())
            try:
                with open(project_root / "service_error.txt", 'a') as f:
                    f.write(f"Logging setup error: {e}\n")
            except:
                pass
    
    def SvcStop(self):
        """Stop the service."""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        self.logger.info("Stopping FMCG AI Trainer service...")
        
        # Stop the uvicorn server
        if self.server:
            try:
                self.server.should_exit = True
            except Exception as e:
                self.logger.error(f"Error stopping server: {e}")
        
        # Signal the stop event
        win32event.SetEvent(self.stop_event)
        self.logger.info("Service stopped")
    
    def SvcDoRun(self):
        """Run the service."""
        error_file = project_root / "service_startup_error.txt"
        
        try:
            # Write to error file immediately
            with open(error_file, 'w') as f:
                f.write("Service starting...\n")
                f.write(f"Python: {sys.executable}\n")
                f.write(f"Initial dir: {os.getcwd()}\n")
            
            # Report that service is starting
            self.ReportServiceStatus(win32service.SERVICE_START_PENDING)
            
            servicemanager.LogMsg(
                servicemanager.EVENTLOG_INFORMATION_TYPE,
                servicemanager.PYS_SERVICE_STARTED,
                (self._svc_name_, '')
            )
            
            with open(error_file, 'a') as f:
                f.write("Service status reported\n")
            
            self.logger.info(f"Starting {self._svc_display_name_}...")
            self.logger.info(f"Python executable: {sys.executable}")
            self.logger.info(f"Python path: {sys.path}")
            
            # Change to project directory
            os.chdir(str(project_root))
            self.logger.info(f"Changed to directory: {os.getcwd()}")
            
            with open(error_file, 'a') as f:
                f.write(f"Changed to: {os.getcwd()}\n")
            
            # Add project root to Python path explicitly
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
                self.logger.info(f"Added {project_root} to Python path")
            
            with open(error_file, 'a') as f:
                f.write("About to import app.main\n")
            
            # Test import before creating uvicorn config
            with open(error_file, 'a') as f:
                f.write("Testing app.main import...\n")
            
            try:
                from app.main import app
                self.logger.info("Successfully imported app.main")
                with open(error_file, 'a') as f:
                    f.write("app.main imported successfully\n")
            except Exception as import_error:
                error_msg = f"Failed to import app.main: {import_error}"
                self.logger.error(error_msg, exc_info=True)
                with open(error_file, 'a') as f:
                    f.write(f"IMPORT ERROR: {error_msg}\n")
                    import traceback
                    f.write(traceback.format_exc())
                raise
            
            # Get configuration
            host = os.getenv("HOST", "0.0.0.0")
            port = int(os.getenv("PORT", "8003"))
            
            self.logger.info(f"Starting server on {host}:{port}")
            
            with open(error_file, 'a') as f:
                f.write(f"Creating uvicorn config for {host}:{port}\n")
            
            # Create uvicorn config
            config = uvicorn.Config(
                "app.main:app",
                host=host,
                port=port,
                log_level="info",
                access_log=True,
                reload=False,  # Disable reload in service mode
                log_config=UVICORN_LOGGING_CONFIG,
            )
            
            with open(error_file, 'a') as f:
                f.write("Uvicorn config created\n")
            
            # Create and start server in a separate thread
            self.server = uvicorn.Server(config)
            
            # Run server in a thread
            self.server_thread = threading.Thread(target=self.server.run, daemon=True)
            self.server_thread.start()
            
            # Give server a moment to start
            time.sleep(2)
            
            # Report that service is running
            self.ReportServiceStatus(win32service.SERVICE_RUNNING)
            
            self.logger.info(f"{self._svc_display_name_} started successfully on port {port}")
            
            # Wait for stop event
            win32event.WaitForSingleObject(self.stop_event, win32event.INFINITE)
            
        except Exception as e:
            error_msg = f"Service error: {e}"
            # Write to error file
            try:
                with open(error_file, 'a') as f:
                    f.write(f"\nFATAL ERROR: {error_msg}\n")
                    import traceback
                    f.write(traceback.format_exc())
            except:
                pass
            
            # Try to log
            try:
                self.logger.error(error_msg, exc_info=True)
            except:
                pass
            
            servicemanager.LogErrorMsg(error_msg)
            # Report service stopped due to error
            try:
                self.ReportServiceStatus(win32service.SERVICE_STOPPED)
            except:
                pass
            raise


def main():
    """Main entry point for service installation/removal."""
    # Check for administrator privileges when installing/removing
    if len(sys.argv) > 1 and sys.argv[1] in ['install', 'remove']:
        try:
            import ctypes
            is_admin = ctypes.windll.shell32.IsUserAnAdmin()
            if not is_admin:
                print("ERROR: Administrator privileges are required to install/remove the service.")
                print("Please run this command as Administrator:")
                print("  1. Right-click Command Prompt")
                print("  2. Select 'Run as administrator'")
                print("  3. Navigate to the project directory")
                print("  4. Run: python windows_service.py install")
                sys.exit(1)
        except ImportError:
            # If ctypes is not available, continue anyway
            pass
    
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(FMCGAITrainerService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        try:
            # Use the standard HandleCommandLine which properly handles installation
            # It will use the current Python executable (from venv if activated)
            win32serviceutil.HandleCommandLine(FMCGAITrainerService)
        except Exception as e:
            if "Access is denied" in str(e) or "5" in str(e):
                print("\nERROR: Access denied. Administrator privileges are required.")
                print("\nTo fix this:")
                print("1. Close this command prompt")
                print("2. Right-click Command Prompt and select 'Run as administrator'")
                print("3. Navigate to: C:\\apps\\fmcg-ai-trainer-aidin")
                print("4. Activate venv: venv\\Scripts\\activate")
                print("5. Run: python windows_service.py install")
                sys.exit(1)
            else:
                raise


if __name__ == "__main__":
    main()

