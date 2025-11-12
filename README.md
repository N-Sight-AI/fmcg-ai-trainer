# FMCG AI Trainer

AI Training API for FMCG recommendations and similar customer analysis.

## 🚀 Quick Start

### Prerequisites

- Python 3.11 (64-bit) or higher
- pip (Python package manager)
- SQL Server accessible from this machine
- **ODBC Driver for SQL Server** (required for database connections)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd fmcg-ai-trainer
   ```

2. **Install ODBC Driver for SQL Server:**
   
   The application requires an ODBC driver to connect to SQL Server. Install the appropriate driver for your operating system:
   
   **Windows:**
   - Download and install [ODBC Driver 17 for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server) or [ODBC Driver 18 for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)
   - The driver name in `config.json` should match the installed driver (e.g., "ODBC Driver 17 for SQL Server" or "ODBC Driver 18 for SQL Server")
   
   **macOS:**
   - Install using Homebrew: `brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release && brew install msodbcsql17`
   - Or download from [Microsoft's download page](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)
   
   **Linux:**
   - Follow the installation instructions for your distribution on [Microsoft's documentation](https://learn.microsoft.com/en-us/sql/connect/odbc/linux-mac/install-microsoft-odbc-driver-sql-server-macos)
   
   **Verify installation:**
   - On Windows: Check in "ODBC Data Source Administrator" (odbcad32.exe)
   - On macOS/Linux: Run `odbcinst -q -d` to list installed drivers

3. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Windows (Command Prompt):
   venv\Scripts\activate.bat
   ```

4. **Install dependencies inside the virtual environment:**
   ```bash
   pip install -r requirements.txt
   ```

## 📝 Configuration

Edit `config.json` to configure your SQL connection and training schedule:

```json
{
  "tenants": {
    "PRODUCTION": {
      "db_server": "your-sql-server.com",
      "db_name": "YourDatabase",
      "db_port": 1433,
      "db_driver": "ODBC Driver 17 for SQL Server",
      "db_user": "your_username",
      "db_password": "your_password",
      "db_trusted_connection": false,
      "db_encrypt": false,
      "db_trust_server_cert": false
    }
  }
}
```

Alternatively, you can set environment variables:

```bash
export TENANT_PRODUCTION_DB_SERVER=your-sql-server.com
export TENANT_PRODUCTION_DB_NAME=YourDatabase
export TENANT_PRODUCTION_DB_USER=your_username
export TENANT_PRODUCTION_DB_PASSWORD=your_password
export TENANT_PRODUCTION_DB_TRUSTED_CONNECTION=false
```

## 🏃 Running the Application

### Option 1: Run Training via CLI

Run training jobs directly from the command line:

```bash
# Run customer recommendation training
python trainer_cli.py --tenant PRODUCTION --type customer_order_recommendation_als

# Run similar customers training
python trainer_cli.py --tenant PRODUCTION --type similar_customers

# Dry run (validate configuration without executing)
python trainer_cli.py --tenant PRODUCTION --type customer_order_recommendation_als --dry-run
```

### Option 2: Run Training via Python Module

```bash
# Run training using the training CLI module
python -m app.train.training_cli --tenant PRODUCTION --training-type customer_order_recommendation_als

# List all available training types
python -m app.train.training_cli --list-types
```

### Option 3: Run FastAPI Server

Start the API server:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload
```

The API will be available at:
- **API**: http://localhost:8003
- **Swagger Docs**: http://localhost:8003/docs
- **ReDoc**: http://localhost:8003/redoc

### Option 4: Install as Windows Service (Windows Only)

To run the API as a Windows service that starts automatically:

1. **Install the service:**
   - Open an elevated Command Prompt (**Run as administrator**)
   - Navigate to the project directory and use the virtual environment interpreter:
     ```cmd
     cd C:\apps\fmcg-ai-trainer-aidin
     venv\Scripts\python.exe windows_service.py install
     ```
   - Alternatively, run the remaining helper script `install_service.bat` as Administrator. It ensures a virtual environment exists, installs dependencies (including `pywin32`), and calls the same command above.

2. **Start the service:**
   ```cmd
   net start FMCGAITrainer
   ```
   
   Or use Windows Services Manager:
   - Press `Win + R`, type `services.msc`, and press Enter
   - Find "FMCG AI Trainer API Service"
   - Right-click and select "Start"

3. **Verify the service is running:**
   ```cmd
   # Check service status
   sc query FMCGAITrainer
   
   # Or check in Services Manager (services.msc)
   # Status should show "Running"
   ```

4. **Test the API:**
   ```cmd
   # Test if the API is responding
   curl http://localhost:8003/docs
   
   # Or open in browser:
   # http://localhost:8003/docs
   # http://localhost:8003/api/v1/training/status
   ```

5. **Check service logs:**
   ```cmd
   # View the service log file (created on first service run)
   type logs\service.log
   
   # Or open in Notepad
   notepad logs\service.log
   ```

6. **Stop the service:**
   ```cmd
   net stop FMCGAITrainer
   ```

7. **Restart the service:**
   ```cmd
   net stop FMCGAITrainer
   net start FMCGAITrainer
   ```

8. **Uninstall the service:**
   - Open an elevated Command Prompt and run:
     ```cmd
     venv\Scripts\python.exe windows_service.py remove
     ```
   - If you prefer a helper script, right-click `uninstall_service.bat` and choose **"Run as administrator"**.

**Service Configuration:**
- Service Name: `FMCGAITrainer`
- Display Name: `FMCG AI Trainer API Service`
- Default Port: `8003` (configurable via `PORT` environment variable)
- Logs: Written to `logs/service.log`

**Manual pywin32 Installation (if needed):**
If you prefer to install pywin32 manually before running the installer, make sure you use the same interpreter that will host the service (the project virtual environment):
```cmd
venv\Scripts\python.exe -m pip install --upgrade --force-reinstall pywin32
venv\Scripts\python.exe venv\Scripts\pywin32_postinstall.py -install
```

**Testing the Service:**

After installation, you can manually verify the service:

```cmd
# 1. Check service status
sc query FMCGAITrainer

# 2. Start the service (if not running)
net start FMCGAITrainer

# 3. Test API in browser
# Open: http://localhost:8003/docs

# 4. Test API with curl
curl http://localhost:8003/api/v1/training/status

# 5. Check logs
type logs\service.log
```

**Manual Service Management:**
```cmd
# Install (ensure you are in an elevated prompt)
venv\Scripts\python.exe windows_service.py install

# Start
venv\Scripts\python.exe windows_service.py start
# or
net start FMCGAITrainer

# Stop
venv\Scripts\python.exe windows_service.py stop
# or
net stop FMCGAITrainer

# Restart
net stop FMCGAITrainer && net start FMCGAITrainer

# Remove
venv\Scripts\python.exe windows_service.py remove
```

**Troubleshooting the Service:**

**If you get "The service name is invalid" error:**

This means the service is not installed. Check installation:

1. **Verify service installation:**
   ```cmd
   # Check if service exists
   sc query FMCGAITrainer
   
   # Optionally remove and reinstall with the commands below.
   ```

2. **Reinstall the service:**
   ```cmd
   # Navigate to project directory
   cd C:\apps\fmcg-ai-trainer-aidin
   
   # Run installer as Administrator
   venv\Scripts\python.exe windows_service.py remove
   venv\Scripts\python.exe windows_service.py install
   ```

3. **Check installation logs:**
   - Review the output from `install_service.bat`
   - Look for any error messages during installation

**If the service fails to start:**

1. **Check service status:**
   ```cmd
   sc query FMCGAITrainer
   ```

2. **Check Windows Event Viewer:**
   - Press `Win + R`, type `eventvwr.msc`
   - Navigate to: Windows Logs → Application
   - Look for errors related to "FMCGAITrainer"

3. **Check service logs:**
   ```cmd
   type logs\service.log
   ```

4. **Verify configuration:**
   - Ensure `config.json` exists and is valid
   - Check database connection settings
   - Verify ODBC driver is installed

5. **Test manually:**
   ```cmd
   # Activate virtual environment
   venv\Scripts\activate
   
   # Run the app manually to see errors
   python main.py
   ```

6. **Run the service in debug mode to surface errors immediately:**
   ```cmd
   venv\Scripts\python.exe windows_service.py debug
   ```

## 📋 Available Training Types

- `customer_order_recommendation_als` - Customer order recommendations using ALS
- `similar_customers` - Similar customer analysis

## 🧪 Testing

Test the training scripts:

```bash
# Dry run to validate configuration
python trainer_cli.py --tenant PRODUCTION --type customer_order_recommendation_als --dry-run
```

## 📁 Project Structure

```
fmcg-ai-trainer/
├── app/
│   ├── api/              # FastAPI endpoints
│   ├── core/             # Core training registry
│   ├── train/            # Training scripts
│   ├── shared/           # Shared utilities
│   └── main.py           # FastAPI app
├── config.json           # Configuration file
├── requirements.txt      # Python dependencies
├── main.py              # API server entry point
├── trainer_cli.py       # CLI entry point
├── windows_service.py    # Windows service wrapper (Windows only)
├── install_service.bat   # Service installation script (Windows only)
└── uninstall_service.bat # Service uninstallation script (Windows only)
```

## 🔧 Helper Scripts (Windows)

- `install_service.bat` – Bootstraps or reuses the virtual environment, installs dependencies, and registers the Windows service via `venv\Scripts\python.exe windows_service.py install`.
- `uninstall_service.bat` – Removes the registered Windows service via the virtual environment interpreter.

All other batch helpers have been removed. Use the commands documented above (`venv\Scripts\python.exe windows_service.py ...`, `net start`, `net stop`, etc.) for additional service management tasks.

## 🔧 Development

For development with auto-reload:

```bash
# Set environment to development
export ENV=development

# Run API server with auto-reload
python main.py
```

## 🔍 Troubleshooting

### ODBC Connection Errors

If you encounter errors like:
```
pyodbc.InterfaceError: ('IM002', '[IM002] [Microsoft][ODBC Driver Manager] Data source name not found and no default driver specified')
```

**Solution:** This error indicates that the ODBC driver is not installed or the driver name in `config.json` doesn't match the installed driver.

1. Verify the ODBC driver is installed (see Installation step 2)
2. Check that the `db_driver` value in `config.json` exactly matches the installed driver name
3. On Windows, you can verify installed drivers by opening "ODBC Data Source Administrator" (search for `odbcad32.exe`)

## 📝 Notes

- The application uses environment variables for configuration, with `config.json` as a fallback
- Runtime logs are written into the `logs/` directory (ignored by Git) when the Windows service is active
- Training jobs can be run independently of the API server

### Windows Service: `pythonservice.exe` Missing

If `win32serviceutil` reports an error similar to:

```
RuntimeError: Can't find 'C:\...\pythonservice.exe'
```

It means the service installer is using a different interpreter than the one that has `pywin32`. Ensure you install/start the service with `venv\Scripts\python.exe`. If the error persists:

1. Reinstall pywin32 inside the virtual environment:
   ```cmd
   venv\Scripts\python.exe -m pip install --upgrade --force-reinstall pywin32
   venv\Scripts\python.exe venv\Scripts\pywin32_postinstall.py -install
   ```
2. Confirm `pythonservice.exe` exists in `venv\` (created automatically after the step above).
