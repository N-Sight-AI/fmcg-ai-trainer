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

Edit `config.json` to configure your SQL connection:

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
uvicorn app.app:app --host 0.0.0.0 --port 8003 --reload
```

The API will be available at:
- **API**: http://localhost:8003
- **Swagger Docs**: http://localhost:8003/docs
- **ReDoc**: http://localhost:8003/redoc


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
│   └── app.py            # FastAPI app
├── config.json           # Configuration file
├── requirements.txt      # Python dependencies
├── main.py              # API server entry point
└── trainer_cli.py       # CLI entry point
```

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
- Runtime logs are written into the `logs/` directory (ignored by Git) when configured in the application
- Training jobs can be run independently of the API server
