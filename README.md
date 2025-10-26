# FMCG AI Trainer - Simple Installation

## 🚀 Quick Start

**ONE command to install everything:**

```powershell
.\install.ps1 -Action install
```

That's it! This will:
- ✅ Create conda environment `nSight`
- ✅ Install all Python packages
- ✅ Set up daily training at 2 AM and 3 AM
- ✅ Create logging system

## 📝 Configure SQL Connection

After installation, edit `C:\fmcg-ai-trainer\run-training.bat` and update these lines:

```batch
set TENANT_PRODUCTION_DB_SERVER=your-sql-server.com
set TENANT_PRODUCTION_DB_NAME=YourDatabase
set TENANT_PRODUCTION_DB_USER=your_username
set TENANT_PRODUCTION_DB_PASSWORD=your_password
set TENANT_PRODUCTION_DB_TRUSTED_CONNECTION=false
```

## 🧪 Test Installation

```powershell
.\install.ps1 -Action test
```

## 📋 Management Commands

```powershell
# View scheduled tasks
Get-ScheduledTask -TaskName "FMCG-*"

# Run training manually
Start-ScheduledTask -TaskName "FMCG-CustomerRecommendationALS-PRODUCTION"

# View logs
Get-Content "C:\fmcg-ai-trainer\logs\training_*.log" | Select-Object -Last 10

# Remove everything
.\install.ps1 -Action remove
```

## 📁 What Gets Installed

- **Base Path**: `C:\fmcg-ai-trainer\`
- **Conda Environment**: `fmcg-ai-trainer`
- **Daily Schedule**: 
  - User Recommendations: 2:00 AM
  - Similar Customers: 3:00 AM
- **Logs**: `C:\fmcg-ai-trainer\logs\`

## ⚠️ Prerequisites

- Anaconda or Miniconda installed
- PowerShell execution policy allows scripts
- SQL Server accessible from this machine

That's it! Simple and clean. 🎉