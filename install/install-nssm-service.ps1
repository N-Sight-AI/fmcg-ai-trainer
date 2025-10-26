Param(
  [string]$ServiceName = "nsight-fmcg-ai-trainer",
  [string]$BasePath = "C:\fmcg-ai-trainer",
  [string]$Port = "8088",
  [string]$PythonExe = ""
)
# Requires NSSM already downloaded: set $NssmExe to its path
# Example: $NssmExe = "C:\tools\nssm\nssm.exe"
$ErrorActionPreference = "Stop"

if (-not (Test-Path $BasePath)) { New-Item -ItemType Directory -Force -Path $BasePath | Out-Null }
Copy-Item -Recurse -Force "$PSScriptRoot\.." $BasePath

# venv
if ($PythonExe -eq "") { $PythonExe = (Get-Command py).Source }
& $PythonExe -3 -m venv "$BasePath\venv"
& "$BasePath\venv\Scripts\pip.exe" install --upgrade pip
& "$BasePath\venv\Scripts\pip.exe" install -r "$BasePath\requirements.txt"

# NSSM
$NssmExe = $env:NSSM_EXE
if (-not $NssmExe) { throw "Please set NSSM_EXE environment variable to nssm.exe path." }

# Install service to run uvicorn
& $NssmExe install $ServiceName "$BasePath\venv\Scripts\python.exe" " -m uvicorn app.main:app --host 0.0.0.0 --port $Port"
& $NssmExe set $ServiceName AppDirectory $BasePath
& $NssmExe set $ServiceName Start SERVICE_AUTO_START
& $NssmExe set $ServiceName AppStdout "$BasePath\logs\svc.out.log"
& $NssmExe set $ServiceName AppStderr "$BasePath\logs\svc.err.log"
& $NssmExe set $ServiceName AppRotateFiles 1
& $NssmExe set $ServiceName AppRotateOnline 1

# Example tenant env (remove/replace with real)
& $NssmExe set $ServiceName AppEnvironmentExtra "TENANT_DEMO_DB_SERVER=SQLHOST"
& $NssmExe set $ServiceName AppEnvironmentExtra "TENANT_DEMO_DB_NAME=Sales"
& $NssmExe set $ServiceName AppEnvironmentExtra "TENANT_DEMO_DB_PORT=1433"
& $NssmExe set $ServiceName AppEnvironmentExtra "TENANT_DEMO_DB_DRIVER=ODBC Driver 17 for SQL Server"

# Logging level
& $NssmExe set $ServiceName AppEnvironmentExtra "NSIGHT_LOG_LEVEL=INFO"

# Start it
& $NssmExe start $ServiceName
Write-Host "Service $ServiceName installed and started on port $Port"
