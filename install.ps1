# FMCG AI Trainer - Simple Conda Installer
# This ONE script does everything: installs conda environment, sets up daily training, configures SQL connections
# Supports multiple training files and both API service and scheduled execution

Param(
  [string]$Action = "help",
  [string]$BasePath = "C:\fmcg-ai-trainer",
  [string]$CondaEnvName = "nSight",
  [string]$Mode = "scheduler"  # "scheduler" or "api"
)

$ErrorActionPreference = "Stop"

function Show-Help {
    Write-Host "FMCG AI Trainer - Simple Conda Installer" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\install.ps1 -Action <action> [-Mode <mode>]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Actions:" -ForegroundColor Cyan
    Write-Host "  install   - Install everything (conda env + daily training OR API service)" -ForegroundColor White
    Write-Host "  remove    - Remove everything" -ForegroundColor White
    Write-Host "  test      - Test training scripts" -ForegroundColor White
    Write-Host "  help      - Show this help" -ForegroundColor White
    Write-Host ""
    Write-Host "Modes:" -ForegroundColor Cyan
    Write-Host "  scheduler - Set up daily scheduled training (default)" -ForegroundColor White
    Write-Host "  api       - Set up FastAPI service with Swagger" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\install.ps1 -Action install                    # Scheduled training" -ForegroundColor White
    Write-Host "  .\install.ps1 -Action install -Mode api          # API service" -ForegroundColor White
    Write-Host "  .\install.ps1 -Action test" -ForegroundColor White
}

function Find-CondaPath {
    # Try common conda locations
    $CondaPaths = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:PROGRAMDATA\miniconda3\Scripts\conda.exe",
        "$env:PROGRAMDATA\anaconda3\Scripts\conda.exe"
    )
    
    foreach ($Path in $CondaPaths) {
        if (Test-Path $Path) {
            return $Path
        }
    }
    
    # Try PATH
    try {
        return (Get-Command conda).Source
    } catch {
        throw "Conda not found. Please install Anaconda or Miniconda."
    }
}

function Install-Everything {
    Write-Host "🚀 Installing FMCG AI Trainer..." -ForegroundColor Green
    
    # Find conda
    $CondaPath = Find-CondaPath
    Write-Host "Using conda: $CondaPath" -ForegroundColor Cyan
    
    # Create directories
    if (-not (Test-Path $BasePath)) { 
        New-Item -ItemType Directory -Force -Path $BasePath | Out-Null 
    }
    $LogPath = "$BasePath\logs"
    if (-not (Test-Path $LogPath)) { 
        New-Item -ItemType Directory -Force -Path $LogPath | Out-Null 
    }
    
    # Copy files
    Write-Host "📁 Copying application files..." -ForegroundColor Yellow
    Copy-Item -Recurse -Force "$PSScriptRoot\.." $BasePath
    
    # Setup conda environment
    Write-Host "🐍 Setting up conda environment: $CondaEnvName" -ForegroundColor Yellow
    $EnvExists = & $CondaPath env list | Select-String $CondaEnvName
    if ($EnvExists) {
        Write-Host "Environment exists, updating..." -ForegroundColor Yellow
        & $CondaPath env update -n $CondaEnvName -f "$BasePath\environment.yml"
    } else {
        Write-Host "Creating new environment..." -ForegroundColor Yellow
        & $CondaPath create -n $CondaEnvName python=3.9 -y
        & $CondaPath run -n $CondaEnvName pip install -r "$BasePath\requirements.txt"
    }
    
    # Load configuration
    $Config = @{}
    if (Test-Path "$BasePath\config.json") {
        Write-Host "Loading configuration from config.json..." -ForegroundColor Yellow
        $Config = Get-Content "$BasePath\config.json" | ConvertFrom-Json
    } else {
        Write-Host "Using default configuration..." -ForegroundColor Yellow
        $Config = @{
            tenants = @{
                "PRODUCTION" = @{
                    db_server = "localhost"
                    db_name = "Sales"
                    db_port = 1433
                    db_driver = "ODBC Driver 17 for SQL Server"
                    db_user = ""
                    db_password = ""
                    db_trusted_connection = $true
                    db_encrypt = $true
                    db_trust_server_cert = $false
                }
            }
            schedule = @{
                customer_order_recommendation_als = @{
                    enabled = $true
                    time = "02:00"
                    days = @("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
                }
                similar_customers = @{
                    enabled = $true
                    time = "03:00"
                    days = @("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
                }
            }
        }
    }
    
    if ($Mode -eq "api") {
        Install-APIService -BasePath $BasePath -CondaPath $CondaPath -CondaEnvName $CondaEnvName -Config $Config
    } else {
        Install-ScheduledTraining -BasePath $BasePath -CondaPath $CondaPath -CondaEnvName $CondaEnvName -Config $Config
    }
}

function Install-APIService {
    param($BasePath, $CondaPath, $CondaEnvName, $Config)
    
    Write-Host "🌐 Installing FastAPI Service..." -ForegroundColor Yellow
    
    # Validate NSSM
    $NssmExe = $env:NSSM_EXE
    if (-not $NssmExe) { 
        throw "Please set NSSM_EXE environment variable to nssm.exe path. Download from: https://nssm.cc/download" 
    }
    
    $ServiceName = "nsight-fmcg-ai-trainer"
    $Port = "8088"
    
    # Install service
    & $NssmExe install $ServiceName "$CondaPath" "run -n $CondaEnvName -m uvicorn app.main:app --host 0.0.0.0 --port $Port"
    & $NssmExe set $ServiceName AppDirectory $BasePath
    & $NssmExe set $ServiceName Start SERVICE_AUTO_START
    & $NssmExe set $ServiceName AppStdout "$BasePath\logs\svc.out.log"
    & $NssmExe set $ServiceName AppStderr "$BasePath\logs\svc.err.log"
    & $NssmExe set $ServiceName AppRotateFiles 1
    & $NssmExe set $ServiceName AppRotateOnline 1
    
    # Set environment variables for logging (fallback support)
    & $NssmExe set $ServiceName AppEnvironmentExtra "NSIGHT_LOG_LEVEL=$($Config.logging.default.level)"
    & $NssmExe set $ServiceName AppEnvironmentExtra "NSIGHT_LOG_FORMAT=$($Config.logging.default.format)"
    
    # Set tenant configurations
    foreach ($tenantName in $Config.tenants.Keys) {
        $tenant = $Config.tenants[$tenantName]
        $prefix = "TENANT_$($tenantName.ToUpper())"
        
        & $NssmExe set $ServiceName AppEnvironmentExtra "$prefix`_DB_SERVER=$($tenant.db_server)"
        & $NssmExe set $ServiceName AppEnvironmentExtra "$prefix`_DB_NAME=$($tenant.db_name)"
        & $NssmExe set $ServiceName AppEnvironmentExtra "$prefix`_DB_PORT=$($tenant.db_port)"
        & $NssmExe set $ServiceName AppEnvironmentExtra "$prefix`_DB_DRIVER=$($tenant.db_driver)"
        
        if ($tenant.db_user) {
            & $NssmExe set $ServiceName AppEnvironmentExtra "$prefix`_DB_USER=$($tenant.db_user)"
        }
        if ($tenant.db_password) {
            & $NssmExe set $ServiceName AppEnvironmentExtra "$prefix`_DB_PASSWORD=$($tenant.db_password)"
        }
        
        & $NssmExe set $ServiceName AppEnvironmentExtra "$prefix`_DB_TRUSTED_CONNECTION=$($tenant.db_trusted_connection.ToString().ToLower())"
        & $NssmExe set $ServiceName AppEnvironmentExtra "$prefix`_DB_ENCRYPT=$($tenant.db_encrypt.ToString().ToLower())"
        & $NssmExe set $ServiceName AppEnvironmentExtra "$prefix`_DB_TRUST_SERVER_CERT=$($tenant.db_trust_server_cert.ToString().ToLower())"
    }
    
    # Start service
    & $NssmExe start $ServiceName
    
    Write-Host "`n🎉 API Service Installed!" -ForegroundColor Green
    Write-Host "   🌐 API URL: http://localhost:$Port" -ForegroundColor White
    Write-Host "   📚 Swagger Docs: http://localhost:$Port/docs" -ForegroundColor White
    Write-Host "   📁 Logs: $BasePath\logs" -ForegroundColor White
}

function Install-ScheduledTraining {
    param($BasePath, $CondaPath, $CondaEnvName, $Config)
    
    Write-Host "⏰ Installing Scheduled Training..." -ForegroundColor Yellow
    
    # Create batch script
    $BatchScript = @"
@echo off
REM FMCG AI Trainer - Scheduled Training Script

set BASE_PATH=$BasePath
set CONDA_PATH=$CondaPath
set CONDA_ENV=$CondaEnvName
set LOG_DIR=%BASE_PATH%\logs

REM Get parameters
set TENANT_NAME=%~1
set TRAINING_TYPE=%~2
set DRY_RUN=%~3

REM Set log file with timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,4%%dt:~4,2%%dt:~6,2%_%dt:~8,2%%dt:~10,2%%dt:~12,2%"
set LOG_FILE=%LOG_DIR%\training_%TRAINING_TYPE%_%TENANT_NAME%_%timestamp%.log

REM Set environment variables for tenant
"@

    # Add tenant environment variables from config
    foreach ($tenantName in $Config.tenants.Keys) {
        $tenant = $Config.tenants[$tenantName]
        $prefix = "TENANT_$($tenantName.ToUpper())"
        
        $BatchScript += @"

if "%TENANT_NAME%"=="$tenantName" (
    set $prefix`_DB_SERVER=$($tenant.db_server)
    set $prefix`_DB_NAME=$($tenant.db_name)
    set $prefix`_DB_PORT=$($tenant.db_port)
    set $prefix`_DB_DRIVER=$($tenant.db_driver)
    set $prefix`_DB_TRUSTED_CONNECTION=$($tenant.db_trusted_connection.ToString().ToLower())
    set $prefix`_DB_ENCRYPT=$($tenant.db_encrypt.ToString().ToLower())
    set $prefix`_DB_TRUST_SERVER_CERT=$($tenant.db_trust_server_cert.ToString().ToLower())
"@
        
        if ($tenant.db_user) {
            $BatchScript += @"
    set $prefix`_DB_USER=$($tenant.db_user)
"@
        }
        if ($tenant.db_password) {
            $BatchScript += @"
    set $prefix`_DB_PASSWORD=$($tenant.db_password)
"@
        }
        
        $BatchScript += @"
)
"@
    }
    
    $BatchScript += @"

REM Set logging environment
set NSIGHT_LOG_LEVEL=$($Config.logging.level)
set NSIGHT_LOG_FORMAT=$($Config.logging.format)
set PYTHONPATH=%BASE_PATH%

REM Log and run
echo [%date% %time%] Starting %TRAINING_TYPE% training for tenant %TENANT_NAME% >> "%LOG_FILE%"
echo [%date% %time%] Using conda environment: %CONDA_ENV% >> "%LOG_FILE%"

REM Run training using the registry system
"%CONDA_PATH%" run -n "%CONDA_ENV%" python "%BASE_PATH%\app\train\training.py" --tenant "%TENANT_NAME%" --training-type "%TRAINING_TYPE%" %DRY_RUN% >> "%LOG_FILE%" 2>&1

echo [%date% %time%] Training completed with exit code %ERRORLEVEL% >> "%LOG_FILE%"

exit /b %ERRORLEVEL%
"@
    
    $BatchScript | Out-File -FilePath "$BasePath\run-training.bat" -Encoding ASCII
    Write-Host "✅ Created training script: $BasePath\run-training.bat" -ForegroundColor Green
    
    # Create scheduled tasks based on config
    foreach ($tenantName in $Config.tenants.Keys) {
        foreach ($trainingType in $Config.schedule.Keys) {
            $schedule = $Config.schedule[$trainingType]
            if ($schedule.enabled) {
                $TaskName = "FMCG-$($trainingType.Replace('_', ''))-$tenantName"
                $Action = New-ScheduledTaskAction -Execute "$BasePath\run-training.bat" -Argument "$tenantName $trainingType"
                $Trigger = New-ScheduledTaskTrigger -Daily -At $schedule.time
                $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
                $Principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
                
                Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Description "FMCG AI Training: $trainingType for $tenantName (Daily $($schedule.time))"
                Write-Host "  ✅ Created: $TaskName (Daily at $($schedule.time))" -ForegroundColor Green
            }
        }
    }
    
    Write-Host "`n🎉 Scheduled Training Installed!" -ForegroundColor Green
    Write-Host "   📁 Base Path: $BasePath" -ForegroundColor White
    Write-Host "   🐍 Conda Env: $CondaEnvName" -ForegroundColor White
    Write-Host "   📝 Logs: $BasePath\logs" -ForegroundColor White
}

function Remove-Everything {
    Write-Host "🗑️  Removing FMCG AI Trainer..." -ForegroundColor Yellow
    
    # Remove scheduled tasks
    $Tasks = Get-ScheduledTask -TaskName "FMCG-*" -ErrorAction SilentlyContinue
    if ($Tasks) {
        foreach ($Task in $Tasks) {
            Unregister-ScheduledTask -TaskName $Task.TaskName -Confirm:$false
            Write-Host "✅ Removed task: $($Task.TaskName)" -ForegroundColor Green
        }
    }
    
    # Remove conda environment
    try {
        $CondaPath = Find-CondaPath
        & $CondaPath env remove -n $CondaEnvName -y
        Write-Host "✅ Removed conda environment: $CondaEnvName" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Could not remove conda environment" -ForegroundColor Yellow
    }
    
    # Remove files
    if (Test-Path $BasePath) {
        Remove-Item -Recurse -Force $BasePath
        Write-Host "✅ Removed files: $BasePath" -ForegroundColor Green
    }
    
    Write-Host "🎉 Cleanup Complete!" -ForegroundColor Green
}

function Test-TrainingScripts {
    Write-Host "🧪 Testing training scripts..." -ForegroundColor Yellow
    
    $BasePath = "C:\fmcg-ai-trainer"
    if (-not (Test-Path "$BasePath\run-training.bat")) {
        Write-Host "❌ Training script not found. Run 'install' first." -ForegroundColor Red
        return
    }
    
    Write-Host "Testing customer recommendation ALS..." -ForegroundColor Cyan
    & "$BasePath\run-training.bat" "PRODUCTION" "customer_order_recommendation_als" "--dry-run"
    
    Write-Host "Testing similar customers..." -ForegroundColor Cyan
    & "$BasePath\run-training.bat" "PRODUCTION" "similar_customers" "--dry-run"
    
    Write-Host "✅ Test completed! Check logs in $BasePath\logs" -ForegroundColor Green
}

# Main execution
switch ($Action.ToLower()) {
    "install" { Install-Everything }
    "remove" { Remove-Everything }
    "test" { Test-TrainingScripts }
    "help" { Show-Help }
    default { Show-Help }
}
