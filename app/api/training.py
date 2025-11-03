"""
Training API endpoints for FMCG AI Trainer.

Provides endpoints to trigger and monitor training jobs using the registry system.
"""

import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.shared.common.tracing import get_logger
from app.shared.common.tenant_utils import get_tenant_config_from_cache_only
from app.core.training_registry import get_training_registry, get_training_config

# Create API router
training_router = APIRouter()
logger = get_logger(__name__)

# Global storage for training job status
training_jobs: Dict[str, Dict[str, Any]] = {}


class TrainingRequest(BaseModel):
    """Request model for triggering training jobs."""
    tenant: str
    dry_run: bool = False


class TrainingResponse(BaseModel):
    """Response model for training operations."""
    job_id: str
    status: str
    message: str
    tenant: str
    training_type: str
    dry_run: bool


class TrainingStatusResponse(BaseModel):
    """Response model for training status queries."""
    job_id: str
    status: str
    tenant: str
    training_type: str
    dry_run: bool
    started_at: str
    completed_at: Optional[str] = None
    duration_seconds: Optional[int] = None
    logs: Optional[str] = None
    error: Optional[str] = None


def run_subprocess_with_streaming(cmd, cwd, env, job_id=None):
    """Run a subprocess with real-time output streaming to console logs."""
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
        universal_newlines=True
    )
    
    stdout_lines = []
    stderr_lines = []
    
    while True:
        return_code = process.poll()
        
        stdout_line = process.stdout.readline()
        if stdout_line:
            stdout_line = stdout_line.strip()
            stdout_lines.append(stdout_line)
            if job_id:
                logger.info(f"Training job {job_id}: {stdout_line}")
        
        stderr_line = process.stderr.readline()
        if stderr_line:
            stderr_line = stderr_line.strip()
            stderr_lines.append(stderr_line)
            if job_id:
                # Check if this is a progress bar line (contains progress indicators)
                if any(indicator in stderr_line for indicator in ['%|', 'it/s', 'ETA:', 'elapsed', 'remaining', 'Neighbors (ALS):', 'Uploading factors']):
                    # Log progress bars as INFO instead of ERROR
                    logger.info(f"Training job {job_id} PROGRESS: {stderr_line}")
                else:
                    logger.error(f"Training job {job_id} ERROR: {stderr_line}")
        
        if return_code is not None and not stdout_line and not stderr_line:
            break
    
    final_return_code = process.wait()
    
    result = type('Result', (), {
        'returncode': final_return_code,
        'stdout': '\n'.join(stdout_lines),
        'stderr': '\n'.join(stderr_lines)
    })()
    
    return result


def run_training_script(tenant: str, training_type: str, dry_run: bool, job_id: str):
    """Run the training script in a separate thread."""
    try:
        # Update job status to running
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
        
        # Verify tenant exists
        tenant_config = get_tenant_config_from_cache_only(tenant)
        if not tenant_config:
            raise ValueError(f"Tenant '{tenant}' not found in cache.")
        
        # Get training configuration from registry
        training_config = get_training_config(training_type)
        if not training_config:
            raise ValueError(f"Training type '{training_type}' not found in registry")
        
        # Get the script command from registry
        try:
            cmd = get_training_registry().get_script_command(training_type, tenant, dry_run)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        logger.info(f"Starting training job {job_id}: {' '.join(cmd)}")
        
        # Run the training script with environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.dirname(os.path.dirname(os.path.dirname(training_config.script_path)))
        env['NSIGHT_LOG_LEVEL'] = os.getenv('NSIGHT_LOG_LEVEL', 'INFO')
        env['NSIGHT_LOG_FORMAT'] = os.getenv('NSIGHT_LOG_FORMAT', 'json')
        
        # Pass tenant configuration via environment variables
        try:
            tenant_config = get_tenant_config_from_cache_only(tenant)
            if tenant_config:
                env[f'TENANT_{tenant.upper()}_DB_SERVER'] = getattr(tenant_config, "db_server", "")
                env[f'TENANT_{tenant.upper()}_DB_PORT'] = str(getattr(tenant_config, "db_port", 1433))
                env[f'TENANT_{tenant.upper()}_DB_NAME'] = getattr(tenant_config, "db_name", "")
                env[f'TENANT_{tenant.upper()}_DB_DRIVER'] = getattr(tenant_config, "db_driver", "ODBC Driver 17 for SQL Server")
                if getattr(tenant_config, "db_user", None):
                    env[f'TENANT_{tenant.upper()}_DB_USER'] = tenant_config.db_user
                if getattr(tenant_config, "db_password", None):
                    env[f'TENANT_{tenant.upper()}_DB_PASSWORD'] = tenant_config.db_password
                env[f'TENANT_{tenant.upper()}_DB_TRUSTED_CONNECTION'] = str(getattr(tenant_config, "db_trusted_connection", False)).lower()
                env[f'TENANT_{tenant.upper()}_DB_ENCRYPT'] = str(getattr(tenant_config, "db_encrypt", True)).lower()
                env[f'TENANT_{tenant.upper()}_DB_TRUST_SERVER_CERT'] = str(getattr(tenant_config, "db_trust_server_cert", False)).lower()
        except Exception as e:
            logger.warning(f"Failed to get tenant config for {tenant}: {e}")
        
        # Run the training script with real-time output streaming
        result = run_subprocess_with_streaming(
            cmd=cmd,
            cwd=os.path.dirname(training_config.script_path),
            env=env,
            job_id=job_id
        )
        
        # Update job status based on result
        if result.returncode == 0:
            training_jobs[job_id]["status"] = "completed"
            training_jobs[job_id]["logs"] = result.stdout
            logger.info(f"Training job {job_id} completed successfully")
        else:
            training_jobs[job_id]["status"] = "failed"
            error_message = ""
            if result.stderr:
                error_message += f"STDERR: {result.stderr}\n"
            if result.stdout and "ERROR" in result.stdout:
                stdout_lines = result.stdout.split('\n')
                error_lines = [line for line in stdout_lines if 'ERROR' in line or 'Exception' in line or 'Traceback' in line]
                if error_lines:
                    error_message += f"STDOUT ERRORS: {' '.join(error_lines)}\n"
            
            if not error_message:
                error_message = f"Process failed with return code {result.returncode}\nSTDOUT: {result.stdout}"
            
            training_jobs[job_id]["error"] = error_message.strip()
            training_jobs[job_id]["logs"] = result.stdout
            logger.error(f"Training job {job_id} failed with return code {result.returncode}")
        
        # Set completion time
        training_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        
        # Calculate duration
        if training_jobs[job_id]["started_at"]:
            start_time = datetime.fromisoformat(training_jobs[job_id]["started_at"].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(training_jobs[job_id]["completed_at"].replace('Z', '+00:00'))
            duration = (end_time - start_time).total_seconds()
            training_jobs[job_id]["duration_seconds"] = int(duration)
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)
        training_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        logger.error(f"Training job {job_id} failed with exception: {str(e)}")
        logger.error(f"Error traceback: {error_traceback}")


def start_training_job(tenant: str, training_type: str, dry_run: bool) -> str:
    """Start a training job and return the job ID."""
    # Verify tenant exists
    tenant_config = get_tenant_config_from_cache_only(tenant)
    if not tenant_config:
        raise ValueError(f"Tenant '{tenant}' not found in cache.")
    
    # Verify training type exists
    training_config = get_training_config(training_type)
    if not training_config:
        raise ValueError(f"Training type '{training_type}' not found in registry")
    
    # Create job
    job_id = f"{training_type}_{tenant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    training_jobs[job_id] = {
        "tenant": tenant,
        "training_type": training_type,
        "dry_run": dry_run,
        "status": "queued",
        "started_at": None,
        "completed_at": None,
        "duration_seconds": None,
        "logs": None,
        "error": None
    }
    
    # Start training in background thread
    thread = threading.Thread(
        target=run_training_script,
        args=(tenant, training_type, dry_run, job_id)
    )
    thread.daemon = True
    thread.start()
    
    logger.info(f"Queued {training_type} training job {job_id} for tenant: {tenant}")
    return job_id


@training_router.post("/training/{training_type}", response_model=TrainingResponse)
def start_training(training_type: str, request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Start a training job for the specified training type.
    
    ## Start Training Job
    
    Triggers a training job for the specified training type and tenant.
    
    ### Parameters
    - **training_type**: Type of training to run (customer_order_recommendation_als, similar_customers, etc.)
    - **tenant**: Tenant identifier
    - **dry_run**: Whether to run in dry-run mode (optional, default: false)
    
    ### Example Request
    ```bash
    curl -X POST "{base_url}/api/v1/training/customer_order_recommendation_als" \
         -H "Content-Type: application/json" \
         -d '{"tenant": "PRODUCTION", "dry_run": false}'
    ```
    
    ### Response
    Returns job information including job_id for status tracking.
    """
    try:
        job_id = start_training_job(request.tenant, training_type, request.dry_run)
        
        return TrainingResponse(
            job_id=job_id,
            status="queued",
            message=f"{training_type} training {'(dry run)' if request.dry_run else ''} queued for tenant: {request.tenant}",
            tenant=request.tenant,
            training_type=training_type,
            dry_run=request.dry_run
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Failed to queue {training_type} training: {str(e)}")
        logger.error(f"Full traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to queue training: {str(e)}\n\nDetails:\n{error_details}")


@training_router.get("/training/types")
def get_training_types():
    """
    Get list of available training types.
    
    ## Get Available Training Types
    
    This endpoint retrieves a list of all available training types
    that can be executed.
    
    ### Example Request
    ```bash
    curl -X GET "{base_url}/api/v1/training/types"
    ```
    
    ### Response
    Returns list of available training types with their configurations.
    """
    try:
        registry = get_training_registry()
        training_configs = registry.get_all()
        
        result = []
        for name, config in training_configs.items():
            result.append({
                "name": config.name,
                "display_name": config.display_name,
                "description": config.description,
                "enabled": config.enabled,
                "script_exists": registry.validate_script_exists(name),
                "default_schedule_time": config.default_schedule_time,
                "default_schedule_days": config.default_schedule_days
            })
        
        return {
            "training_types": result,
            "total": len(result)
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Failed to get training types: {e}")
        logger.error(f"Full traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to get training types: {str(e)}\n\nDetails:\n{error_details}")


@training_router.get("/training/status/{job_id}", response_model=TrainingStatusResponse)
def get_training_status(job_id: str):
    """
    Get status of a specific training job.
    
    ## Get Training Job Status
    
    Retrieves the current status and details of a training job.
    
    ### Parameters
    - **job_id**: Unique identifier for the training job
    
    ### Example Request
    ```bash
    curl -X GET "{base_url}/api/v1/training/status/{job_id}"
    ```
    
    ### Response
    Returns detailed job status including logs and error information.
    """
    try:
        if job_id not in training_jobs:
            raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
        
        job = training_jobs[job_id]
        return TrainingStatusResponse(
            job_id=job_id,
            status=job["status"],
            tenant=job["tenant"],
            training_type=job["training_type"],
            dry_run=job["dry_run"],
            started_at=job["started_at"] or "",
            completed_at=job["completed_at"],
            duration_seconds=job["duration_seconds"],
            logs=job["logs"],
            error=job["error"]
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Failed to get training status: {e}")
        logger.error(f"Full traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}\n\nDetails:\n{error_details}")


@training_router.get("/training/jobs")
def get_training_jobs(
    tenant: Optional[str] = None,
    training_type: Optional[str] = None,
    status: Optional[str] = None
):
    """
    List training jobs with optional filtering.
    
    ## Get Training Jobs
    
    Retrieves a list of training jobs with optional filtering by tenant,
    training type, or status.
    
    ### Query Parameters
    - **tenant**: Filter by tenant (optional)
    - **training_type**: Filter by training type (optional)
    - **status**: Filter by status (optional)
    
    ### Example Request
    ```bash
    curl -X GET "{base_url}/api/v1/training/jobs?tenant=DEMO&status=completed"
    ```
    
    ### Response
    Returns filtered list of training jobs with metadata.
    """
    try:
        filtered_jobs = []
        for job_id, job in training_jobs.items():
            if tenant and job["tenant"] != tenant:
                continue
            if training_type and job["training_type"] != training_type:
                continue
            if status and job["status"] != status:
                continue
            filtered_jobs.append({"job_id": job_id, **job})
        
        filtered_jobs.sort(key=lambda x: x["started_at"] or "", reverse=True)
        return {
            "jobs": filtered_jobs,
            "total": len(filtered_jobs),
            "filters": {
                "tenant": tenant,
                "training_type": training_type,
                "status": status
            }
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Failed to get training jobs: {e}")
        logger.error(f"Full traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to get training jobs: {str(e)}\n\nDetails:\n{error_details}")


@training_router.delete("/training/jobs/{job_id}")
def delete_training_job(job_id: str):
    """
    Delete a training job record.
    
    ## Delete Training Job
    
    Removes a training job record from the system. This does not cancel
    running jobs, only removes the record.
    
    ### Parameters
    - **job_id**: Unique identifier for the training job
    
    ### Example Request
    ```bash
    curl -X DELETE "{base_url}/api/v1/training/jobs/{job_id}"
    ```
    
    ### Response
    Confirms deletion of the training job record.
    """
    try:
        if job_id not in training_jobs:
            raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
        
        del training_jobs[job_id]
        logger.info(f"Deleted training job: {job_id}")
        return {"message": f"Training job {job_id} deleted successfully", "job_id": job_id}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Failed to delete training job: {e}")
        logger.error(f"Full traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to delete training job: {str(e)}\n\nDetails:\n{error_details}")


@training_router.post("/training/{training_type}/run-direct", response_model=dict)
async def run_training_direct(
    training_type: str,
    tenant: str = "production",
    dry_run: bool = False
):
    """
    Run training directly without queuing.
    
    This endpoint runs the training script synchronously and returns the result immediately.
    Use this for immediate execution without job tracking.
    
    ## Direct Training Execution
    
    Executes training immediately without creating a job in the queue system.
    Returns the result directly in the response.
    
    ### Parameters
    - **training_type**: Type of training to run (customer_order_recommendation_als, similar_customers, etc.)
    - **tenant**: Tenant identifier (default: production)
    - **dry_run**: Whether to run in dry-run mode (optional, default: false)
    
    ### Example Request
    ```bash
    curl -X POST "{base_url}/api/v1/training/customer_order_recommendation_als/run-direct" \
         -H "Content-Type: application/json" \
         -d '{"tenant": "production", "dry_run": false}'
    ```
    
    ### Response
    Returns the training execution result directly including output and status.
    """
    try:
        # Verify tenant exists
        tenant_config = get_tenant_config_from_cache_only(tenant)
        if not tenant_config:
            raise HTTPException(status_code=404, detail=f"Tenant '{tenant}' not found in cache.")
        
        # Verify training type exists
        training_config = get_training_config(training_type)
        if not training_config:
            raise HTTPException(status_code=404, detail=f"Training type '{training_type}' not found in registry")
        
        logger.info(f"Starting direct training execution: {training_type} for tenant {tenant}")
        
        # Set tenant configuration via environment variables
        import os
        env = os.environ.copy()
        env['NSIGHT_LOG_LEVEL'] = os.getenv('NSIGHT_LOG_LEVEL', 'INFO')
        env['NSIGHT_LOG_FORMAT'] = os.getenv('NSIGHT_LOG_FORMAT', 'json')
        
        # Pass tenant configuration via environment variables
        try:
            if tenant_config:
                env[f'TENANT_{tenant.upper()}_DB_SERVER'] = getattr(tenant_config, "db_server", "")
                env[f'TENANT_{tenant.upper()}_DB_PORT'] = str(getattr(tenant_config, "db_port", 1433))
                env[f'TENANT_{tenant.upper()}_DB_NAME'] = getattr(tenant_config, "db_name", "")
                env[f'TENANT_{tenant.upper()}_DB_DRIVER'] = getattr(tenant_config, "db_driver", "ODBC Driver 17 for SQL Server")
                if getattr(tenant_config, "db_user", None):
                    env[f'TENANT_{tenant.upper()}_DB_USER'] = tenant_config.db_user
                if getattr(tenant_config, "db_password", None):
                    env[f'TENANT_{tenant.upper()}_DB_PASSWORD'] = tenant_config.db_password
                env[f'TENANT_{tenant.upper()}_DB_TRUSTED_CONNECTION'] = str(getattr(tenant_config, "db_trusted_connection", False)).lower()
                env[f'TENANT_{tenant.upper()}_DB_ENCRYPT'] = str(getattr(tenant_config, "db_encrypt", True)).lower()
                env[f'TENANT_{tenant.upper()}_DB_TRUST_SERVER_CERT'] = str(getattr(tenant_config, "db_trust_server_cert", False)).lower()
        except Exception as e:
            logger.warning(f"Failed to set tenant config for {tenant}: {e}")
        
        # Import and run training directly (no subprocess!)
        from app.train.training_cli import run_training
        import traceback
        
        start_time = datetime.now()
        try:
            # Run training directly in-process (will block, but that's OK)
            # Run in executor to avoid blocking the event loop completely
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                await loop.run_in_executor(executor, run_training, tenant, training_type, dry_run)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"Direct training execution completed successfully in {duration:.2f} seconds")
            return {
                "status": "success",
                "message": f"Training {training_type} completed successfully for tenant {tenant}",
                "training_type": training_type,
                "tenant": tenant,
                "dry_run": dry_run,
                "execution_time_seconds": duration
            }
        except Exception as e:
            error_tb = traceback.format_exc()
            logger.error("=" * 60)
            logger.error(f"TRAINING FAILED - {training_type} for tenant {tenant}")
            logger.error(f"Error: {str(e)}")
            logger.error(f"Traceback:\n{error_tb}")
            logger.error("=" * 60)
            
            # Raise HTTPException
            raise HTTPException(
                status_code=500,
                detail=f"Training failed: {str(e)}\n\nTraceback:\n{error_tb}"
            )
            
    except HTTPException:
        # Re-raise HTTPExceptions as-is (these are intentional error responses)
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Unexpected error during training execution: {str(e)}")
        logger.error(f"Full traceback:\n{error_details}")
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error during training execution: {str(e)}\n\nTraceback:\n{error_details}"
        )
