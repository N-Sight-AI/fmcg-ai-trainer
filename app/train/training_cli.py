#!/usr/bin/env python3
"""
Command-line interface for FMCG AI Trainer

This script provides a command-line interface for running training jobs
directly without the API server. It uses direct module imports for in-process execution.
"""

import argparse
import sys
import os
import importlib
from datetime import datetime
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.training_registry import get_training_registry, get_training_config
from app.shared.common.tracing import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="FMCG AI Trainer - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training.py --tenant PRODUCTION --training-type customer_order_recommendation_als
  python training.py --tenant PRODUCTION --training-type similar_customers --dry-run
  python training.py --list-types
        """
    )
    
    parser.add_argument(
        "--tenant", 
        required=False,
        help="Tenant identifier (required for training execution)"
    )
    
    parser.add_argument(
        "--training-type",
        required=False,
        help="Type of training to run (customer_order_recommendation_als, similar_customers)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode without actual execution"
    )
    
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="List all available training types"
    )
    
    args = parser.parse_args()
    
    try:
        if args.list_types:
            list_training_types()
        elif args.tenant and args.training_type:
            run_training(args.tenant, args.training_type, args.dry_run)
        else:
            parser.print_help()
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        sys.exit(1)

def list_training_types():
    """List all available training types."""
    registry = get_training_registry()
    training_configs = registry.get_all()
    
    print("Available Training Types:")
    print("=" * 50)
    
    for name, config in training_configs.items():
        status = "✅" if registry.validate_script_exists(name) else "❌"
        enabled = "Enabled" if config.enabled else "Disabled"
        
        print(f"{status} {config.display_name}")
        print(f"   Name: {config.name}")
        print(f"   Description: {config.description}")
        print(f"   Status: {enabled}")
        print(f"   Default Schedule: {config.default_schedule_time}")
        print(f"   Script Exists: {registry.validate_script_exists(name)}")
        print()

def run_training(tenant: str, training_type: str, dry_run: bool = False):
    """Run a specific training job using direct imports (no subprocess)."""
    logger.info(f"Starting training: {training_type} for tenant {tenant}")
    
    if dry_run:
        logger.info("Running in DRY-RUN mode")
    
    # Get training configuration
    training_config = get_training_config(training_type)
    if not training_config:
        raise ValueError(f"Training type '{training_type}' not found in registry")
    
    # Validate script exists
    registry = get_training_registry()
    if not registry.validate_script_exists(training_type):
        raise FileNotFoundError(f"Training script not found: {training_config.script_path}")
    
    # Run training using direct class import (no subprocess)
    if not training_config.class_name:
        raise ValueError(f"Training type '{training_type}' must have a class_name defined")
    
    logger.info(f"Running training class '{training_config.class_name}' from {training_config.script_path}")
    run_training_class(training_config, tenant, dry_run)

def run_training_class(training_config, tenant: str, dry_run: bool):
    """Run training using the class-based approach with direct imports."""
    try:
        # Get the training module name from script path
        # e.g., "app/train/customer_order_recommendations_als.py" -> "app.train.customer_order_recommendations_als"
        script_path = Path(training_config.script_path)
        
        # Convert path to module name
        # Remove .py extension and convert / to .
        parts = script_path.with_suffix('').parts
        
        # Find 'app' in the path and use everything after it
        try:
            app_idx = parts.index('app')
            module_name_parts = parts[app_idx:]
            module_name = '.'.join(module_name_parts)
        except ValueError:
            # Fallback: use the full path relative to current directory
            module_name = str(script_path.with_suffix('')).replace(os.sep, '.')
        
        logger.info(f"Importing training module: {module_name}")
        
        # Import the module dynamically using importlib
        module = importlib.import_module(module_name)
        
        # Get the class
        if not hasattr(module, training_config.class_name):
            raise AttributeError(f"Class '{training_config.class_name}' not found in module '{module_name}'")
        
        trainer_class = getattr(module, training_config.class_name)
        
        if dry_run:
            logger.info(f"Dry run: Would train {training_config.display_name} for tenant {tenant}")
            return
        
        # Create trainer instance and run training
        logger.info(f"Starting {training_config.display_name} training for tenant {tenant}")
        trainer = trainer_class(tenant)
        trainer.train_and_export()
        logger.info(f"Training completed successfully")
            
    except Exception as e:
        logger.error(f"Failed to run training class: {str(e)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
