#!/usr/bin/env python3
"""
Training CLI Entry Point for FMCG AI Trainer

This is a standalone CLI that runs training jobs.
It is separate from the FastAPI service to avoid conflicts.
"""

import argparse
import sys
from pathlib import Path

# Add app directory to path
BASE_DIR = Path(__file__).parent
if (BASE_DIR / "app").exists():
    sys.path.insert(0, str(BASE_DIR))

from app.train.training_cli import run_training
from app.shared.common.tracing import get_logger

logger = get_logger(__name__)

def main():
    """Main entry point for training CLI."""
    parser = argparse.ArgumentParser(
        description="FMCG AI Trainer - Run training jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--tenant",
        required=True,
        help="Tenant identifier (e.g., PRODUCTION)"
    )
    
    parser.add_argument(
        "--type",
        dest="training_type",
        required=True,
        help="Training type (e.g., customer_order_recommendation_als)"
    )
    
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Run in dry-run mode (validate configuration without executing)"
    )
    
    args = parser.parse_args()
    
    # Log startup
    logger.info(f"Starting training CLI")
    logger.info(f"Tenant: {args.tenant}")
    logger.info(f"Training Type: {args.training_type}")
    logger.info(f"Dry Run: {args.dry_run}")
    
    # Run the training
    try:
        run_training(args.tenant, args.training_type, args.dry_run)
        logger.info("Training completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())

