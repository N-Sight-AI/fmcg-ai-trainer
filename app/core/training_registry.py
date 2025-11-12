"""
Training Registry for FMCG AI Trainer

This module provides a registry system for managing multiple training files
and their configurations. It allows easy addition of new training types
without modifying the core API code.
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
import os
import sys

@dataclass
class TrainingConfig:
    """Configuration for a training type."""
    name: str
    display_name: str
    script_path: str
    class_name: Optional[str] = None
    description: str = ""
    enabled: bool = True

class TrainingRegistry:
    """Registry for managing training configurations."""
    
    def __init__(self):
        self._trainings: Dict[str, TrainingConfig] = {}
        self._register_default_trainings()
    
    def _register_default_trainings(self):
        """Register default training configurations."""
        base_path = os.path.dirname(os.path.dirname(__file__))
        
        # Customer Order Recommendations Training
        self.register(TrainingConfig(
            name="customer_order_recommendation_als",
            display_name="Customer Order Recommendation ALS",
            script_path=os.path.join(base_path, "train", "customer_order_recommendations_als.py"),
            class_name="ALSTrainer",
            description="Train ALS model for customer order recommendations using customer-item interactions"
        ))
        
        # Similar Customers Training
        self.register(TrainingConfig(
            name="similar_customers",
            display_name="Similar Customers",
            script_path=os.path.join(base_path, "train", "train_als_export_similar_customer.py"),
            class_name="ALSTrainer",
            description="Train ALS model for similar customer analysis"
        ))
    
    def register(self, config: TrainingConfig):
        """Register a new training configuration."""
        self._trainings[config.name] = config
    
    def get(self, name: str) -> Optional[TrainingConfig]:
        """Get training configuration by name."""
        return self._trainings.get(name)
    
    def get_all(self) -> Dict[str, TrainingConfig]:
        """Get all registered training configurations."""
        return self._trainings.copy()
    
    def get_enabled(self) -> Dict[str, TrainingConfig]:
        """Get all enabled training configurations."""
        return {name: config for name, config in self._trainings.items() if config.enabled}
    
    def list_names(self) -> List[str]:
        """Get list of all training names."""
        return list(self._trainings.keys())
    
    def validate_script_exists(self, name: str) -> bool:
        """Check if the training script file exists."""
        config = self.get(name)
        if not config:
            return False
        return os.path.exists(config.script_path)
    
    def get_script_command(self, name: str, tenant: str, dry_run: bool = False) -> List[str]:
        """Get the command to run a training script."""
        config = self.get(name)
        if not config:
            raise ValueError(f"Training '{name}' not found in registry")
        
        if not self.validate_script_exists(name):
            raise FileNotFoundError(f"Training script not found: {config.script_path}")
        
        # Get the Python executable to use
        # If running in EXE (frozen), we need to use the Python executable
        # from within the EXE, not the EXE itself
        import sys
        
        # Check if we're running in a PyInstaller bundle
        if getattr(sys, 'frozen', False):
            # Running as compiled EXE - use embedded Python
            python_exe = sys.executable
        else:
            # Running as Python script - use normal Python
            python_exe = sys.executable
        
        # Check if we should call the training class directly via Python module
        # Instead of running as subprocess, we can use the CLI module
        if config.class_name:
            # Use the training_cli.py module
            base_path = os.path.dirname(os.path.dirname(__file__))
            cli_path = os.path.join(base_path, "train", "training_cli.py")
            if os.path.exists(cli_path):
                cmd = [python_exe, cli_path, "--tenant", tenant, "--training-type", name]
                if dry_run:
                    cmd.append("--dry-run")
                return cmd
        
        # Fallback: run the script directly
        cmd = [python_exe, config.script_path, "--tenant", tenant]
        if dry_run:
            cmd.append("--dry-run")
        
        return cmd

# Global registry instance
training_registry = TrainingRegistry()

def get_training_registry() -> TrainingRegistry:
    """Get the global training registry instance."""
    return training_registry

def register_training(config: TrainingConfig):
    """Convenience function to register a training configuration."""
    training_registry.register(config)

def get_training_config(name: str) -> Optional[TrainingConfig]:
    """Convenience function to get a training configuration."""
    return training_registry.get(name)

def list_available_trainings() -> List[str]:
    """Convenience function to list available training names."""
    return training_registry.list_names()
