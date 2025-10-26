"""
Core module for FMCG AI Trainer

This module contains core functionality including:
- Training registry for managing multiple training files
- Configuration management
- Common utilities
"""

from .training_registry import (
    TrainingConfig,
    TrainingRegistry,
    training_registry,
    get_training_registry,
    register_training,
    get_training_config,
    list_available_trainings
)

__all__ = [
    "TrainingConfig",
    "TrainingRegistry", 
    "training_registry",
    "get_training_registry",
    "register_training",
    "get_training_config",
    "list_available_trainings"
]
