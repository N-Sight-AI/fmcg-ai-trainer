import logging, os
from .tenant_utils import get_logging_config

def get_logger(name: str = None, profile: str = "default"):
    # Try to get logging config from config.json, fallback to environment variables
    try:
        log_config = get_logging_config(profile)
        level = log_config.get("level", "INFO").upper()
        log_format = log_config.get("format", "json")
    except:
        # Fallback to environment variables
        level = os.getenv("NSIGHT_LOG_LEVEL", "INFO").upper()
        log_format = os.getenv("NSIGHT_LOG_FORMAT", "json")
    
    # Set up logging format based on configuration
    if log_format.lower() == "json":
        format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    else:
        format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    # Create logs directory if it doesn't exist
    # Use the project root logs directory
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up file logging
    log_file = os.path.join(log_dir, "api.log")
    
    # Configure logging with both console and file handlers
    logger = logging.getLogger(name or __name__)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(getattr(logging, level, logging.INFO))
        
        # Create formatter
        formatter = logging.Formatter(format_str)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
