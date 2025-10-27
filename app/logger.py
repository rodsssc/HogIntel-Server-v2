"""
HogIntel Logger
Centralized logging configuration with UTF-8 support for Windows
"""
import logging
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from config import settings


def setup_logger(name: str = "hogintel"):
    """
    Configure and return logger instance with UTF-8 support
    
    Args:
        name: Logger name (defaults to "hogintel")
        
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # ============================================================
    # Console Handler with UTF-8 Support for Windows
    # ============================================================
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_format)
    
    # Fix Windows console encoding issues
    if sys.platform == 'win32':
        try:
            # Try to reconfigure stdout to use UTF-8 encoding
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            elif hasattr(sys.stdout, 'buffer'):
                # Fallback for older Python versions
                import codecs
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        except Exception as e:
            # If reconfiguration fails, log warning (after handler is added)
            pass
    
    logger.addHandler(console_handler)
    
    # ============================================================
    # File Handler with Rotation and UTF-8 Encoding
    # ============================================================
    try:
        # Create logs directory if not exists
        log_path = Path(settings.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use RotatingFileHandler instead of FileHandler
        # Max file size: 10MB, Keep 5 backup files
        file_handler = RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8',  # UTF-8 encoding for file
            errors='replace'   # Replace unencodable characters
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
    except Exception as e:
        logger.warning(f"Failed to setup file logging: {e}")
    
    return logger


# Default logger instance
logger = setup_logger()