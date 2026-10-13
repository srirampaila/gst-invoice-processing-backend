import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance.
    Logs sensitive information should be avoided by the caller.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        
        logger.setLevel(log_level)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add formatter to ch
        ch.setFormatter(formatter)
        
        # Add ch to logger
        logger.addHandler(ch)
        
    return logger
