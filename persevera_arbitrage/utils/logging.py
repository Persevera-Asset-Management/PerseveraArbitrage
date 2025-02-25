import logging
import sys
from typing import Optional

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with the specified name and level.
    
    Args:
        name: The name of the logger, typically using dot notation (e.g., 'persevera_arbitrage.trading')
        level: The logging level (e.g., logging.INFO, logging.DEBUG)
        
    Returns:
        A configured logger
    """
    logger = logging.getLogger(name)
    
    if level is not None:
        logger.setLevel(level)
        
    return logger

def configure_logger(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    format_str: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> None:
    """
    Configure the root logger for the package.
    
    Args:
        level: The logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional path to a log file
        console: Whether to log to the console
        format_str: The format string for log messages
    """
    # Get the root logger for the package
    root_logger = logging.getLogger('persevera_arbitrage')
    root_logger.setLevel(level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if a log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def set_log_level(level: int) -> None:
    """
    Set the log level for the package.
    
    Args:
        level: The logging level (e.g., logging.INFO, logging.DEBUG)
    """
    logging.getLogger('persevera_arbitrage').setLevel(level) 