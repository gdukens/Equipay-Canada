"""
Centralized Logging Configuration for EquiPay Canada
=====================================================

Provides consistent logging setup across all modules.
"""

import logging
import logging.config
from pathlib import Path
from typing import Optional


LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'simple': {
            'format': '%(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'detailed',
            'level': 'DEBUG',
            'filename': 'logs/equipay.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'detailed',
            'level': 'ERROR',
            'filename': 'logs/equipay_errors.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
        },
    },
    'loggers': {
        'src': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False,
        },
        'api': {
            'level': 'INFO',
            'handlers': ['console', 'file'],
            'propagate': False,
        },
        'app': {
            'level': 'INFO',
            'handlers': ['console', 'file'],
            'propagate': False,
        },
    },
    'root': {
        'handlers': ['console', 'file', 'error_file'],
        'level': 'WARNING',
    },
}


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    use_file_logging: bool = True
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Custom log file path (optional)
        use_file_logging: Whether to enable file logging
    """
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Update config if custom log file provided
    config = LOGGING_CONFIG.copy()
    
    if log_file:
        config['handlers']['file']['filename'] = log_file
    
    if not use_file_logging:
        # Remove file handlers if file logging disabled
        config['handlers'] = {'console': config['handlers']['console']}
        config['root']['handlers'] = ['console']
        for logger in config['loggers'].values():
            logger['handlers'] = ['console']
    
    # Update log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    config['handlers']['console']['level'] = level.upper()
    
    # Apply configuration
    try:
        logging.config.dictConfig(config)
    except Exception as e:
        # Fallback to basic config if dictConfig fails
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.warning(f"Failed to apply logging config, using basic config: {e}")
    
    logging.info(f"Logging configured at level {level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Module-level logger
logger = get_logger(__name__)
