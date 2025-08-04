"""
Structured logging configuration for Akasha system.

This module provides comprehensive logging setup using structlog for
structured, contextualized logging with performance tracking and
correlation IDs for request tracing.
"""

import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
from contextvars import ContextVar
from uuid import uuid4

import structlog
from structlog.stdlib import LoggerFactory

from .config import LoggingConfig, get_config


# Context variable for correlation ID tracking
correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class CorrelationIDProcessor:
    """Processor to add correlation ID to log entries."""
    
    def __call__(self, logger, method_name, event_dict):
        corr_id = correlation_id.get()
        if corr_id:
            event_dict["correlation_id"] = corr_id
        return event_dict


class PerformanceProcessor:
    """Processor to add performance metrics to log entries."""
    
    def __call__(self, logger, method_name, event_dict):
        # Add timestamp
        event_dict["timestamp"] = time.time()
        
        # Add performance context if available
        if "duration" in event_dict:
            # Round duration to 3 decimal places
            event_dict["duration"] = round(event_dict["duration"], 3)
        
        return event_dict


class MemoryUsageProcessor:
    """Processor to add memory usage information."""
    
    def __init__(self):
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            self.psutil = None
    
    def __call__(self, logger, method_name, event_dict):
        if self.psutil and method_name in ["error", "critical"]:
            try:
                process = self.psutil.Process()
                memory_info = process.memory_info()
                event_dict["memory_usage_mb"] = round(memory_info.rss / 1024 / 1024, 2)
                event_dict["memory_percent"] = round(process.memory_percent(), 2)
            except Exception:
                # Don't fail logging if memory info unavailable
                pass
        
        return event_dict


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    Set up structured logging for the Akasha system.
    
    Args:
        config: Optional logging configuration. If None, uses global config.
    """
    if config is None:
        config = get_config().logging
    
    # Create log directory if needed
    if config.output in ["file", "both"]:
        log_path = Path(config.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.level),
    )
    
    # Set up processors based on configuration
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        CorrelationIDProcessor(),
        PerformanceProcessor(),
        MemoryUsageProcessor(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    # Add appropriate final processor based on format
    if config.format == "structured":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=config.output == "console"
            )
        )
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Set up handlers based on output configuration
    root_logger = logging.getLogger()
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    if config.output in ["console", "both"]:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.level))
        root_logger.addHandler(console_handler)
    
    if config.output in ["file", "both"]:
        # Use rotating file handler to manage log file size
        file_handler = logging.handlers.RotatingFileHandler(
            filename=config.file_path,
            maxBytes=config.max_file_size_mb * 1024 * 1024,
            backupCount=config.backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, config.level))
        root_logger.addHandler(file_handler)
    
    # Set third-party library log levels to reduce noise
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def set_correlation_id(corr_id: Optional[str] = None) -> str:
    """
    Set correlation ID for request tracing.
    
    Args:
        corr_id: Optional correlation ID. If None, generates a new UUID.
        
    Returns:
        The correlation ID that was set
    """
    if corr_id is None:
        corr_id = str(uuid4())
    
    correlation_id.set(corr_id)
    return corr_id


def get_correlation_id() -> Optional[str]:
    """
    Get the current correlation ID.
    
    Returns:
        Current correlation ID or None if not set
    """
    return correlation_id.get()


class LoggingContext:
    """Context manager for adding structured context to logs."""
    
    def __init__(self, **context):
        self.context = context
        self.logger = get_logger(self.__class__.__module__)
    
    def __enter__(self):
        self.bound_logger = self.logger.bind(**self.context)
        return self.bound_logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.bound_logger.error(
                "Exception in logging context",
                exc_type=exc_type.__name__,
                exc_msg=str(exc_val)
            )


class PerformanceLogger:
    """Context manager for performance logging."""
    
    def __init__(self, operation: str, logger: Optional[structlog.stdlib.BoundLogger] = None, **context):
        self.operation = operation
        self.logger = logger or get_logger(self.__class__.__module__)
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(
            f"Starting {self.operation}",
            operation=self.operation,
            **self.context
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type:
            self.logger.error(
                f"Failed {self.operation}",
                operation=self.operation,
                duration=duration,
                exc_type=exc_type.__name__,
                exc_msg=str(exc_val),
                **self.context
            )
        else:
            self.logger.info(
                f"Completed {self.operation}",
                operation=self.operation,
                duration=duration,
                **self.context
            )
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self.__enter__()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return self.__exit__(exc_type, exc_val, exc_tb)


def log_function_call(func_name: str = None):
    """
    Decorator for logging function calls with performance metrics.
    
    Args:
        func_name: Optional function name override
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = func_name or f"{func.__module__}.{func.__name__}"
            logger = get_logger(func.__module__)
            
            with PerformanceLogger(f"function_call:{name}", logger):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = func_name or f"{func.__module__}.{func.__name__}"
            logger = get_logger(func.__module__)
            
            with PerformanceLogger(f"function_call:{name}", logger):
                return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Convenience functions for common logging patterns
def log_api_request(method: str, path: str, status_code: int, duration: float, **extra):
    """Log API request with standard format."""
    logger = get_logger("akasha.api")
    
    level = "info"
    if status_code >= 500:
        level = "error"
    elif status_code >= 400:
        level = "warning"
    
    getattr(logger, level)(
        "API request",
        method=method,
        path=path,
        status_code=status_code,
        duration=duration,
        **extra
    )


def log_document_processing(document_id: str, operation: str, status: str, **extra):
    """Log document processing events."""
    logger = get_logger("akasha.processing")
    
    level = "error" if status == "failed" else "info"
    
    getattr(logger, level)(
        "Document processing",
        document_id=document_id,
        operation=operation,
        status=status,
        **extra
    )


def log_model_operation(model_name: str, operation: str, **extra):
    """Log model loading/inference operations."""
    logger = get_logger("akasha.models")
    
    logger.info(
        "Model operation",
        model_name=model_name,
        operation=operation,
        **extra
    )


def log_search_query(query: str, results_count: int, duration: float, **extra):
    """Log search query with results."""
    logger = get_logger("akasha.search")
    
    logger.info(
        "Search query",
        query_length=len(query),
        results_count=results_count,
        duration=duration,
        **extra
    )


def log_system_metric(metric_name: str, value: float, unit: str = None, **extra):
    """Log system metrics."""
    logger = get_logger("akasha.metrics")
    
    logger.info(
        "System metric",
        metric_name=metric_name,
        value=value,
        unit=unit,
        **extra
    )