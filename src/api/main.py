"""
Main FastAPI application for Akasha.

This module creates the FastAPI application instance with all routes,
middleware, and error handlers configured for the Akasha system.
"""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from ..core.config import get_config, AkashaConfig
from ..core.logging import (
    setup_logging, 
    get_logger, 
    set_correlation_id, 
    log_api_request
)
from ..core.exceptions import AkashaError
from .auth import router as auth_router
from .middleware import setup_security_middleware


# Global logger
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    
    Handles startup and shutdown tasks for the application.
    """
    # Startup
    config = get_config()
    setup_logging(config.logging)
    
    logger.info(
        "Starting Akasha application",
        version=config.system.version,
        environment=config.system.environment
    )
    
    # TODO: Initialize other services (plugins, models, etc.)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Akasha application")
    
    # TODO: Cleanup services


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    config = get_config()
    
    app = FastAPI(
        title="Akasha",
        description="A state-of-the-art, modular, local-first multimodal RAG system",
        version=config.system.version,
        docs_url="/docs" if config.system.debug else None,
        redoc_url="/redoc" if config.system.debug else None,
        lifespan=lifespan
    )
    
    # Add middleware (order matters - last added is first executed)
    setup_middleware(app, config)
    setup_security_middleware(app, config)
    
    # Add error handlers
    setup_error_handlers(app)
    
    # Add routes
    setup_routes(app)
    
    return app


def setup_middleware(app: FastAPI, config: AkashaConfig) -> None:
    """Set up middleware for the FastAPI application."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware (if not in debug mode)
    if not config.system.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=[config.api.host, "localhost", "127.0.0.1"]
        )
    
    # Request logging middleware
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        # Set correlation ID for request tracing
        correlation_id = set_correlation_id()
        
        start_time = time.time()
        
        # Log request start
        logger.debug(
            "Request started",
            method=request.method,
            path=request.url.path,
            correlation_id=correlation_id
        )
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log API request
            log_api_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration=duration,
                correlation_id=correlation_id
            )
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error(
                "Request failed with exception",
                method=request.method,
                path=request.url.path,
                duration=duration,
                error=str(e),
                correlation_id=correlation_id
            )
            raise


def setup_error_handlers(app: FastAPI) -> None:
    """Set up error handlers for the FastAPI application."""
    
    logger.debug("Setting up error handlers")
    
    # Add a catch-all route for 404s
    @app.middleware("http")
    async def catch_404_middleware(request: Request, call_next):
        response = await call_next(request)
        if response.status_code == 404:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": {
                        "code": "HTTP_404",
                        "message": "Not Found"
                    },
                    "metadata": {
                        "timestamp": time.time(),
                        "path": request.url.path
                    }
                }
            )
        return response
    
    @app.exception_handler(AkashaError)
    async def akasha_error_handler(request: Request, exc: AkashaError):
        """Handle custom Akasha errors."""
        logger.error(
            "Akasha error occurred",
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
            context=exc.context,
            path=request.url.path
        )
        
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": exc.to_dict(),
                "metadata": {
                    "timestamp": time.time(),
                    "path": request.url.path
                }
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        logger.warning(
            "Request validation failed",
            errors=exc.errors(),
            path=request.url.path
        )
        
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": exc.errors()
                },
                "metadata": {
                    "timestamp": time.time(),
                    "path": request.url.path
                }
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        logger.warning(
            "HTTP exception occurred",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail
                },
                "metadata": {
                    "timestamp": time.time(),
                    "path": request.url.path
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(
            "Unhandled exception occurred",
            error_type=type(exc).__name__,
            error_message=str(exc),
            path=request.url.path,
            exc_info=True
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An internal server error occurred"
                },
                "metadata": {
                    "timestamp": time.time(),
                    "path": request.url.path
                }
            }
        )


def setup_routes(app: FastAPI) -> None:
    """Set up routes for the FastAPI application."""
    
    # Include authentication routes
    app.include_router(auth_router)
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        config = get_config()
        return {
            "success": True,
            "data": {
                "name": config.system.name,
                "version": config.system.version,
                "environment": config.system.environment,
                "message": "Welcome to Akasha - Multimodal RAG System"
            },
            "metadata": {
                "timestamp": time.time()
            }
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        config = get_config()
        
        # Basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": config.system.version,
            "environment": config.system.environment,
            "checks": {
                "api": "pass",
                "config": "pass",
                "logging": "pass"
            }
        }
        
        # TODO: Add checks for other services (database, models, etc.)
        
        return {
            "success": True,
            "data": health_status
        }
    
    @app.get("/status")
    async def system_status():
        """Detailed system status endpoint."""
        config = get_config()
        
        try:
            import psutil
            
            # Get system resource usage
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')
            
            status = {
                "status": "healthy",
                "version": config.system.version,
                "uptime": time.time(),  # TODO: Calculate actual uptime
                "timestamp": time.time(),
                "components": {
                    "api_server": "healthy",
                    "configuration": "healthy",
                    "logging": "healthy",
                    # TODO: Add other components
                },
                "resources": {
                    "cpu_usage_percent": cpu_percent,
                    "memory_usage_percent": memory.percent,
                    "memory_available_gb": round(memory.available / 1024 / 1024 / 1024, 2),
                    "memory_total_gb": round(memory.total / 1024 / 1024 / 1024, 2),
                    "disk_usage_percent": round((disk.used / disk.total) * 100, 2),
                    "disk_free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
                },
                "configuration": {
                    "max_memory_gb": config.system.max_memory_gb,
                    "debug_mode": config.system.debug,
                    "llm_backend": config.llm.backend,
                    "llm_model": config.llm.model_name,
                    "vector_store": config.vector_store.backend,
                }
            }
            
        except ImportError:
            # psutil not available, return basic status
            status = {
                "status": "healthy",
                "version": config.system.version,
                "timestamp": time.time(),
                "components": {
                    "api_server": "healthy",
                    "configuration": "healthy",
                    "logging": "healthy",
                },
                "message": "Resource monitoring unavailable (psutil not installed)"
            }
        
        return {
            "success": True,
            "data": status
        }
    
    @app.get("/config")
    async def get_configuration():
        """Get system configuration (non-sensitive parts)."""
        config = get_config()
        
        # Return safe configuration without sensitive data
        safe_config = {
            "system": {
                "name": config.system.name,
                "version": config.system.version,
                "environment": config.system.environment,
                "debug": config.system.debug,
                "max_memory_gb": config.system.max_memory_gb,
            },
            "api": {
                "host": config.api.host,
                "port": config.api.port,
                "max_request_size": config.api.max_request_size,
            },
            "llm": {
                "backend": config.llm.backend,
                "model_name": config.llm.model_name,
                "quantization_bits": config.llm.quantization_bits,
                "max_tokens": config.llm.max_tokens,
                "memory_limit_gb": config.llm.memory_limit_gb,
            },
            "embedding": {
                "model": config.embedding.model,
                "dimensions": config.embedding.dimensions,
                "batch_size": config.embedding.batch_size,
                "memory_limit_gb": config.embedding.memory_limit_gb,
            },
            "vector_store": {
                "backend": config.vector_store.backend,
                "collection_name": config.vector_store.collection_name,
                "memory_limit_gb": config.vector_store.memory_limit_gb,
            },
            "features": {
                "multimodal_search": True,
                "streaming_chat": True,
                "plugin_support": True,
                "graph_rag": False,  # TODO: Update when implemented
            }
        }
        
        return {
            "success": True,
            "data": safe_config
        }


# Create the FastAPI application instance
app = create_app()


def main():
    """Run the application."""
    config = get_config()
    
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload and config.system.debug,
        workers=config.api.workers if not config.api.reload else 1,
        log_config=None,  # We handle logging ourselves
        access_log=False,  # We handle access logging ourselves
    )


if __name__ == "__main__":
    main()