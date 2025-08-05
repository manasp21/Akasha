"""
Security middleware for Akasha API.

Provides rate limiting, security headers, input validation, and other security features.
"""

import time
import json
from collections import defaultdict, deque
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta, timezone

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.config import get_config
from ..core.logging import get_logger

logger = get_logger(__name__)

class RateLimiter:
    """
    Token bucket rate limiter with sliding window.
    """
    
    def __init__(self):
        self.clients: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "tokens": 0,
            "last_update": time.time(),
            "requests": deque()
        })
    
    def is_allowed(self, client_id: str, max_requests: int, window_seconds: int = 60) -> bool:
        """
        Check if client is allowed to make a request.
        
        Args:
            client_id: Client identifier (IP, user ID, etc.)
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
            
        Returns:
            True if request is allowed, False otherwise
        """
        now = time.time()
        client_data = self.clients[client_id]
        
        # Clean old requests outside the window
        window_start = now - window_seconds
        while client_data["requests"] and client_data["requests"][0] < window_start:
            client_data["requests"].popleft()
        
        # Check if under limit
        if len(client_data["requests"]) < max_requests:
            client_data["requests"].append(now)
            return True
        
        return False
    
    def get_reset_time(self, client_id: str, window_seconds: int = 60) -> float:
        """Get time until rate limit resets."""
        client_data = self.clients[client_id]
        if not client_data["requests"]:
            return 0
        
        oldest_request = client_data["requests"][0]
        return oldest_request + window_seconds - time.time()

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to responses.
    """
    
    def __init__(self, app, config: Optional[Any] = None):
        super().__init__(app)
        self.config = config or get_config()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        if self.config.auth.enable_security_headers:
            # Content Security Policy
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: blob:; "
                "font-src 'self' data:; "
                "connect-src 'self'; "
                "frame-ancestors 'none'"
            )
            
            # Security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Permissions-Policy"] = (
                "geolocation=(), microphone=(), camera=(), "
                "payment=(), usb=(), magnetometer=(), gyroscope=()"
            )
            
            # HSTS (only if HTTPS)
            if request.url.scheme == "https":
                response.headers["Strict-Transport-Security"] = (
                    "max-age=31536000; includeSubDomains; preload"
                )
        
        return response

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API rate limiting.
    """
    
    def __init__(self, app, config: Optional[Any] = None):
        super().__init__(app)
        self.config = config or get_config()
        self.rate_limiter = RateLimiter()
        self.login_limiter = RateLimiter()
    
    def get_client_identifier(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get user ID from token first
        if hasattr(request.state, "user_id"):
            return f"user:{request.state.user_id}"
        
        # Fall back to IP address
        client_ip = request.client.host
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.config.auth.enable_rate_limiting:
            return await call_next(request)
        
        client_id = self.get_client_identifier(request)
        path = request.url.path
        
        # Different limits for different endpoints
        if path.startswith("/auth/login"):
            # Stricter limits for login attempts
            max_requests = self.config.auth.login_attempts_per_minute
            if not self.login_limiter.is_allowed(client_id, max_requests, 60):
                reset_time = self.login_limiter.get_reset_time(client_id, 60)
                
                logger.warning(
                    "Login rate limit exceeded",
                    client_id=client_id,
                    path=path,
                    reset_in=reset_time
                )
                
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "success": False,
                        "error": {
                            "code": "RATE_LIMIT_EXCEEDED",
                            "message": "Too many login attempts. Please try again later.",
                            "retry_after": int(reset_time) + 1
                        },
                        "metadata": {
                            "timestamp": time.time(),
                            "path": path
                        }
                    },
                    headers={
                        "Retry-After": str(int(reset_time) + 1),
                        "X-RateLimit-Limit": str(max_requests),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(time.time() + reset_time))
                    }
                )
        
        else:
            # General API rate limiting
            max_requests = self.config.auth.api_requests_per_minute
            if not self.rate_limiter.is_allowed(client_id, max_requests, 60):
                reset_time = self.rate_limiter.get_reset_time(client_id, 60)
                
                logger.warning(
                    "API rate limit exceeded",
                    client_id=client_id,
                    path=path,
                    reset_in=reset_time
                )
                
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "success": False,
                        "error": {
                            "code": "RATE_LIMIT_EXCEEDED",
                            "message": "Too many requests. Please try again later.",
                            "retry_after": int(reset_time) + 1
                        },
                        "metadata": {
                            "timestamp": time.time(),
                            "path": path
                        }
                    },
                    headers={
                        "Retry-After": str(int(reset_time) + 1),
                        "X-RateLimit-Limit": str(max_requests),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(time.time() + reset_time))
                    }
                )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        if response.status_code < 400:
            response.headers["X-RateLimit-Limit"] = str(self.config.auth.api_requests_per_minute)
        
        return response

class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for input validation and sanitization.
    """
    
    def __init__(self, app, config: Optional[Any] = None):
        super().__init__(app)
        self.config = config or get_config()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > self.config.security.max_request_size_mb:
                logger.warning(
                    "Request size limit exceeded",
                    size_mb=size_mb,
                    limit_mb=self.config.security.max_request_size_mb,
                    path=request.url.path
                )
                
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "success": False,
                        "error": {
                            "code": "REQUEST_TOO_LARGE",
                            "message": f"Request size exceeds limit of {self.config.security.max_request_size_mb}MB"
                        },
                        "metadata": {
                            "timestamp": time.time(),
                            "path": request.url.path
                        }
                    }
                )
        
        # Basic header validation
        user_agent = request.headers.get("user-agent", "")
        if len(user_agent) > 500:  # Suspiciously long user agent
            logger.warning(
                "Suspicious user agent detected",
                user_agent_length=len(user_agent),
                path=request.url.path,
                client_ip=request.client.host
            )
        
        # Check for common attack patterns in query parameters
        query_string = str(request.url.query)
        suspicious_patterns = [
            "script>", "<iframe", "javascript:", "vbscript:",
            "onload=", "onerror=", "eval(", "expression("
        ]
        
        for pattern in suspicious_patterns:
            if pattern.lower() in query_string.lower():
                logger.warning(
                    "Suspicious query parameter detected",
                    pattern=pattern,
                    query=query_string[:200],  # Log first 200 chars
                    path=request.url.path,
                    client_ip=request.client.host
                )
                
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "success": False,
                        "error": {
                            "code": "INVALID_REQUEST",
                            "message": "Request contains invalid characters"
                        },
                        "metadata": {
                            "timestamp": time.time(),
                            "path": request.url.path
                        }
                    }
                )
        
        return await call_next(request)

class SecurityAuditMiddleware(BaseHTTPMiddleware):
    """
    Middleware for security auditing and logging.
    """
    
    def __init__(self, app, config: Optional[Any] = None):
        super().__init__(app)
        self.config = config or get_config()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log security-relevant requests
        security_paths = ["/auth/", "/admin/", "/config", "/status"]
        is_security_endpoint = any(request.url.path.startswith(path) for path in security_paths)
        
        if is_security_endpoint:
            logger.info(
                "Security endpoint accessed",
                method=request.method,
                path=request.url.path,
                client_ip=request.client.host,
                user_agent=request.headers.get("user-agent", ""),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        
        # Log failed authentication attempts
        if request.url.path.startswith("/auth/") and response.status_code in [401, 403]:
            logger.warning(
                "Authentication failure",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                client_ip=request.client.host,
                duration=duration,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        # Log suspicious activity
        if response.status_code in [400, 429] and duration < 0.01:  # Very fast error responses
            logger.warning(
                "Potential automated attack",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                client_ip=request.client.host,
                duration=duration
            )
        
        return response

def setup_security_middleware(app, config: Optional[Any] = None):
    """
    Set up all security middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        config: Configuration object
    """
    if config is None:
        config = get_config()
    
    # Add middleware in reverse order (last added = first executed)
    app.add_middleware(SecurityAuditMiddleware, config=config)
    app.add_middleware(InputValidationMiddleware, config=config)
    app.add_middleware(RateLimitingMiddleware, config=config)
    app.add_middleware(SecurityHeadersMiddleware, config=config)
    
    logger.info("Security middleware configured")