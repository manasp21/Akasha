"""
Custom exceptions for the Akasha system.

This module defines a hierarchy of custom exceptions that provide
structured error handling throughout the application with proper
context and error codes.
"""

from typing import Any, Dict, Optional


class AkashaError(Exception):
    """
    Base exception class for all Akasha-specific errors.
    
    Provides structured error information including error codes,
    context, and user-friendly messages.
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        details: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details
        self.context = context or {}
        self.user_message = user_message or message
        
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error_code": self.error_code,
            "message": self.user_message,
            "details": self.details,
            "context": self.context
        }


# Configuration Errors
class ConfigurationError(AkashaError):
    """Raised when there are configuration-related issues."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "CONFIGURATION_ERROR")
        super().__init__(message, **kwargs)


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration validation fails."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "INVALID_CONFIGURATION")
        super().__init__(message, **kwargs)


# Document Processing Errors
class DocumentProcessingError(AkashaError):
    """Base class for document processing errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "DOCUMENT_PROCESSING_ERROR")
        super().__init__(message, **kwargs)


class DocumentNotFoundError(DocumentProcessingError):
    """Raised when a requested document cannot be found."""
    
    def __init__(self, document_id: str, **kwargs):
        message = f"Document not found: {document_id}"
        kwargs.setdefault("error_code", "DOCUMENT_NOT_FOUND")
        kwargs.setdefault("context", {}).update({"document_id": document_id})
        super().__init__(message, **kwargs)


class DocumentFormatError(DocumentProcessingError):
    """Raised when document format is not supported or invalid."""
    
    def __init__(self, message: str, file_type: str = None, **kwargs):
        kwargs.setdefault("error_code", "DOCUMENT_FORMAT_ERROR")
        if file_type:
            kwargs.setdefault("context", {}).update({"file_type": file_type})
        super().__init__(message, **kwargs)


class DocumentTooLargeError(DocumentProcessingError):
    """Raised when document exceeds size limits."""
    
    def __init__(self, file_size: int, max_size: int, **kwargs):
        message = f"Document size ({file_size} bytes) exceeds maximum allowed size ({max_size} bytes)"
        kwargs.setdefault("error_code", "DOCUMENT_TOO_LARGE")
        kwargs.setdefault("context", {}).update({
            "file_size": file_size,
            "max_size": max_size
        })
        super().__init__(message, **kwargs)


class ExtractionError(DocumentProcessingError):
    """Raised when content extraction fails."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "EXTRACTION_ERROR")
        super().__init__(message, **kwargs)


# Model and Embedding Errors
class ModelError(AkashaError):
    """Base class for model-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "MODEL_ERROR")
        super().__init__(message, **kwargs)


class ModelNotFoundError(ModelError):
    """Raised when a requested model cannot be found."""
    
    def __init__(self, model_name: str, **kwargs):
        message = f"Model not found: {model_name}"
        kwargs.setdefault("error_code", "MODEL_NOT_FOUND")
        kwargs.setdefault("context", {}).update({"model_name": model_name})
        super().__init__(message, **kwargs)


class ModelLoadError(ModelError):
    """Raised when model loading fails."""
    
    def __init__(self, model_name: str, reason: str = None, **kwargs):
        message = f"Failed to load model: {model_name}"
        if reason:
            message += f" - {reason}"
        kwargs.setdefault("error_code", "MODEL_LOAD_ERROR")
        kwargs.setdefault("context", {}).update({"model_name": model_name})
        if reason:
            kwargs["details"] = reason
        super().__init__(message, **kwargs)


class InferenceError(ModelError):
    """Raised when model inference fails."""
    
    def __init__(self, message: str, model_name: str = None, **kwargs):
        kwargs.setdefault("error_code", "INFERENCE_ERROR")
        if model_name:
            kwargs.setdefault("context", {}).update({"model_name": model_name})
        super().__init__(message, **kwargs)


class EmbeddingError(ModelError):
    """Raised when embedding generation fails."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "EMBEDDING_ERROR")
        super().__init__(message, **kwargs)


# Vector Store Errors
class VectorStoreError(AkashaError):
    """Base class for vector store errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "VECTOR_STORE_ERROR")
        super().__init__(message, **kwargs)


class VectorStoreConnectionError(VectorStoreError):
    """Raised when vector store connection fails."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "VECTOR_STORE_CONNECTION_ERROR")
        super().__init__(message, **kwargs)


class CollectionNotFoundError(VectorStoreError):
    """Raised when a requested collection cannot be found."""
    
    def __init__(self, collection_name: str, **kwargs):
        message = f"Collection not found: {collection_name}"
        kwargs.setdefault("error_code", "COLLECTION_NOT_FOUND")
        kwargs.setdefault("context", {}).update({"collection_name": collection_name})
        super().__init__(message, **kwargs)


class SearchError(VectorStoreError):
    """Raised when vector search fails."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "SEARCH_ERROR")
        super().__init__(message, **kwargs)


# Plugin Errors
class PluginError(AkashaError):
    """Base class for plugin-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "PLUGIN_ERROR")
        super().__init__(message, **kwargs)


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin cannot be found."""
    
    def __init__(self, plugin_name: str, **kwargs):
        message = f"Plugin not found: {plugin_name}"
        kwargs.setdefault("error_code", "PLUGIN_NOT_FOUND")
        kwargs.setdefault("context", {}).update({"plugin_name": plugin_name})
        super().__init__(message, **kwargs)


class PluginLoadError(PluginError):
    """Raised when plugin loading fails."""
    
    def __init__(self, plugin_name: str, reason: str = None, **kwargs):
        message = f"Failed to load plugin: {plugin_name}"
        if reason:
            message += f" - {reason}"
        kwargs.setdefault("error_code", "PLUGIN_LOAD_ERROR")
        kwargs.setdefault("context", {}).update({"plugin_name": plugin_name})
        if reason:
            kwargs["details"] = reason
        super().__init__(message, **kwargs)


class PluginExecutionError(PluginError):
    """Raised when plugin execution fails."""
    
    def __init__(self, plugin_name: str, operation: str, reason: str = None, **kwargs):
        message = f"Plugin execution failed: {plugin_name}.{operation}"
        if reason:
            message += f" - {reason}"
        kwargs.setdefault("error_code", "PLUGIN_EXECUTION_ERROR")
        kwargs.setdefault("context", {}).update({
            "plugin_name": plugin_name,
            "operation": operation
        })
        if reason:
            kwargs["details"] = reason
        super().__init__(message, **kwargs)


class PluginTimeoutError(PluginError):
    """Raised when plugin execution times out."""
    
    def __init__(self, plugin_name: str, timeout_seconds: float, **kwargs):
        message = f"Plugin execution timed out: {plugin_name} (timeout: {timeout_seconds}s)"
        kwargs.setdefault("error_code", "PLUGIN_TIMEOUT")
        kwargs.setdefault("context", {}).update({
            "plugin_name": plugin_name,
            "timeout_seconds": timeout_seconds
        })
        super().__init__(message, **kwargs)


# API and Authentication Errors
class APIError(AkashaError):
    """Base class for API-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "API_ERROR")
        super().__init__(message, **kwargs)


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication required", **kwargs):
        kwargs.setdefault("error_code", "AUTHENTICATION_ERROR")
        kwargs.setdefault("user_message", "Authentication required")
        super().__init__(message, **kwargs)


class AuthorizationError(APIError):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Access denied", **kwargs):
        kwargs.setdefault("error_code", "AUTHORIZATION_ERROR")
        kwargs.setdefault("user_message", "Access denied")
        super().__init__(message, **kwargs)


class ValidationError(APIError):
    """Raised when request validation fails."""
    
    def __init__(self, message: str, field_errors: Optional[Dict[str, Any]] = None, **kwargs):
        kwargs.setdefault("error_code", "VALIDATION_ERROR")
        if field_errors:
            kwargs.setdefault("context", {}).update({"field_errors": field_errors})
        super().__init__(message, **kwargs)


class RateLimitError(APIError):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, retry_after: int = None, **kwargs):
        message = "Rate limit exceeded"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        kwargs.setdefault("error_code", "RATE_LIMIT_EXCEEDED")
        if retry_after:
            kwargs.setdefault("context", {}).update({"retry_after": retry_after})
        super().__init__(message, **kwargs)


# Resource and Memory Errors
class ResourceError(AkashaError):
    """Base class for resource-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "RESOURCE_ERROR")
        super().__init__(message, **kwargs)


class InsufficientMemoryError(ResourceError):
    """Raised when insufficient memory is available."""
    
    def __init__(self, required_mb: float, available_mb: float = None, **kwargs):
        message = f"Insufficient memory: requires {required_mb}MB"
        if available_mb:
            message += f", available {available_mb}MB"
        kwargs.setdefault("error_code", "INSUFFICIENT_MEMORY")
        kwargs.setdefault("context", {}).update({"required_mb": required_mb})
        if available_mb:
            kwargs["context"]["available_mb"] = available_mb
        super().__init__(message, **kwargs)


class StorageError(ResourceError):
    """Raised when storage operations fail."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "STORAGE_ERROR")
        super().__init__(message, **kwargs)


class InsufficientStorageError(StorageError):
    """Raised when insufficient storage space is available."""
    
    def __init__(self, required_mb: float, available_mb: float = None, **kwargs):
        message = f"Insufficient storage space: requires {required_mb}MB"
        if available_mb:
            message += f", available {available_mb}MB"
        kwargs.setdefault("error_code", "INSUFFICIENT_STORAGE")
        kwargs.setdefault("context", {}).update({"required_mb": required_mb})
        if available_mb:
            kwargs["context"]["available_mb"] = available_mb
        super().__init__(message, **kwargs)


# System and Internal Errors
class SystemError(AkashaError):
    """Raised for internal system errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "SYSTEM_ERROR")
        kwargs.setdefault("user_message", "An internal system error occurred")
        super().__init__(message, **kwargs)


class ServiceUnavailableError(SystemError):
    """Raised when a service is temporarily unavailable."""
    
    def __init__(self, service_name: str, **kwargs):
        message = f"Service unavailable: {service_name}"
        kwargs.setdefault("error_code", "SERVICE_UNAVAILABLE")
        kwargs.setdefault("context", {}).update({"service_name": service_name})
        kwargs.setdefault("user_message", "Service is temporarily unavailable")
        super().__init__(message, **kwargs)


class DependencyError(SystemError):
    """Raised when a required dependency is missing or incompatible."""
    
    def __init__(self, dependency: str, reason: str = None, **kwargs):
        message = f"Dependency error: {dependency}"
        if reason:
            message += f" - {reason}"
        kwargs.setdefault("error_code", "DEPENDENCY_ERROR")
        kwargs.setdefault("context", {}).update({"dependency": dependency})
        if reason:
            kwargs["details"] = reason
        super().__init__(message, **kwargs)