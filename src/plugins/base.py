"""
Base plugin interfaces and abstract classes for the Akasha plugin system.

This module defines the core interfaces that all plugins must implement,
providing a standardized way to extend Akasha's functionality while
maintaining security and isolation.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from ..core.exceptions import PluginError


class PluginType(str, Enum):
    """Enumeration of plugin types."""
    DOCUMENT_PROCESSOR = "document_processor"
    EMBEDDING_MODEL = "embedding_model"
    VECTOR_STORE = "vector_store"
    LLM_BACKEND = "llm_backend"
    RETRIEVER = "retriever"
    RERANKER = "reranker"
    PREPROCESSOR = "preprocessor"
    POSTPROCESSOR = "postprocessor"
    UI_COMPONENT = "ui_component"
    MIDDLEWARE = "middleware"
    CUSTOM = "custom"


class PluginStatus(str, Enum):
    """Plugin status enumeration."""
    INACTIVE = "inactive"
    LOADING = "loading"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginCapability:
    """Represents a capability that a plugin provides."""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    parameters: Dict[str, Any]


class PluginMetadata(BaseModel):
    """Plugin metadata and configuration."""
    
    # Basic information
    name: str = Field(..., description="Plugin name")
    version: str = Field(..., description="Plugin version")
    description: str = Field(..., description="Plugin description")
    author: str = Field(..., description="Plugin author")
    
    # Plugin classification
    plugin_type: PluginType = Field(..., description="Type of plugin")
    capabilities: List[PluginCapability] = Field(default=[], description="Plugin capabilities")
    
    # Requirements and compatibility
    akasha_version: str = Field(..., description="Required Akasha version")
    python_version: str = Field(default=">=3.11", description="Required Python version")
    dependencies: List[str] = Field(default=[], description="Required dependencies")
    
    # Configuration
    config_schema: Dict[str, Any] = Field(default={}, description="Configuration schema")
    default_config: Dict[str, Any] = Field(default={}, description="Default configuration")
    
    # Resources and limits
    memory_limit_mb: Optional[int] = Field(default=None, description="Memory limit in MB")
    cpu_limit_percent: Optional[float] = Field(default=None, description="CPU limit percentage")
    timeout_seconds: int = Field(default=30, description="Execution timeout")
    
    # Security and permissions
    sandbox_enabled: bool = Field(default=True, description="Enable sandboxing")
    required_permissions: List[str] = Field(default=[], description="Required permissions")
    network_access: bool = Field(default=False, description="Requires network access")
    file_access: List[str] = Field(default=[], description="Required file access paths")


class PluginInterface(ABC):
    """
    Base interface that all plugins must implement.
    
    This interface provides the fundamental methods for plugin lifecycle
    management and execution.
    """
    
    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self.status = PluginStatus.INACTIVE
        self.config: Dict[str, Any] = {}
        self.logger = None  # Will be set by plugin manager
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the plugin with configuration.
        
        Args:
            config: Plugin configuration dictionary
            
        Raises:
            PluginError: If initialization fails
        """
        pass
    
    @abstractmethod
    async def process(self, input_data: Any, **kwargs) -> Any:
        """
        Main processing method for the plugin.
        
        Args:
            input_data: Input data to process
            **kwargs: Additional keyword arguments
            
        Returns:
            Processed output data
            
        Raises:
            PluginError: If processing fails
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up plugin resources.
        
        Called when the plugin is being unloaded or the system is shutting down.
        """
        pass
    
    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate plugin configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            PluginError: If configuration is invalid
        """
        # Default implementation - override in subclasses for custom validation
        return True
    
    async def get_capabilities(self) -> List[PluginCapability]:
        """
        Get plugin capabilities.
        
        Returns:
            List of plugin capabilities
        """
        return self.metadata.capabilities
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform plugin health check.
        
        Returns:
            Dictionary with health status information
        """
        return {
            "status": self.status.value,
            "name": self.metadata.name,
            "version": self.metadata.version,
            "healthy": self.status == PluginStatus.ACTIVE
        }
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return self.metadata


class DocumentProcessorInterface(PluginInterface):
    """Interface for document processing plugins."""
    
    @abstractmethod
    async def extract_content(self, document_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Extract content from a document.
        
        Args:
            document_bytes: Raw document bytes
            filename: Original filename
            
        Returns:
            Dictionary containing extracted content, metadata, and segments
        """
        pass
    
    @abstractmethod
    async def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions (e.g., ['.pdf', '.docx'])
        """
        pass


class EmbeddingModelInterface(PluginInterface):
    """Interface for embedding model plugins."""
    
    @abstractmethod
    async def embed_text(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text inputs.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    async def embed_image(self, images: List[bytes]) -> List[List[float]]:
        """
        Generate embeddings for image inputs.
        
        Args:
            images: List of image bytes to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    async def embed_multimodal(self, content: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Generate embeddings for multimodal content.
        
        Args:
            content: List of content dictionaries with 'type' and 'data' keys
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    async def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        pass


class VectorStoreInterface(PluginInterface):
    """Interface for vector store plugins."""
    
    @abstractmethod
    async def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        """
        Create a new vector collection.
        
        Args:
            name: Collection name
            dimension: Vector dimension
            **kwargs: Additional parameters
            
        Returns:
            True if collection created successfully
        """
        pass
    
    @abstractmethod
    async def insert_vectors(
        self, 
        collection: str, 
        vectors: List[List[float]], 
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Insert vectors into a collection.
        
        Args:
            collection: Collection name
            vectors: List of vectors to insert
            metadata: List of metadata dictionaries
            ids: Optional list of IDs
            
        Returns:
            List of inserted vector IDs
        """
        pass
    
    @abstractmethod
    async def search_vectors(
        self, 
        collection: str, 
        query_vector: List[float], 
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            collection: Collection name
            query_vector: Query vector
            limit: Maximum number of results
            filters: Optional metadata filters
            
        Returns:
            List of search results with scores and metadata
        """
        pass
    
    @abstractmethod
    async def delete_vectors(self, collection: str, ids: List[str]) -> bool:
        """
        Delete vectors from a collection.
        
        Args:
            collection: Collection name
            ids: List of vector IDs to delete
            
        Returns:
            True if deletion successful
        """
        pass


class LLMBackendInterface(PluginInterface):
    """Interface for LLM backend plugins."""
    
    @abstractmethod
    async def load_model(self, model_path: str, **kwargs) -> bool:
        """
        Load a language model.
        
        Args:
            model_path: Path to model files
            **kwargs: Additional loading parameters
            
        Returns:
            True if model loaded successfully
        """
        pass
    
    @abstractmethod
    async def generate_text(
        self, 
        prompt: str, 
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    async def generate_text_stream(
        self, 
        prompt: str, 
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate streaming text completion.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            **kwargs: Additional generation parameters
            
        Yields:
            Generated text chunks
        """
        pass
    
    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        pass


class RetrieverInterface(PluginInterface):
    """Interface for retrieval plugins."""
    
    @abstractmethod
    async def retrieve(
        self, 
        query: str, 
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filters
            **kwargs: Additional parameters
            
        Returns:
            List of retrieved documents with scores
        """
        pass


class RerankerInterface(PluginInterface):
    """Interface for reranking plugins."""
    
    @abstractmethod
    async def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            **kwargs: Additional parameters
            
        Returns:
            Reranked list of documents
        """
        pass


# Registry of plugin interfaces by type
PLUGIN_INTERFACES = {
    PluginType.DOCUMENT_PROCESSOR: DocumentProcessorInterface,
    PluginType.EMBEDDING_MODEL: EmbeddingModelInterface,
    PluginType.VECTOR_STORE: VectorStoreInterface,
    PluginType.LLM_BACKEND: LLMBackendInterface,
    PluginType.RETRIEVER: RetrieverInterface,
    PluginType.RERANKER: RerankerInterface,
    PluginType.CUSTOM: PluginInterface,
}


def get_plugin_interface(plugin_type: PluginType) -> Type[PluginInterface]:
    """
    Get the appropriate interface class for a plugin type.
    
    Args:
        plugin_type: Type of plugin
        
    Returns:
        Plugin interface class
        
    Raises:
        PluginError: If plugin type not found
    """
    interface_class = PLUGIN_INTERFACES.get(plugin_type)
    if interface_class is None:
        raise PluginError(f"No interface defined for plugin type: {plugin_type}")
    
    return interface_class