"""
Vector storage system for Akasha RAG using ChromaDB.

This module provides vector database operations using ChromaDB for storing and 
retrieving document embeddings with metadata support and efficient similarity search.
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

from pydantic import BaseModel, Field

from ..core.logging import get_logger, PerformanceLogger
from ..core.exceptions import AkashaError
from .ingestion import DocumentChunk, DocumentMetadata


class VectorStorageBackend(str, Enum):
    """Supported vector storage backends."""
    CHROMADB = "chromadb"
    QDRANT = "qdrant"
    FAISS = "faiss"
    PINECONE = "pinecone"


class DistanceMetric(str, Enum):
    """Supported distance metrics for vector similarity."""
    COSINE = "cosine"
    EUCLIDEAN = "l2"
    MANHATTAN = "l1"
    DOT_PRODUCT = "ip"


class StorageConfig(BaseModel):
    """Configuration for vector storage."""
    backend: VectorStorageBackend = Field(default=VectorStorageBackend.CHROMADB)
    persist_directory: Optional[str] = Field(default=None, description="Directory for persistent storage")
    collection_name: str = Field(default="akasha_documents", description="Name of the collection")
    distance_metric: DistanceMetric = Field(default=DistanceMetric.COSINE)
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=8000, description="Database port")
    api_key: Optional[str] = Field(default=None, description="API key for cloud services")
    embedding_dimensions: int = Field(default=384, description="Embedding vector dimensions")
    max_batch_size: int = Field(default=1000, description="Maximum batch size for operations")
    enable_hnsw: bool = Field(default=True, description="Enable HNSW indexing for fast retrieval")
    hnsw_ef_construction: int = Field(default=200, description="HNSW ef_construction parameter")
    hnsw_m: int = Field(default=16, description="HNSW M parameter")


class SearchResult(BaseModel):
    """Search result with chunk and similarity score."""
    chunk: DocumentChunk = Field(..., description="Document chunk")
    score: float = Field(..., description="Similarity score")
    distance: float = Field(..., description="Distance metric value") 
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class VectorStorageProvider(ABC):
    """Abstract base class for vector storage providers."""
    
    def __init__(self, config: StorageConfig, logger=None):
        self.config = config
        self.logger = logger or get_logger(__name__)
        self.client = None
        self.collection = None
        self.initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage provider."""
        pass
    
    @abstractmethod
    async def create_collection(self, name: str, force_recreate: bool = False) -> None:
        """Create a collection."""
        pass
    
    @abstractmethod
    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to storage."""
        pass
    
    @abstractmethod
    async def search_similar(self, query_embedding: List[float], 
                           top_k: int = 10, 
                           filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a specific chunk by ID."""
        pass
    
    @abstractmethod
    async def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete chunks by IDs."""
        pass
    
    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        pass


class ChromaDBProvider(VectorStorageProvider):
    """ChromaDB vector storage provider."""
    
    def __init__(self, config: StorageConfig, logger=None):
        super().__init__(config, logger)
        self._chromadb = None
    
    async def _import_chromadb(self):
        """Lazy import ChromaDB."""
        if self._chromadb is None:
            try:
                import chromadb
                from chromadb.config import Settings
                self._chromadb = chromadb
                self._settings = Settings
            except ImportError:
                raise AkashaError("chromadb not installed. Install with: pip install chromadb")
    
    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        if self.initialized:
            return
        
        await self._import_chromadb()
        
        async with PerformanceLogger("chromadb_initialization", self.logger):
            # Configure ChromaDB settings
            settings = self._settings()
            
            if self.config.persist_directory:
                persist_path = Path(self.config.persist_directory)
                persist_path.mkdir(parents=True, exist_ok=True)
                settings = self._settings(
                    persist_directory=str(persist_path),
                    is_persistent=True
                )
            
            # Create client
            def _create_client():
                return self._chromadb.Client(settings)
            
            self.client = await asyncio.get_event_loop().run_in_executor(None, _create_client)
            
            # Create or get collection
            await self.create_collection(self.config.collection_name)
            
            self.initialized = True
            self.logger.info(
                "ChromaDB initialized successfully",
                collection_name=self.config.collection_name,
                persist_directory=self.config.persist_directory
            )
    
    async def create_collection(self, name: str, force_recreate: bool = False) -> None:
        """Create or get ChromaDB collection."""
        if not self.client:
            await self.initialize()
        
        def _create_collection():
            try:
                if force_recreate:
                    try:
                        self.client.delete_collection(name)
                        self.logger.info(f"Deleted existing collection: {name}")
                    except Exception:
                        pass  # Collection might not exist
                
                # Map distance metric to ChromaDB format
                distance_function = {
                    DistanceMetric.COSINE: "cosine",
                    DistanceMetric.EUCLIDEAN: "l2", 
                    DistanceMetric.MANHATTAN: "l1",
                    DistanceMetric.DOT_PRODUCT: "ip"
                }.get(self.config.distance_metric, "cosine")
                
                metadata = {
                    "hnsw:space": distance_function,
                    "hnsw:construction_ef": self.config.hnsw_ef_construction,
                    "hnsw:M": self.config.hnsw_m,
                }
                
                collection = self.client.get_or_create_collection(
                    name=name,
                    metadata=metadata
                )
                return collection
                
            except Exception as e:
                raise AkashaError(f"Failed to create ChromaDB collection: {e}")
        
        self.collection = await asyncio.get_event_loop().run_in_executor(None, _create_collection)
        
        self.logger.info(
            "ChromaDB collection ready",
            collection_name=name,
            distance_metric=self.config.distance_metric
        )
    
    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to ChromaDB."""
        if not self.initialized:
            await self.initialize()
        
        if not chunks:
            return
        
        async with PerformanceLogger(f"chromadb_add_chunks:count_{len(chunks)}", self.logger):
            # Process chunks in batches
            batch_size = min(self.config.max_batch_size, 1000)  # ChromaDB batch limit
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Prepare data for ChromaDB
                ids = []
                embeddings = []
                documents = []
                metadatas = []
                
                for chunk in batch:
                    if not chunk.embedding:
                        raise AkashaError(f"Chunk {chunk.id} has no embedding")
                    
                    ids.append(chunk.id)
                    embeddings.append(chunk.embedding)
                    documents.append(chunk.content)
                    
                    # Prepare metadata (ChromaDB requires flat dictionary)
                    metadata = {
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "start_offset": chunk.start_offset,
                        "end_offset": chunk.end_offset,
                        "content_length": len(chunk.content),
                        "created_at": time.time()
                    }
                    
                    # Add custom metadata (flatten nested structures)
                    for key, value in chunk.metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            metadata[f"custom_{key}"] = value
                        else:
                            metadata[f"custom_{key}"] = json.dumps(value)
                    
                    metadatas.append(metadata)
                
                # Add to ChromaDB
                def _add_batch():
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas
                    )
                
                await asyncio.get_event_loop().run_in_executor(None, _add_batch)
                
                self.logger.debug(
                    "Added chunk batch to ChromaDB",
                    batch_size=len(batch),
                    total_processed=min(i + batch_size, len(chunks))
                )
            
            self.logger.info(
                "Successfully added chunks to ChromaDB",
                chunk_count=len(chunks),
                collection_name=self.config.collection_name
            )
    
    async def search_similar(self, query_embedding: List[float], 
                           top_k: int = 10, 
                           filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Search for similar vectors in ChromaDB."""
        if not self.initialized:
            await self.initialize()
        
        if not query_embedding:
            return []
        
        async with PerformanceLogger(f"chromadb_search:top_k_{top_k}", self.logger):
            # Prepare where filters for ChromaDB
            where_filter = None
            if filters:
                where_filter = {}
                for key, value in filters.items():
                    if isinstance(value, list):
                        where_filter[key] = {"$in": value}
                    elif isinstance(value, dict):
                        # Handle range queries
                        if "min" in value or "max" in value:
                            range_filter = {}
                            if "min" in value:
                                range_filter["$gte"] = value["min"]
                            if "max" in value:
                                range_filter["$lte"] = value["max"]
                            where_filter[key] = range_filter
                        else:
                            where_filter[key] = value
                    else:
                        where_filter[key] = {"$eq": value}
            
            def _search():
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"]
                )
                return results
            
            raw_results = await asyncio.get_event_loop().run_in_executor(None, _search)
            
            # Convert ChromaDB results to SearchResult objects
            search_results = []
            
            if raw_results and raw_results["documents"]:
                documents = raw_results["documents"][0]
                metadatas = raw_results["metadatas"][0]
                distances = raw_results["distances"][0]
                ids = raw_results["ids"][0]
                
                for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
                    # Reconstruct DocumentChunk
                    chunk_metadata = {}
                    custom_metadata = {}
                    
                    for key, value in metadata.items():
                        if key.startswith("custom_"):
                            custom_key = key[7:]  # Remove "custom_" prefix
                            try:
                                # Try to parse JSON for complex values
                                custom_metadata[custom_key] = json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                custom_metadata[custom_key] = value
                        else:
                            chunk_metadata[key] = value
                    
                    chunk = DocumentChunk(
                        id=doc_id,
                        content=document,
                        document_id=chunk_metadata.get("document_id", ""),
                        chunk_index=chunk_metadata.get("chunk_index", 0),
                        start_offset=chunk_metadata.get("start_offset", 0),
                        end_offset=chunk_metadata.get("end_offset", 0),
                        metadata=custom_metadata
                    )
                    
                    # Convert distance to similarity score (for cosine distance)
                    if self.config.distance_metric == DistanceMetric.COSINE:
                        score = 1.0 - distance  # Cosine similarity = 1 - cosine distance
                    else:
                        score = 1.0 / (1.0 + distance)  # General similarity conversion
                    
                    search_result = SearchResult(
                        chunk=chunk,
                        score=score,
                        distance=distance,
                        metadata={"created_at": chunk_metadata.get("created_at")}
                    )
                    
                    search_results.append(search_result)
            
            self.logger.debug(
                "ChromaDB search completed",
                query_embedding_dim=len(query_embedding),
                results_count=len(search_results),
                top_k=top_k
            )
            
            return search_results
    
    async def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a specific chunk by ID."""
        if not self.initialized:
            await self.initialize()
        
        def _get_chunk():
            try:
                results = self.collection.get(
                    ids=[chunk_id],
                    include=["documents", "metadatas"]
                )
                return results
            except Exception:
                return None
        
        results = await asyncio.get_event_loop().run_in_executor(None, _get_chunk)
        
        if not results or not results["documents"]:
            return None
        
        document = results["documents"][0]
        metadata = results["metadatas"][0]
        
        # Reconstruct chunk
        custom_metadata = {}
        for key, value in metadata.items():
            if key.startswith("custom_"):
                custom_key = key[7:]
                try:
                    custom_metadata[custom_key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    custom_metadata[custom_key] = value
        
        chunk = DocumentChunk(
            id=chunk_id,
            content=document,
            document_id=metadata.get("document_id", ""),
            chunk_index=metadata.get("chunk_index", 0),
            start_offset=metadata.get("start_offset", 0),
            end_offset=metadata.get("end_offset", 0),
            metadata=custom_metadata
        )
        
        return chunk
    
    async def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete chunks by IDs."""
        if not self.initialized:
            await self.initialize()
        
        if not chunk_ids:
            return
        
        def _delete_chunks():
            self.collection.delete(ids=chunk_ids)
        
        await asyncio.get_event_loop().run_in_executor(None, _delete_chunks)
        
        self.logger.info(
            "Deleted chunks from ChromaDB",
            chunk_count=len(chunk_ids)
        )
    
    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        if not self.initialized:
            await self.initialize()
        
        def _delete_document():
            # Find all chunks for this document
            results = self.collection.get(
                where={"document_id": {"$eq": document_id}},
                include=["metadatas"]
            )
            
            if results and results["ids"]:
                chunk_ids = results["ids"]
                self.collection.delete(ids=chunk_ids)
                return len(chunk_ids)
            return 0
        
        deleted_count = await asyncio.get_event_loop().run_in_executor(None, _delete_document)
        
        self.logger.info(
            "Deleted document chunks from ChromaDB",
            document_id=document_id,
            chunks_deleted=deleted_count
        )
        
        return deleted_count
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.initialized:
            await self.initialize()
        
        def _get_stats():
            try:
                count_result = self.collection.count()
                return {"total_chunks": count_result}
            except Exception as e:
                self.logger.warning(f"Failed to get collection stats: {e}")
                return {"total_chunks": 0}
        
        stats = await asyncio.get_event_loop().run_in_executor(None, _get_stats)
        
        stats.update({
            "collection_name": self.config.collection_name,
            "distance_metric": self.config.distance_metric,
            "embedding_dimensions": self.config.embedding_dimensions,
            "backend": "chromadb"
        })
        
        return stats
    
    async def list_documents(self) -> List[str]:
        """List all unique document IDs in the collection."""
        if not self.initialized:
            await self.initialize()
        
        def _list_documents():
            # Get all metadatas to extract unique document IDs
            results = self.collection.get(include=["metadatas"])
            
            if results and results["metadatas"]:
                document_ids = set()
                for metadata in results["metadatas"]:
                    if "document_id" in metadata:
                        document_ids.add(metadata["document_id"])
                return list(document_ids)
            return []
        
        document_ids = await asyncio.get_event_loop().run_in_executor(None, _list_documents)
        return document_ids


class VectorStore:
    """Main vector storage interface with support for multiple backends."""
    
    def __init__(self, config: StorageConfig = None, logger=None):
        self.config = config or StorageConfig()
        self.logger = logger or get_logger(__name__)
        
        # Initialize provider based on backend
        if self.config.backend == VectorStorageBackend.CHROMADB:
            self.provider = ChromaDBProvider(self.config, self.logger)
        else:
            # Default to ChromaDB for now
            self.provider = ChromaDBProvider(self.config, self.logger)
        
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the vector store."""
        if self.initialized:
            return
        
        await self.provider.initialize()
        self.initialized = True
        
        self.logger.info(
            "Vector store initialized",
            backend=self.config.backend,
            collection_name=self.config.collection_name
        )
    
    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector store."""
        if not self.initialized:
            await self.initialize()
        
        return await self.provider.add_chunks(chunks)
    
    async def add_document(self, metadata: DocumentMetadata, chunks: List[DocumentChunk]) -> None:
        """Add a complete document with metadata and chunks."""
        if not self.initialized:
            await self.initialize()
        
        # Store document metadata separately or as part of chunk metadata
        for chunk in chunks:
            # Add document metadata to chunk metadata
            chunk.metadata.update({
                "file_name": metadata.file_name,
                "file_path": metadata.file_path,
                "file_size": metadata.file_size,
                "file_hash": metadata.file_hash,
                "mime_type": metadata.mime_type,
                "format": metadata.format.value,
                "processed_at": metadata.processed_at,
                "processing_time": metadata.processing_time,
                "source": metadata.source
            })
            
            # Add custom metadata
            if metadata.custom_metadata:
                chunk.metadata.update(metadata.custom_metadata)
        
        await self.add_chunks(chunks)
        
        self.logger.info(
            "Added document to vector store",
            document_id=chunks[0].document_id if chunks else "unknown",
            file_name=metadata.file_name,
            chunk_count=len(chunks)
        )
    
    async def search_similar(self, query_embedding: List[float], 
                           top_k: int = 10, 
                           filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Search for similar chunks."""
        if not self.initialized:
            await self.initialize()
        
        return await self.provider.search_similar(query_embedding, top_k, filters)
    
    async def search_by_document(self, query_embedding: List[float],
                               document_ids: List[str],
                               top_k: int = 10) -> List[SearchResult]:
        """Search within specific documents."""
        filters = {"document_id": document_ids}
        return await self.search_similar(query_embedding, top_k, filters)
    
    async def search_by_metadata(self, query_embedding: List[float],
                               metadata_filters: Dict[str, Any],
                               top_k: int = 10) -> List[SearchResult]:
        """Search with metadata filters."""
        # Add custom_ prefix to metadata filters for ChromaDB
        chromadb_filters = {}
        for key, value in metadata_filters.items():
            chromadb_filters[f"custom_{key}"] = value
        
        return await self.search_similar(query_embedding, top_k, chromadb_filters)
    
    async def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a specific chunk by ID."""
        if not self.initialized:
            await self.initialize()
        
        return await self.provider.get_chunk(chunk_id)
    
    async def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete chunks by IDs."""
        if not self.initialized:
            await self.initialize()
        
        return await self.provider.delete_chunks(chunk_ids)
    
    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        if not self.initialized:
            await self.initialize()
        
        if hasattr(self.provider, 'delete_document'):
            return await self.provider.delete_document(document_id)
        else:
            # Fallback: search for chunks and delete them
            # This is a simplified version - full implementation would depend on the provider
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if not self.initialized:
            await self.initialize()
        
        return await self.provider.get_collection_stats()
    
    async def list_documents(self) -> List[str]:
        """List all document IDs in the store."""
        if not self.initialized:
            await self.initialize()
        
        if hasattr(self.provider, 'list_documents'):
            return await self.provider.list_documents()
        else:
            return []
    
    async def backup_collection(self, backup_path: str) -> None:
        """Backup the collection to a file."""
        # Implementation would depend on the specific backend
        # For now, this is a placeholder
        self.logger.info(f"Backup functionality not yet implemented for {self.config.backend}")
    
    async def restore_collection(self, backup_path: str) -> None:
        """Restore the collection from a backup file."""
        # Implementation would depend on the specific backend
        # For now, this is a placeholder
        self.logger.info(f"Restore functionality not yet implemented for {self.config.backend}")