"""
Hybrid Search Implementation for Akasha RAG System.

This module implements hybrid search capabilities combining semantic vector search
with traditional keyword search, as specified in Phase 2 requirements.
"""

import asyncio
import os
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
import re

from ..core.logging import get_logger, PerformanceLogger
from ..core.exceptions import AkashaError
from .storage import VectorStore, SearchResult
from .ingestion import DocumentChunk


class SearchBackend(str, Enum):
    """Supported search backends."""
    WHOOSH = "whoosh"
    ELASTICSEARCH = "elasticsearch" 
    BM25 = "bm25"


class FusionMethod(str, Enum):
    """Result fusion methods."""
    RRF = "reciprocal_rank_fusion"  # Reciprocal Rank Fusion
    WEIGHTED = "weighted_sum"
    LINEAR_COMBINATION = "linear_combination"


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""
    keyword_backend: SearchBackend = SearchBackend.BM25
    fusion_method: FusionMethod = FusionMethod.RRF
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    rrf_k: int = 60  # RRF parameter
    max_keyword_results: int = 100
    max_vector_results: int = 100
    enable_query_expansion: bool = True
    whoosh_index_dir: Optional[str] = None
    elasticsearch_host: str = "localhost:9200"
    elasticsearch_index: str = "akasha_documents"


class SearchQuery(BaseModel):
    """Search query specification."""
    text: str = Field(..., description="Query text")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Metadata filters")
    top_k: int = Field(default=10, description="Number of results to return")
    enable_vector_search: bool = Field(default=True, description="Enable vector search")
    enable_keyword_search: bool = Field(default=True, description="Enable keyword search")
    boost_fields: Dict[str, float] = Field(default_factory=dict, description="Field boost factors")


class HybridSearchResult(BaseModel):
    """Hybrid search result with scoring breakdown."""
    chunk: DocumentChunk = Field(..., description="Document chunk")
    final_score: float = Field(..., description="Final fused score")
    vector_score: float = Field(default=0.0, description="Vector similarity score")
    keyword_score: float = Field(default=0.0, description="Keyword relevance score")
    vector_rank: int = Field(default=0, description="Rank in vector search")
    keyword_rank: int = Field(default=0, description="Rank in keyword search")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class KeywordSearchProvider(ABC):
    """Abstract base class for keyword search providers."""
    
    def __init__(self, config: HybridSearchConfig, logger=None):
        self.config = config
        self.logger = logger or get_logger(__name__)
        self.initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the keyword search provider."""
        pass
    
    @abstractmethod
    async def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Index document chunks for keyword search."""
        pass
    
    @abstractmethod
    async def search(self, query: str, top_k: int = 10, filters: Dict[str, Any] = None) -> List[Tuple[str, float]]:
        """Search for documents and return (chunk_id, score) tuples."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get search index statistics."""
        pass


class BM25SearchProvider(KeywordSearchProvider):
    """BM25-based keyword search provider."""
    
    def __init__(self, config: HybridSearchConfig, logger=None):
        super().__init__(config, logger)
        self.bm25 = None
        self.chunk_map = {}  # chunk_id -> chunk
        self.documents = []  # Tokenized documents
        self.chunk_ids = []  # Corresponding chunk IDs
    
    async def initialize(self) -> None:
        """Initialize BM25 search."""
        if self.initialized:
            return
        
        self.chunk_map = {}
        self.documents = []
        self.chunk_ids = []
        self.bm25 = None
        
        self.initialized = True
        self.logger.info("BM25 keyword search provider initialized")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    async def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Index document chunks for BM25 search."""
        if not self.initialized:
            await self.initialize()
        
        async with PerformanceLogger(f"bm25_index_chunks:count_{len(chunks)}", self.logger):
            for chunk in chunks:
                # Store chunk mapping
                self.chunk_map[chunk.id] = chunk
                
                # Tokenize content for BM25
                tokens = self._tokenize(chunk.content)
                self.documents.append(tokens)
                self.chunk_ids.append(chunk.id)
            
            # Build BM25 index
            if self.documents:
                self.bm25 = BM25Okapi(self.documents)
            
            self.logger.info(
                "Indexed chunks for BM25 search",
                chunk_count=len(chunks),
                total_indexed=len(self.documents)
            )
    
    async def search(self, query: str, top_k: int = 10, filters: Dict[str, Any] = None) -> List[Tuple[str, float]]:
        """Search using BM25."""
        if not self.bm25 or not self.documents:
            return []
        
        async with PerformanceLogger(f"bm25_search:query_len_{len(query)}", self.logger):
            # Tokenize query
            query_tokens = self._tokenize(query)
            
            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)
            
            # Create (chunk_id, score) pairs
            chunk_scores = []
            for i, score in enumerate(scores):
                if score > 0:  # Only include non-zero scores
                    chunk_id = self.chunk_ids[i]
                    
                    # Apply filters if specified
                    if filters:
                        chunk = self.chunk_map[chunk_id]
                        if not self._matches_filters(chunk, filters):
                            continue
                    
                    chunk_scores.append((chunk_id, float(score)))
            
            # Sort by score and return top_k
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            results = chunk_scores[:top_k]
            
            self.logger.debug(
                "BM25 search completed",
                query_length=len(query),
                results_count=len(results),
                top_k=top_k
            )
            
            return results
    
    def _matches_filters(self, chunk: DocumentChunk, filters: Dict[str, Any]) -> bool:
        """Check if chunk matches the given filters."""
        for key, value in filters.items():
            if key == "document_id":
                if chunk.document_id != value:
                    return False
            elif key in chunk.metadata:
                if chunk.metadata[key] != value:
                    return False
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get BM25 search statistics."""
        return {
            "backend": "bm25",
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunk_map),
            "average_doc_length": np.mean([len(doc) for doc in self.documents]) if self.documents else 0,
            "vocabulary_size": len(set(token for doc in self.documents for token in doc)) if self.documents else 0
        }


class WhooshSearchProvider(KeywordSearchProvider):
    """Whoosh-based keyword search provider."""
    
    def __init__(self, config: HybridSearchConfig, logger=None):
        super().__init__(config, logger)
        self.index = None
        self.index_dir = None
        self._whoosh = None
    
    async def _import_whoosh(self):
        """Lazy import Whoosh."""
        if self._whoosh is None:
            try:
                from whoosh import fields, index, qparser
                from whoosh.analysis import StandardAnalyzer
                self._whoosh = {
                    'fields': fields,
                    'index': index,
                    'qparser': qparser,
                    'StandardAnalyzer': StandardAnalyzer
                }
            except ImportError:
                raise AkashaError("Whoosh not installed. Install with: pip install whoosh")
    
    async def initialize(self) -> None:
        """Initialize Whoosh search."""
        if self.initialized:
            return
        
        await self._import_whoosh()
        
        # Set up index directory
        if self.config.whoosh_index_dir:
            self.index_dir = Path(self.config.whoosh_index_dir)
        else:
            self.index_dir = Path(tempfile.mkdtemp(prefix="akasha_whoosh_"))
        
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Create schema
        schema = self._whoosh['fields'].Schema(
            chunk_id=self._whoosh['fields'].ID(stored=True, unique=True),
            content=self._whoosh['fields'].TEXT(analyzer=self._whoosh['StandardAnalyzer'](), stored=True),
            document_id=self._whoosh['fields'].ID(stored=True),
            metadata=self._whoosh['fields'].STORED()
        )
        
        # Create or open index
        if self._whoosh['index'].exists_in(str(self.index_dir)):
            self.index = self._whoosh['index'].open_dir(str(self.index_dir))
        else:
            self.index = self._whoosh['index'].create_in(str(self.index_dir), schema)
        
        self.initialized = True
        self.logger.info(
            "Whoosh keyword search provider initialized",
            index_dir=str(self.index_dir)
        )
    
    async def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Index document chunks in Whoosh."""
        if not self.initialized:
            await self.initialize()
        
        async with PerformanceLogger(f"whoosh_index_chunks:count_{len(chunks)}", self.logger):
            def _index_chunks():
                writer = self.index.writer()
                try:
                    for chunk in chunks:
                        writer.add_document(
                            chunk_id=chunk.id,
                            content=chunk.content,
                            document_id=chunk.document_id,
                            metadata=chunk.metadata
                        )
                    writer.commit()
                except Exception as e:
                    writer.cancel()
                    raise e
            
            await asyncio.get_event_loop().run_in_executor(None, _index_chunks)
            
            self.logger.info(
                "Indexed chunks in Whoosh",
                chunk_count=len(chunks)
            )
    
    async def search(self, query: str, top_k: int = 10, filters: Dict[str, Any] = None) -> List[Tuple[str, float]]:
        """Search using Whoosh."""
        if not self.index:
            return []
        
        async with PerformanceLogger(f"whoosh_search:query_len_{len(query)}", self.logger):
            def _search():
                with self.index.searcher() as searcher:
                    # Parse query
                    parser = self._whoosh['qparser'].QueryParser("content", self.index.schema)
                    parsed_query = parser.parse(query)
                    
                    # Execute search
                    results = searcher.search(parsed_query, limit=top_k)
                    
                    # Extract results
                    chunk_scores = []
                    for result in results:
                        chunk_id = result['chunk_id']
                        score = result.score
                        
                        # Apply filters if specified
                        if filters:
                            if not self._matches_filters_whoosh(result, filters):
                                continue
                        
                        chunk_scores.append((chunk_id, float(score)))
                    
                    return chunk_scores
            
            results = await asyncio.get_event_loop().run_in_executor(None, _search)
            
            self.logger.debug(
                "Whoosh search completed",
                query_length=len(query),
                results_count=len(results),
                top_k=top_k
            )
            
            return results
    
    def _matches_filters_whoosh(self, result, filters: Dict[str, Any]) -> bool:
        """Check if Whoosh result matches filters."""
        for key, value in filters.items():
            if key == "document_id":
                if result['document_id'] != value:
                    return False
            elif key in result.get('metadata', {}):
                if result['metadata'][key] != value:
                    return False
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Whoosh search statistics."""
        if not self.index:
            return {"backend": "whoosh", "doc_count": 0}
        
        def _get_stats():
            with self.index.searcher() as searcher:
                return {
                    "backend": "whoosh",
                    "doc_count": searcher.doc_count(),
                    "term_count": len(list(searcher.field_terms("content"))),
                    "index_dir": str(self.index_dir)
                }
        
        return await asyncio.get_event_loop().run_in_executor(None, _get_stats)


class HybridSearchEngine:
    """Main hybrid search engine combining vector and keyword search."""
    
    def __init__(self, vector_store: VectorStore, config: HybridSearchConfig = None, logger=None):
        self.vector_store = vector_store
        self.config = config or HybridSearchConfig()
        self.logger = logger or get_logger(__name__)
        
        # Initialize keyword search provider
        if self.config.keyword_backend == SearchBackend.BM25:
            self.keyword_provider = BM25SearchProvider(self.config, self.logger)
        elif self.config.keyword_backend == SearchBackend.WHOOSH:
            self.keyword_provider = WhooshSearchProvider(self.config, self.logger)
        else:
            raise AkashaError(f"Unsupported keyword backend: {self.config.keyword_backend}")
        
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the hybrid search engine."""
        if self.initialized:
            return
        
        await self.keyword_provider.initialize()
        self.initialized = True
        
        self.logger.info(
            "Hybrid search engine initialized",
            keyword_backend=self.config.keyword_backend,
            fusion_method=self.config.fusion_method
        )
    
    async def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Index chunks for both vector and keyword search."""
        if not self.initialized:
            await self.initialize()
        
        async with PerformanceLogger(f"hybrid_index_chunks:count_{len(chunks)}", self.logger):
            # Index for keyword search
            await self.keyword_provider.index_chunks(chunks)
            
            self.logger.info(
                "Indexed chunks for hybrid search",
                chunk_count=len(chunks),
                keyword_backend=self.config.keyword_backend
            )
    
    async def search(self, query: SearchQuery, embedding_generator=None) -> List[HybridSearchResult]:
        """Perform hybrid search combining vector and keyword search."""
        if not self.initialized:
            await self.initialize()
        
        async with PerformanceLogger(f"hybrid_search:query_len_{len(query.text)}", self.logger):
            vector_results = []
            keyword_results = []
            
            # Perform vector search if enabled
            if query.enable_vector_search and embedding_generator:
                query_embedding = await embedding_generator.embed_text(query.text)
                vector_search_results = await self.vector_store.search_similar(
                    query_embedding, 
                    top_k=self.config.max_vector_results,
                    filters=query.filters
                )
                vector_results = [(r.chunk.id, r.score) for r in vector_search_results]
            
            # Perform keyword search if enabled
            if query.enable_keyword_search:
                keyword_results = await self.keyword_provider.search(
                    query.text,
                    top_k=self.config.max_keyword_results,
                    filters=query.filters
                )
            
            # Fuse results
            fused_results = await self._fuse_results(
                vector_results, 
                keyword_results, 
                query.top_k
            )
            
            self.logger.info(
                "Hybrid search completed",
                query_length=len(query.text),
                vector_results=len(vector_results),
                keyword_results=len(keyword_results),
                fused_results=len(fused_results),
                fusion_method=self.config.fusion_method
            )
            
            return fused_results
    
    async def _fuse_results(self, vector_results: List[Tuple[str, float]], 
                          keyword_results: List[Tuple[str, float]], 
                          top_k: int) -> List[HybridSearchResult]:
        """Fuse vector and keyword search results."""
        if self.config.fusion_method == FusionMethod.RRF:
            return await self._reciprocal_rank_fusion(vector_results, keyword_results, top_k)
        elif self.config.fusion_method == FusionMethod.WEIGHTED:
            return await self._weighted_fusion(vector_results, keyword_results, top_k)
        else:
            return await self._linear_combination_fusion(vector_results, keyword_results, top_k)
    
    async def _reciprocal_rank_fusion(self, vector_results: List[Tuple[str, float]], 
                                    keyword_results: List[Tuple[str, float]], 
                                    top_k: int) -> List[HybridSearchResult]:
        """Reciprocal Rank Fusion (RRF) algorithm."""
        k = self.config.rrf_k
        chunk_scores = {}
        
        # Process vector results
        vector_map = {chunk_id: (rank + 1, score) for rank, (chunk_id, score) in enumerate(vector_results)}
        
        # Process keyword results  
        keyword_map = {chunk_id: (rank + 1, score) for rank, (chunk_id, score) in enumerate(keyword_results)}
        
        # Combine all chunk IDs
        all_chunk_ids = set(vector_map.keys()) | set(keyword_map.keys())
        
        # Calculate RRF scores
        for chunk_id in all_chunk_ids:
            rrf_score = 0
            vector_rank = vector_score = keyword_rank = keyword_score = 0
            
            if chunk_id in vector_map:
                vector_rank, vector_score = vector_map[chunk_id]
                rrf_score += 1 / (k + vector_rank)
            
            if chunk_id in keyword_map:
                keyword_rank, keyword_score = keyword_map[chunk_id]
                rrf_score += 1 / (k + keyword_rank)
            
            chunk_scores[chunk_id] = {
                'final_score': rrf_score,
                'vector_score': vector_score,
                'keyword_score': keyword_score,
                'vector_rank': vector_rank,
                'keyword_rank': keyword_rank
            }
        
        # Sort by RRF score and get top_k
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1]['final_score'], reverse=True)[:top_k]
        
        # Create HybridSearchResult objects
        results = []
        for chunk_id, scores in sorted_chunks:
            # Get chunk from vector store or keyword provider
            chunk = await self._get_chunk_by_id(chunk_id)
            if chunk:
                result = HybridSearchResult(
                    chunk=chunk,
                    final_score=scores['final_score'],
                    vector_score=scores['vector_score'],
                    keyword_score=scores['keyword_score'],
                    vector_rank=scores['vector_rank'],
                    keyword_rank=scores['keyword_rank'],
                    metadata={"fusion_method": "rrf", "rrf_k": k}
                )
                results.append(result)
        
        return results
    
    async def _weighted_fusion(self, vector_results: List[Tuple[str, float]], 
                             keyword_results: List[Tuple[str, float]], 
                             top_k: int) -> List[HybridSearchResult]:
        """Weighted sum fusion."""
        chunk_scores = {}
        
        # Normalize scores to [0, 1] range
        vector_max = max([score for _, score in vector_results], default=1.0)
        keyword_max = max([score for _, score in keyword_results], default=1.0)
        
        # Process vector results
        for rank, (chunk_id, score) in enumerate(vector_results):
            normalized_score = score / vector_max if vector_max > 0 else 0
            chunk_scores[chunk_id] = {
                'vector_score': score,
                'vector_norm': normalized_score,
                'vector_rank': rank + 1,
                'keyword_score': 0,
                'keyword_norm': 0,
                'keyword_rank': 0
            }
        
        # Process keyword results
        for rank, (chunk_id, score) in enumerate(keyword_results):
            normalized_score = score / keyword_max if keyword_max > 0 else 0
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id].update({
                    'keyword_score': score,
                    'keyword_norm': normalized_score,
                    'keyword_rank': rank + 1
                })
            else:
                chunk_scores[chunk_id] = {
                    'vector_score': 0,
                    'vector_norm': 0,
                    'vector_rank': 0,
                    'keyword_score': score,
                    'keyword_norm': normalized_score,
                    'keyword_rank': rank + 1
                }
        
        # Calculate weighted scores
        for chunk_id, scores in chunk_scores.items():
            weighted_score = (
                self.config.vector_weight * scores['vector_norm'] + 
                self.config.keyword_weight * scores['keyword_norm']
            )
            scores['final_score'] = weighted_score
        
        # Sort and get top_k
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1]['final_score'], reverse=True)[:top_k]
        
        # Create results
        results = []
        for chunk_id, scores in sorted_chunks:
            chunk = await self._get_chunk_by_id(chunk_id)
            if chunk:
                result = HybridSearchResult(
                    chunk=chunk,
                    final_score=scores['final_score'],
                    vector_score=scores['vector_score'],
                    keyword_score=scores['keyword_score'],
                    vector_rank=scores['vector_rank'],
                    keyword_rank=scores['keyword_rank'],
                    metadata={
                        "fusion_method": "weighted",
                        "vector_weight": self.config.vector_weight,
                        "keyword_weight": self.config.keyword_weight
                    }
                )
                results.append(result)
        
        return results
    
    async def _linear_combination_fusion(self, vector_results: List[Tuple[str, float]], 
                                       keyword_results: List[Tuple[str, float]], 
                                       top_k: int) -> List[HybridSearchResult]:
        """Linear combination fusion (similar to weighted but different normalization)."""
        # For now, use weighted fusion with different normalization
        return await self._weighted_fusion(vector_results, keyword_results, top_k)
    
    async def _get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get chunk by ID from vector store or keyword provider."""
        # Try vector store first
        chunk = await self.vector_store.get_chunk(chunk_id)
        if chunk:
            return chunk
        
        # Try keyword provider if it has chunk mapping
        if hasattr(self.keyword_provider, 'chunk_map') and chunk_id in self.keyword_provider.chunk_map:
            return self.keyword_provider.chunk_map[chunk_id]
        
        return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get hybrid search statistics."""
        vector_stats = await self.vector_store.get_stats()
        keyword_stats = await self.keyword_provider.get_stats()
        
        return {
            "hybrid_search": {
                "config": {
                    "keyword_backend": self.config.keyword_backend,
                    "fusion_method": self.config.fusion_method,
                    "vector_weight": self.config.vector_weight,
                    "keyword_weight": self.config.keyword_weight
                },
                "vector_search": vector_stats,
                "keyword_search": keyword_stats
            }
        }