"""
Cross-encoder reranking for Akasha RAG system.

This module provides sophisticated reranking using cross-encoder models
that score query-document pairs directly for improved relevance.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from pydantic import BaseModel, Field

from ..core.logging import get_logger, PerformanceLogger
from ..core.exceptions import AkashaError
from .ingestion import DocumentChunk


class CrossEncoderModel(str, Enum):
    """Supported cross-encoder models."""
    MS_MARCO_MINI = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    MS_MARCO_BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MS_MARCO_LARGE = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    SBERT_BASE = "cross-encoder/sbert-distilbert-base"
    QUORA_DISTILBERT = "cross-encoder/quora-distilbert-base"


@dataclass
class CrossEncoderConfig:
    """Configuration for cross-encoder reranking."""
    model_name: CrossEncoderModel = CrossEncoderModel.MS_MARCO_MINI
    max_query_length: int = 256
    max_passage_length: int = 512
    batch_size: int = 16
    device: str = "auto"
    cache_predictions: bool = True
    score_threshold: float = 0.0
    use_fp16: bool = False
    model_cache_dir: Optional[str] = None


class RerankingResult(BaseModel):
    """Result of cross-encoder reranking."""
    reranked_chunks: List[DocumentChunk] = Field(..., description="Chunks ordered by relevance")
    relevance_scores: List[float] = Field(..., description="Cross-encoder relevance scores")
    original_order: List[int] = Field(..., description="Original order indices")
    processing_time: float = Field(..., description="Time taken for reranking")
    model_name: str = Field(..., description="Model used for reranking")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CrossEncoderProvider(ABC):
    """Abstract base class for cross-encoder providers."""
    
    def __init__(self, config: CrossEncoderConfig, logger=None):
        self.config = config
        self.logger = logger or get_logger(__name__)
        self.model = None
        self.model_loaded = False
        self._prediction_cache = {} if config.cache_predictions else None
    
    @abstractmethod
    async def load_model(self) -> None:
        """Load the cross-encoder model."""
        pass
    
    @abstractmethod
    async def score_pairs(self, query: str, passages: List[str]) -> List[float]:
        """Score query-passage pairs."""
        pass
    
    async def rerank(self, query: str, chunks: List[DocumentChunk]) -> RerankingResult:
        """Rerank chunks using cross-encoder scoring."""
        if not self.model_loaded:
            await self.load_model()
        
        if not chunks:
            return RerankingResult(
                reranked_chunks=[],
                relevance_scores=[],
                original_order=[],
                processing_time=0.0,
                model_name=self.config.model_name.value
            )
        
        start_time = time.time()
        
        # Extract passages from chunks
        passages = [chunk.content for chunk in chunks]
        
        # Score query-passage pairs
        scores = await self.score_pairs(query, passages)
        
        # Create (chunk, score, original_index) tuples and sort by score
        scored_chunks = list(zip(chunks, scores, range(len(chunks))))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Extract reranked components
        reranked_chunks = [chunk for chunk, _, _ in scored_chunks]
        relevance_scores = [score for _, score, _ in scored_chunks]
        original_order = [idx for _, _, idx in scored_chunks]
        
        processing_time = time.time() - start_time
        
        self.logger.debug(
            "Cross-encoder reranking completed",
            query=query[:50],
            num_chunks=len(chunks),
            processing_time=processing_time,
            model=self.config.model_name.value
        )
        
        return RerankingResult(
            reranked_chunks=reranked_chunks,
            relevance_scores=relevance_scores,
            original_order=original_order,
            processing_time=processing_time,
            model_name=self.config.model_name.value,
            metadata={
                "max_score": max(relevance_scores) if relevance_scores else 0,
                "min_score": min(relevance_scores) if relevance_scores else 0,
                "avg_score": np.mean(relevance_scores) if relevance_scores else 0
            }
        )


class SentenceTransformersCrossEncoder(CrossEncoderProvider):
    """Cross-encoder implementation using sentence-transformers."""
    
    def __init__(self, config: CrossEncoderConfig, logger=None):
        super().__init__(config, logger)
        self._sentence_transformers = None
        self._cross_encoder = None
    
    async def _import_sentence_transformers(self):
        """Lazy import sentence-transformers."""
        if self._sentence_transformers is None:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder
            except ImportError:
                raise AkashaError(
                    "sentence-transformers not installed. Install with: pip install sentence-transformers"
                )
    
    async def load_model(self) -> None:
        """Load the cross-encoder model."""
        if self.model_loaded:
            return
        
        async with PerformanceLogger(f"load_cross_encoder:{self.config.model_name}", self.logger):
            await self._import_sentence_transformers()
            
            model_name = self.config.model_name.value
            device = "mps" if self.config.device == "auto" else self.config.device
            
            # Load model in a thread to avoid blocking
            def _load_model():
                return self._cross_encoder(
                    model_name,
                    device=device,
                    max_length=self.config.max_passage_length
                )
            
            self.model = await asyncio.get_event_loop().run_in_executor(None, _load_model)
            self.model_loaded = True
            
            self.logger.info(
                "Cross-encoder model loaded",
                model_name=model_name,
                device=device,
                max_length=self.config.max_passage_length
            )
    
    async def score_pairs(self, query: str, passages: List[str]) -> List[float]:
        """Score query-passage pairs using the cross-encoder."""
        if not self.model_loaded:
            await self.load_model()
        
        if not passages:
            return []
        
        # Truncate query and passages to max lengths
        truncated_query = query[:self.config.max_query_length]
        truncated_passages = [passage[:self.config.max_passage_length] for passage in passages]
        
        # Create query-passage pairs
        pairs = [(truncated_query, passage) for passage in truncated_passages]
        
        # Check cache if enabled
        if self._prediction_cache is not None:
            cached_scores = []
            uncached_pairs = []
            uncached_indices = []
            
            for i, pair in enumerate(pairs):
                pair_key = hash((pair[0], pair[1]))
                if pair_key in self._prediction_cache:
                    cached_scores.append((i, self._prediction_cache[pair_key]))
                else:
                    uncached_pairs.append(pair)
                    uncached_indices.append(i)
            
            # Predict uncached pairs
            if uncached_pairs:
                new_scores = await self._predict_batch(uncached_pairs)
                
                # Cache new predictions
                for pair, score in zip(uncached_pairs, new_scores):
                    pair_key = hash((pair[0], pair[1]))
                    self._prediction_cache[pair_key] = score
            else:
                new_scores = []
            
            # Combine cached and new scores in correct order
            all_scores = [0.0] * len(pairs)
            
            # Place cached scores
            for orig_idx, score in cached_scores:
                all_scores[orig_idx] = score
            
            # Place new scores
            for uncached_idx, score in zip(uncached_indices, new_scores):
                all_scores[uncached_idx] = score
            
            return all_scores
        else:
            return await self._predict_batch(pairs)
    
    async def _predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Predict scores for a batch of query-passage pairs."""
        if not pairs:
            return []
        
        # Process in batches to manage memory
        all_scores = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            
            # Predict in a thread to avoid blocking
            def _predict_batch():
                return self.model.predict(batch_pairs).tolist()
            
            batch_scores = await asyncio.get_event_loop().run_in_executor(None, _predict_batch)
            all_scores.extend(batch_scores)
        
        return all_scores


class MockCrossEncoder(CrossEncoderProvider):
    """Mock cross-encoder for testing and fallback."""
    
    async def load_model(self) -> None:
        """Mock model loading."""
        self.model_loaded = True
        self.logger.info("Mock cross-encoder loaded")
    
    async def score_pairs(self, query: str, passages: List[str]) -> List[float]:
        """Generate mock relevance scores based on simple text overlap."""
        if not passages:
            return []
        
        query_words = set(query.lower().split())
        scores = []
        
        for passage in passages:
            passage_words = set(passage.lower().split())
            # Simple Jaccard similarity as mock relevance
            intersection = len(query_words.intersection(passage_words))
            union = len(query_words.union(passage_words))
            jaccard = intersection / union if union > 0 else 0.0
            
            # Scale to cross-encoder-like range
            mock_score = jaccard * 2.0 - 1.0  # Range roughly [-1, 1]
            scores.append(mock_score)
        
        return scores


class CrossEncoderReranker:
    """Main cross-encoder reranking service."""
    
    def __init__(self, config: CrossEncoderConfig = None, logger=None):
        self.config = config or CrossEncoderConfig()
        self.logger = logger or get_logger(__name__)
        
        # Initialize provider based on availability
        try:
            self.provider = SentenceTransformersCrossEncoder(self.config, self.logger)
        except ImportError:
            self.logger.warning("sentence-transformers not available, using mock cross-encoder")
            self.provider = MockCrossEncoder(self.config, self.logger)
    
    async def initialize(self) -> None:
        """Initialize the cross-encoder reranker."""
        await self.provider.load_model()
        self.logger.info(
            "Cross-encoder reranker initialized",
            model=self.config.model_name.value,
            device=self.config.device
        )
    
    async def rerank(self, query: str, chunks: List[DocumentChunk], top_k: int = None) -> RerankingResult:
        """Rerank chunks using cross-encoder scoring."""
        async with PerformanceLogger(f"cross_encoder_rerank:chunks_{len(chunks)}", self.logger):
            result = await self.provider.rerank(query, chunks)
            
            # Apply top_k filtering if specified
            if top_k is not None and top_k < len(result.reranked_chunks):
                result.reranked_chunks = result.reranked_chunks[:top_k]
                result.relevance_scores = result.relevance_scores[:top_k]
                result.original_order = result.original_order[:top_k]
            
            # Filter by score threshold
            if self.config.score_threshold > float('-inf'):
                filtered_chunks = []
                filtered_scores = []
                filtered_order = []
                
                for chunk, score, order in zip(result.reranked_chunks, result.relevance_scores, result.original_order):
                    if score >= self.config.score_threshold:
                        filtered_chunks.append(chunk)
                        filtered_scores.append(score)
                        filtered_order.append(order)
                
                result.reranked_chunks = filtered_chunks
                result.relevance_scores = filtered_scores
                result.original_order = filtered_order
            
            self.logger.info(
                "Cross-encoder reranking completed",
                query=query[:50],
                input_chunks=len(chunks),
                output_chunks=len(result.reranked_chunks),
                processing_time=result.processing_time
            )
            
            return result
    
    async def get_reranker_stats(self) -> Dict[str, Any]:
        """Get statistics about the reranker."""
        return {
            "model_name": self.config.model_name.value,
            "model_loaded": self.provider.model_loaded,
            "batch_size": self.config.batch_size,
            "max_query_length": self.config.max_query_length,
            "max_passage_length": self.config.max_passage_length,
            "cache_enabled": self.config.cache_predictions,
            "cache_size": len(self.provider._prediction_cache) if self.provider._prediction_cache else 0,
            "score_threshold": self.config.score_threshold
        }