"""
Document retrieval system for Akasha RAG.

This module provides advanced retrieval strategies including multi-stage retrieval,
query preprocessing, hybrid search, and intelligent result ranking.
"""

import asyncio
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Set
import math

from pydantic import BaseModel, Field

from ..core.logging import get_logger, PerformanceLogger
from ..core.exceptions import AkashaError
from .storage import VectorStore, SearchResult, StorageConfig
from .embeddings import EmbeddingGenerator, EmbeddingConfig
from .ingestion import DocumentChunk


class RetrievalStrategy(str, Enum):
    """Supported retrieval strategies."""
    VECTOR_ONLY = "vector_only"
    HYBRID = "hybrid"
    MULTI_STAGE = "multi_stage"
    SEMANTIC_SEARCH = "semantic_search"
    KEYWORD_SEARCH = "keyword_search"


class QueryType(str, Enum):
    """Types of queries for different processing strategies."""
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"
    SUMMARY = "summary"
    GENERAL = "general"


class RerankingMethod(str, Enum):
    """Methods for reranking search results."""
    NONE = "none"
    CROSS_ENCODER = "cross_encoder"
    BM25 = "bm25"
    RECIPROCAL_RANK_FUSION = "rrf"
    WEIGHTED_SCORE = "weighted_score"


@dataclass
class RetrievalConfig:
    """Configuration for document retrieval."""
    strategy: RetrievalStrategy = RetrievalStrategy.MULTI_STAGE
    initial_top_k: int = 50
    final_top_k: int = 10
    rerank_top_k: int = 20
    reranking_method: RerankingMethod = RerankingMethod.BM25
    min_similarity_threshold: float = 0.3
    max_context_length: int = 4000
    query_expansion: bool = True
    use_query_classification: bool = True
    enable_semantic_chunking: bool = True
    chunk_overlap_penalty: float = 0.1
    diversity_lambda: float = 0.3
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    rrf_k: int = 60


class QueryContext(BaseModel):
    """Context information for query processing."""
    original_query: str = Field(..., description="Original user query")
    processed_query: str = Field(..., description="Processed/expanded query")
    query_type: QueryType = Field(default=QueryType.GENERAL, description="Classified query type")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    entities: List[str] = Field(default_factory=list, description="Named entities")
    intent: str = Field(default="", description="Detected user intent")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Metadata filters")
    session_context: Dict[str, Any] = Field(default_factory=dict, description="Session context")


class RetrievalResult(BaseModel):
    """Enhanced retrieval result with additional metadata."""
    chunks: List[DocumentChunk] = Field(..., description="Retrieved chunks")
    scores: List[float] = Field(..., description="Relevance scores")
    total_score: float = Field(..., description="Combined relevance score")
    retrieval_method: str = Field(..., description="Method used for retrieval")
    processing_time: float = Field(..., description="Time taken for retrieval")
    query_context: QueryContext = Field(..., description="Query processing context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QueryProcessor:
    """Handles query preprocessing and expansion."""
    
    def __init__(self, config: RetrievalConfig, logger=None):
        self.config = config
        self.logger = logger or get_logger(__name__)
        self._stop_words = self._load_stop_words()
    
    def _load_stop_words(self) -> Set[str]:
        """Load common English stop words."""
        return {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "will", "with", "the", "this", "but", "they", "have",
            "had", "what", "said", "each", "which", "she", "do", "how", "their",
            "if", "up", "out", "many", "then", "them", "these", "so", "some",
            "her", "would", "make", "like", "into", "him", "time", "two", "more",
            "go", "no", "way", "could", "my", "than", "first", "been", "call",
            "who", "oil", "sit", "now", "find", "down", "day", "did", "get",
            "come", "made", "may", "part"
        }
    
    async def process_query(self, query: str, context: Dict[str, Any] = None) -> QueryContext:
        """Process and analyze the user query."""
        async with PerformanceLogger("query_processing", self.logger):
            # Create query context
            query_context = QueryContext(
                original_query=query,
                processed_query=query,
                session_context=context or {}
            )
            
            # Clean and normalize query
            query_context.processed_query = self._clean_query(query)
            
            # Extract keywords
            query_context.keywords = self._extract_keywords(query_context.processed_query)
            
            # Classify query type if enabled
            if self.config.use_query_classification:
                query_context.query_type = self._classify_query(query_context.processed_query)
            
            # Extract named entities (simple implementation)
            query_context.entities = self._extract_entities(query_context.processed_query)
            
            # Expand query if enabled
            if self.config.query_expansion:
                query_context.processed_query = await self._expand_query(query_context)
            
            # Detect intent
            query_context.intent = self._detect_intent(query_context.processed_query)
            
            self.logger.debug(
                "Query processed",
                original_query=query_context.original_query,
                processed_query=query_context.processed_query,
                query_type=query_context.query_type,
                keywords=query_context.keywords,
                intent=query_context.intent
            )
            
            return query_context
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query."""
        # Convert to lowercase
        cleaned = query.lower().strip()
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove special characters but keep important punctuation
        cleaned = re.sub(r'[^\w\s\?\.\!\-\,\:\;]', '', cleaned)
        
        return cleaned
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from the query."""
        # Simple keyword extraction - could be enhanced with NLP libraries
        words = query.split()
        
        # Filter stop words and short words
        keywords = [
            word for word in words 
            if len(word) > 2 and word.lower() not in self._stop_words
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify the query type based on patterns."""
        query_lower = query.lower()
        
        # Factual queries
        factual_patterns = [
            r'^what is', r'^who is', r'^when is', r'^where is',
            r'^define', r'^definition of', r'^meaning of'
        ]
        if any(re.search(pattern, query_lower) for pattern in factual_patterns):
            return QueryType.FACTUAL
        
        # Comparative queries
        comparative_patterns = [
            r'compare', r'difference between', r'versus', r'vs',
            r'better than', r'worse than', r'similar to'
        ]
        if any(re.search(pattern, query_lower) for pattern in comparative_patterns):
            return QueryType.COMPARATIVE
        
        # Analytical queries
        analytical_patterns = [
            r'analyze', r'analysis of', r'why', r'how does',
            r'explain', r'reason for', r'cause of'
        ]
        if any(re.search(pattern, query_lower) for pattern in analytical_patterns):
            return QueryType.ANALYTICAL
        
        # Summary queries
        summary_patterns = [
            r'summarize', r'summary of', r'overview of',
            r'key points', r'main ideas'
        ]
        if any(re.search(pattern, query_lower) for pattern in summary_patterns):
            return QueryType.SUMMARY
        
        # Conceptual queries
        conceptual_patterns = [
            r'concept of', r'theory of', r'principle of',
            r'understand', r'learn about'
        ]
        if any(re.search(pattern, query_lower) for pattern in conceptual_patterns):
            return QueryType.CONCEPTUAL
        
        return QueryType.GENERAL
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from the query (simplified implementation)."""
        # This is a simple implementation - could be enhanced with spaCy or similar
        entities = []
        
        # Find capitalized words (potential proper nouns)
        words = query.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 2:
                entities.append(word)
        
        # Find quoted phrases
        quoted_matches = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted_matches)
        
        return entities
    
    async def _expand_query(self, query_context: QueryContext) -> str:
        """Expand the query with synonyms and related terms."""
        # Simple query expansion - could be enhanced with Word2Vec, WordNet, etc.
        expanded_terms = []
        
        # Add original query
        expanded_terms.append(query_context.processed_query)
        
        # Add synonyms for key terms (simplified)
        synonym_map = {
            "ai": ["artificial intelligence", "machine learning", "ML"],
            "ml": ["machine learning", "artificial intelligence", "AI"],
            "nlp": ["natural language processing", "text processing"],
            "paper": ["document", "research paper", "article", "publication"],
            "method": ["approach", "technique", "methodology"],
            "result": ["outcome", "finding", "conclusion"],
            "analysis": ["evaluation", "assessment", "examination"],
            "model": ["algorithm", "system", "framework"],
        }
        
        for keyword in query_context.keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in synonym_map:
                # Add synonyms as alternative query terms
                synonyms = synonym_map[keyword_lower]
                for synonym in synonyms:
                    if synonym not in query_context.processed_query:
                        expanded_terms.append(f"{query_context.processed_query} {synonym}")
        
        # For now, return the original query (expansion can be improved)
        return query_context.processed_query
    
    def _detect_intent(self, query: str) -> str:
        """Detect user intent from the query."""
        query_lower = query.lower()
        
        # Define intent patterns
        intent_patterns = {
            "search": ["find", "search", "look for", "show me"],
            "explain": ["explain", "describe", "tell me about", "what is"],
            "compare": ["compare", "difference", "versus", "vs"],
            "summarize": ["summarize", "summary", "overview", "brief"],
            "analyze": ["analyze", "analysis", "examine", "evaluate"],
            "define": ["define", "definition", "meaning", "what does"],
        }
        
        for intent, patterns in intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent
        
        return "general"


class BM25Ranker:
    """BM25 ranking implementation for keyword-based scoring."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}
        self.idf = {}
        self.doc_len = {}
        self.avgdl = 0
        self.corpus = []
        self.initialized = False
    
    def fit(self, corpus: List[str]) -> None:
        """Fit the BM25 model on a corpus."""
        self.corpus = corpus
        self.doc_len = {}
        self.doc_freqs = {}
        
        # Calculate document frequencies and lengths
        for idx, doc in enumerate(corpus):
            doc_tokens = doc.lower().split()
            self.doc_len[idx] = len(doc_tokens)
            
            freq = {}
            for token in doc_tokens:
                freq[token] = freq.get(token, 0) + 1
            self.doc_freqs[idx] = freq
        
        # Calculate average document length
        self.avgdl = sum(self.doc_len.values()) / len(self.doc_len) if self.doc_len else 0
        
        # Calculate IDF values
        self.idf = {}
        all_tokens = set()
        for doc_freq in self.doc_freqs.values():
            all_tokens.update(doc_freq.keys())
        
        for token in all_tokens:
            containing_docs = sum(1 for doc_freq in self.doc_freqs.values() if token in doc_freq)
            self.idf[token] = math.log((len(corpus) - containing_docs + 0.5) / (containing_docs + 0.5))
        
        self.initialized = True
    
    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score for a query-document pair."""
        if not self.initialized or doc_idx not in self.doc_freqs:
            return 0.0
        
        query_tokens = query.lower().split()
        doc_freqs = self.doc_freqs[doc_idx]
        doc_len = self.doc_len[doc_idx]
        
        score = 0.0
        for token in query_tokens:
            if token in doc_freqs:
                freq = doc_freqs[token]
                idf = self.idf.get(token, 0)
                
                # BM25 formula
                score += idf * (freq * (self.k1 + 1)) / (
                    freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                )
        
        return score


class DocumentRetriever:
    """Main document retrieval system with multiple strategies."""
    
    def __init__(self, 
                 vector_store: VectorStore,
                 embedding_generator: EmbeddingGenerator,
                 config: RetrievalConfig = None,
                 logger=None):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.config = config or RetrievalConfig()
        self.logger = logger or get_logger(__name__)
        
        self.query_processor = QueryProcessor(self.config, self.logger)
        self.bm25_ranker = BM25Ranker(self.config.bm25_k1, self.config.bm25_b)
        self.bm25_fitted = False
    
    async def initialize(self) -> None:
        """Initialize the retriever."""
        await self.vector_store.initialize()
        await self.embedding_generator.initialize()
        
        self.logger.info("Document retriever initialized")
    
    async def retrieve(self, query: str, 
                      top_k: int = None,
                      filters: Dict[str, Any] = None,
                      context: Dict[str, Any] = None) -> RetrievalResult:
        """Retrieve relevant documents for a query."""
        start_time = time.time()
        top_k = top_k or self.config.final_top_k
        
        async with PerformanceLogger(f"document_retrieval:top_k_{top_k}", self.logger):
            # Process query
            query_context = await self.query_processor.process_query(query, context)
            
            # Apply any additional filters from query context
            if filters is None:
                filters = {}
            filters.update(query_context.filters)
            
            # Choose retrieval strategy
            if self.config.strategy == RetrievalStrategy.VECTOR_ONLY:
                result = await self._vector_retrieval(query_context, top_k, filters)
            elif self.config.strategy == RetrievalStrategy.HYBRID:
                result = await self._hybrid_retrieval(query_context, top_k, filters)
            elif self.config.strategy == RetrievalStrategy.MULTI_STAGE:
                result = await self._multi_stage_retrieval(query_context, top_k, filters)
            else:
                # Default to vector retrieval
                result = await self._vector_retrieval(query_context, top_k, filters)
            
            processing_time = time.time() - start_time
            
            # Create retrieval result
            retrieval_result = RetrievalResult(
                chunks=result["chunks"],
                scores=result["scores"],
                total_score=sum(result["scores"]),
                retrieval_method=result["method"],
                processing_time=processing_time,
                query_context=query_context,
                metadata=result.get("metadata", {})
            )
            
            self.logger.info(
                "Document retrieval completed",
                query=query[:100],
                retrieved_chunks=len(retrieval_result.chunks),
                processing_time=processing_time,
                strategy=self.config.strategy
            )
            
            return retrieval_result
    
    async def _vector_retrieval(self, query_context: QueryContext, 
                               top_k: int, 
                               filters: Dict[str, Any]) -> Dict[str, Any]:
        """Pure vector similarity retrieval."""
        # Generate query embedding
        query_embedding = await self.embedding_generator.embed_query(query_context.processed_query)
        
        # Search vector store
        search_results = await self.vector_store.search_similar(
            query_embedding, 
            top_k=top_k, 
            filters=filters
        )
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in search_results
            if result.score >= self.config.min_similarity_threshold
        ]
        
        chunks = [result.chunk for result in filtered_results]
        scores = [result.score for result in filtered_results]
        
        return {
            "chunks": chunks,
            "scores": scores,
            "method": "vector_similarity",
            "metadata": {"total_candidates": len(search_results)}
        }
    
    async def _hybrid_retrieval(self, query_context: QueryContext,
                               top_k: int,
                               filters: Dict[str, Any]) -> Dict[str, Any]:
        """Hybrid retrieval combining vector and keyword search."""
        # Get initial candidates with higher top_k
        initial_top_k = min(self.config.initial_top_k, top_k * 3)
        
        # Vector retrieval
        vector_result = await self._vector_retrieval(query_context, initial_top_k, filters)
        
        # Keyword scoring using BM25
        if not self.bm25_fitted and vector_result["chunks"]:
            corpus = [chunk.content for chunk in vector_result["chunks"]]
            self.bm25_ranker.fit(corpus)
            self.bm25_fitted = True
        
        # Calculate BM25 scores
        bm25_scores = []
        for idx, chunk in enumerate(vector_result["chunks"]):
            bm25_score = self.bm25_ranker.score(query_context.processed_query, idx)
            bm25_scores.append(bm25_score)
        
        # Combine scores (weighted combination)
        combined_scores = []
        vector_scores = vector_result["scores"]
        
        # Normalize scores to [0, 1] range
        if vector_scores:
            max_vector = max(vector_scores)
            min_vector = min(vector_scores)
            vector_range = max_vector - min_vector if max_vector != min_vector else 1
            norm_vector_scores = [(score - min_vector) / vector_range for score in vector_scores]
        else:
            norm_vector_scores = []
        
        if bm25_scores:
            max_bm25 = max(bm25_scores)
            min_bm25 = min(bm25_scores)
            bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1
            norm_bm25_scores = [(score - min_bm25) / bm25_range for score in bm25_scores]
        else:
            norm_bm25_scores = [0] * len(vector_scores)
        
        # Weight combination (70% vector, 30% BM25)
        vector_weight = 0.7
        bm25_weight = 0.3
        
        for v_score, b_score in zip(norm_vector_scores, norm_bm25_scores):
            combined_score = vector_weight * v_score + bm25_weight * b_score
            combined_scores.append(combined_score)
        
        # Sort by combined score and select top_k
        scored_results = list(zip(vector_result["chunks"], combined_scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        final_chunks = [chunk for chunk, _ in scored_results[:top_k]]
        final_scores = [score for _, score in scored_results[:top_k]]
        
        return {
            "chunks": final_chunks,
            "scores": final_scores,
            "method": "hybrid_vector_bm25",
            "metadata": {
                "initial_candidates": len(vector_result["chunks"]),
                "vector_weight": vector_weight,
                "bm25_weight": bm25_weight
            }
        }
    
    async def _multi_stage_retrieval(self, query_context: QueryContext,
                                   top_k: int,
                                   filters: Dict[str, Any]) -> Dict[str, Any]:
        """Multi-stage retrieval with initial broad search and reranking."""
        # Stage 1: Initial broad retrieval
        initial_result = await self._hybrid_retrieval(
            query_context, 
            self.config.initial_top_k, 
            filters
        )
        
        if not initial_result["chunks"]:
            return initial_result
        
        # Stage 2: Reranking
        if self.config.reranking_method == RerankingMethod.NONE:
            # No reranking, just return top_k
            reranked_chunks = initial_result["chunks"][:top_k]
            reranked_scores = initial_result["scores"][:top_k]
        elif self.config.reranking_method == RerankingMethod.BM25:
            # Additional BM25 reranking
            reranked_chunks, reranked_scores = await self._bm25_rerank(
                query_context, 
                initial_result["chunks"][:self.config.rerank_top_k]
            )
            reranked_chunks = reranked_chunks[:top_k]
            reranked_scores = reranked_scores[:top_k]
        elif self.config.reranking_method == RerankingMethod.RECIPROCAL_RANK_FUSION:
            # RRF reranking
            reranked_chunks, reranked_scores = await self._rrf_rerank(
                query_context,
                initial_result["chunks"][:self.config.rerank_top_k],
                initial_result["scores"][:self.config.rerank_top_k]
            )
            reranked_chunks = reranked_chunks[:top_k]
            reranked_scores = reranked_scores[:top_k]
        else:
            # Default to no reranking
            reranked_chunks = initial_result["chunks"][:top_k]
            reranked_scores = initial_result["scores"][:top_k]
        
        # Stage 3: Diversity filtering (optional)
        if self.config.diversity_lambda > 0:
            diverse_chunks, diverse_scores = await self._apply_diversity_filter(
                reranked_chunks, 
                reranked_scores,
                self.config.diversity_lambda
            )
        else:
            diverse_chunks = reranked_chunks
            diverse_scores = reranked_scores
        
        return {
            "chunks": diverse_chunks,
            "scores": diverse_scores,
            "method": f"multi_stage_{self.config.reranking_method}",
            "metadata": {
                "initial_candidates": len(initial_result["chunks"]),
                "rerank_candidates": min(self.config.rerank_top_k, len(initial_result["chunks"])),
                "diversity_applied": self.config.diversity_lambda > 0
            }
        }
    
    async def _bm25_rerank(self, query_context: QueryContext, 
                          chunks: List[DocumentChunk]) -> Tuple[List[DocumentChunk], List[float]]:
        """Rerank chunks using BM25 scoring."""
        if not chunks:
            return [], []
        
        # Fit BM25 on the candidate chunks
        corpus = [chunk.content for chunk in chunks]
        ranker = BM25Ranker(self.config.bm25_k1, self.config.bm25_b)
        ranker.fit(corpus)
        
        # Calculate BM25 scores
        scores = []
        for idx in range(len(chunks)):
            score = ranker.score(query_context.processed_query, idx)
            scores.append(score)
        
        # Sort by BM25 score
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        reranked_chunks = [chunk for chunk, _ in scored_chunks]
        reranked_scores = [score for _, score in scored_chunks]
        
        return reranked_chunks, reranked_scores
    
    async def _rrf_rerank(self, query_context: QueryContext,
                         chunks: List[DocumentChunk],
                         vector_scores: List[float]) -> Tuple[List[DocumentChunk], List[float]]:
        """Rerank using Reciprocal Rank Fusion."""
        if not chunks:
            return [], []
        
        # Get BM25 ranking
        bm25_chunks, bm25_scores = await self._bm25_rerank(query_context, chunks)
        
        # Create rank mappings
        vector_ranks = {chunk.id: rank for rank, chunk in enumerate(chunks)}
        bm25_ranks = {chunk.id: rank for rank, chunk in enumerate(bm25_chunks)}
        
        # Calculate RRF scores
        rrf_scores = {}
        k = self.config.rrf_k
        
        for chunk in chunks:
            chunk_id = chunk.id
            vector_rank = vector_ranks.get(chunk_id, len(chunks))
            bm25_rank = bm25_ranks.get(chunk_id, len(chunks))
            
            rrf_score = 1 / (k + vector_rank + 1) + 1 / (k + bm25_rank + 1)
            rrf_scores[chunk_id] = rrf_score
        
        # Sort by RRF score
        sorted_chunks = sorted(chunks, key=lambda x: rrf_scores[x.id], reverse=True)
        sorted_scores = [rrf_scores[chunk.id] for chunk in sorted_chunks]
        
        return sorted_chunks, sorted_scores
    
    async def _apply_diversity_filter(self, chunks: List[DocumentChunk], 
                                    scores: List[float],
                                    diversity_lambda: float) -> Tuple[List[DocumentChunk], List[float]]:
        """Apply diversity filtering to reduce redundant results."""
        if not chunks or len(chunks) <= 1:
            return chunks, scores
        
        selected_chunks = []
        selected_scores = []
        remaining_chunks = list(zip(chunks, scores))
        
        while remaining_chunks and len(selected_chunks) < len(chunks):
            if not selected_chunks:
                # Select the highest scoring chunk first
                best_chunk, best_score = remaining_chunks.pop(0)
                selected_chunks.append(best_chunk)
                selected_scores.append(best_score)
            else:
                # Select chunk that balances relevance and diversity
                best_idx = 0
                best_score_adjusted = -float('inf')
                
                for idx, (chunk, score) in enumerate(remaining_chunks):
                    # Calculate similarity with already selected chunks
                    max_similarity = 0.0
                    
                    for selected_chunk in selected_chunks:
                        # Simple content similarity (could use embedding similarity)
                        similarity = self._calculate_content_similarity(
                            chunk.content, 
                            selected_chunk.content
                        )
                        max_similarity = max(max_similarity, similarity)
                    
                    # Adjust score for diversity
                    adjusted_score = score - diversity_lambda * max_similarity
                    
                    if adjusted_score > best_score_adjusted:
                        best_score_adjusted = adjusted_score
                        best_idx = idx
                
                # Select the best diverse chunk
                best_chunk, best_score = remaining_chunks.pop(best_idx)
                selected_chunks.append(best_chunk)
                selected_scores.append(best_score)
        
        return selected_chunks, selected_scores
    
    def _calculate_content_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple content similarity between two texts."""
        # Simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def search_by_document_type(self, query: str, 
                                    document_types: List[str],
                                    top_k: int = None) -> RetrievalResult:
        """Search within specific document types."""
        filters = {"custom_format": document_types}
        return await self.retrieve(query, top_k, filters)
    
    async def search_by_date_range(self, query: str,
                                 start_date: float,
                                 end_date: float,
                                 top_k: int = None) -> RetrievalResult:
        """Search within a specific date range."""
        filters = {
            "custom_processed_at": {
                "min": start_date,
                "max": end_date
            }
        }
        return await self.retrieve(query, top_k, filters)
    
    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics."""
        vector_stats = await self.vector_store.get_stats()
        embedding_info = await self.embedding_generator.get_embedding_info()
        
        return {
            "storage": vector_stats,
            "embeddings": embedding_info,
            "config": {
                "strategy": self.config.strategy,
                "reranking_method": self.config.reranking_method,
                "initial_top_k": self.config.initial_top_k,
                "final_top_k": self.config.final_top_k,
                "min_similarity_threshold": self.config.min_similarity_threshold
            },
            "bm25_fitted": self.bm25_fitted
        }