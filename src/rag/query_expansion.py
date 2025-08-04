"""
Advanced query expansion for Akasha RAG system.

This module provides sophisticated query expansion techniques including
semantic similarity, corpus-specific expansion, and contextual reformulation.
"""

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np

from pydantic import BaseModel, Field

from ..core.logging import get_logger, PerformanceLogger
from ..core.exceptions import AkashaError


class ExpansionStrategy(str, Enum):
    """Query expansion strategies."""
    NONE = "none"
    SYNONYMS = "synonyms"
    SEMANTIC = "semantic"
    CORPUS_BASED = "corpus_based"
    CONTEXTUAL = "contextual"
    HYBRID = "hybrid"


class QueryExpansionConfig(BaseModel):
    """Configuration for query expansion."""
    strategy: ExpansionStrategy = ExpansionStrategy.HYBRID
    max_expansion_terms: int = 5
    min_similarity_threshold: float = 0.6
    expansion_weight: float = 0.3
    use_stemming: bool = True
    use_pos_filtering: bool = True
    corpus_expansion_ratio: float = 0.2
    enable_phrase_expansion: bool = True
    context_window_size: int = 3


class ExpandedQuery(BaseModel):
    """Result of query expansion."""
    original_query: str = Field(..., description="Original user query")
    expanded_query: str = Field(..., description="Expanded query with additional terms")
    expansion_terms: List[str] = Field(default_factory=list, description="Added expansion terms")
    expansion_weights: Dict[str, float] = Field(default_factory=dict, description="Weight for each term")
    expansion_method: str = Field(..., description="Method used for expansion")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QueryExpander(ABC):
    """Abstract base class for query expansion strategies."""
    
    def __init__(self, config: QueryExpansionConfig, logger=None):
        self.config = config
        self.logger = logger or get_logger(__name__)
    
    @abstractmethod
    async def expand_query(self, query: str, context: Dict[str, Any] = None) -> ExpandedQuery:
        """Expand a query using the specific strategy."""
        pass


class SynonymExpander(QueryExpander):
    """Expands queries using predefined synonym mappings."""
    
    def __init__(self, config: QueryExpansionConfig, logger=None):
        super().__init__(config, logger)
        self.synonym_db = self._load_synonym_database()
    
    def _load_synonym_database(self) -> Dict[str, List[str]]:
        """Load comprehensive synonym database."""
        return {
            # AI/ML domain
            "ai": ["artificial intelligence", "machine learning", "ML", "deep learning", "neural networks"],
            "ml": ["machine learning", "artificial intelligence", "AI", "statistical learning", "predictive modeling"],
            "nlp": ["natural language processing", "text processing", "computational linguistics", "language understanding"],
            "llm": ["large language model", "language model", "generative model", "transformer model"],
            "gpt": ["generative pre-trained transformer", "language model", "AI model"],
            
            # Research domain
            "paper": ["document", "research paper", "article", "publication", "study", "manuscript"],
            "research": ["study", "investigation", "analysis", "examination", "exploration"],
            "method": ["approach", "technique", "methodology", "procedure", "strategy"],
            "result": ["outcome", "finding", "conclusion", "output", "discovery"],
            "analysis": ["evaluation", "assessment", "examination", "investigation", "study"],
            "model": ["algorithm", "system", "framework", "architecture", "approach"],
            "dataset": ["data", "corpus", "collection", "database", "repository"],
            "evaluation": ["assessment", "testing", "validation", "measurement", "benchmarking"],
            "performance": ["accuracy", "effectiveness", "efficiency", "quality", "results"],
            
            # Academic terms
            "hypothesis": ["theory", "assumption", "proposition", "conjecture"],
            "experiment": ["test", "trial", "study", "investigation"],
            "conclusion": ["result", "finding", "outcome", "determination"],
            "literature": ["publications", "research", "studies", "works"],
            "survey": ["review", "overview", "study", "analysis"],
            
            # Technical terms
            "algorithm": ["method", "procedure", "technique", "approach"],
            "implementation": ["development", "coding", "programming", "construction"],
            "optimization": ["improvement", "enhancement", "refinement", "tuning"],
            "architecture": ["design", "structure", "framework", "organization"],
            "pipeline": ["workflow", "process", "sequence", "chain"],
            
            # General academic
            "significant": ["important", "notable", "substantial", "considerable"],
            "novel": ["new", "innovative", "original", "unique"],
            "effective": ["successful", "efficient", "productive", "powerful"],
            "robust": ["reliable", "stable", "strong", "resilient"],
            "comprehensive": ["thorough", "complete", "extensive", "detailed"],
        }
    
    async def expand_query(self, query: str, context: Dict[str, Any] = None) -> ExpandedQuery:
        """Expand query using synonym mappings."""
        async with PerformanceLogger("synonym_expansion", self.logger):
            original_query = query.lower()
            expansion_terms = []
            expansion_weights = {}
            
            # Extract words from query
            words = re.findall(r'\b\w+\b', original_query)
            
            for word in words:
                if word in self.synonym_db:
                    synonyms = self.synonym_db[word][:self.config.max_expansion_terms]
                    for synonym in synonyms:
                        if synonym not in original_query and synonym not in expansion_terms:
                            expansion_terms.append(synonym)
                            expansion_weights[synonym] = self.config.expansion_weight
            
            # Create expanded query
            if expansion_terms:
                expanded_query = f"{query} {' '.join(expansion_terms[:self.config.max_expansion_terms])}"
            else:
                expanded_query = query
            
            return ExpandedQuery(
                original_query=query,
                expanded_query=expanded_query,
                expansion_terms=expansion_terms[:self.config.max_expansion_terms],
                expansion_weights=expansion_weights,
                expansion_method="synonyms",
                metadata={"synonym_matches": len(expansion_terms)}
            )


class SemanticExpander(QueryExpander):
    """Expands queries using semantic similarity from embeddings."""
    
    def __init__(self, config: QueryExpansionConfig, embedding_generator=None, logger=None):
        super().__init__(config, logger)
        self.embedding_generator = embedding_generator
        self.term_embeddings_cache = {}
        self.corpus_terms = set()
    
    async def _get_corpus_terms(self, vector_store) -> Set[str]:
        """Extract terms from the document corpus."""
        if not self.corpus_terms:
            # This would ideally come from analyzing the stored documents
            # For now, return a set of common academic/research terms
            self.corpus_terms = {
                "algorithm", "model", "dataset", "training", "testing", "validation",
                "accuracy", "precision", "recall", "f1-score", "performance", "evaluation",
                "neural", "network", "deep", "learning", "transformer", "attention",
                "embedding", "vector", "similarity", "classification", "regression",
                "clustering", "dimensionality", "reduction", "feature", "extraction",
                "preprocessing", "tokenization", "normalization", "optimization",
                "gradient", "descent", "backpropagation", "activation", "function",
                "loss", "cost", "objective", "metric", "benchmark", "baseline"
            }
        return self.corpus_terms
    
    async def expand_query(self, query: str, context: Dict[str, Any] = None) -> ExpandedQuery:
        """Expand query using semantic similarity."""
        if not self.embedding_generator:
            # Fallback to synonym expansion if no embedding generator
            synonym_expander = SynonymExpander(self.config, self.logger)
            return await synonym_expander.expand_query(query, context)
        
        async with PerformanceLogger("semantic_expansion", self.logger):
            # Get query embedding
            query_embedding = await self.embedding_generator.embed_query(query)
            
            expansion_terms = []
            expansion_weights = {}
            
            # Get corpus terms for semantic comparison
            vector_store = context.get("vector_store") if context else None
            if vector_store:
                corpus_terms = await self._get_corpus_terms(vector_store)
            else:
                corpus_terms = self.corpus_terms
            
            # Find semantically similar terms
            similar_terms = []
            for term in corpus_terms:
                if term.lower() not in query.lower():
                    # Get or compute term embedding
                    if term not in self.term_embeddings_cache:
                        term_embedding = await self.embedding_generator.embed_text(term)
                        self.term_embeddings_cache[term] = term_embedding
                    else:
                        term_embedding = self.term_embeddings_cache[term]
                    
                    # Compute similarity
                    similarity = await self.embedding_generator.compute_similarity(
                        query_embedding, term_embedding
                    )
                    
                    if similarity >= self.config.min_similarity_threshold:
                        similar_terms.append((term, similarity))
            
            # Sort by similarity and select top terms
            similar_terms.sort(key=lambda x: x[1], reverse=True)
            
            for term, similarity in similar_terms[:self.config.max_expansion_terms]:
                expansion_terms.append(term)
                expansion_weights[term] = similarity * self.config.expansion_weight
            
            # Create expanded query
            if expansion_terms:
                expanded_query = f"{query} {' '.join(expansion_terms)}"
            else:
                expanded_query = query
            
            return ExpandedQuery(
                original_query=query,
                expanded_query=expanded_query,
                expansion_terms=expansion_terms,
                expansion_weights=expansion_weights,
                expansion_method="semantic",
                metadata={
                    "semantic_matches": len(similar_terms),
                    "avg_similarity": np.mean([sim for _, sim in similar_terms]) if similar_terms else 0
                }
            )


class ContextualExpander(QueryExpander):
    """Expands queries based on conversation context and user history."""
    
    async def expand_query(self, query: str, context: Dict[str, Any] = None) -> ExpandedQuery:
        """Expand query using contextual information."""
        async with PerformanceLogger("contextual_expansion", self.logger):
            expansion_terms = []
            expansion_weights = {}
            
            if context:
                # Extract terms from conversation history
                conversation_history = context.get("conversation_history", [])
                for turn in conversation_history[-self.config.context_window_size:]:
                    if "query" in turn:
                        prev_query = turn["query"]
                        # Extract key terms from previous queries
                        prev_terms = re.findall(r'\b\w{4,}\b', prev_query.lower())
                        for term in prev_terms:
                            if term not in query.lower() and term not in expansion_terms:
                                expansion_terms.append(term)
                                expansion_weights[term] = self.config.expansion_weight * 0.5
                
                # Extract terms from previously relevant documents
                relevant_docs = context.get("relevant_documents", [])
                doc_terms = set()
                for doc in relevant_docs:
                    if "content" in doc:
                        content_terms = re.findall(r'\b\w{4,}\b', doc["content"].lower())
                        doc_terms.update(content_terms[:10])  # Top 10 terms per doc
                
                # Add contextually relevant terms
                for term in list(doc_terms)[:self.config.max_expansion_terms]:
                    if term not in query.lower() and term not in expansion_terms:
                        expansion_terms.append(term)
                        expansion_weights[term] = self.config.expansion_weight * 0.3
            
            # Limit expansion terms
            expansion_terms = expansion_terms[:self.config.max_expansion_terms]
            
            # Create expanded query
            if expansion_terms:
                expanded_query = f"{query} {' '.join(expansion_terms)}"
            else:
                expanded_query = query
            
            return ExpandedQuery(
                original_query=query,
                expanded_query=expanded_query,
                expansion_terms=expansion_terms,
                expansion_weights=expansion_weights,
                expansion_method="contextual",
                metadata={
                    "context_sources": len(context.get("conversation_history", [])) if context else 0,
                    "relevant_docs": len(context.get("relevant_documents", [])) if context else 0
                }
            )


class HybridExpander(QueryExpander):
    """Combines multiple expansion strategies for optimal results."""
    
    def __init__(self, config: QueryExpansionConfig, embedding_generator=None, logger=None):
        super().__init__(config, logger)
        self.synonym_expander = SynonymExpander(config, logger)
        self.semantic_expander = SemanticExpander(config, embedding_generator, logger)
        self.contextual_expander = ContextualExpander(config, logger)
    
    async def expand_query(self, query: str, context: Dict[str, Any] = None) -> ExpandedQuery:
        """Expand query using hybrid approach combining multiple strategies."""
        async with PerformanceLogger("hybrid_expansion", self.logger):
            # Run all expansion strategies
            synonym_result = await self.synonym_expander.expand_query(query, context)
            semantic_result = await self.semantic_expander.expand_query(query, context)
            contextual_result = await self.contextual_expander.expand_query(query, context)
            
            # Combine expansion terms with weighted scoring
            all_terms = {}
            
            # Add synonym terms with base weight
            for term in synonym_result.expansion_terms:
                all_terms[term] = all_terms.get(term, 0) + synonym_result.expansion_weights.get(term, 0.3)
            
            # Add semantic terms with higher weight
            for term in semantic_result.expansion_terms:
                all_terms[term] = all_terms.get(term, 0) + semantic_result.expansion_weights.get(term, 0.5)
            
            # Add contextual terms with medium weight
            for term in contextual_result.expansion_terms:
                all_terms[term] = all_terms.get(term, 0) + contextual_result.expansion_weights.get(term, 0.4)
            
            # Sort terms by combined weight and select top terms
            sorted_terms = sorted(all_terms.items(), key=lambda x: x[1], reverse=True)
            expansion_terms = [term for term, _ in sorted_terms[:self.config.max_expansion_terms]]
            expansion_weights = dict(sorted_terms[:self.config.max_expansion_terms])
            
            # Create expanded query
            if expansion_terms:
                expanded_query = f"{query} {' '.join(expansion_terms)}"
            else:
                expanded_query = query
            
            return ExpandedQuery(
                original_query=query,
                expanded_query=expanded_query,
                expansion_terms=expansion_terms,
                expansion_weights=expansion_weights,
                expansion_method="hybrid",
                metadata={
                    "synonym_terms": len(synonym_result.expansion_terms),
                    "semantic_terms": len(semantic_result.expansion_terms),
                    "contextual_terms": len(contextual_result.expansion_terms),
                    "total_candidates": len(all_terms)
                }
            )


class QueryExpansionService:
    """Main service for query expansion with strategy management."""
    
    def __init__(self, config: QueryExpansionConfig = None, embedding_generator=None, logger=None):
        self.config = config or QueryExpansionConfig()
        self.logger = logger or get_logger(__name__)
        self.embedding_generator = embedding_generator
        
        # Initialize expanders
        self.expanders = {
            ExpansionStrategy.SYNONYMS: SynonymExpander(self.config, self.logger),
            ExpansionStrategy.SEMANTIC: SemanticExpander(self.config, embedding_generator, self.logger),
            ExpansionStrategy.CONTEXTUAL: ContextualExpander(self.config, self.logger),
            ExpansionStrategy.HYBRID: HybridExpander(self.config, embedding_generator, self.logger),
        }
    
    async def expand_query(self, query: str, 
                          strategy: ExpansionStrategy = None, 
                          context: Dict[str, Any] = None) -> ExpandedQuery:
        """Expand a query using the specified strategy."""
        strategy = strategy or self.config.strategy
        
        if strategy == ExpansionStrategy.NONE:
            return ExpandedQuery(
                original_query=query,
                expanded_query=query,
                expansion_terms=[],
                expansion_weights={},
                expansion_method="none"
            )
        
        if strategy not in self.expanders:
            self.logger.warning(f"Unknown expansion strategy: {strategy}, falling back to hybrid")
            strategy = ExpansionStrategy.HYBRID
        
        expander = self.expanders[strategy]
        result = await expander.expand_query(query, context)
        
        self.logger.info(
            "Query expansion completed",
            original_query=query[:50],
            expansion_method=result.expansion_method,
            terms_added=len(result.expansion_terms),
            strategy=strategy
        )
        
        return result
    
    async def get_expansion_stats(self) -> Dict[str, Any]:
        """Get statistics about query expansion performance."""
        return {
            "available_strategies": list(self.expanders.keys()),
            "current_strategy": self.config.strategy,
            "max_expansion_terms": self.config.max_expansion_terms,
            "min_similarity_threshold": self.config.min_similarity_threshold,
            "expansion_weight": self.config.expansion_weight,
            "embedding_generator_available": self.embedding_generator is not None
        }