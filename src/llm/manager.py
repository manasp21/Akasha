"""
LLM Manager for Akasha RAG system.

This module provides high-level management of LLM providers, including
load balancing, fallback handling, and performance monitoring.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, AsyncIterator, Union
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from ..core.logging import get_logger, PerformanceLogger
from ..core.exceptions import AkashaError
from .config import LLMConfig, LLMBackend
from .provider import LLMProvider, LLMResponse, StreamingEvent, create_llm_provider
from .templates import TemplateManager, ModelFamily, TemplateType, ContextChunk
from ..rag.retrieval import RetrievalResult


class ProviderStatus(str, Enum):
    """Status of LLM provider."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    INITIALIZING = "initializing"


@dataclass
class ProviderStats:
    """Statistics for an LLM provider."""
    requests_total: int = 0
    requests_successful: int = 0
    requests_failed: int = 0
    avg_response_time: float = 0.0
    total_tokens_generated: int = 0
    last_request_time: Optional[float] = None
    status: ProviderStatus = ProviderStatus.HEALTHY


class LLMManager:
    """Manages multiple LLM providers with fallback and load balancing."""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger(__name__)
        self.providers: Dict[str, LLMProvider] = {}
        self.provider_configs: Dict[str, LLMConfig] = {}
        self.stats: Dict[str, ProviderStats] = {}
        self.template_managers: Dict[str, TemplateManager] = {}
        self.primary_provider: Optional[str] = None
        self.fallback_providers: List[str] = []
        self._initialization_lock = asyncio.Lock()
        self.initialized = False
    
    async def add_provider(self, name: str, config: LLMConfig, is_primary: bool = False) -> None:
        """Add an LLM provider."""
        async with self._initialization_lock:
            try:
                # Create provider
                provider = create_llm_provider(config, self.logger)
                
                # Store provider and config
                self.providers[name] = provider
                self.provider_configs[name] = config
                self.stats[name] = ProviderStats(status=ProviderStatus.INITIALIZING)
                
                # Create template manager based on model family
                model_family = self._get_model_family(config)
                self.template_managers[name] = TemplateManager(model_family)
                
                # Set as primary if specified
                if is_primary or not self.primary_provider:
                    self.primary_provider = name
                else:
                    self.fallback_providers.append(name)
                
                self.logger.info(
                    "LLM provider added",
                    provider_name=name,
                    backend=config.backend,
                    model=config.model.name,
                    is_primary=is_primary
                )
                
            except Exception as e:
                self.logger.error(f"Failed to add provider {name}: {e}")
                raise AkashaError(f"Failed to add LLM provider {name}: {e}")
    
    async def initialize(self) -> None:
        """Initialize all providers."""
        if self.initialized:
            return
        
        async with self._initialization_lock:
            if not self.providers:
                raise AkashaError("No LLM providers configured")
            
            # Initialize all providers concurrently
            init_tasks = []
            for name, provider in self.providers.items():
                task = self._initialize_provider(name, provider)
                init_tasks.append(task)
            
            # Wait for all initializations to complete
            results = await asyncio.gather(*init_tasks, return_exceptions=True)
            
            # Check results
            successful_providers = []
            for name, result in zip(self.providers.keys(), results):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to initialize provider {name}: {result}")
                    self.stats[name].status = ProviderStatus.UNHEALTHY
                else:
                    successful_providers.append(name)
                    self.stats[name].status = ProviderStatus.HEALTHY
            
            if not successful_providers:
                raise AkashaError("Failed to initialize any LLM providers")
            
            # Ensure primary provider is healthy
            if self.primary_provider not in successful_providers:
                # Promote first healthy provider to primary
                self.primary_provider = successful_providers[0]
                self.fallback_providers = [p for p in successful_providers[1:]]
                self.logger.warning(
                    f"Primary provider unhealthy, promoted {self.primary_provider} to primary"
                )
            
            self.initialized = True
            self.logger.info(
                "LLM Manager initialized successfully",
                primary_provider=self.primary_provider,
                fallback_providers=self.fallback_providers,
                total_providers=len(successful_providers)
            )
    
    async def _initialize_provider(self, name: str, provider: LLMProvider) -> None:
        """Initialize a single provider."""
        try:
            async with PerformanceLogger(f"provider_init:{name}", self.logger):
                await provider.initialize()
                self.logger.info(f"Provider {name} initialized successfully")
        except Exception as e:
            self.logger.error(f"Provider {name} initialization failed: {e}")
            raise
    
    async def generate(self, prompt: str, provider_name: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate response using specified or best available provider."""
        if not self.initialized:
            await self.initialize()
        
        # Determine which provider to use
        if provider_name and provider_name in self.providers:
            providers_to_try = [provider_name]
        else:
            providers_to_try = self._get_provider_priority_list()
        
        last_exception = None
        
        for provider_name in providers_to_try:
            if self.stats[provider_name].status == ProviderStatus.UNHEALTHY:
                continue
            
            try:
                start_time = time.time()
                provider = self.providers[provider_name]
                
                # Generate response
                response = await provider.generate(prompt, **kwargs)
                
                # Update stats
                self._update_provider_stats(provider_name, True, time.time() - start_time, response.total_tokens)
                
                # Add provider info to response metadata
                response.metadata.update({
                    "provider_name": provider_name,
                    "provider_backend": self.provider_configs[provider_name].backend.value
                })
                
                return response
                
            except Exception as e:
                self.logger.warning(f"Provider {provider_name} failed: {e}")
                self._update_provider_stats(provider_name, False, time.time() - start_time if 'start_time' in locals() else 0.0, 0)
                last_exception = e
                
                # Mark provider as degraded after multiple failures
                if self.stats[provider_name].requests_failed >= 3:
                    self.stats[provider_name].status = ProviderStatus.DEGRADED
                
                continue
        
        # All providers failed
        raise AkashaError(f"All LLM providers failed. Last error: {last_exception}")
    
    async def generate_stream(self, prompt: str, provider_name: Optional[str] = None, **kwargs) -> AsyncIterator[StreamingEvent]:
        """Generate streaming response."""
        if not self.initialized:
            await self.initialize()
        
        # Determine which provider to use
        if provider_name and provider_name in self.providers:
            providers_to_try = [provider_name]
        else:
            providers_to_try = self._get_provider_priority_list()
        
        for provider_name in providers_to_try:
            if self.stats[provider_name].status == ProviderStatus.UNHEALTHY:
                continue
            
            try:
                provider = self.providers[provider_name]
                start_time = time.time()
                total_tokens = 0
                
                async for event in provider.generate_stream(prompt, **kwargs):
                    # Add provider info to event metadata
                    event.metadata.update({
                        "provider_name": provider_name,
                        "provider_backend": self.provider_configs[provider_name].backend.value
                    })
                    
                    if event.type == "finish":
                        total_tokens = event.metadata.get("total_tokens", 0)
                        self._update_provider_stats(provider_name, True, time.time() - start_time, total_tokens)
                    elif event.type == "error":
                        self._update_provider_stats(provider_name, False, time.time() - start_time, 0)
                    
                    yield event
                
                return  # Successfully completed
                
            except Exception as e:
                self.logger.warning(f"Streaming provider {provider_name} failed: {e}")
                self._update_provider_stats(provider_name, False, 0.0, 0)
                
                # Yield error event
                yield StreamingEvent(
                    type="error",
                    content=f"Provider {provider_name} failed: {e}",
                    metadata={"provider_name": provider_name, "error_type": type(e).__name__}
                )
                
                continue
        
        # All providers failed
        yield StreamingEvent(
            type="error",
            content="All LLM providers failed",
            metadata={"error_type": "AllProvidersFailed"}
        )
    
    async def generate_rag_response(self, 
                                   query: str,
                                   retrieval_result: RetrievalResult,
                                   template_type: TemplateType = TemplateType.RAG_QA,
                                   provider_name: Optional[str] = None,
                                   **kwargs) -> LLMResponse:
        """Generate RAG response using retrieved context."""
        if not self.initialized:
            await self.initialize()
        
        # Determine provider
        target_provider = provider_name or self.primary_provider
        if target_provider not in self.template_managers:
            raise AkashaError(f"Provider {target_provider} not found")
        
        # Create RAG prompt
        template_manager = self.template_managers[target_provider]
        prompt = template_manager.create_rag_prompt(
            query=query,
            retrieval_result=retrieval_result,
            template_type=template_type,
            **kwargs
        )
        
        # Generate response
        response = await self.generate(prompt, target_provider, **kwargs)
        
        # Add RAG metadata
        response.metadata.update({
            "query": query,
            "template_type": template_type.value,
            "context_chunks": len(retrieval_result.chunks),
            "retrieval_method": retrieval_result.retrieval_method,
            "retrieval_score": retrieval_result.total_score
        })
        
        return response
    
    async def generate_rag_stream(self,
                                 query: str,
                                 retrieval_result: RetrievalResult,
                                 template_type: TemplateType = TemplateType.RAG_QA,
                                 provider_name: Optional[str] = None,
                                 **kwargs) -> AsyncIterator[StreamingEvent]:
        """Generate streaming RAG response."""
        if not self.initialized:
            await self.initialize()
        
        # Determine provider
        target_provider = provider_name or self.primary_provider
        if target_provider not in self.template_managers:
            raise AkashaError(f"Provider {target_provider} not found")
        
        # Create RAG prompt
        template_manager = self.template_managers[target_provider]
        prompt = template_manager.create_rag_prompt(
            query=query,
            retrieval_result=retrieval_result,
            template_type=template_type,
            **kwargs
        )
        
        # Generate streaming response
        async for event in self.generate_stream(prompt, target_provider, **kwargs):
            # Add RAG metadata to events
            event.metadata.update({
                "query": query,
                "template_type": template_type.value,
                "context_chunks": len(retrieval_result.chunks),
                "retrieval_method": retrieval_result.retrieval_method
            })
            yield event
    
    def _get_provider_priority_list(self) -> List[str]:
        """Get providers in priority order (primary first, then healthy fallbacks)."""
        priority_list = []
        
        # Add primary provider if healthy
        if (self.primary_provider and 
            self.stats[self.primary_provider].status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]):
            priority_list.append(self.primary_provider)
        
        # Add healthy fallback providers
        for provider in self.fallback_providers:
            if self.stats[provider].status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]:
                priority_list.append(provider)
        
        return priority_list
    
    def _update_provider_stats(self, provider_name: str, success: bool, response_time: float, tokens: int):
        """Update provider statistics."""
        stats = self.stats[provider_name]
        stats.requests_total += 1
        stats.last_request_time = time.time()
        
        if success:
            stats.requests_successful += 1
            stats.total_tokens_generated += tokens
            
            # Update average response time
            if stats.requests_successful == 1:
                stats.avg_response_time = response_time
            else:
                stats.avg_response_time = (
                    (stats.avg_response_time * (stats.requests_successful - 1) + response_time) 
                    / stats.requests_successful
                )
            
            # Recover status if needed
            if stats.status == ProviderStatus.DEGRADED:
                if stats.requests_successful >= stats.requests_failed:
                    stats.status = ProviderStatus.HEALTHY
        else:
            stats.requests_failed += 1
    
    def _get_model_family(self, config: LLMConfig) -> ModelFamily:
        """Determine model family from config."""
        model_name = config.model.name.lower()
        
        if "gemma" in model_name:
            return ModelFamily.GEMMA
        elif "llama" in model_name:
            return ModelFamily.LLAMA
        elif "mistral" in model_name:
            return ModelFamily.MISTRAL
        elif config.backend.value == "openai":
            return ModelFamily.OPENAI
        elif config.backend.value == "anthropic":
            return ModelFamily.ANTHROPIC
        else:
            return ModelFamily.GENERIC
    
    def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all providers."""
        stats = {}
        for name, provider_stats in self.stats.items():
            stats[name] = {
                "status": provider_stats.status.value,
                "requests_total": provider_stats.requests_total,
                "requests_successful": provider_stats.requests_successful,
                "requests_failed": provider_stats.requests_failed,
                "success_rate": (
                    provider_stats.requests_successful / provider_stats.requests_total
                    if provider_stats.requests_total > 0 else 0.0
                ),
                "avg_response_time": provider_stats.avg_response_time,
                "total_tokens_generated": provider_stats.total_tokens_generated,
                "last_request_time": provider_stats.last_request_time,
                "backend": self.provider_configs[name].backend.value,
                "model": self.provider_configs[name].model.name
            }
        return stats
    
    def get_healthy_providers(self) -> List[str]:
        """Get list of healthy provider names."""
        return [
            name for name, stats in self.stats.items()
            if stats.status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all providers."""
        if not self.initialized:
            return {"initialized": False, "providers": {}}
        
        health_status = {
            "initialized": True,
            "primary_provider": self.primary_provider,
            "healthy_providers": self.get_healthy_providers(),
            "providers": {}
        }
        
        for name, provider in self.providers.items():
            try:
                # Simple health check with token estimation
                test_text = "Hello, world!"
                token_count = provider.estimate_tokens(test_text)
                
                health_status["providers"][name] = {
                    "status": self.stats[name].status.value,
                    "initialized": provider.initialized,
                    "token_estimation_working": token_count > 0,
                    "model_info": provider.get_model_info()
                }
            except Exception as e:
                health_status["providers"][name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return health_status
    
    async def estimate_tokens(self, text: str, provider_name: Optional[str] = None) -> int:
        """Estimate token count using specified or best available provider."""
        if not self.initialized:
            await self.initialize()
        
        target_provider = provider_name or self.primary_provider
        if target_provider not in self.providers:
            raise AkashaError(f"Provider {target_provider} not found")
        
        provider = self.providers[target_provider]
        return provider.estimate_tokens(text)
    
    def list_providers(self) -> List[Dict[str, Any]]:
        """List all configured providers."""
        providers = []
        for name in self.providers.keys():
            config = self.provider_configs[name]
            stats = self.stats[name]
            
            providers.append({
                "name": name,
                "backend": config.backend.value,
                "model": config.model.name,
                "status": stats.status.value,
                "is_primary": name == self.primary_provider,
                "initialized": self.providers[name].initialized
            })
        
        return providers