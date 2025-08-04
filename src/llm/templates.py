"""
Prompt templates for Akasha RAG system.

This module provides prompt templates for various LLM interactions,
including RAG-specific templates and general conversation templates.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from pydantic import BaseModel, Field

from ..rag.ingestion import DocumentChunk
from ..rag.retrieval import RetrievalResult


class TemplateType(str, Enum):
    """Types of prompt templates."""
    RAG_QA = "rag_qa"
    RAG_SUMMARY = "rag_summary"
    RAG_ANALYSIS = "rag_analysis"
    RAG_COMPARISON = "rag_comparison"
    CONVERSATION = "conversation"
    SYSTEM = "system"
    INSTRUCTION = "instruction"


class ModelFamily(str, Enum):
    """Model families with different prompt formats."""
    GEMMA = "gemma"
    LLAMA = "llama"
    MISTRAL = "mistral"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GENERIC = "generic"


@dataclass
class ContextChunk:
    """Context chunk for prompt template."""
    content: str
    source: str
    relevance_score: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PromptTemplate(ABC):
    """Abstract base class for prompt templates."""
    
    def __init__(self, model_family: ModelFamily = ModelFamily.GENERIC):
        self.model_family = model_family
    
    @abstractmethod
    def format(self, **kwargs) -> str:
        """Format the template with provided arguments."""
        pass
    
    def get_token_estimate(self, formatted_prompt: str) -> int:
        """Estimate token count for a formatted prompt."""
        # Rough estimation: ~4 characters per token
        return len(formatted_prompt) // 4
    
    def truncate_to_context(self, content: str, max_tokens: int) -> str:
        """Truncate content to fit within token limit."""
        max_chars = max_tokens * 4  # Rough estimation
        if len(content) <= max_chars:
            return content
        
        # Truncate at sentence boundary if possible
        truncated = content[:max_chars]
        last_sentence = truncated.rfind('.')
        if last_sentence > max_chars * 0.8:  # If we can keep 80% and end at sentence
            return truncated[:last_sentence + 1]
        
        return truncated + "..."


class RAGTemplate(PromptTemplate):
    """Template for RAG-based question answering."""
    
    def __init__(self, model_family: ModelFamily = ModelFamily.GENERIC, template_type: TemplateType = TemplateType.RAG_QA):
        super().__init__(model_family)
        self.template_type = template_type
    
    def format(self, 
               query: str,
               context_chunks: List[ContextChunk],
               max_context_tokens: int = 3000,
               include_metadata: bool = True,
               **kwargs) -> str:
        """Format RAG prompt with query and context."""
        
        # Select and format context
        formatted_context = self._format_context(
            context_chunks, 
            max_context_tokens,
            include_metadata
        )
        
        # Get template based on model family and type
        if self.model_family == ModelFamily.GEMMA:
            return self._format_gemma(query, formatted_context, **kwargs)
        elif self.model_family == ModelFamily.LLAMA:
            return self._format_llama(query, formatted_context, **kwargs)
        elif self.model_family == ModelFamily.MISTRAL:
            return self._format_mistral(query, formatted_context, **kwargs)
        elif self.model_family == ModelFamily.OPENAI:
            return self._format_openai(query, formatted_context, **kwargs)
        elif self.model_family == ModelFamily.ANTHROPIC:
            return self._format_anthropic(query, formatted_context, **kwargs)
        else:
            return self._format_generic(query, formatted_context, **kwargs)
    
    def _format_context(self, 
                       context_chunks: List[ContextChunk], 
                       max_tokens: int,
                       include_metadata: bool = True) -> str:
        """Format context chunks into a cohesive context string."""
        if not context_chunks:
            return "No relevant context found."
        
        context_parts = []
        current_tokens = 0
        
        for i, chunk in enumerate(context_chunks):
            # Format individual chunk
            chunk_text = f"[Document {i+1}]"
            
            if include_metadata and chunk.metadata:
                # Add relevant metadata
                if "file_name" in chunk.metadata:
                    chunk_text += f" (Source: {chunk.metadata['file_name']})"
                if "custom_page_count" in chunk.metadata:
                    chunk_text += f" (Page {chunk.metadata.get('custom_page_count', 'Unknown')})"
            
            chunk_text += f"\n{chunk.content}\n"
            
            # Check token limit
            chunk_tokens = self.get_token_estimate(chunk_text)
            if current_tokens + chunk_tokens > max_tokens:
                if current_tokens == 0:  # First chunk is too long
                    # Truncate first chunk to fit
                    available_tokens = max_tokens - self.get_token_estimate(f"[Document 1]\n\n")
                    truncated_content = self.truncate_to_context(chunk.content, available_tokens)
                    chunk_text = f"[Document 1]\n{truncated_content}\n"
                    context_parts.append(chunk_text)
                break
            
            context_parts.append(chunk_text)
            current_tokens += chunk_tokens
        
        return "\n".join(context_parts)
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for RAG."""
        return """You are an intelligent research assistant. Your task is to answer questions based on the provided context documents. Please follow these guidelines:

1. Answer based on the provided context - don't use external knowledge unless specifically asked
2. If the context doesn't contain enough information, clearly state this
3. Cite your sources when possible by referencing document numbers
4. Be precise and factual in your responses
5. If asked to summarize, provide a concise but comprehensive overview
6. If asked to analyze, provide detailed insights and connections
7. If asked to compare, highlight similarities and differences clearly"""
    
    def _format_gemma(self, query: str, context: str, **kwargs) -> str:
        """Format prompt for Gemma models."""
        system_prompt = kwargs.get("system_prompt", self._get_default_system_prompt())
        
        if self.template_type == TemplateType.RAG_SUMMARY:
            instruction = "Please provide a comprehensive summary based on the provided context."
        elif self.template_type == TemplateType.RAG_ANALYSIS:
            instruction = "Please analyze the information in the provided context and answer the question."
        elif self.template_type == TemplateType.RAG_COMPARISON:
            instruction = "Please compare and contrast the information in the provided context."
        else:
            instruction = "Please answer the question based on the provided context."
        
        prompt = f"""<bos><start_of_turn>user
{system_prompt}

Context:
{context}

Question: {query}

{instruction}
<end_of_turn>
<start_of_turn>model
"""
        return prompt
    
    def _format_llama(self, query: str, context: str, **kwargs) -> str:
        """Format prompt for Llama models."""
        system_prompt = kwargs.get("system_prompt", self._get_default_system_prompt())
        
        prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

Context:
{context}

Question: {query}

Please answer the question based on the provided context. If the context doesn't contain enough information, please say so. [/INST]
"""
        return prompt
    
    def _format_mistral(self, query: str, context: str, **kwargs) -> str:
        """Format prompt for Mistral models."""
        system_prompt = kwargs.get("system_prompt", self._get_default_system_prompt())
        
        prompt = f"""<s>[INST] {system_prompt}

Context:
{context}

Question: {query}

Please answer the question based on the provided context. [/INST]
"""
        return prompt
    
    def _format_openai(self, query: str, context: str, **kwargs) -> str:
        """Format prompt for OpenAI models."""
        system_prompt = kwargs.get("system_prompt", self._get_default_system_prompt())
        
        # OpenAI uses message format, but we return as single string for consistency
        prompt = f"""System: {system_prompt}

Context:
{context}

User: {query}

Assistant:"""
        return prompt
    
    def _format_anthropic(self, query: str, context: str, **kwargs) -> str:
        """Format prompt for Anthropic models."""
        system_prompt = kwargs.get("system_prompt", self._get_default_system_prompt())
        
        prompt = f"""Human: {system_prompt}

Context:
{context}

Question: {query}

Please answer the question based on the provided context.

Assistant:"""
        return prompt
    
    def _format_generic(self, query: str, context: str, **kwargs) -> str:
        """Format prompt for generic models."""
        system_prompt = kwargs.get("system_prompt", self._get_default_system_prompt())
        
        prompt = f"""{system_prompt}

Context:
{context}

Question: {query}

Answer:"""
        return prompt
    
    @classmethod
    def from_retrieval_result(cls, 
                            query: str, 
                            retrieval_result: RetrievalResult,
                            model_family: ModelFamily = ModelFamily.GENERIC,
                            template_type: TemplateType = TemplateType.RAG_QA) -> str:
        """Create prompt from retrieval result."""
        template = cls(model_family, template_type)
        
        # Convert chunks to ContextChunk format
        context_chunks = []
        for chunk, score in zip(retrieval_result.chunks, retrieval_result.scores):
            context_chunk = ContextChunk(
                content=chunk.content,
                source=chunk.metadata.get('file_name', 'Unknown'),
                relevance_score=score,
                metadata=chunk.metadata
            )
            context_chunks.append(context_chunk)
        
        return template.format(query=query, context_chunks=context_chunks)


class ConversationTemplate(PromptTemplate):
    """Template for conversational interactions."""
    
    def __init__(self, model_family: ModelFamily = ModelFamily.GENERIC):
        super().__init__(model_family)
        self.conversation_history = []
    
    def format(self, message: str, **kwargs) -> str:
        """Format conversational prompt."""
        system_prompt = kwargs.get("system_prompt", "You are a helpful AI assistant.")
        
        if self.model_family == ModelFamily.GEMMA:
            return self._format_gemma_conversation(message, system_prompt)
        elif self.model_family == ModelFamily.LLAMA:
            return self._format_llama_conversation(message, system_prompt)
        elif self.model_family == ModelFamily.OPENAI:
            return self._format_openai_conversation(message, system_prompt)
        else:
            return self._format_generic_conversation(message, system_prompt)
    
    def _format_gemma_conversation(self, message: str, system_prompt: str) -> str:
        """Format conversation for Gemma."""
        conversation = f"<bos><start_of_turn>user\n{system_prompt}\n\n{message}<end_of_turn>\n<start_of_turn>model\n"
        return conversation
    
    def _format_llama_conversation(self, message: str, system_prompt: str) -> str:
        """Format conversation for Llama."""
        conversation = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{message} [/INST]"
        return conversation
    
    def _format_openai_conversation(self, message: str, system_prompt: str) -> str:
        """Format conversation for OpenAI."""
        conversation = f"System: {system_prompt}\n\nUser: {message}\n\nAssistant:"
        return conversation
    
    def _format_generic_conversation(self, message: str, system_prompt: str) -> str:
        """Format generic conversation."""
        conversation = f"{system_prompt}\n\nUser: {message}\n\nAssistant:"
        return conversation
    
    def add_exchange(self, user_message: str, assistant_response: str):
        """Add a conversation exchange to history."""
        self.conversation_history.append({
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()


class TemplateManager:
    """Manages different prompt templates."""
    
    def __init__(self, model_family: ModelFamily = ModelFamily.GENERIC):
        self.model_family = model_family
        self.templates = {}
    
    def get_rag_template(self, template_type: TemplateType = TemplateType.RAG_QA) -> RAGTemplate:
        """Get RAG template."""
        key = f"rag_{template_type.value}"
        if key not in self.templates:
            self.templates[key] = RAGTemplate(self.model_family, template_type)
        return self.templates[key]
    
    def get_conversation_template(self) -> ConversationTemplate:
        """Get conversation template."""
        key = "conversation"
        if key not in self.templates:
            self.templates[key] = ConversationTemplate(self.model_family)
        return self.templates[key]
    
    def create_rag_prompt(self, 
                         query: str,
                         retrieval_result: RetrievalResult,
                         template_type: TemplateType = TemplateType.RAG_QA,
                         **kwargs) -> str:
        """Create RAG prompt from retrieval result."""
        template = self.get_rag_template(template_type)
        return RAGTemplate.from_retrieval_result(
            query, retrieval_result, self.model_family, template_type
        )
    
    def create_conversation_prompt(self, message: str, **kwargs) -> str:
        """Create conversation prompt."""
        template = self.get_conversation_template()
        return template.format(message, **kwargs)