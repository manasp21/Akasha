#!/usr/bin/env python3
"""Debug script to analyze chunking behavior."""

import asyncio
from src.rag.ingestion import DocumentChunker, ChunkingConfig, ChunkingStrategy

async def debug_chunking():
    """Debug the chunking issue with detailed analysis."""
    
    # Simple test content
    test_content = """Section 1: Introduction

This is section one with some content.

Section 2: Features  

This is section two with different content.

Section 3: Implementation

This is section three with more content."""
    
    print(f"Original content length: {len(test_content)} characters")
    print(f"Original content:\n{repr(test_content)}\n")
    
    # Configure chunker with small sizes to see the issue
    config = ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=50,  # Small size to force multiple chunks
        chunk_overlap=10,  # Small overlap
        min_chunk_size=5
    )
    
    chunker = DocumentChunker(config)
    chunks = await chunker.chunk_text(test_content, "debug_doc")
    
    print(f"Number of chunks created: {len(chunks)}")
    
    total_length = 0
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}:")
        print(f"  Length: {len(chunk.content)} chars")
        print(f"  Content: {repr(chunk.content[:100])}...")
        print(f"  Start: {chunk.start_offset}, End: {chunk.end_offset}")
        total_length += len(chunk.content)
    
    print(f"\nTotal content across all chunks: {total_length} characters")
    print(f"Expected maximum with overlap: {len(test_content) + (len(chunks) * config.chunk_overlap)}")
    print(f"Duplication factor: {total_length / len(test_content):.2f}x")

if __name__ == "__main__":
    asyncio.run(debug_chunking())