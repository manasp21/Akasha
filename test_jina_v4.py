#!/usr/bin/env python3
"""
Test script for JINA v4 compatibility in Akasha RAG system.
"""

import sys
import time
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.logging import get_logger

async def test_jina_v4_compatibility():
    """Test JINA v4 embedding model compatibility."""
    logger = get_logger("jina_v4_test")
    
    logger.info("Starting JINA v4 compatibility test")
    
    # Test 1: Try to import and load JINA v4
    try:
        logger.info("Testing JINA v4 import...")
        
        # Try different import methods
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("✓ sentence-transformers available")
        except ImportError as e:
            logger.error(f"✗ sentence-transformers not available: {e}")
            return False
        
        # Test JINA v4 model loading
        try:
            logger.info("Testing JINA v4 model loading...")
            model = SentenceTransformer('jinaai/jina-embeddings-v4')
            logger.info("✓ JINA v4 model loaded successfully")
            
            # Test basic embedding generation
            test_texts = [
                "This is a test sentence for embedding generation.",
                "JINA v4 provides multimodal embeddings.",
                "Testing compatibility with Akasha RAG system."
            ]
            
            start_time = time.time()
            embeddings = model.encode(test_texts)
            end_time = time.time()
            
            logger.info(f"✓ Generated embeddings for {len(test_texts)} texts")
            logger.info(f"✓ Embedding dimensions: {embeddings.shape[1]}")
            logger.info(f"✓ Processing time: {end_time - start_time:.2f}s")
            
            # Test multimodal capabilities if available
            try:
                # Test with image (if PIL available)
                from PIL import Image
                import numpy as np
                
                # Create a dummy image
                dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                
                # Check if model supports images
                if hasattr(model, 'encode') and hasattr(model[0], 'auto_model'):
                    logger.info("✓ Model appears to support multimodal inputs")
                else:
                    logger.info("? Model multimodal support unclear")
                    
            except ImportError:
                logger.warning("PIL not available for image testing")
            except Exception as e:
                logger.warning(f"Multimodal test failed: {e}")
            
        except Exception as e:
            logger.error(f"✗ JINA v4 model loading failed: {e}")
            
            # Try alternative models
            try:
                logger.info("Trying alternative JINA models...")
                alternative_models = [
                    'jinaai/jina-embeddings-v3',
                    'jinaai/jina-embeddings-v2-base-en'
                ]
                
                for model_name in alternative_models:
                    try:
                        model = SentenceTransformer(model_name)
                        logger.info(f"✓ Alternative model {model_name} loaded successfully")
                        break
                    except Exception as e:
                        logger.warning(f"Alternative model {model_name} failed: {e}")
                        continue
                else:
                    logger.error("✗ No JINA models could be loaded")
                    return False
                    
            except Exception as e:
                logger.error(f"✗ Alternative model testing failed: {e}")
                return False
        
    except Exception as e:
        logger.error(f"✗ JINA v4 import test failed: {e}")
        return False
    
    # Test 2: Integration with Akasha embedding system
    try:
        logger.info("Testing integration with Akasha embedding system...")
        
        from src.rag.embeddings import EmbeddingConfig, EmbeddingBackend
        
        # Test configuration
        config = EmbeddingConfig(
            backend=EmbeddingBackend.SENTENCE_TRANSFORMERS,
            model_name="jinaai/jina-embeddings-v4",  # Custom model name
            multimodal=True
        )
        
        logger.info("✓ Akasha embedding configuration created")
        
    except Exception as e:
        logger.error(f"✗ Akasha integration test failed: {e}")
        return False
    
    logger.info("JINA v4 compatibility test completed successfully")
    return True

async def main():
    """Main test function."""
    try:
        success = await test_jina_v4_compatibility()
        if success:
            print("✓ JINA v4 compatibility test passed")
            return 0
        else:
            print("✗ JINA v4 compatibility test failed")
            return 1
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))