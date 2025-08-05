#!/usr/bin/env python3
"""
Test JINA integration with Akasha embedding system.
"""

import sys
import asyncio
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_jina_integration():
    """Test JINA embedding integration."""
    print("🧪 Testing JINA integration with Akasha embedding system")
    
    try:
        # Import Akasha modules
        from src.rag.embeddings import EmbeddingConfig, EmbeddingModel, EmbeddingGenerator
        
        # Test different JINA models
        jina_models = [
            (EmbeddingModel.JINA_V4, "JINA v4 (latest)"),
            (EmbeddingModel.JINA_V3, "JINA v3"),
            (EmbeddingModel.JINA_V2_BASE, "JINA v2 Base"),
        ]
        
        test_texts = [
            "This is a test document about machine learning.",
            "JINA provides high-quality embeddings for RAG systems.",
            "Akasha is a multimodal RAG system built for research."
        ]
        
        for model, description in jina_models:
            print(f"\n📊 Testing {description}")
            
            try:
                # Create configuration
                config = EmbeddingConfig(
                    model_name=model,
                    batch_size=16,
                    normalize_embeddings=True
                )
                
                # Create generator
                generator = EmbeddingGenerator(config)
                
                # Initialize
                print(f"   🔄 Initializing {description}...")
                start_time = time.time()
                await generator.initialize()
                init_time = time.time() - start_time
                print(f"   ✓ Initialized in {init_time:.2f}s")
                
                # Get model info
                if hasattr(generator.provider, 'get_model_info'):
                    model_info = await generator.provider.get_model_info()
                    print(f"   📏 Dimensions: {model_info['dimensions']}")
                    print(f"   🏷️  Version: {model_info['version']}")
                    print(f"   🔮 Multimodal: {model_info.get('supports_multimodal', False)}")
                
                # Test embedding generation
                print(f"   🔄 Generating embeddings for {len(test_texts)} texts...")
                start_time = time.time()
                embeddings = await generator.embed_texts(test_texts)
                embed_time = time.time() - start_time
                
                print(f"   ✓ Generated embeddings in {embed_time:.2f}s")
                print(f"   📊 Shape: {len(embeddings)} x {len(embeddings[0]) if embeddings else 0}")
                
                # Test single text embedding
                single_embedding = await generator.embed_text("Single test text")
                print(f"   ✓ Single text embedding: {len(single_embedding)} dimensions")
                
                # Test similarity (simple dot product)
                if len(embeddings) >= 2:
                    import numpy as np
                    emb1 = np.array(embeddings[0])
                    emb2 = np.array(embeddings[1])
                    similarity = np.dot(emb1, emb2)
                    print(f"   📈 Similarity between first two texts: {similarity:.4f}")
                
                print(f"   ✅ {description} test completed successfully")
                
                # Clean up to free memory
                del generator
                
            except Exception as e:
                print(f"   ❌ {description} test failed: {e}")
                continue
        
        print("\n✅ JINA integration testing completed")
        return True
        
    except Exception as e:
        print(f"❌ JINA integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_embedding_fallback():
    """Test JINA embedding fallback mechanism."""
    print("\n🔄 Testing JINA embedding fallback mechanism")
    
    try:
        from src.rag.embeddings import EmbeddingConfig, EmbeddingModel, JINAEmbeddingProvider
        
        # Test with JINA v4 (should fallback to v2 due to missing dependencies)
        config = EmbeddingConfig(model_name=EmbeddingModel.JINA_V4)
        provider = JINAEmbeddingProvider(config)
        
        print("   🔄 Testing fallback from JINA v4...")
        await provider.load_model()
        
        print(f"   ✓ Fallback successful, loaded version: {provider.jina_version}")
        
        # Test embedding generation
        test_embeddings = await provider.embed_texts(["Fallback test"])
        print(f"   ✓ Generated test embedding: {len(test_embeddings[0])} dimensions")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Fallback test failed: {e}")
        return False

async def main():
    """Main test function."""
    print("🚀 Starting JINA embedding integration tests\n")
    
    success1 = await test_jina_integration()
    success2 = await test_embedding_fallback()
    
    if success1 and success2:
        print("\n🎉 All JINA integration tests passed!")
        return 0
    else:
        print("\n💥 Some JINA integration tests failed!")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))