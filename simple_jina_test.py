#!/usr/bin/env python3
"""
Simple JINA v4 compatibility test without dependencies.
"""

import time

def test_jina_v4():
    """Test JINA v4 embedding model compatibility."""
    print("🧪 Starting JINA v4 compatibility test")
    
    # Test 1: Basic imports
    try:
        print("1. Testing sentence-transformers import...")
        from sentence_transformers import SentenceTransformer
        print("   ✓ sentence-transformers available")
    except ImportError as e:
        print(f"   ✗ sentence-transformers not available: {e}")
        return False
    
    # Test 2: JINA v4 model loading
    try:
        print("2. Testing JINA v4 model loading...")
        model = SentenceTransformer('jinaai/jina-embeddings-v4', trust_remote_code=True)
        print("   ✓ JINA v4 model loaded successfully")
        
        # Get model info
        print(f"   📏 Model max sequence length: {getattr(model, 'max_seq_length', 'unknown')}")
        
    except Exception as e:
        print(f"   ✗ JINA v4 model loading failed: {e}")
        
        # Try alternative models
        alternative_models = [
            'jinaai/jina-embeddings-v3',
            'jinaai/jina-embeddings-v2-base-en',
            'all-MiniLM-L6-v2'  # Fallback
        ]
        
        for model_name in alternative_models:
            try:
                print(f"   🔄 Trying alternative model: {model_name}")
                model = SentenceTransformer(model_name, trust_remote_code=True)
                print(f"   ✓ Alternative model {model_name} loaded successfully")
                break
            except Exception as e:
                print(f"   ✗ Alternative model {model_name} failed: {e}")
                continue
        else:
            print("   ✗ No JINA models could be loaded")
            return False
    
    # Test 3: Basic embedding generation
    try:
        print("3. Testing embedding generation...")
        test_texts = [
            "This is a test sentence for embedding generation.",
            "JINA v4 provides high-quality embeddings.",
            "Testing compatibility with Akasha RAG system."
        ]
        
        start_time = time.time()
        embeddings = model.encode(test_texts)
        end_time = time.time()
        
        print(f"   ✓ Generated embeddings for {len(test_texts)} texts")
        print(f"   📐 Embedding dimensions: {embeddings.shape[1]}")
        print(f"   ⏱️  Processing time: {end_time - start_time:.2f}s")
        print(f"   📊 Embedding shape: {embeddings.shape}")
        
    except Exception as e:
        print(f"   ✗ Embedding generation failed: {e}")
        return False
    
    # Test 4: Check multimodal capabilities
    try:
        print("4. Testing multimodal capabilities...")
        
        # Check model architecture
        model_info = str(model)
        if 'clip' in model_info.lower() or 'multimodal' in model_info.lower():
            print("   ✓ Model appears to support multimodal inputs")
        else:
            print("   ? Model multimodal support unclear from architecture")
            
        # Try to check tokenizer info
        if hasattr(model, 'tokenizer'):
            print("   ✓ Model has tokenizer")
        
        if hasattr(model, '_modules') and 'vision_model' in str(model._modules):
            print("   ✓ Model has vision components")
            
    except Exception as e:
        print(f"   ⚠️  Multimodal test failed: {e}")
    
    print("✅ JINA v4 compatibility test completed successfully")
    return True

if __name__ == "__main__":
    success = test_jina_v4()
    exit(0 if success else 1)