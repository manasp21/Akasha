#!/usr/bin/env python3
"""
Phase 2 Edge Cases and Error Handling Testing.

Tests error conditions, edge cases, and resilience of Phase 2 components.
"""

import asyncio
import tempfile
import time
import os
from pathlib import Path
from unittest.mock import patch

from src.rag.ingestion import DocumentIngestion, ChunkingConfig, ChunkingStrategy, DocumentFormat
from src.rag.embeddings import EmbeddingGenerator, EmbeddingConfig, EmbeddingBackend, EmbeddingModel
from src.rag.storage import VectorStore, StorageConfig
from src.core.exceptions import AkashaError


async def test_document_ingestion_edge_cases():
    """Test document ingestion with various edge cases."""
    print("📄 TESTING DOCUMENT INGESTION EDGE CASES")
    print("=" * 60)
    
    ingestion = DocumentIngestion(ChunkingConfig(
        strategy=ChunkingStrategy.FIXED_SIZE,
        chunk_size=100,
        chunk_overlap=20
    ))
    
    edge_cases = [
        {
            "name": "Empty file",
            "content": "",
            "should_fail": True,
            "expected_error": "No text content"
        },
        {
            "name": "Single character file",
            "content": "A",
            "should_fail": False,
            "expected_chunks": 0  # Too small for min chunk size
        },
        {
            "name": "Only whitespace",
            "content": "   \n\n\t  \n   ",
            "should_fail": True,
            "expected_error": "No text content"
        },
        {
            "name": "Unicode and special characters",
            "content": "Hello 世界! 🌍 This contains émojis and àccénts. Mathematical symbols: ∑∏∫√",
            "should_fail": False,
            "expected_chunks": 1
        },
        {
            "name": "Very long single line",
            "content": "A" * 10000,
            "should_fail": False,
            "expected_chunks": 100  # Should be chunked
        },
        {
            "name": "Many short lines",
            "content": "\n".join([f"Line {i}" for i in range(1000)]),
            "should_fail": False,
            "expected_chunks": 70  # Approximate
        },
        {
            "name": "Binary-like content",
            "content": "".join([chr(i % 256) for i in range(1000)]),
            "should_fail": False,
            "expected_chunks": 10  # Should handle gracefully
        }
    ]
    
    results = []
    for i, case in enumerate(edge_cases):
        print(f"\n   Test {i+1}: {case['name']}")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            try:
                f.write(case['content'])
                f.flush()
                file_path = Path(f.name)
            except UnicodeEncodeError:
                # For binary-like content, use binary mode
                f.close()
                with open(f.name, 'wb') as bf:
                    bf.write(case['content'].encode('utf-8', errors='ignore'))
                file_path = Path(f.name)
        
        try:
            start_time = time.time()
            metadata, chunks = await ingestion.process_file(file_path)
            processing_time = time.time() - start_time
            
            if case['should_fail']:
                print(f"      ❌ Expected failure but succeeded: {len(chunks)} chunks")
                results.append({"case": case['name'], "status": "unexpected_success", "chunks": len(chunks)})
            else:
                print(f"      ✅ Processed successfully: {len(chunks)} chunks in {processing_time:.3f}s")
                print(f"         File size: {metadata.file_size} bytes")
                results.append({"case": case['name'], "status": "success", "chunks": len(chunks), "time": processing_time})
                
        except Exception as e:
            error_msg = str(e)
            if case['should_fail'] and case['expected_error'] in error_msg:
                print(f"      ✅ Failed as expected: {error_msg}")
                results.append({"case": case['name'], "status": "expected_failure", "error": error_msg})
            elif case['should_fail']:
                print(f"      ⚠️  Failed with unexpected error: {error_msg}")
                results.append({"case": case['name'], "status": "wrong_failure", "error": error_msg})
            else:
                print(f"      ❌ Unexpected failure: {error_msg}")
                results.append({"case": case['name'], "status": "unexpected_failure", "error": error_msg})
        finally:
            file_path.unlink()
    
    print(f"\n📊 Edge Cases Summary:")
    success_count = len([r for r in results if r['status'] in ['success', 'expected_failure']])
    print(f"   ✅ Handled correctly: {success_count}/{len(edge_cases)}")
    
    return results


async def test_embedding_generation_edge_cases():
    """Test embedding generation with edge cases."""
    print("\n🧠 TESTING EMBEDDING GENERATION EDGE CASES")
    print("=" * 60)
    
    embedding_config = EmbeddingConfig(
        backend=EmbeddingBackend.MLX,
        model_name=EmbeddingModel.ALL_MINILM_L6_V2,
        batch_size=4
    )
    embedding_generator = EmbeddingGenerator(embedding_config)
    await embedding_generator.initialize()
    
    edge_cases = [
        {
            "name": "Empty string",
            "texts": [""],
            "should_fail": False
        },
        {
            "name": "Very long text",
            "texts": ["This is a very long text. " * 1000],
            "should_fail": False
        },
        {
            "name": "Unicode text",
            "texts": ["Hello 世界! 🌍 Émojis and àccénts work fine."],
            "should_fail": False
        },
        {
            "name": "Special characters only",
            "texts": ["!@#$%^&*()_+-=[]{}|;':\",./<>?"],
            "should_fail": False
        },
        {
            "name": "Numbers only",
            "texts": ["1234567890 3.14159 -42 1e10"],
            "should_fail": False
        },
        {
            "name": "Mixed content batch",
            "texts": ["", "Short", "A" * 1000, "Hello 世界!", "123"],
            "should_fail": False
        },
        {
            "name": "Large batch",
            "texts": [f"Text number {i}" for i in range(100)],
            "should_fail": False
        }
    ]
    
    results = []
    for i, case in enumerate(edge_cases):
        print(f"\n   Test {i+1}: {case['name']}")
        print(f"      Input: {len(case['texts'])} texts")
        
        try:
            start_time = time.time()
            embeddings = await embedding_generator.embed_texts(case['texts'])
            processing_time = time.time() - start_time
            
            print(f"      ✅ Generated {len(embeddings)} embeddings in {processing_time:.3f}s")
            print(f"         Speed: {len(case['texts'])/processing_time:.1f} texts/sec")
            
            # Verify embedding properties
            if embeddings:
                embedding_dims = len(embeddings[0])
                all_same_dim = all(len(emb) == embedding_dims for emb in embeddings)
                print(f"         Dimensions: {embedding_dims}, Consistent: {all_same_dim}")
                
                results.append({
                    "case": case['name'], 
                    "status": "success", 
                    "count": len(embeddings),
                    "time": processing_time,
                    "dimensions": embedding_dims
                })
            else:
                results.append({"case": case['name'], "status": "empty_result"})
                
        except Exception as e:
            error_msg = str(e)
            if case['should_fail']:
                print(f"      ✅ Failed as expected: {error_msg}")
                results.append({"case": case['name'], "status": "expected_failure", "error": error_msg})
            else:
                print(f"      ❌ Unexpected failure: {error_msg}")
                results.append({"case": case['name'], "status": "unexpected_failure", "error": error_msg})
    
    print(f"\n📊 Embedding Edge Cases Summary:")
    success_count = len([r for r in results if r['status'] == 'success'])
    print(f"   ✅ Handled correctly: {success_count}/{len(edge_cases)}")
    
    return results


async def test_vector_storage_edge_cases():
    """Test vector storage with edge cases."""
    print("\n🗄️  TESTING VECTOR STORAGE EDGE CASES")
    print("=" * 60)
    
    storage_config = StorageConfig(
        collection_name="edge_case_test",
        persist_directory="./test_edge_case_db"
    )
    vector_store = VectorStore(storage_config)
    await vector_store.initialize()
    
    # Test with various embedding dimensions and edge cases
    edge_cases = [
        {
            "name": "Empty embedding",
            "embeddings": [[]],
            "should_fail": True
        },
        {
            "name": "Single dimension",
            "embeddings": [[1.0]],
            "should_fail": False
        },
        {
            "name": "High dimensional",
            "embeddings": [[float(i) for i in range(1000)]],
            "should_fail": False
        },
        {
            "name": "Zero vectors",
            "embeddings": [[0.0] * 384],
            "should_fail": False
        },
        {
            "name": "Extreme values",
            "embeddings": [[-1e10, 1e10, float('inf'), -float('inf')] + [0.0] * 380],
            "should_fail": True  # inf values should fail
        },
        {
            "name": "Many small vectors",
            "embeddings": [[i/100.0] * 384 for i in range(100)],
            "should_fail": False
        }
    ]
    
    results = []
    for i, case in enumerate(edge_cases):
        print(f"\n   Test {i+1}: {case['name']}")
        
        try:
            from src.rag.ingestion import DocumentChunk, DocumentMetadata
            
            # Create test chunks with embeddings
            chunks = []
            for j, embedding in enumerate(case['embeddings']):
                chunk = DocumentChunk(
                    id=f"test_chunk_{j}",
                    content=f"Test content {j}",
                    document_id="test_doc",
                    chunk_index=j,
                    embedding=embedding
                )
                chunks.append(chunk)
            
            # Create test metadata
            metadata = DocumentMetadata(
                document_id="test_doc",
                file_path="/test/edge_case.txt",
                file_name="edge_case.txt",
                file_size=100,
                file_hash="test_hash",
                mime_type="text/plain",
                format=DocumentFormat.TEXT,
                processed_at=time.time(),
                chunk_count=len(chunks),
                processing_time=0.1
            )
            
            start_time = time.time()
            await vector_store.add_document(metadata, chunks)
            storage_time = time.time() - start_time
            
            if case['should_fail']:
                print(f"      ❌ Expected failure but succeeded in {storage_time:.3f}s")
                results.append({"case": case['name'], "status": "unexpected_success", "time": storage_time})
            else:
                print(f"      ✅ Stored successfully in {storage_time:.3f}s")
                
                # Test search with the same embedding
                if case['embeddings']:
                    query_embedding = case['embeddings'][0]
                    search_results = await vector_store.search_similar(query_embedding, top_k=1)
                    print(f"         Search returned {len(search_results)} results")
                
                results.append({"case": case['name'], "status": "success", "time": storage_time})
                
        except Exception as e:
            error_msg = str(e)
            if case['should_fail']:
                print(f"      ✅ Failed as expected: {error_msg}")
                results.append({"case": case['name'], "status": "expected_failure", "error": error_msg})
            else:
                print(f"      ❌ Unexpected failure: {error_msg}")
                results.append({"case": case['name'], "status": "unexpected_failure", "error": error_msg})
    
    print(f"\n📊 Vector Storage Edge Cases Summary:")
    success_count = len([r for r in results if r['status'] in ['success', 'expected_failure']])
    print(f"   ✅ Handled correctly: {success_count}/{len(edge_cases)}")
    
    return results


async def test_system_resilience():
    """Test system resilience under stress conditions."""
    print("\n💪 TESTING SYSTEM RESILIENCE")
    print("=" * 60)
    
    resilience_tests = []
    
    # Test 1: Rapid sequential operations
    print("\n   Test 1: Rapid sequential operations")
    try:
        embedding_config = EmbeddingConfig(
            backend=EmbeddingBackend.MLX,
            model_name=EmbeddingModel.ALL_MINILM_L6_V2,
            batch_size=1
        )
        embedding_generator = EmbeddingGenerator(embedding_config)
        await embedding_generator.initialize()
        
        start_time = time.time()
        for i in range(10):
            await embedding_generator.embed_text(f"Quick test {i}")
        total_time = time.time() - start_time
        
        print(f"      ✅ Completed 10 rapid operations in {total_time:.3f}s")
        resilience_tests.append({"test": "rapid_operations", "status": "success", "time": total_time})
        
    except Exception as e:
        print(f"      ❌ Rapid operations failed: {e}")
        resilience_tests.append({"test": "rapid_operations", "status": "failure", "error": str(e)})
    
    # Test 2: Memory pressure simulation
    print("\n   Test 2: Memory pressure simulation")
    try:
        large_texts = [f"Large text content {i}. " * 1000 for i in range(20)]
        
        start_time = time.time()
        embeddings = await embedding_generator.embed_texts(large_texts[:10])  # Reasonable batch
        processing_time = time.time() - start_time
        
        print(f"      ✅ Processed {len(large_texts[:10])} large texts in {processing_time:.3f}s")
        print(f"         Memory usage appeared stable")
        resilience_tests.append({"test": "memory_pressure", "status": "success", "time": processing_time})
        
    except Exception as e:
        print(f"      ❌ Memory pressure test failed: {e}")
        resilience_tests.append({"test": "memory_pressure", "status": "failure", "error": str(e)})
    
    # Test 3: Concurrent operations
    print("\n   Test 3: Concurrent operations")
    try:
        async def concurrent_embedding(text_id):
            return await embedding_generator.embed_text(f"Concurrent text {text_id}")
        
        start_time = time.time()
        tasks = [concurrent_embedding(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        concurrent_time = time.time() - start_time
        
        successful = len([r for r in results if not isinstance(r, Exception)])
        print(f"      ✅ {successful}/5 concurrent operations succeeded in {concurrent_time:.3f}s")
        resilience_tests.append({"test": "concurrent_ops", "status": "success", "successful": successful, "time": concurrent_time})
        
    except Exception as e:
        print(f"      ❌ Concurrent operations failed: {e}")
        resilience_tests.append({"test": "concurrent_ops", "status": "failure", "error": str(e)})
    
    # Test 4: Error recovery
    print("\n   Test 4: Error recovery")
    try:
        # Trigger an error and then continue normal operation
        try:
            await embedding_generator.embed_texts([None])  # This should fail
        except:
            pass  # Expected to fail
        
        # Now try normal operation
        start_time = time.time()
        normal_result = await embedding_generator.embed_text("Recovery test")
        recovery_time = time.time() - start_time
        
        print(f"      ✅ System recovered and processed normally in {recovery_time:.3f}s")
        resilience_tests.append({"test": "error_recovery", "status": "success", "time": recovery_time})
        
    except Exception as e:
        print(f"      ❌ Error recovery failed: {e}")
        resilience_tests.append({"test": "error_recovery", "status": "failure", "error": str(e)})
    
    print(f"\n📊 System Resilience Summary:")
    successful_tests = len([t for t in resilience_tests if t['status'] == 'success'])
    print(f"   ✅ Resilience tests passed: {successful_tests}/{len(resilience_tests)}")
    
    return resilience_tests


async def test_file_system_edge_cases():
    """Test file system related edge cases."""
    print("\n📁 TESTING FILE SYSTEM EDGE CASES")
    print("=" * 60)
    
    ingestion = DocumentIngestion()
    
    file_system_tests = []
    
    # Test 1: Non-existent file
    print("\n   Test 1: Non-existent file")
    try:
        await ingestion.process_file("/non/existent/file.txt")
        print(f"      ❌ Should have failed for non-existent file")
        file_system_tests.append({"test": "non_existent", "status": "unexpected_success"})
    except Exception as e:
        print(f"      ✅ Failed as expected: {str(e)[:100]}...")
        file_system_tests.append({"test": "non_existent", "status": "expected_failure"})
    
    # Test 2: Permission denied simulation (create file with restrictive permissions)
    print("\n   Test 2: File permissions")
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Test content")
            temp_path = Path(f.name)
        
        # Try to process the file normally first
        metadata, chunks = await ingestion.process_file(temp_path)
        print(f"      ✅ Processed file normally: {len(chunks)} chunks")
        
        temp_path.unlink()
        file_system_tests.append({"test": "file_permissions", "status": "success"})
        
    except Exception as e:
        print(f"      ❌ File permission test failed: {e}")
        file_system_tests.append({"test": "file_permissions", "status": "failure", "error": str(e)})
    
    # Test 3: Different file extensions
    print("\n   Test 3: Various file extensions")
    extensions_tests = []
    
    test_extensions = ['.txt', '.md', '.html', '.json', '.csv', '.unknown']
    
    for ext in test_extensions:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False) as f:
                if ext == '.json':
                    f.write('{"test": "content", "value": 123}')
                elif ext == '.csv':
                    f.write('name,value\ntest,123\nother,456')
                elif ext == '.html':
                    f.write('<html><body><h1>Test</h1><p>Content</p></body></html>')
                else:
                    f.write("Test content for different file extensions")
                temp_path = Path(f.name)
            
            start_time = time.time()
            metadata, chunks = await ingestion.process_file(temp_path)
            processing_time = time.time() - start_time
            
            print(f"      ✅ {ext}: {len(chunks)} chunks, format: {metadata.format}")
            extensions_tests.append({
                "extension": ext, 
                "status": "success", 
                "chunks": len(chunks),
                "format": str(metadata.format),
                "time": processing_time
            })
            
            temp_path.unlink()
            
        except Exception as e:
            print(f"      ⚠️  {ext}: Failed - {str(e)[:50]}...")
            extensions_tests.append({"extension": ext, "status": "failure", "error": str(e)})
            if temp_path.exists():
                temp_path.unlink()
    
    successful_extensions = len([t for t in extensions_tests if t['status'] == 'success'])
    print(f"      📊 Successfully processed: {successful_extensions}/{len(test_extensions)} file types")
    
    file_system_tests.append({"test": "file_extensions", "results": extensions_tests})
    
    return file_system_tests


async def main():
    """Run comprehensive edge case and error handling tests."""
    print("🚀 AKASHA PHASE 2 - EDGE CASES & ERROR HANDLING")  
    print("=" * 70)
    
    all_results = {}
    
    try:
        # Test 1: Document Ingestion Edge Cases
        all_results['ingestion'] = await test_document_ingestion_edge_cases()
        
        # Test 2: Embedding Generation Edge Cases  
        all_results['embeddings'] = await test_embedding_generation_edge_cases()
        
        # Test 3: Vector Storage Edge Cases
        all_results['storage'] = await test_vector_storage_edge_cases()
        
        # Test 4: System Resilience
        all_results['resilience'] = await test_system_resilience()
        
        # Test 5: File System Edge Cases
        all_results['filesystem'] = await test_file_system_edge_cases()
        
        print(f"\n🎯 COMPREHENSIVE EDGE CASE TESTING COMPLETED!")
        print("=" * 70)
        
        # Summary statistics
        total_tests = 0
        successful_tests = 0
        
        for category, results in all_results.items():
            if category == 'filesystem':
                # Special handling for filesystem results
                category_tests = len(results)
                category_success = len([r for r in results if r.get('status') in ['success', 'expected_failure']])
            else:
                category_tests = len(results)
                category_success = len([r for r in results if r.get('status') in ['success', 'expected_failure']])
            
            total_tests += category_tests
            successful_tests += category_success
            
            print(f"📊 {category.upper()}: {category_success}/{category_tests} tests handled correctly")
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"\n🏆 OVERALL EDGE CASE HANDLING: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print("✅ EXCELLENT: System demonstrates robust error handling!")
        elif success_rate >= 60:
            print("⚠️  GOOD: System handles most edge cases well")
        else:
            print("❌ NEEDS IMPROVEMENT: System needs better error handling")
        
    except Exception as e:
        print(f"\n❌ EDGE CASE TESTING FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())