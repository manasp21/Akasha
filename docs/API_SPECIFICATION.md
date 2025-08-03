# Akasha API Specification

## Table of Contents
1. [Overview](#1-overview)
2. [Authentication](#2-authentication)
3. [Request/Response Format](#3-requestresponse-format)
4. [Error Handling](#4-error-handling)
5. [Rate Limiting](#5-rate-limiting)
6. [Document Management API](#6-document-management-api)
7. [Search API](#7-search-api)
8. [Chat API](#8-chat-api)
9. [Model Management API](#9-model-management-api)
10. [Plugin Management API](#10-plugin-management-api)
11. [System API](#11-system-api)
12. [WebSocket API](#12-websocket-api)
13. [Webhooks](#13-webhooks)
14. [SDK and Client Libraries](#14-sdk-and-client-libraries)

---

## 1. Overview

The Akasha API provides a comprehensive REST and WebSocket interface for accessing all system functionality. The API follows RESTful conventions with JSON payloads and supports both synchronous and real-time interactions.

### 1.1 Base URL
```
Production: https://api.akasha.ai/v1
Development: http://localhost:8000/api/v1
```

### 1.2 API Versioning
- Current version: `v1`
- Version specified in URL path: `/api/v1/`
- Backwards compatibility maintained within major versions

### 1.3 Content Types
- Request Content-Type: `application/json`
- File uploads: `multipart/form-data`
- Response Content-Type: `application/json`
- Streaming responses: `text/plain` or `application/x-ndjson`

---

## 2. Authentication

### 2.1 API Key Authentication
```http
Authorization: Bearer your-api-key-here
```

### 2.2 Authentication Flow
```bash
# Example request with authentication
curl -X GET \
  "https://api.akasha.ai/v1/status" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json"
```

### 2.3 API Key Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/keys` | GET | List API keys |
| `/auth/keys` | POST | Create new API key |
| `/auth/keys/{key_id}` | DELETE | Revoke API key |

### 2.4 Error Responses
```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid API key",
    "details": "The provided API key is invalid or expired"
  }
}
```

---

## 3. Request/Response Format

### 3.1 Standard Response Structure
```json
{
  "success": true,
  "data": {
    // Response payload
  },
  "metadata": {
    "timestamp": "2025-01-15T10:30:00Z",
    "request_id": "req_123456789",
    "processing_time": 150
  }
}
```

### 3.2 Error Response Structure
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": "Additional error details",
    "field_errors": {
      "field_name": ["Field-specific error messages"]
    }
  },
  "metadata": {
    "timestamp": "2025-01-15T10:30:00Z",
    "request_id": "req_123456789"
  }
}
```

### 3.3 Pagination
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 150,
    "total_pages": 8,
    "has_next": true,
    "has_prev": false
  }
}
```

---

## 4. Error Handling

### 4.1 HTTP Status Codes
| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Access denied |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource conflict |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

### 4.2 Error Codes
| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Request validation failed |
| `DOCUMENT_NOT_FOUND` | Document not found |
| `PROCESSING_FAILED` | Document processing failed |
| `MODEL_NOT_AVAILABLE` | Requested model not available |
| `PLUGIN_ERROR` | Plugin operation failed |
| `RATE_LIMIT_EXCEEDED` | Rate limit exceeded |
| `INSUFFICIENT_STORAGE` | Storage quota exceeded |

---

## 5. Rate Limiting

### 5.1 Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642694400
X-RateLimit-Window: 3600
```

### 5.2 Rate Limits by Endpoint
| Endpoint Category | Limit | Window |
|-------------------|-------|--------|
| Document Upload | 100 requests | 1 hour |
| Search | 1000 requests | 1 hour |
| Chat | 500 requests | 1 hour |
| System | 200 requests | 1 hour |

---

## 6. Document Management API

### 6.1 Upload Document

**Endpoint:** `POST /documents`

**Description:** Upload and process a document

**Request:**
```http
POST /api/v1/documents
Content-Type: multipart/form-data
Authorization: Bearer your-api-key

--boundary123
Content-Disposition: form-data; name="file"; filename="document.pdf"
Content-Type: application/pdf

[PDF file contents]
--boundary123
Content-Disposition: form-data; name="metadata"

{
  "title": "Research Paper",
  "tags": ["machine-learning", "nlp"],
  "category": "research"
}
--boundary123--
```

**Response:**
```json
{
  "success": true,
  "data": {
    "document_id": "doc_123456789",
    "filename": "document.pdf",
    "status": "processing",
    "processing_time": 2.5,
    "metadata": {
      "title": "Research Paper",
      "tags": ["machine-learning", "nlp"],
      "category": "research",
      "file_size": 1024000,
      "page_count": 15,
      "detected_language": "en"
    },
    "processing_stats": {
      "text_segments": 45,
      "images_extracted": 8,
      "tables_extracted": 3,
      "formulas_extracted": 12
    }
  }
}
```

### 6.2 Get Document

**Endpoint:** `GET /documents/{document_id}`

**Request:**
```http
GET /api/v1/documents/doc_123456789
Authorization: Bearer your-api-key
```

**Response:**
```json
{
  "success": true,
  "data": {
    "document_id": "doc_123456789",
    "filename": "document.pdf",
    "status": "processed",
    "created_at": "2025-01-15T10:30:00Z",
    "updated_at": "2025-01-15T10:32:30Z",
    "metadata": {
      "title": "Research Paper",
      "authors": ["John Doe", "Jane Smith"],
      "abstract": "This paper presents...",
      "publication_date": "2024-12-15",
      "doi": "10.1000/xyz123",
      "tags": ["machine-learning", "nlp"],
      "file_size": 1024000,
      "page_count": 15
    },
    "processing_stats": {
      "text_segments": 45,
      "images_extracted": 8,
      "tables_extracted": 3,
      "formulas_extracted": 12,
      "entities_extracted": 127,
      "relations_extracted": 89
    },
    "content_preview": {
      "abstract": "This paper presents a novel approach...",
      "first_page_text": "Introduction\n\nThe field of machine learning..."
    }
  }
}
```

### 6.3 List Documents

**Endpoint:** `GET /documents`

**Query Parameters:**
- `page` (integer): Page number (default: 1)
- `per_page` (integer): Items per page (default: 20, max: 100)
- `status` (string): Filter by status (`processing`, `processed`, `failed`)
- `category` (string): Filter by category
- `tags` (array): Filter by tags
- `search` (string): Search in title and content
- `sort` (string): Sort by (`created_at`, `updated_at`, `title`, `size`)
- `order` (string): Sort order (`asc`, `desc`)

**Request:**
```http
GET /api/v1/documents?page=1&per_page=20&status=processed&sort=created_at&order=desc
Authorization: Bearer your-api-key
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "document_id": "doc_123456789",
      "filename": "document.pdf",
      "title": "Research Paper",
      "status": "processed",
      "created_at": "2025-01-15T10:30:00Z",
      "metadata": {
        "file_size": 1024000,
        "page_count": 15,
        "tags": ["machine-learning", "nlp"]
      }
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 150,
    "total_pages": 8,
    "has_next": true,
    "has_prev": false
  }
}
```

### 6.4 Delete Document

**Endpoint:** `DELETE /documents/{document_id}`

**Request:**
```http
DELETE /api/v1/documents/doc_123456789
Authorization: Bearer your-api-key
```

**Response:**
```json
{
  "success": true,
  "data": {
    "message": "Document deleted successfully",
    "document_id": "doc_123456789"
  }
}
```

### 6.5 Get Document Content

**Endpoint:** `GET /documents/{document_id}/content`

**Query Parameters:**
- `format` (string): Response format (`segments`, `markdown`, `text`)
- `include_images` (boolean): Include image data
- `include_metadata` (boolean): Include segment metadata

**Request:**
```http
GET /api/v1/documents/doc_123456789/content?format=segments&include_images=true
Authorization: Bearer your-api-key
```

**Response:**
```json
{
  "success": true,
  "data": {
    "document_id": "doc_123456789",
    "format": "segments",
    "segments": [
      {
        "id": "seg_001",
        "type": "text",
        "content": "Introduction\n\nThis paper presents...",
        "page_number": 1,
        "order_index": 0,
        "metadata": {
          "section": "introduction",
          "font_size": 12,
          "bounding_box": {"x": 72, "y": 200, "width": 450, "height": 100}
        }
      },
      {
        "id": "seg_002",
        "type": "image",
        "content": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
        "page_number": 2,
        "order_index": 1,
        "metadata": {
          "caption": "Figure 1: System Architecture",
          "format": "png",
          "width": 400,
          "height": 300
        }
      }
    ],
    "total_segments": 45
  }
}
```

---

## 7. Search API

### 7.1 Document Search

**Endpoint:** `POST /search`

**Request:**
```json
{
  "query": "machine learning algorithms for natural language processing",
  "filters": {
    "document_type": ["research_paper"],
    "tags": ["machine-learning", "nlp"],
    "date_range": {
      "start": "2023-01-01",
      "end": "2025-01-01"
    },
    "authors": ["John Doe"]
  },
  "options": {
    "search_type": "hybrid",
    "limit": 10,
    "include_metadata": true,
    "include_content": true,
    "include_highlights": true,
    "similarity_threshold": 0.7
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "query": "machine learning algorithms for natural language processing",
    "results": [
      {
        "document_id": "doc_123456789",
        "title": "Advanced ML Techniques for NLP",
        "score": 0.95,
        "rank": 1,
        "metadata": {
          "authors": ["John Doe", "Jane Smith"],
          "publication_date": "2024-06-15",
          "tags": ["machine-learning", "nlp", "deep-learning"]
        },
        "content_matches": [
          {
            "segment_id": "seg_015",
            "content": "...machine learning algorithms have shown remarkable success in natural language processing tasks...",
            "highlights": [
              {"text": "machine learning algorithms", "start": 3, "end": 30},
              {"text": "natural language processing", "start": 65, "end": 92}
            ],
            "page_number": 5,
            "score": 0.92
          }
        ],
        "image_matches": [
          {
            "segment_id": "seg_022",
            "caption": "Figure 3: ML Pipeline for NLP",
            "description": "Diagram showing the machine learning pipeline for natural language processing",
            "page_number": 8,
            "score": 0.85
          }
        ]
      }
    ],
    "total_results": 42,
    "search_time": 0.15,
    "query_id": "query_123456789"
  }
}
```

### 7.2 Semantic Search

**Endpoint:** `POST /search/semantic`

**Request:**
```json
{
  "query": "neural networks for text classification",
  "options": {
    "limit": 5,
    "similarity_threshold": 0.8,
    "include_cross_modal": true
  }
}
```

### 7.3 Image Similarity Search

**Endpoint:** `POST /search/image`

**Request:**
```http
POST /api/v1/search/image
Content-Type: multipart/form-data

--boundary123
Content-Disposition: form-data; name="image"; filename="query_image.png"
Content-Type: image/png

[Image file contents]
--boundary123
Content-Disposition: form-data; name="options"

{
  "limit": 10,
  "similarity_threshold": 0.7,
  "include_metadata": true
}
--boundary123--
```

### 7.4 Find Similar Documents

**Endpoint:** `GET /search/similar/{document_id}`

**Query Parameters:**
- `limit` (integer): Maximum results (default: 10)
- `threshold` (float): Similarity threshold (default: 0.7)
- `exclude_self` (boolean): Exclude the source document (default: true)

**Request:**
```http
GET /api/v1/search/similar/doc_123456789?limit=5&threshold=0.8
Authorization: Bearer your-api-key
```

---

## 8. Chat API

### 8.1 Chat with Documents

**Endpoint:** `POST /chat`

**Request:**
```json
{
  "message": "What are the main contributions of this paper?",
  "conversation_id": "conv_123456789",
  "context": {
    "document_ids": ["doc_123456789", "doc_987654321"],
    "max_context_length": 4000,
    "include_images": true
  },
  "model_config": {
    "model": "gemma-3-27b",
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 0.9
  },
  "response_format": {
    "stream": false,
    "include_sources": true,
    "include_citations": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "response": "Based on the documents provided, the main contributions of this paper are:\n\n1. **Novel Architecture**: The paper introduces a new transformer-based architecture that significantly improves performance on NLP tasks [Source 1, p. 3].\n\n2. **Efficiency Improvements**: The proposed method achieves 40% faster training times compared to existing approaches [Source 1, p. 8].\n\n3. **Benchmark Results**: The model achieves state-of-the-art results on multiple benchmark datasets [Source 1, Table 2].\n\nThese contributions represent significant advances in the field of natural language processing.",
    "conversation_id": "conv_123456789",
    "message_id": "msg_123456789",
    "sources": [
      {
        "document_id": "doc_123456789",
        "title": "Advanced ML Techniques for NLP",
        "page_number": 3,
        "segment_id": "seg_015",
        "relevance_score": 0.92,
        "citation_id": "Source 1"
      }
    ],
    "generation_stats": {
      "tokens_generated": 156,
      "generation_time": 2.3,
      "model_used": "gemma-3-27b"
    },
    "retrieval_stats": {
      "contexts_retrieved": 5,
      "retrieval_time": 0.4,
      "search_queries": ["main contributions", "novel architecture", "efficiency improvements"]
    }
  }
}
```

### 8.2 Streaming Chat

**Endpoint:** `POST /chat/stream`

**Request:**
```json
{
  "message": "Explain the methodology used in this research",
  "conversation_id": "conv_123456789",
  "stream": true
}
```

**Response (Server-Sent Events):**
```
data: {"type": "start", "conversation_id": "conv_123456789", "message_id": "msg_123456789"}

data: {"type": "chunk", "content": "The methodology"}

data: {"type": "chunk", "content": " described in this"}

data: {"type": "chunk", "content": " research paper"}

data: {"type": "sources", "sources": [{"document_id": "doc_123456789", "title": "Research Paper", "page": 5}]}

data: {"type": "end", "generation_stats": {"tokens_generated": 243, "generation_time": 3.1}}
```

### 8.3 Get Conversation History

**Endpoint:** `GET /conversations/{conversation_id}`

**Request:**
```http
GET /api/v1/conversations/conv_123456789
Authorization: Bearer your-api-key
```

**Response:**
```json
{
  "success": true,
  "data": {
    "conversation_id": "conv_123456789",
    "created_at": "2025-01-15T10:30:00Z",
    "updated_at": "2025-01-15T10:45:00Z",
    "message_count": 6,
    "messages": [
      {
        "message_id": "msg_001",
        "role": "user",
        "content": "What are the main contributions of this paper?",
        "timestamp": "2025-01-15T10:30:00Z"
      },
      {
        "message_id": "msg_002",
        "role": "assistant",
        "content": "Based on the documents provided, the main contributions...",
        "timestamp": "2025-01-15T10:30:15Z",
        "sources": [
          {
            "document_id": "doc_123456789",
            "title": "Advanced ML Techniques for NLP"
          }
        ]
      }
    ]
  }
}
```

### 8.4 Delete Conversation

**Endpoint:** `DELETE /conversations/{conversation_id}`

---

## 9. Model Management API

### 9.1 List Available Models

**Endpoint:** `GET /models`

**Request:**
```http
GET /api/v1/models
Authorization: Bearer your-api-key
```

**Response:**
```json
{
  "success": true,
  "data": {
    "models": [
      {
        "model_id": "gemma-3-27b",
        "name": "Gemma 3 27B",
        "type": "llm",
        "status": "loaded",
        "description": "Large language model optimized for instruction following",
        "capabilities": ["text-generation", "chat", "reasoning"],
        "parameters": "27B",
        "context_window": 8192,
        "supported_backends": ["mlx", "llama_cpp"],
        "memory_usage": "16GB",
        "performance": {
          "tokens_per_second": 25,
          "latency_ms": 150
        }
      },
      {
        "model_id": "jina-embeddings-v4",
        "name": "JINA Embeddings v4",
        "type": "embedding",
        "status": "loaded",
        "description": "Multimodal embedding model for text and images",
        "capabilities": ["text-embedding", "image-embedding", "multimodal"],
        "dimensions": 512,
        "supported_modalities": ["text", "image"],
        "max_input_length": 8192
      }
    ],
    "current_llm": "gemma-3-27b",
    "current_embedding": "jina-embeddings-v4"
  }
}
```

### 9.2 Load Model

**Endpoint:** `POST /models/{model_id}/load`

**Request:**
```json
{
  "backend": "mlx",
  "config": {
    "quantization": "4bit",
    "max_memory": "16GB"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "model_id": "gemma-3-27b",
    "status": "loading",
    "estimated_time": 30,
    "progress": 0
  }
}
```

### 9.3 Get Model Status

**Endpoint:** `GET /models/{model_id}/status`

### 9.4 Unload Model

**Endpoint:** `POST /models/{model_id}/unload`

---

## 10. Plugin Management API

### 10.1 List Plugins

**Endpoint:** `GET /plugins`

**Response:**
```json
{
  "success": true,
  "data": {
    "plugins": [
      {
        "plugin_id": "mineru-enhanced",
        "name": "Enhanced MinerU Processor",
        "version": "1.2.0",
        "status": "active",
        "category": "document-processor",
        "description": "Enhanced PDF processing with advanced OCR",
        "author": "Akasha Team",
        "capabilities": ["pdf-processing", "ocr", "table-extraction"],
        "config": {
          "ocr_engine": "paddleocr",
          "enable_tables": true,
          "languages": ["en", "fr", "de"]
        }
      }
    ],
    "active_plugins": 3,
    "available_plugins": 12
  }
}
```

### 10.2 Activate Plugin

**Endpoint:** `POST /plugins/{plugin_id}/activate`

**Request:**
```json
{
  "config": {
    "ocr_engine": "paddleocr",
    "enable_tables": true,
    "batch_size": 4
  }
}
```

### 10.3 Deactivate Plugin

**Endpoint:** `POST /plugins/{plugin_id}/deactivate`

### 10.4 Get Plugin Info

**Endpoint:** `GET /plugins/{plugin_id}`

---

## 11. System API

### 11.1 System Status

**Endpoint:** `GET /status`

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "uptime": 86400,
    "timestamp": "2025-01-15T10:30:00Z",
    "components": {
      "ingestion_engine": "healthy",
      "embedding_service": "healthy",
      "vector_store": "healthy",
      "llm_service": "healthy",
      "cache_manager": "healthy",
      "plugin_manager": "healthy"
    },
    "performance": {
      "total_documents": 1247,
      "total_vectors": 48329,
      "cache_hit_rate": 0.85,
      "average_response_time": 250,
      "active_connections": 12
    },
    "resources": {
      "cpu_usage": 45.2,
      "memory_usage": 62.8,
      "disk_usage": 34.1,
      "gpu_usage": 78.5
    }
  }
}
```

### 11.2 Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00Z",
  "checks": {
    "database": "pass",
    "vector_store": "pass",
    "cache": "pass",
    "models": "pass"
  }
}
```

### 11.3 System Metrics

**Endpoint:** `GET /metrics`

**Response:**
```json
{
  "success": true,
  "data": {
    "request_metrics": {
      "total_requests": 150420,
      "requests_per_minute": 45,
      "average_response_time": 250,
      "error_rate": 0.02
    },
    "processing_metrics": {
      "documents_processed": 1247,
      "vectors_generated": 48329,
      "searches_performed": 8432,
      "chat_messages": 3421
    },
    "resource_metrics": {
      "cpu_usage": 45.2,
      "memory_usage": 62.8,
      "disk_usage": 34.1,
      "gpu_usage": 78.5,
      "network_io": {
        "bytes_sent": 1024000000,
        "bytes_received": 512000000
      }
    },
    "cache_metrics": {
      "hit_rate": 0.85,
      "miss_rate": 0.15,
      "evictions": 123,
      "size_mb": 512
    }
  }
}
```

### 11.4 Configuration

**Endpoint:** `GET /config`

**Response:**
```json
{
  "success": true,
  "data": {
    "system": {
      "version": "1.0.0",
      "environment": "production"
    },
    "features": {
      "multimodal_search": true,
      "streaming_chat": true,
      "plugin_support": true,
      "graph_rag": true
    },
    "limits": {
      "max_file_size_mb": 100,
      "max_documents": 10000,
      "max_concurrent_requests": 100
    }
  }
}
```

---

## 12. WebSocket API

### 12.1 Connection

**Endpoint:** `wss://api.akasha.ai/ws`

**Connection:**
```javascript
const ws = new WebSocket('wss://api.akasha.ai/ws');

// Send authentication after connection
ws.onopen = function() {
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'your-api-key'
  }));
};
```

### 12.2 Message Types

#### 12.2.1 Authentication
```json
{
  "type": "auth",
  "token": "your-api-key"
}
```

#### 12.2.2 Chat Message
```json
{
  "type": "chat",
  "data": {
    "message": "What are the key findings?",
    "conversation_id": "conv_123456789",
    "stream": true
  }
}
```

#### 12.2.3 Search Request
```json
{
  "type": "search",
  "data": {
    "query": "machine learning",
    "filters": {},
    "limit": 10
  }
}
```

#### 12.2.4 Status Request
```json
{
  "type": "status"
}
```

### 12.3 Server Responses

#### 12.3.1 Authentication Response
```json
{
  "type": "auth_response",
  "success": true,
  "user_id": "user_123456789"
}
```

#### 12.3.2 Chat Response
```json
{
  "type": "chat_response",
  "data": {
    "message_id": "msg_123456789",
    "content": "The key findings include...",
    "sources": [...],
    "streaming": false
  }
}
```

#### 12.3.3 Chat Streaming
```json
{
  "type": "chat_chunk",
  "data": {
    "message_id": "msg_123456789",
    "chunk": "The key findings",
    "is_final": false
  }
}
```

#### 12.3.4 Error Response
```json
{
  "type": "error",
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Invalid message format"
  }
}
```

### 12.4 Connection Management

#### 12.4.1 Heartbeat
```json
{
  "type": "ping"
}
```

**Response:**
```json
{
  "type": "pong",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

#### 12.4.2 Connection Limits
- Maximum concurrent connections: 100
- Idle timeout: 5 minutes
- Message rate limit: 60 messages per minute

---

## 13. Webhooks

### 13.1 Webhook Configuration

**Endpoint:** `POST /webhooks`

**Request:**
```json
{
  "url": "https://your-app.com/webhooks/akasha",
  "events": ["document.processed", "search.completed", "model.loaded"],
  "secret": "your-webhook-secret",
  "active": true
}
```

### 13.2 Webhook Events

#### 13.2.1 Document Processed
```json
{
  "event": "document.processed",
  "timestamp": "2025-01-15T10:30:00Z",
  "data": {
    "document_id": "doc_123456789",
    "status": "processed",
    "processing_time": 45.2,
    "metadata": {
      "page_count": 15,
      "text_segments": 45,
      "images_extracted": 8
    }
  }
}
```

#### 13.2.2 Model Loaded
```json
{
  "event": "model.loaded",
  "timestamp": "2025-01-15T10:30:00Z",
  "data": {
    "model_id": "gemma-3-27b",
    "status": "ready",
    "load_time": 30.5
  }
}
```

### 13.3 Webhook Security

- HMAC-SHA256 signature in `X-Akasha-Signature` header
- Signature calculated using webhook secret
- Payload verification required

---

## 14. SDK and Client Libraries

### 14.1 Python SDK

**Installation:**
```bash
pip install akasha-sdk
```

**Usage:**
```python
from akasha import AkashaClient

client = AkashaClient(api_key='your-api-key')

# Upload document
with open('document.pdf', 'rb') as f:
    result = client.documents.upload(f, metadata={'title': 'Research Paper'})
    document_id = result['document_id']

# Search documents
results = client.search('machine learning algorithms')

# Chat with documents
response = client.chat('What are the main findings?', 
                      conversation_id='conv_123')
```

### 14.2 JavaScript SDK

**Installation:**
```bash
npm install @akasha/sdk
```

**Usage:**
```javascript
import { AkashaClient } from '@akasha/sdk';

const client = new AkashaClient({ apiKey: 'your-api-key' });

// Upload document
const result = await client.documents.upload(file, {
  title: 'Research Paper'
});

// Search documents
const results = await client.search('machine learning algorithms');

// Streaming chat
const stream = client.chat.stream('Explain the methodology', {
  conversationId: 'conv_123'
});

for await (const chunk of stream) {
  console.log(chunk.content);
}
```

### 14.3 cURL Examples

#### Upload Document
```bash
curl -X POST \
  "https://api.akasha.ai/v1/documents" \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@document.pdf" \
  -F 'metadata={"title":"Research Paper"}'
```

#### Search Documents
```bash
curl -X POST \
  "https://api.akasha.ai/v1/search" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "limit": 10
  }'
```

#### Chat with Documents
```bash
curl -X POST \
  "https://api.akasha.ai/v1/chat" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the main contributions?",
    "conversation_id": "conv_123456789"
  }'
```

---

## 15. Best Practices

### 15.1 Authentication
- Store API keys securely
- Use environment variables for keys
- Rotate keys regularly
- Monitor key usage

### 15.2 Error Handling
- Always check response status codes
- Implement retry logic with exponential backoff
- Handle rate limiting gracefully
- Log errors for debugging

### 15.3 Performance
- Use pagination for large result sets
- Cache frequently accessed data
- Use streaming for real-time responses
- Monitor API usage and performance

### 15.4 Security
- Use HTTPS for all requests
- Validate webhook signatures
- Sanitize user inputs
- Follow rate limiting guidelines

This comprehensive API specification provides complete documentation for integrating with the Akasha system, covering all endpoints, request/response formats, authentication, and best practices.