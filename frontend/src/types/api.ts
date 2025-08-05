// API Response Types
export interface ApiResponse<T = any> {
  data: T;
  message?: string;
  status: 'success' | 'error';
  timestamp: string;
}

// Document Types
export interface Document {
  id: string;
  title: string;
  filename: string;
  file_path: string;
  content_type: string;
  size: number;
  page_count?: number;
  upload_date: string;
  last_modified: string;
  processing_status: 'pending' | 'processing' | 'completed' | 'failed';
  processing_progress?: number;
  metadata: {
    author?: string;
    subject?: string;
    keywords?: string[];
    language?: string;
    creation_date?: string;
  };
  tags: string[];
  embeddings_generated: boolean;
  indexed: boolean;
}

// Document Processing Types
export interface ProcessingJob {
  id: string;
  document_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  current_step: string;
  total_steps: number;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
}

// Search and Retrieval Types
export interface SearchQuery {
  query: string;
  filters?: {
    document_ids?: string[];
    tags?: string[];
    date_range?: {
      start: string;
      end: string;
    };
    content_type?: string[];
  };
  limit?: number;
  offset?: number;
  search_type?: 'semantic' | 'keyword' | 'hybrid';
}

export interface SearchResult {
  id: string;
  document_id: string;
  document_title: string;
  content: string;
  score: number;
  page_number?: number;
  chunk_id: string;
  highlights: string[];
  metadata: Record<string, any>;
}

export interface SearchResponse {
  results: SearchResult[];
  total_count: number;
  query_time: number;
  search_method: string;
}

// Chat Types
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources?: {
    document_id: string;
    document_title: string;
    page_number?: number;
    chunk_id: string;
    relevance_score: number;
  }[];
  processing_time?: number;
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  last_message_at: string;
  message_count: number;
  messages: ChatMessage[];
}

export interface StreamingResponse {
  type: 'chunk' | 'sources' | 'complete' | 'error';
  content?: string;
  sources?: ChatMessage['sources'];
  error?: string;
  message_id?: string;
}

// User and Authentication Types
export interface User {
  id: string;
  username: string;
  email: string;
  full_name: string;
  created_at: string;
  last_login?: string;
  preferences: {
    theme: 'light' | 'dark' | 'auto';
    language: string;
    default_search_type: 'semantic' | 'keyword' | 'hybrid';
    results_per_page: number;
  };
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse extends ApiResponse<{
  user: User;
  tokens: AuthTokens;
}> {}

// System Status Types
export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  services: {
    api: 'up' | 'down';
    database: 'up' | 'down';
    vector_store: 'up' | 'down';
    llm_service: 'up' | 'down';
    embedding_service: 'up' | 'down';
  };
  performance: {
    average_query_time: number;
    documents_processed_today: number;
    active_processing_jobs: number;
  };
  system_info: {
    version: string;
    uptime: number;
    memory_usage: number;
    disk_usage: number;
  };
}

// Analytics Types
export interface UsageAnalytics {
  period: {
    start: string;
    end: string;
  };
  metrics: {
    total_queries: number;
    unique_users: number;
    documents_uploaded: number;
    average_response_time: number;
    most_queried_documents: Array<{
      document_id: string;
      title: string;
      query_count: number;
    }>;
    popular_search_terms: Array<{
      term: string;
      count: number;
    }>;
  };
}

// Error Types
export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, any>;
  timestamp: string;
}