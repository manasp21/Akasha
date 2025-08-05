import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { 
  ApiResponse, 
  Document, 
  SearchQuery, 
  SearchResponse, 
  ChatMessage, 
  ChatSession,
  ProcessingJob,
  SystemHealth,
  UsageAnalytics,
  LoginRequest,
  LoginResponse,
  User
} from '../types/api';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API_TIMEOUT = 30000; // 30 seconds

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: API_TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor to add auth token
    this.client.interceptors.request.use(
      (config) => {
        const token = this.getAuthToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config;

        // Handle 401 errors (token expired)
        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;
          
          try {
            await this.refreshToken();
            const token = this.getAuthToken();
            if (token) {
              originalRequest.headers.Authorization = `Bearer ${token}`;
              return this.client(originalRequest);
            }
          } catch (refreshError) {
            // Refresh failed, redirect to login
            this.clearAuthToken();
            window.location.href = '/login';
            return Promise.reject(refreshError);
          }
        }

        return Promise.reject(error);
      }
    );
  }

  // Auth token management
  private getAuthToken(): string | null {
    const authData = localStorage.getItem('akasha-auth');
    if (authData) {
      const parsed = JSON.parse(authData);
      return parsed.state?.tokens?.access_token || null;
    }
    return null;
  }

  private getRefreshToken(): string | null {
    const authData = localStorage.getItem('akasha-auth');
    if (authData) {
      const parsed = JSON.parse(authData);
      return parsed.state?.tokens?.refresh_token || null;
    }
    return null;
  }

  private clearAuthToken(): void {
    localStorage.removeItem('akasha-auth');
  }

  private async refreshToken(): Promise<void> {
    const refreshToken = this.getRefreshToken();
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await this.client.post('/auth/refresh', {
      refresh_token: refreshToken,
    });

    // Update stored tokens
    const authData = localStorage.getItem('akasha-auth');
    if (authData) {
      const parsed = JSON.parse(authData);
      parsed.state.tokens = response.data.data.tokens;
      localStorage.setItem('akasha-auth', JSON.stringify(parsed));
    }
  }

  // Generic request method
  private async request<T>(config: AxiosRequestConfig): Promise<T> {
    try {
      const response: AxiosResponse<ApiResponse<T>> = await this.client(config);
      return response.data.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const message = error.response?.data?.message || error.message;
        throw new Error(message);
      }
      throw error;
    }
  }

  // Authentication endpoints
  async login(credentials: LoginRequest): Promise<LoginResponse['data']> {
    return this.request({
      method: 'POST',
      url: '/auth/login',
      data: credentials,
    });
  }

  async logout(): Promise<void> {
    await this.request({
      method: 'POST',
      url: '/auth/logout',
    });
    this.clearAuthToken();
  }

  async getCurrentUser(): Promise<User> {
    return this.request({
      method: 'GET',
      url: '/auth/me',
    });
  }

  // Document management endpoints
  async getDocuments(params?: {
    page?: number;
    per_page?: number;
    search?: string;
    tags?: string[];
    status?: string;
  }): Promise<{ documents: Document[]; total: number; page: number; per_page: number }> {
    return this.request({
      method: 'GET',
      url: '/documents',
      params,
    });
  }

  async getDocument(id: string): Promise<Document> {
    return this.request({
      method: 'GET',
      url: `/documents/${id}`,
    });
  }

  async uploadDocument(file: File, metadata?: Record<string, any>): Promise<Document> {
    const formData = new FormData();
    formData.append('file', file);
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata));
    }

    return this.request({
      method: 'POST',
      url: '/documents/upload',
      data: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  }

  async deleteDocument(id: string): Promise<void> {
    return this.request({
      method: 'DELETE',
      url: `/documents/${id}`,
    });
  }

  async updateDocument(id: string, updates: Partial<Document>): Promise<Document> {
    return this.request({
      method: 'PATCH',
      url: `/documents/${id}`,
      data: updates,
    });
  }

  // Processing endpoints
  async getProcessingJobs(documentId?: string): Promise<ProcessingJob[]> {
    return this.request({
      method: 'GET',
      url: '/processing/jobs',
      params: documentId ? { document_id: documentId } : undefined,
    });
  }

  async getProcessingJob(jobId: string): Promise<ProcessingJob> {
    return this.request({
      method: 'GET',
      url: `/processing/jobs/${jobId}`,
    });
  }

  // Search endpoints
  async search(query: SearchQuery): Promise<SearchResponse> {
    return this.request({
      method: 'POST',
      url: '/search',
      data: query,
    });
  }

  async getSimilarDocuments(documentId: string, limit = 10): Promise<SearchResponse> {
    return this.request({
      method: 'GET',
      url: `/search/similar/${documentId}`,
      params: { limit },
    });
  }

  // Chat endpoints
  async getChatSessions(): Promise<ChatSession[]> {
    return this.request({
      method: 'GET',
      url: '/chat/sessions',
    });
  }

  async getChatSession(sessionId: string): Promise<ChatSession> {
    return this.request({
      method: 'GET',
      url: `/chat/sessions/${sessionId}`,
    });
  }

  async createChatSession(title?: string): Promise<ChatSession> {
    return this.request({
      method: 'POST',
      url: '/chat/sessions',
      data: { title },
    });
  }

  async deleteChatSession(sessionId: string): Promise<void> {
    return this.request({
      method: 'DELETE',
      url: `/chat/sessions/${sessionId}`,
    });
  }

  async sendChatMessage(sessionId: string, message: string): Promise<ChatMessage> {
    return this.request({
      method: 'POST',
      url: `/chat/sessions/${sessionId}/messages`,
      data: { message },
    });
  }

  // Streaming chat
  async *streamChatMessage(sessionId: string, message: string): AsyncGenerator<any, void, unknown> {
    const response = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}/messages/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.getAuthToken()}`,
      },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No reader available');
    }

    const decoder = new TextDecoder();
    
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') return;
            
            try {
              const parsed = JSON.parse(data);
              yield parsed;
            } catch (e) {
              // Skip invalid JSON
              continue;
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  // System endpoints
  async getSystemHealth(): Promise<SystemHealth> {
    return this.request({
      method: 'GET',
      url: '/system/health',
    });
  }

  async getUsageAnalytics(period: 'day' | 'week' | 'month' = 'week'): Promise<UsageAnalytics> {
    return this.request({
      method: 'GET',
      url: '/system/analytics',
      params: { period },
    });
  }

  // Utility methods
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    return this.request({
      method: 'GET',
      url: '/health',
    });
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default apiService;