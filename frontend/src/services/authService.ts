/**
 * Authentication service for Akasha frontend.
 * 
 * Handles JWT authentication, token management, and user session state.
 */

import axios, { AxiosResponse } from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export interface User {
  id: string;
  email: string;
  full_name: string;
  role: 'admin' | 'user' | 'viewer';
  is_active: boolean;
  created_at: string;
  last_login?: string;
  status: 'active' | 'inactive' | 'suspended';
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  full_name: string;
  password: string;
  role?: 'user' | 'admin' | 'viewer';
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
  metadata?: {
    timestamp: number;
    path?: string;
  };
}

class AuthService {
  private accessToken: string | null = null;
  private refreshToken: string | null = null;
  private tokenExpiryTime: number | null = null;

  constructor() {
    // Load tokens from localStorage on initialization
    this.loadTokensFromStorage();
    this.setupAxiosInterceptors();
  }

  private loadTokensFromStorage(): void {
    this.accessToken = localStorage.getItem('access_token');
    this.refreshToken = localStorage.getItem('refresh_token');
    const expiryTime = localStorage.getItem('token_expiry');
    this.tokenExpiryTime = expiryTime ? parseInt(expiryTime) : null;
  }

  private saveTokensToStorage(tokenResponse: TokenResponse): void {
    this.accessToken = tokenResponse.access_token;
    this.refreshToken = tokenResponse.refresh_token;
    
    // Calculate expiry time
    this.tokenExpiryTime = Date.now() + (tokenResponse.expires_in * 1000);

    localStorage.setItem('access_token', this.accessToken);
    localStorage.setItem('refresh_token', this.refreshToken);
    localStorage.setItem('token_expiry', this.tokenExpiryTime.toString());
  }

  private clearTokensFromStorage(): void {
    this.accessToken = null;
    this.refreshToken = null;
    this.tokenExpiryTime = null;

    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('token_expiry');
  }

  private setupAxiosInterceptors(): void {
    // Request interceptor to add auth header
    axios.interceptors.request.use(
      (config) => {
        if (this.accessToken && this.isTokenValid()) {
          config.headers.Authorization = `Bearer ${this.accessToken}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor to handle token refresh
    axios.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config;

        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;

          try {
            await this.refreshAccessToken();
            originalRequest.headers.Authorization = `Bearer ${this.accessToken}`;
            return axios(originalRequest);
          } catch (refreshError) {
            this.logout();
            window.location.href = '/login';
            return Promise.reject(refreshError);
          }
        }

        return Promise.reject(error);
      }
    );
  }

  public async login(credentials: LoginRequest): Promise<User> {
    try {
      const response: AxiosResponse<TokenResponse> = await axios.post(
        `${API_BASE_URL}/auth/login`,
        credentials
      );

      this.saveTokensToStorage(response.data);
      
      // Get user info after successful login
      const userInfo = await this.getCurrentUser();
      return userInfo;
    } catch (error: any) {
      if (error.response?.data?.error) {
        throw new Error(error.response.data.error.message || 'Login failed');
      }
      throw new Error('Network error during login');
    }
  }

  public async register(userData: RegisterRequest): Promise<User> {
    try {
      const response: AxiosResponse<User> = await axios.post(
        `${API_BASE_URL}/auth/register`,
        userData
      );

      return response.data;
    } catch (error: any) {
      if (error.response?.data?.error) {
        throw new Error(error.response.data.error.message || 'Registration failed');
      }
      throw new Error('Network error during registration');
    }
  }

  public async logout(): Promise<void> {
    try {
      if (this.refreshToken) {
        await axios.post(`${API_BASE_URL}/auth/logout`, {
          refresh_token: this.refreshToken
        });
      }
    } catch (error) {
      console.error('Error during logout:', error);
    } finally {
      this.clearTokensFromStorage();
    }
  }

  public async refreshAccessToken(): Promise<void> {
    if (!this.refreshToken) {
      throw new Error('No refresh token available');
    }

    try {
      const response: AxiosResponse<TokenResponse> = await axios.post(
        `${API_BASE_URL}/auth/refresh`,
        { refresh_token: this.refreshToken }
      );

      this.saveTokensToStorage(response.data);
    } catch (error: any) {
      this.clearTokensFromStorage();
      throw new Error('Token refresh failed');
    }
  }

  public async getCurrentUser(): Promise<User> {
    try {
      const response: AxiosResponse<User> = await axios.get(
        `${API_BASE_URL}/auth/me`
      );

      return response.data;
    } catch (error: any) {
      if (error.response?.status === 401) {
        this.clearTokensFromStorage();
        throw new Error('Authentication required');
      }
      throw new Error('Failed to get user information');
    }
  }

  public isAuthenticated(): boolean {
    return this.accessToken !== null && this.isTokenValid();
  }

  private isTokenValid(): boolean {
    if (!this.tokenExpiryTime) return false;
    
    // Add 5 minutes buffer before expiry
    return Date.now() < (this.tokenExpiryTime - 5 * 60 * 1000);
  }

  public getAccessToken(): string | null {
    return this.isTokenValid() ? this.accessToken : null;
  }

  public hasRole(requiredRole: string): boolean {
    // Role hierarchy: admin > user > viewer
    const roleHierarchy = { admin: 3, user: 2, viewer: 1 };
    
    // Get current user role from token
    if (!this.accessToken) return false;
    
    try {
      const payload = JSON.parse(atob(this.accessToken.split('.')[1]));
      const userRole = payload.role;
      
      return (roleHierarchy[userRole as keyof typeof roleHierarchy] || 0) >= 
             (roleHierarchy[requiredRole as keyof typeof roleHierarchy] || 0);
    } catch {
      return false;
    }
  }

  public getCurrentUserRole(): string | null {
    if (!this.accessToken) return null;
    
    try {
      const payload = JSON.parse(atob(this.accessToken.split('.')[1]));
      return payload.role || null;
    } catch {
      return null;
    }
  }
}

// Export singleton instance
export const authService = new AuthService();
export default authService;