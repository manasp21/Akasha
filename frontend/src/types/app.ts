// Application State Types
export interface AppTheme {
  mode: 'light' | 'dark';
  primary_color?: string;
  secondary_color?: string;
}

export interface AppSettings {
  theme: AppTheme;
  language: string;
  results_per_page: number;
  default_search_type: 'semantic' | 'keyword' | 'hybrid';
  auto_save_chat: boolean;
  show_source_citations: boolean;
  enable_streaming: boolean;
}

// Navigation Types
export interface NavigationItem {
  id: string;
  label: string;
  path: string;
  icon?: string;
  exact?: boolean;
  badge?: number;
}

// Upload Types
export interface FileUpload {
  id: string;
  file: File;
  name: string;
  size: number;
  type: string;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'failed';
  progress: number;
  error?: string;
  document_id?: string;
}

// Notification Types
export interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  action?: {
    label: string;
    onClick: () => void;
  };
  auto_dismiss?: boolean;
  duration?: number;
}

// Modal Types
export interface ModalState {
  isOpen: boolean;
  type?: 'document-upload' | 'document-preview' | 'settings' | 'help' | 'confirmation';
  data?: any;
  title?: string;
  onConfirm?: () => void;
  onCancel?: () => void;
}

// Loading States
export interface LoadingState {
  global: boolean;
  documents: boolean;
  search: boolean;
  chat: boolean;
  upload: boolean;
  auth: boolean;
}

// Error States
export interface ErrorState {
  global?: string;
  documents?: string;
  search?: string;
  chat?: string;
  upload?: string;
  auth?: string;
  network?: string;
}

// View States
export interface ViewState {
  sidebar_collapsed: boolean;
  current_page: string;
  search_panel_open: boolean;
  document_preview_open: boolean;
  chat_panel_open: boolean;
  filters_panel_open: boolean;
}

// Pagination Types
export interface PaginationState {
  page: number;
  per_page: number;
  total: number;
  total_pages: number;
}

// Search State Types
export interface SearchState {
  query: string;
  filters: {
    document_ids: string[];
    tags: string[];
    date_range?: {
      start: string;
      end: string;
    };
    content_types: string[];
  };
  search_type: 'semantic' | 'keyword' | 'hybrid';
  results: any[];
  pagination: PaginationState;
  loading: boolean;
  error?: string;
  last_search_time?: number;
}

// Filter Types
export interface FilterOption {
  id: string;
  label: string;
  value: string;
  count?: number;
  selected: boolean;
}

export interface FilterGroup {
  id: string;
  label: string;
  type: 'checkbox' | 'radio' | 'date' | 'range';
  options: FilterOption[];
  expanded: boolean;
}

// Component Props Types
export interface BaseComponentProps {
  className?: string;
  style?: React.CSSProperties;
  testId?: string;
}

export interface PageProps extends BaseComponentProps {
  title?: string;
  subtitle?: string;
  loading?: boolean;
  error?: string;
}

// Hook Return Types
export interface UseApiReturn<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export interface UseUploadReturn {
  upload: (files: File[]) => Promise<void>;
  uploads: FileUpload[];
  clearCompleted: () => void;
  cancelUpload: (id: string) => void;
  retryUpload: (id: string) => void;
}

// Form Types
export interface FormField {
  name: string;
  label: string;
  type: 'text' | 'email' | 'password' | 'select' | 'textarea' | 'checkbox' | 'file';
  required?: boolean;
  validation?: {
    min?: number;
    max?: number;
    pattern?: RegExp;
    message?: string;
  };
  options?: Array<{ label: string; value: string }>;
  placeholder?: string;
  help_text?: string;
}