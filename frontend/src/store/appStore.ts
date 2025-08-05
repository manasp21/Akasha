import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { AppSettings, AppTheme, Notification, ModalState, ViewState, LoadingState, ErrorState } from '../types/app';

interface AppState {
  // Settings
  settings: AppSettings;
  
  // UI State
  theme: AppTheme;
  viewState: ViewState;
  loadingState: LoadingState;
  errorState: ErrorState;
  
  // Notifications
  notifications: Notification[];
  
  // Modals
  modal: ModalState;
  
  // Actions
  updateSettings: (settings: Partial<AppSettings>) => void;
  setTheme: (theme: Partial<AppTheme>) => void;
  toggleTheme: () => void;
  
  // View state actions
  toggleSidebar: () => void;
  setCurrentPage: (page: string) => void;
  toggleSearchPanel: () => void;
  toggleDocumentPreview: () => void;
  toggleChatPanel: () => void;
  toggleFiltersPanel: () => void;
  
  // Loading state actions
  setLoading: (key: keyof LoadingState, loading: boolean) => void;
  
  // Error state actions
  setError: (key: keyof ErrorState, error: string | null) => void;
  clearErrors: () => void;
  
  // Notification actions
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => void;
  removeNotification: (id: string) => void;
  markNotificationRead: (id: string) => void;
  clearNotifications: () => void;
  
  // Modal actions
  openModal: (type: ModalState['type'], data?: any, options?: Partial<ModalState>) => void;
  closeModal: () => void;
}

const defaultSettings: AppSettings = {
  theme: {
    mode: 'light',
  },
  language: 'en',
  results_per_page: 20,
  default_search_type: 'hybrid',
  auto_save_chat: true,
  show_source_citations: true,
  enable_streaming: true,
};

const defaultViewState: ViewState = {
  sidebar_collapsed: false,
  current_page: 'dashboard',
  search_panel_open: false,
  document_preview_open: false,
  chat_panel_open: false,
  filters_panel_open: false,
};

const defaultLoadingState: LoadingState = {
  global: false,
  documents: false,
  search: false,
  chat: false,
  upload: false,
  auth: false,
};

const defaultErrorState: ErrorState = {};

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Initial state
      settings: defaultSettings,
      theme: defaultSettings.theme,
      viewState: defaultViewState,
      loadingState: defaultLoadingState,
      errorState: defaultErrorState,
      notifications: [],
      modal: { isOpen: false },

      // Settings actions
      updateSettings: (newSettings: Partial<AppSettings>) => {
        const currentSettings = get().settings;
        const updatedSettings = { ...currentSettings, ...newSettings };
        
        set({
          settings: updatedSettings,
          theme: updatedSettings.theme,
        });
      },

      setTheme: (themeUpdate: Partial<AppTheme>) => {
        const currentTheme = get().theme;
        const newTheme = { ...currentTheme, ...themeUpdate };
        
        set({
          theme: newTheme,
          settings: {
            ...get().settings,
            theme: newTheme,
          },
        });
      },

      toggleTheme: () => {
        const currentTheme = get().theme;
        const newMode = currentTheme.mode === 'light' ? 'dark' : 'light';
        
        get().setTheme({ mode: newMode });
      },

      // View state actions
      toggleSidebar: () => {
        set({
          viewState: {
            ...get().viewState,
            sidebar_collapsed: !get().viewState.sidebar_collapsed,
          },
        });
      },

      setCurrentPage: (page: string) => {
        set({
          viewState: {
            ...get().viewState,
            current_page: page,
          },
        });
      },

      toggleSearchPanel: () => {
        set({
          viewState: {
            ...get().viewState,
            search_panel_open: !get().viewState.search_panel_open,
          },
        });
      },

      toggleDocumentPreview: () => {
        set({
          viewState: {
            ...get().viewState,
            document_preview_open: !get().viewState.document_preview_open,
          },
        });
      },

      toggleChatPanel: () => {
        set({
          viewState: {
            ...get().viewState,
            chat_panel_open: !get().viewState.chat_panel_open,
          },
        });
      },

      toggleFiltersPanel: () => {
        set({
          viewState: {
            ...get().viewState,
            filters_panel_open: !get().viewState.filters_panel_open,
          },
        });
      },

      // Loading state actions
      setLoading: (key: keyof LoadingState, loading: boolean) => {
        set({
          loadingState: {
            ...get().loadingState,
            [key]: loading,
          },
        });
      },

      // Error state actions
      setError: (key: keyof ErrorState, error: string | null) => {
        set({
          errorState: {
            ...get().errorState,
            [key]: error,
          },
        });
      },

      clearErrors: () => {
        set({ errorState: {} });
      },

      // Notification actions
      addNotification: (notification) => {
        const newNotification: Notification = {
          ...notification,
          id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
          timestamp: new Date().toISOString(),
          read: false,
        };

        set({
          notifications: [newNotification, ...get().notifications],
        });

        // Auto-dismiss if specified
        if (notification.auto_dismiss !== false) {
          const duration = notification.duration || 5000;
          setTimeout(() => {
            get().removeNotification(newNotification.id);
          }, duration);
        }
      },

      removeNotification: (id: string) => {
        set({
          notifications: get().notifications.filter(n => n.id !== id),
        });
      },

      markNotificationRead: (id: string) => {
        set({
          notifications: get().notifications.map(n =>
            n.id === id ? { ...n, read: true } : n
          ),
        });
      },

      clearNotifications: () => {
        set({ notifications: [] });
      },

      // Modal actions
      openModal: (type, data, options = {}) => {
        set({
          modal: {
            isOpen: true,
            type,
            data,
            ...options,
          },
        });
      },

      closeModal: () => {
        set({
          modal: { isOpen: false },
        });
      },
    }),
    {
      name: 'akasha-app',
      partialize: (state) => ({
        settings: state.settings,
        theme: state.theme,
        viewState: state.viewState,
      }),
    }
  )
);