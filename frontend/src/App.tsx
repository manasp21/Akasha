import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import { createAkashaTheme } from './styles/theme';
import { useAppStore } from './store/appStore';

// Layout
import { AppLayout } from './components/Layout/AppLayout';

// Pages
import { Dashboard } from './pages/Dashboard';
import { Documents } from './pages/Documents';
import { Search } from './pages/Search';
import { Chat } from './pages/Chat';
import { Settings } from './pages/Settings';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  const { theme } = useAppStore();
  
  // Create theme based on current mode
  const muiTheme = React.useMemo(
    () => createAkashaTheme(theme.mode),
    [theme.mode]
  );

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={muiTheme}>
        <CssBaseline />
        <Router>
          <Routes>
            {/* Main Application Routes */}
            <Route path="/" element={<AppLayout />}>
              <Route index element={<Dashboard />} />
              <Route path="documents" element={<Documents />} />
              <Route path="search" element={<Search />} />
              <Route path="chat" element={<Chat />} />
              <Route path="settings" element={<Settings />} />
            </Route>

            {/* Fallback Route */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Router>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
