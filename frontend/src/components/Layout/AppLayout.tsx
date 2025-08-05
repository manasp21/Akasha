import React from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  useTheme,
  useMediaQuery,
  Badge,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Description as DocumentIcon,
  Search as SearchIcon,
  Chat as ChatIcon,
  Settings as SettingsIcon,
  Notifications as NotificationsIcon,
  AccountCircle as AccountIcon,
  LightMode,
  DarkMode,
} from '@mui/icons-material';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { useAppStore } from '../../store/appStore';

const DRAWER_WIDTH = 280;

const navigationItems = [
  { id: 'dashboard', label: 'Dashboard', path: '/', icon: DashboardIcon },
  { id: 'documents', label: 'Documents', path: '/documents', icon: DocumentIcon },
  { id: 'search', label: 'Search', path: '/search', icon: SearchIcon },
  { id: 'chat', label: 'Chat', path: '/chat', icon: ChatIcon },
  { id: 'settings', label: 'Settings', path: '/settings', icon: SettingsIcon },
];

export const AppLayout: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const navigate = useNavigate();
  const location = useLocation();

  const {
    viewState,
    theme: appTheme,
    notifications,
    toggleSidebar,
    toggleTheme,
    setCurrentPage,
  } = useAppStore();

  const handleNavigation = (path: string, pageId: string) => {
    navigate(path);
    setCurrentPage(pageId);
    
    // Close sidebar on mobile after navigation
    if (isMobile && !viewState.sidebar_collapsed) {
      toggleSidebar();
    }
  };


  const unreadNotifications = notifications.filter(n => !n.read).length;

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Logo/Brand */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="h6" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
          Akasha
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Multimodal RAG System
        </Typography>
      </Box>

      {/* Navigation */}
      <List sx={{ flex: 1, pt: 1 }}>
        {navigationItems.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.path;

          return (
            <ListItem key={item.id} disablePadding sx={{ px: 1 }}>
              <ListItemButton
                onClick={() => handleNavigation(item.path, item.id)}
                selected={isActive}
                sx={{
                  borderRadius: 1,
                  mx: 1,
                  '&.Mui-selected': {
                    backgroundColor: 'primary.main',
                    color: 'primary.contrastText',
                    '&:hover': {
                      backgroundColor: 'primary.dark',
                    },
                    '& .MuiListItemIcon-root': {
                      color: 'primary.contrastText',
                    },
                  },
                }}
              >
                <ListItemIcon>
                  <Icon />
                </ListItemIcon>
                <ListItemText primary={item.label} />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          zIndex: theme.zIndex.drawer + 1,
          backgroundColor: 'background.paper',
          color: 'text.primary',
          boxShadow: 1,
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="toggle sidebar"
            onClick={toggleSidebar}
            edge="start"
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>

          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            {navigationItems.find(item => location.pathname === item.path)?.label || 'Akasha'}
          </Typography>

          {/* Theme Toggle */}
          <IconButton color="inherit" onClick={toggleTheme} sx={{ mr: 1 }}>
            {appTheme.mode === 'light' ? <DarkMode /> : <LightMode />}
          </IconButton>

          {/* Notifications */}
          <IconButton color="inherit" sx={{ mr: 1 }}>
            <Badge badgeContent={unreadNotifications} color="error">
              <NotificationsIcon />
            </Badge>
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Sidebar */}
      <Drawer
        variant={isMobile ? 'temporary' : 'persistent'}
        open={!viewState.sidebar_collapsed}
        onClose={toggleSidebar}
        sx={{
          width: DRAWER_WIDTH,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: DRAWER_WIDTH,
            boxSizing: 'border-box',
            top: isMobile ? 0 : '64px',
            height: isMobile ? '100%' : 'calc(100% - 64px)',
            borderRight: 1,
            borderColor: 'divider',
          },
        }}
        ModalProps={{
          keepMounted: true, // Better mobile performance
        }}
      >
        {drawer}
      </Drawer>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { sm: `calc(100% - ${viewState.sidebar_collapsed ? 0 : DRAWER_WIDTH}px)` },
          ml: { sm: viewState.sidebar_collapsed ? 0 : `${DRAWER_WIDTH}px` },
          mt: '64px',
          height: 'calc(100vh - 64px)',
          overflow: 'auto',
        }}
      >
        <Outlet />
      </Box>
    </Box>
  );
};