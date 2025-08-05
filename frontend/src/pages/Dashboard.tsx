import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
} from '@mui/material';
import {
  Upload as UploadIcon,
  Description as DocumentIcon,
  Search as SearchIcon,
  Chat as ChatIcon,
  TrendingUp as TrendingUpIcon,
  Schedule as ScheduleIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

// Mock data for demonstration
const mockStats = {
  totalDocuments: 127,
  processedDocuments: 119,
  pendingProcessing: 8,
  totalQueries: 1543,
  avgResponseTime: 2.1,
  storageUsed: 0.75, // 75%
};

const mockRecentDocuments = [
  {
    id: '1',
    title: 'Research Paper on AI Ethics',
    status: 'completed',
    uploadDate: '2025-01-15',
  },
  {
    id: '2',
    title: 'Technical Documentation v2.3',
    status: 'processing',
    uploadDate: '2025-01-15',
  },
  {
    id: '3',
    title: 'Market Analysis Report',
    status: 'completed',
    uploadDate: '2025-01-14',
  },
];

const mockRecentQueries = [
  'What are the latest developments in machine learning?',
  'How does the RAG system handle multimodal content?',
  'Explain the architecture of vector databases',
];

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'processing':
        return 'warning';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Welcome Section */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Welcome to Akasha
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Your intelligent document processing and retrieval system
        </Typography>
      </Box>

      {/* Quick Actions */}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, mb: 4 }}>
        <Box sx={{ flex: { xs: '1 1 100%', sm: '1 1 45%', md: '1 1 20%' }, minWidth: 0 }}>
          <Card sx={{ height: '100%', cursor: 'pointer' }} onClick={() => navigate('/documents/upload')}>
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <UploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Upload Documents
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Add new documents to your knowledge base
              </Typography>
            </CardContent>
          </Card>
        </Box>

        <Box sx={{ flex: { xs: '1 1 100%', sm: '1 1 45%', md: '1 1 20%' }, minWidth: 0 }}>
          <Card sx={{ height: '100%', cursor: 'pointer' }} onClick={() => navigate('/search')}>
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <SearchIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Search Knowledge
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Find information across all documents
              </Typography>
            </CardContent>
          </Card>
        </Box>

        <Box sx={{ flex: { xs: '1 1 100%', sm: '1 1 45%', md: '1 1 20%' }, minWidth: 0 }}>
          <Card sx={{ height: '100%', cursor: 'pointer' }} onClick={() => navigate('/chat')}>
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <ChatIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Start Chat
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Ask questions and get AI-powered answers
              </Typography>
            </CardContent>
          </Card>
        </Box>

        <Box sx={{ flex: { xs: '1 1 100%', sm: '1 1 45%', md: '1 1 20%' }, minWidth: 0 }}>
          <Card sx={{ height: '100%', cursor: 'pointer' }} onClick={() => navigate('/documents')}>
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <DocumentIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Manage Documents
              </Typography>
              <Typography variant="body2" color="text.secondary">
                View and organize your document library
              </Typography>
            </CardContent>
          </Card>
        </Box>
      </Box>

      {/* Statistics */}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, mb: 4 }}>
        <Box sx={{ flex: { xs: '1 1 100%', md: '1 1 65%' }, minWidth: 0 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Overview
              </Typography>
              
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
                <Box sx={{ flex: { xs: '1 1 45%', md: '1 1 20%' }, textAlign: 'center' }}>
                  <Typography variant="h4" color="primary.main">
                    {mockStats.totalDocuments}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Documents
                  </Typography>
                </Box>
                
                <Box sx={{ flex: { xs: '1 1 45%', md: '1 1 20%' }, textAlign: 'center' }}>
                  <Typography variant="h4" color="success.main">
                    {mockStats.processedDocuments}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Processed
                  </Typography>
                </Box>
                
                <Box sx={{ flex: { xs: '1 1 45%', md: '1 1 20%' }, textAlign: 'center' }}>
                  <Typography variant="h4" color="warning.main">
                    {mockStats.pendingProcessing}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Processing
                  </Typography>
                </Box>
                
                <Box sx={{ flex: { xs: '1 1 45%', md: '1 1 20%' }, textAlign: 'center' }}>
                  <Typography variant="h4" color="info.main">
                    {mockStats.totalQueries}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Queries
                  </Typography>
                </Box>
              </Box>

              <Box sx={{ mt: 3 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Storage Usage
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={mockStats.storageUsed * 100} 
                  sx={{ height: 8, borderRadius: 4 }}
                />
                <Typography variant="caption" color="text.secondary">
                  {Math.round(mockStats.storageUsed * 100)}% of available storage
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Box>

        <Box sx={{ flex: { xs: '1 1 100%', md: '1 1 30%' }, minWidth: 0 }}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Metrics
              </Typography>
              
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <TrendingUpIcon sx={{ mr: 1, color: 'success.main' }} />
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Avg Response Time
                  </Typography>
                  <Typography variant="h6">
                    {mockStats.avgResponseTime}s
                  </Typography>
                </Box>
              </Box>

              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <ScheduleIcon sx={{ mr: 1, color: 'info.main' }} />
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    System Uptime
                  </Typography>
                  <Typography variant="h6">
                    99.9%
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Box>
      </Box>

      {/* Recent Activity */}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
        <Box sx={{ flex: { xs: '1 1 100%', md: '1 1 45%' }, minWidth: 0 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Documents
              </Typography>
              
              <List>
                {mockRecentDocuments.map((doc, index) => (
                  <React.Fragment key={doc.id}>
                    <ListItem>
                      <ListItemIcon>
                        <DocumentIcon />
                      </ListItemIcon>
                      <ListItemText
                        primary={doc.title}
                        secondary={`Uploaded ${doc.uploadDate}`}
                      />
                      <Chip
                        label={doc.status}
                        size="small"
                        color={getStatusColor(doc.status) as any}
                        variant="outlined"
                      />
                    </ListItem>
                    {index < mockRecentDocuments.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>

              <Button
                fullWidth
                variant="outlined"
                onClick={() => navigate('/documents')}
                sx={{ mt: 2 }}
              >
                View All Documents
              </Button>
            </CardContent>
          </Card>
        </Box>

        <Box sx={{ flex: { xs: '1 1 100%', md: '1 1 45%' }, minWidth: 0 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Queries
              </Typography>
              
              <List>
                {mockRecentQueries.map((query, index) => (
                  <React.Fragment key={index}>
                    <ListItem>
                      <ListItemIcon>
                        <SearchIcon />
                      </ListItemIcon>
                      <ListItemText
                        primary={query}
                        secondary="Today"
                      />
                    </ListItem>
                    {index < mockRecentQueries.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>

              <Button
                fullWidth
                variant="outlined"
                onClick={() => navigate('/search')}
                sx={{ mt: 2 }}
              >
                Go to Search
              </Button>
            </CardContent>
          </Card>
        </Box>
      </Box>
    </Box>
  );
};