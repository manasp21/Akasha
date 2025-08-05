import React from 'react';
import { Box, Typography, Button, Container } from '@mui/material';
import { Upload as UploadIcon } from '@mui/icons-material';

export const Documents: React.FC = () => {
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ textAlign: 'center' }}>
        <Typography variant="h4" gutterBottom>
          Document Management
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
          This page will contain the document library, upload functionality, and document management features.
        </Typography>
        
        <Button
          variant="contained"
          startIcon={<UploadIcon />}
          size="large"
        >
          Upload Documents
        </Button>
        
        <Box sx={{ mt: 4, p: 3, bgcolor: 'background.paper', borderRadius: 2, boxShadow: 1 }}>
          <Typography variant="h6" gutterBottom>
            Coming Soon:
          </Typography>
          <Typography variant="body2" component="div" sx={{ textAlign: 'left', ml: 2 }}>
            • Drag-and-drop document upload<br/>
            • Document library with search and filtering<br/>
            • Document preview and viewer<br/>
            • Metadata editing and tagging<br/>
            • Processing status tracking<br/>
            • Bulk operations
          </Typography>
        </Box>
      </Box>
    </Container>
  );
};