import React from 'react';
import { Box, Typography, Container } from '@mui/material';

export const Settings: React.FC = () => {
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ textAlign: 'center' }}>
        <Typography variant="h4" gutterBottom>
          Settings & Preferences
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
          This page will contain user preferences, system settings, and configuration options.
        </Typography>
        
        <Box sx={{ mt: 4, p: 3, bgcolor: 'background.paper', borderRadius: 2, boxShadow: 1 }}>
          <Typography variant="h6" gutterBottom>
            Coming Soon:
          </Typography>
          <Typography variant="body2" component="div" sx={{ textAlign: 'left', ml: 2 }}>
            • User profile and preferences<br/>
            • Theme and appearance settings<br/>
            • Search and chat preferences<br/>
            • System configuration<br/>
            • Data export and import<br/>
            • Account management
          </Typography>
        </Box>
      </Box>
    </Container>
  );
};