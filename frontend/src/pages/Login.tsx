import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Alert,
  InputAdornment,
  IconButton,
  Divider,
  CircularProgress,
  Link,
} from '@mui/material';
import {
  Visibility,
  VisibilityOff,
  Email,
  Lock,
  Login as LoginIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import authService from '../services/authService';

interface LoginForm {
  email: string;
  password: string;
}

export const Login: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  const [form, setForm] = useState<LoginForm>({
    email: '',
    password: '',
  });
  
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showRegister, setShowRegister] = useState(false);

  // Check if user is already authenticated
  useEffect(() => {
    if (authService.isAuthenticated()) {
      const from = (location.state as any)?.from?.pathname || '/';
      navigate(from, { replace: true });
    }
  }, [navigate, location]);

  const handleInputChange = (field: keyof LoginForm) => (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setForm(prev => ({
      ...prev,
      [field]: event.target.value,
    }));
    // Clear error when user starts typing
    if (error) setError(null);
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    
    if (!form.email || !form.password) {
      setError('Please fill in all fields');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await authService.login({
        email: form.email,
        password: form.password,
      });

      // Redirect to intended page or dashboard
      const from = (location.state as any)?.from?.pathname || '/';
      navigate(from, { replace: true });
    } catch (err: any) {
      setError(err.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  const handleRegisterSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    
    if (!form.email || !form.password) {
      setError('Please fill in all fields');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await authService.register({
        email: form.email,
        full_name: form.email.split('@')[0], // Use email prefix as default name
        password: form.password,
        role: 'user',
      });

      // Auto-login after registration
      await authService.login({
        email: form.email,
        password: form.password,
      });

      const from = (location.state as any)?.from?.pathname || '/';
      navigate(from, { replace: true });
    } catch (err: any) {
      setError(err.message || 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  const handleDemoLogin = () => {
    setForm({
      email: 'admin@example.com',
      password: 'admin123',
    });
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        p: 2,
      }}
    >
      <Card
        sx={{
          maxWidth: 400,
          width: '100%',
          boxShadow: '0 20px 40px rgba(0,0,0,0.1)',
        }}
      >
        <CardContent sx={{ p: 4 }}>
          <Box sx={{ textAlign: 'center', mb: 3 }}>
            <Typography variant="h4" gutterBottom sx={{ fontWeight: 600 }}>
              {showRegister ? 'Create Account' : 'Welcome Back'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {showRegister 
                ? 'Sign up to access Akasha' 
                : 'Sign in to your Akasha account'
              }
            </Typography>
          </Box>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <Box
            component="form"
            onSubmit={showRegister ? handleRegisterSubmit : handleSubmit}
            sx={{ width: '100%' }}
          >
            <TextField
              fullWidth
              label="Email Address"
              type="email"
              value={form.email}
              onChange={handleInputChange('email')}
              disabled={loading}
              required
              sx={{ mb: 2 }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Email />
                  </InputAdornment>
                ),
              }}
            />

            <TextField
              fullWidth
              label="Password"
              type={showPassword ? 'text' : 'password'}
              value={form.password}
              onChange={handleInputChange('password')}
              disabled={loading}
              required
              sx={{ mb: 3 }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Lock />
                  </InputAdornment>
                ),
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      onClick={() => setShowPassword(!showPassword)}
                      edge="end"
                    >
                      {showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />

            <Button
              type="submit"
              fullWidth
              variant="contained"
              size="large"
              disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : <LoginIcon />}
              sx={{
                mb: 2,
                py: 1.5,
                background: 'linear-gradient(45deg, #667eea 30%, #764ba2 90%)',
                '&:hover': {
                  background: 'linear-gradient(45deg, #5a6fd8 30%, #6a4190 90%)',
                },
              }}
            >
              {loading 
                ? 'Please wait...' 
                : showRegister 
                  ? 'Create Account' 
                  : 'Sign In'
              }
            </Button>

            <Divider sx={{ my: 2 }}>
              <Typography variant="body2" color="text.secondary">
                or
              </Typography>
            </Divider>

            <Button
              fullWidth
              variant="outlined"
              onClick={handleDemoLogin}
              disabled={loading}
              sx={{ mb: 2 }}
            >
              Try Demo Account
            </Button>

            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="body2">
                {showRegister ? 'Already have an account?' : "Don't have an account?"}{' '}
                <Link
                  component="button"
                  type="button"
                  variant="body2"
                  onClick={() => {
                    setShowRegister(!showRegister);
                    setError(null);
                  }}
                  sx={{ fontWeight: 600 }}
                >
                  {showRegister ? 'Sign In' : 'Sign Up'}
                </Link>
              </Typography>
            </Box>
          </Box>

          {process.env.NODE_ENV === 'development' && (
            <Box sx={{ mt: 3, p: 2, bgcolor: 'grey.100', borderRadius: 1 }}>
              <Typography variant="caption" display="block" gutterBottom>
                Development Mode - Demo Credentials:
              </Typography>
              <Typography variant="caption" display="block">
                Email: admin@example.com
              </Typography>
              <Typography variant="caption" display="block">
                Password: admin123
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};