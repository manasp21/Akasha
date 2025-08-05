"""
Authentication module for Akasha API.

Provides JWT-based authentication, user management, and role-based access control.
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from enum import Enum

from jose import jwt
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field

from ..core.config import get_config
from ..core.logging import get_logger
from ..core.exceptions import AkashaError

logger = get_logger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token security
security = HTTPBearer()

class UserRole(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"

class UserStatus(str, Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"

# Pydantic models
class UserBase(BaseModel):
    """Base user model."""
    email: EmailStr
    full_name: str = Field(..., min_length=1, max_length=100)
    role: UserRole = UserRole.USER
    is_active: bool = True

class UserCreate(UserBase):
    """User creation model."""
    password: str = Field(..., min_length=8, max_length=100)

class UserUpdate(BaseModel):
    """User update model."""
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None

class UserResponse(UserBase):
    """User response model."""
    id: str
    created_at: datetime
    last_login: Optional[datetime] = None
    status: UserStatus

class LoginRequest(BaseModel):
    """Login request model."""
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenRefreshRequest(BaseModel):
    """Token refresh request model."""
    refresh_token: str

# In-memory user store (for demo - replace with database)
users_db: Dict[str, Dict[str, Any]] = {}
refresh_tokens_db: Dict[str, Dict[str, Any]] = {}

class AuthenticationError(AkashaError):
    """Authentication-related errors."""
    pass

class AuthorizationError(AkashaError):
    """Authorization-related errors."""
    pass

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def generate_user_id() -> str:
    """Generate a unique user ID."""
    return secrets.token_urlsafe(16)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    config = get_config()
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=config.auth.access_token_expire_minutes)
    
    to_encode.update({"exp": expire, "type": "access"})
    
    encoded_jwt = jwt.encode(
        to_encode, 
        config.auth.secret_key, 
        algorithm=config.auth.algorithm
    )
    
    return encoded_jwt

def create_refresh_token(user_id: str) -> str:
    """Create a refresh token."""
    config = get_config()
    token_id = secrets.token_urlsafe(32)
    
    # Store refresh token in database
    refresh_tokens_db[token_id] = {
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc),
        "expires_at": datetime.now(timezone.utc) + timedelta(days=config.auth.refresh_token_expire_days),
        "is_active": True
    }
    
    return token_id

def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode a JWT token."""
    config = get_config()
    
    try:
        payload = jwt.decode(
            token, 
            config.auth.secret_key, 
            algorithms=[config.auth.algorithm]
        )
        
        if payload.get("type") != "access":
            raise AuthenticationError("AUTH_001", "Invalid token type")
            
        return payload
        
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("AUTH_002", "Token has expired")
    except jwt.JWTError:
        raise AuthenticationError("AUTH_003", "Invalid token")

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email."""
    for user_id, user in users_db.items():
        if user["email"] == email:
            return {"id": user_id, **user}
    return None

def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user by ID."""
    user = users_db.get(user_id)
    if user:
        return {"id": user_id, **user}
    return None

def create_user(user_data: UserCreate) -> Dict[str, Any]:
    """Create a new user."""
    # Check if user already exists
    if get_user_by_email(user_data.email):
        raise AuthenticationError("AUTH_004", "User with this email already exists")
    
    user_id = generate_user_id()
    hashed_password = hash_password(user_data.password)
    
    user = {
        "email": user_data.email,
        "full_name": user_data.full_name,
        "role": user_data.role,
        "is_active": user_data.is_active,
        "hashed_password": hashed_password,
        "created_at": datetime.now(timezone.utc),
        "last_login": None,
        "status": UserStatus.ACTIVE
    }
    
    users_db[user_id] = user
    
    logger.info(f"User created: {user_data.email}", user_id=user_id)
    
    return {"id": user_id, **user}

def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user with email and password."""
    user = get_user_by_email(email)
    if not user:
        return None
    
    if not verify_password(password, user["hashed_password"]):
        return None
    
    if not user["is_active"] or user["status"] != UserStatus.ACTIVE:
        return None
    
    # Update last login
    users_db[user["id"]]["last_login"] = datetime.now(timezone.utc)
    
    return user

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current authenticated user."""
    try:
        payload = verify_token(credentials.credentials)
        user_id = payload.get("sub")
        
        if user_id is None:
            raise AuthenticationError("AUTH_005", "Invalid token payload")
        
        user = get_user_by_id(user_id)
        if user is None:
            raise AuthenticationError("AUTH_006", "User not found")
        
        if not user["is_active"] or user["status"] != UserStatus.ACTIVE:
            raise AuthenticationError("AUTH_007", "User account is inactive")
        
        return user
        
    except AuthenticationError:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise AuthenticationError("AUTH_008", "Authentication failed")

def require_role(required_roles: List[UserRole]):
    """Dependency to require specific roles."""
    async def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        user_role = UserRole(current_user["role"])
        
        if user_role not in required_roles:
            raise AuthorizationError(
                "AUTH_009", 
                f"Insufficient permissions. Required roles: {[r.value for r in required_roles]}"
            )
        
        return current_user
    
    return role_checker

# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate, request: Request):
    """Register a new user."""
    try:
        user = create_user(user_data)
        
        # Remove sensitive data from response
        user_response = {k: v for k, v in user.items() if k != "hashed_password"}
        
        logger.info(f"User registered: {user_data.email}", 
                   user_id=user["id"], 
                   ip=request.client.host)
        
        return UserResponse(**user_response)
        
    except AuthenticationError:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/login", response_model=TokenResponse)
async def login(login_data: LoginRequest, request: Request):
    """Login user and return tokens."""
    try:
        user = authenticate_user(login_data.email, login_data.password)
        
        if not user:
            logger.warning(f"Failed login attempt: {login_data.email}", 
                          ip=request.client.host)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        config = get_config()
        
        # Create tokens
        access_token = create_access_token(
            data={"sub": user["id"], "email": user["email"], "role": user["role"]}
        )
        refresh_token = create_refresh_token(user["id"])
        
        logger.info(f"User logged in: {login_data.email}", 
                   user_id=user["id"], 
                   ip=request.client.host)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=config.auth.access_token_expire_minutes * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_data: TokenRefreshRequest, request: Request):
    """Refresh access token using refresh token."""
    try:
        token_data = refresh_tokens_db.get(refresh_data.refresh_token)
        
        if not token_data or not token_data["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Check if token has expired
        if datetime.now(timezone.utc) > token_data["expires_at"]:
            # Mark token as inactive
            refresh_tokens_db[refresh_data.refresh_token]["is_active"] = False
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has expired"
            )
        
        user = get_user_by_id(token_data["user_id"])
        if not user or not user["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        config = get_config()
        
        # Create new access token
        access_token = create_access_token(
            data={"sub": user["id"], "email": user["email"], "role": user["role"]}
        )
        
        # Create new refresh token
        new_refresh_token = create_refresh_token(user["id"])
        
        # Invalidate old refresh token
        refresh_tokens_db[refresh_data.refresh_token]["is_active"] = False
        
        logger.info(f"Token refreshed: {user['email']}", 
                   user_id=user["id"], 
                   ip=request.client.host)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=config.auth.access_token_expire_minutes * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@router.post("/logout")
async def logout(
    refresh_data: TokenRefreshRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    request: Request = None
):
    """Logout user by invalidating refresh token."""
    try:
        # Invalidate refresh token
        if refresh_data.refresh_token in refresh_tokens_db:
            refresh_tokens_db[refresh_data.refresh_token]["is_active"] = False
        
        logger.info(f"User logged out: {current_user['email']}", 
                   user_id=current_user["id"], 
                   ip=request.client.host if request else None)
        
        return {"success": True, "message": "Successfully logged out"}
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current user information."""
    user_response = {k: v for k, v in current_user.items() if k != "hashed_password"}
    return UserResponse(**user_response)

@router.get("/users", response_model=List[UserResponse])
async def list_users(
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN]))
):
    """List all users (admin only)."""
    users = []
    for user_id, user_data in users_db.items():
        user_response = {
            "id": user_id,
            **{k: v for k, v in user_data.items() if k != "hashed_password"}
        }
        users.append(UserResponse(**user_response))
    
    return users

@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    user_update: UserUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Update user information."""
    # Users can only update themselves, unless they are admin
    if user_id != current_user["id"] and current_user["role"] != UserRole.ADMIN:
        raise AuthorizationError("AUTH_010", "Cannot update other users")
    
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update user data
    update_data = user_update.dict(exclude_unset=True)
    users_db[user_id].update(update_data)
    
    # Get updated user
    updated_user = get_user_by_id(user_id)
    user_response = {k: v for k, v in updated_user.items() if k != "hashed_password"}
    
    logger.info(f"User updated: {updated_user['email']}", 
               user_id=user_id,
               updated_by=current_user["id"])
    
    return UserResponse(**user_response)

# Initialize default admin user
def init_default_users():
    """Initialize default users."""
    config = get_config()
    
    # Create default admin user if none exists
    if not users_db and hasattr(config.auth, 'default_admin_email'):
        try:
            admin_user = UserCreate(
                email=config.auth.default_admin_email,
                full_name="System Administrator",
                role=UserRole.ADMIN,
                password=config.auth.default_admin_password
            )
            create_user(admin_user)
            logger.info("Default admin user created", email=config.auth.default_admin_email)
        except Exception as e:
            logger.error(f"Failed to create default admin user: {str(e)}")

# Initialize default users on module import
init_default_users()