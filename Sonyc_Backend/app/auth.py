from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from .database import get_db
from .models import User
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

import bcrypt
import hashlib
import base64

# Password hashing - using direct bcrypt with SHA-256 pre-hashing
# Removed passlib/pwd_context due to compatibility issues

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30 days

# HTTP Bearer scheme for token extraction
security = HTTPBearer(auto_error=False)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash using direct bcrypt + SHA256 pre-hash"""
    # 1. SHA256 Pre-hash to ensure fixed length input to bcrypt
    m = hashlib.sha256()
    m.update(plain_password.encode('utf-8'))
    sha_bytes = m.digest()
    # 2. Base64 encode (to get safe bytes for bcrypt, keep as bytes)
    pwd_bytes = base64.b64encode(sha_bytes)
    
    # 3. Verify
    try:
        return bcrypt.checkpw(pwd_bytes, hashed_password.encode('utf-8'))
    except Exception as e:
        logger.error(f"Error verifying password: {e}")
        return False


def get_password_hash(password: str) -> str:
    """Hash a password using direct bcrypt + SHA256 pre-hash"""
    # 1. SHA256 Pre-hash
    m = hashlib.sha256()
    m.update(password.encode('utf-8'))
    sha_bytes = m.digest()
    # 2. Base64 encode
    pwd_bytes = base64.b64encode(sha_bytes)
    
    # 3. Hash
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode('utf-8')


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def extract_token_from_header(request: Request) -> Optional[str]:
    """Manually extract token from Authorization header as fallback"""
    authorization = request.headers.get("Authorization")
    if not authorization:
        return None
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            return None
        return token
    except ValueError:
        return None


def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Try to get token from HTTPBearer first
    token = None
    if credentials is not None:
        token = credentials.credentials
    
    # Fallback: manually extract from header if HTTPBearer didn't work
    if not token:
        token = extract_token_from_header(request)
    
    if not token:
        logger.warning("No token provided in request")
        raise credentials_exception
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            logger.warning("Token payload missing 'sub' field")
            raise credentials_exception
        user_id_int = int(user_id)
    except JWTError as e:
        # Token is invalid or expired
        logger.warning(f"JWT validation failed: {str(e)}")
        raise credentials_exception
    except ValueError as e:
        # Invalid user ID format
        logger.warning(f"Invalid user ID format in token: {str(e)}")
        raise credentials_exception
    
    try:
        user = db.query(User).filter(User.id == user_id_int).first()
        if user is None:
            logger.warning(f"User not found for ID: {user_id_int}")
            raise credentials_exception
        return user
    except Exception as e:
        # Database error
        logger.error(f"Database error while fetching user: {str(e)}")
        raise credentials_exception

