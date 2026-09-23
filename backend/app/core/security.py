"""
Seguridad: hashing de contraseñas y creación/validación de JWT.

Usamos la librería `bcrypt` directamente (en vez de passlib) porque las
versiones recientes de bcrypt (4.x/5.x) ya no exponen el atributo interno
que passlib usa para detectar la versión, lo que rompe passlib con un
error confuso ("password cannot be longer than 72 bytes...") incluso con
contraseñas cortas. Usar bcrypt directo evita ese problema de compatibilidad.
"""

from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models.user import User

# bcrypt trabaja internamente con bytes y trunca en 72 bytes; validamos
# nosotros mismos para dar un mensaje claro en vez de un error críptico.
MAX_PASSWORD_BYTES = 72

# Esta URL es la que Swagger usa para el botón "Authorize" en /docs.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_PREFIX}/auth/login")


def get_password_hash(password: str) -> str:
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise ValueError(f"La contraseña no puede superar {MAX_PASSWORD_BYTES} bytes.")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(subject: str, expires_minutes: int | None = None) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes or settings.JWT_EXPIRE_MINUTES
    )
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    """Busca al usuario y valida la contraseña. Devuelve None si algo falla."""
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Dependencia para proteger endpoints: exige un JWT válido en el header
    Authorization: Bearer <token> y devuelve el usuario correspondiente.
    Esta es la pieza central del 'control de acceso' del commit 2."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudo validar la credencial.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user
