from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import (
    authenticate_user,
    create_access_token,
    get_current_user,
    get_password_hash,
)
from app.models.user import User
from app.schemas.common import ErrorResponse
from app.schemas.user import Token, UserCreate, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Crear una cuenta",
    description="Crea un usuario nuevo. El nombre de usuario debe ser único en todo el sistema.",
    responses={
        400: {"model": ErrorResponse, "description": "El nombre de usuario ya está registrado."},
        422: {"description": "Datos inválidos (ej. contraseña menor a 6 caracteres)."},
    },
)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == payload.username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ese nombre de usuario ya está registrado.",
        )

    user = User(
        username=payload.username,
        hashed_password=get_password_hash(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post(
    "/login",
    response_model=Token,
    summary="Iniciar sesión (obtener JWT)",
    description=(
        "Login estilo OAuth2: username y password van como **form-data**, no JSON "
        "(así lo exige `OAuth2PasswordRequestForm`). Esto es justo lo que permite que "
        "el botón **Authorize** de este Swagger funcione automáticamente: pruébalo con "
        "un usuario creado en `/auth/register` y luego ya podrás llamar los endpoints "
        "protegidos desde aquí mismo."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Usuario o contraseña incorrectos."},
    },
)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contraseña incorrectos.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(subject=user.username)
    return Token(access_token=token)


@router.get(
    "/me",
    response_model=UserOut,
    summary="Ver mi usuario (endpoint protegido)",
    description=(
        "Devuelve el usuario dueño del token enviado. Es el ejemplo más simple de "
        "'control de acceso': si el JWT falta, es inválido o expiró, responde 401 "
        "en vez del usuario."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "No se pudo validar la credencial (token ausente, inválido o expirado)."},
    },
)
def read_current_user(current_user: User = Depends(get_current_user)):
    return current_user
