from pydantic import BaseModel, ConfigDict, Field


class UserCreate(BaseModel):
    """Datos para crear una cuenta nueva."""

    username: str = Field(
        min_length=3,
        max_length=50,
        description="Nombre de usuario único (3 a 50 caracteres).",
        examples=["angelo"],
    )
    password: str = Field(
        min_length=6,
        max_length=72,
        description="Contraseña en texto plano (se guarda hasheada con bcrypt, nunca en texto plano).",
        examples=["contrasena123"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"username": "angelo", "password": "contrasena123"}]
        }
    )


class UserOut(BaseModel):
    """Datos públicos de un usuario (nunca incluye la contraseña)."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="Identificador interno del usuario.", examples=[1])
    username: str = Field(description="Nombre de usuario.", examples=["angelo"])


class Token(BaseModel):
    """Respuesta de un login exitoso: el JWT a usar en el header Authorization."""

    access_token: str = Field(
        description="JWT firmado. Enviar como header 'Authorization: Bearer <access_token>'.",
        examples=["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."],
    )
    token_type: str = Field(
        default="bearer",
        description="Tipo de token, siempre 'bearer' en este proyecto.",
        examples=["bearer"],
    )
