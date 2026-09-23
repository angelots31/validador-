from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Forma estándar de las respuestas de error de la API (la que genera
    FastAPI automáticamente con HTTPException)."""

    detail: str = Field(
        description="Mensaje explicando qué salió mal.",
        examples=["Usuario o contraseña incorrectos."],
    )
