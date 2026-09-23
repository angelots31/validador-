from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuracion central del backend, cargada desde variables de entorno (.env)."""

    PROJECT_NAME: str = "Validador de Calidad de Frutas API"
    API_V1_PREFIX: str = "/api/v1"

    # Cadena de conexion de la base de datos. En produccion (Vercel) apunta a
    # un Postgres administrado (Neon); vacia significa "usa SQLite local".
    # Ver app/core/database.py para el porque.
    DATABASE_URL: str = ""

    # Origenes permitidos para CORS. Se configuran como una lista separada por
    # comas en la variable de entorno, ej:
    #     CORS_ORIGINS=https://mi-frontend.vercel.app,http://localhost:8080
    # El valor por defecto solo sirve para desarrollo local.
    CORS_ORIGINS: list[str] = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]

    # JWT
    JWT_SECRET_KEY: str = "cambia-esta-clave-en-produccion"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_cors_origins(cls, value):
        """Permite escribir los origenes como CSV en vez de JSON.

        pydantic-settings intenta interpretar una lista como JSON, lo que
        obliga a escribir `["https://a", "https://b"]` en el panel de Vercel.
        Aceptar tambien comas es mucho menos fragil al configurar el despliegue.
        """
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


settings = Settings()
