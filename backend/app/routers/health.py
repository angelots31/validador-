from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", summary="Estado de la API", description="Devuelve `ok` si la API está corriendo. Útil para monitoreo y para el chequeo de despliegue en Vercel.")
def health_check():
    return {"status": "ok"}
