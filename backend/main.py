import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.database import Base, apply_sqlite_migrations, engine
from app.models.inspection import Inspection  # noqa: F401 - registra el modelo en Base.metadata
from app.models.user import User  # noqa: F401 - registra el modelo en Base.metadata
from app.routers import auth, health, inspect

logger = logging.getLogger("validador_frutas")

# Crea las tablas si no existen (suficiente para este proyecto; un esquema que
# evolucione en serio pediría Alembic) y aplica las columnas agregadas después
# de la primera versión del esquema en SQLite.
#
# Si la base de datos no responde, NO dejamos que reviente la importación del
# módulo: una Serverless Function que falla al importar devuelve 500 en TODAS
# las rutas, incluidas /health y /docs, sin ninguna pista de qué pasó. Con el
# try/except el arranque en frío sobrevive, /health sigue respondiendo y el
# error real queda en el log de la función.
try:
    Base.metadata.create_all(bind=engine)
    apply_sqlite_migrations()
except Exception:  # noqa: BLE001 - queremos arrancar igual para poder diagnosticar
    logger.exception("No se pudo preparar el esquema de la base de datos")

# Metadata de tags: agrupa y describe cada sección en Swagger (/docs).
# El orden acá también define el orden en que aparecen las secciones.
tags_metadata = [
    {
        "name": "health",
        "description": "Endpoint de salud para confirmar que la API está viva.",
    },
    {
        "name": "auth",
        "description": (
            "Registro, login (JWT) y control de acceso. Usa **Authorize** "
            "(arriba a la derecha) con un usuario creado en `/auth/register` "
            "para probar los endpoints protegidos del resto del Swagger."
        ),
    },
    {
        "name": "inspection",
        "description": (
            "Inspección de frutas y verduras: sube una foto y obtén "
            "item/quality/ripeness, y consulta el historial de inspecciones. "
            "Requiere autenticación.\n\n"
            "El campo `method` de la respuesta dice cómo se identificó el "
            "producto: `resnet18-imagenet` (modelo preentrenado), "
            "`color-shape-heuristics` (clasificador de respaldo por color y "
            "forma, para productos que ImageNet no conoce) o `unrecognized`."
        ),
    },
]

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "API del **Validador de Calidad de Frutas y Verduras** (Taller 4 — SENA "
        "ADSO). Recibe la foto de un producto y devuelve `item`, `quality` y "
        "`ripeness`.\n\n"
        "La identificación usa una **cascada**: primero el modelo ResNet18 "
        "preentrenado en ImageNet (ejecutado con ONNX Runtime), y si el "
        "producto no está entre las clases de ImageNet (caso de la papaya, el "
        "mango, la guayaba, la sandía...), un clasificador de respaldo por "
        "color y forma. El campo `method` indica cuál de los dos actuó.\n\n"
        "Ver el `README.md` del repositorio para el detalle de cómo funciona."
    ),
    version="0.7.0",
    docs_url="/docs",
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optimización de respuesta: comprime las respuestas JSON grandes (ej. el
# historial con muchos registros). No comprime el POST de subida de la
# imagen (eso lo decide el cliente), solo lo que la API devuelve.
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(health.router, prefix=settings.API_V1_PREFIX)
app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
app.include_router(inspect.router, prefix=settings.API_V1_PREFIX)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Los errores 422 de Pydantic son correctos pero verbosos. Armamos un
    'detail' de una sola línea (mismo formato que el resto de la API),
    y dejamos la lista completa en 'errors' para quien la quiera ver en
    detalle (ej. en Swagger)."""
    errors = exc.errors()
    first = errors[0]
    field = ".".join(str(p) for p in first["loc"] if p != "body")
    friendly = f"{field}: {first['msg']}" if field else first["msg"]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": friendly, "errors": errors},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Manejo de errores (commit 7): si algo no esperado revienta (ej. una
    imagen rara que rompe el preprocesamiento), no queremos que el cliente
    reciba un traceback ni un HTML de error genérico — devolvemos el mismo
    formato {"detail": ...} que el resto de la API, y dejamos el detalle
    técnico solo en el log del servidor."""
    logger.exception("Error no manejado en %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Ocurrió un error inesperado procesando la solicitud."},
    )


@app.get("/", tags=["health"], summary="Info básica de la API")
def root():
    """Endpoint raíz informativo: confirma que la API corre y dónde está el Swagger."""
    return {"message": "Validador de Calidad de Frutas API — ver /docs para el Swagger"}
