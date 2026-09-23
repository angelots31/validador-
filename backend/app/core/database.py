"""
Configuración de la base de datos (Postgres en producción, SQLite en local).

**Por qué Postgres y no SQLite en Vercel.** El sistema de archivos de las
Serverless Functions es **de solo lectura** salvo `/tmp`, así que un SQLite
dentro del proyecto ni siquiera se puede crear: al importar `main.py` la
aplicación reventaba en el arranque en frío. Moverlo a `/tmp` lo haría
escribible, pero ese directorio es efímero y por instancia: los usuarios y las
inspecciones se perderían entre invocaciones. Por eso, cuando existe la
variable de entorno `DATABASE_URL` (un Postgres administrado; en este proyecto
Neon) esa es la base de datos que se usa.

En local, si `DATABASE_URL` está vacía, se sigue usando SQLite para no obligar
a levantar un servidor de base de datos para trabajar en el taller.
"""

import logging
import os
import tempfile
from pathlib import Path

from sqlalchemy import create_engine, inspect as sa_inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

from app.core.config import settings

logger = logging.getLogger("validador_frutas")

# Base de datos local: un archivo junto al proyecto (está en .gitignore).
LOCAL_SQLITE_URL = "sqlite:///./app.db"

# Último recurso en Vercel si alguien olvidó configurar DATABASE_URL. En una
# Serverless Function lo único escribible es el directorio temporal (`/tmp`),
# aunque los datos no sobrevivan entre invocaciones. Se resuelve con `tempfile`
# en vez de escribir la cadena "/tmp" a mano porque el equivalente en Windows es
# otra carpeta: así la misma ruta funciona en los dos sistemas.
_EPHEMERAL_DB_PATH = Path(tempfile.gettempdir()) / "app.db"
EPHEMERAL_SQLITE_URL = f"sqlite:///{_EPHEMERAL_DB_PATH.as_posix()}"


def normalize_database_url(url: str) -> str:
    """Adapta la cadena de conexión al driver que tenemos instalado.

    Los proveedores administrados (Neon, Supabase, Render...) entregan la URL
    con el esquema genérico `postgres://` o `postgresql://`, que SQLAlchemy
    interpreta como psycopg2. Aquí usamos psycopg 3 (el driver mantenido y con
    ruedas para las versiones de Python que ofrece Vercel), así que hay que
    pedirlo explícitamente con `postgresql+psycopg://`.
    """
    url = url.strip()
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def _resolve_database_url() -> str:
    """Elige la base de datos: DATABASE_URL > SQLite efímero > SQLite local."""
    configured = (settings.DATABASE_URL or "").strip()
    if configured:
        return normalize_database_url(configured)

    if os.environ.get("VERCEL"):
        logger.warning(
            "Sin DATABASE_URL en Vercel: se usará SQLite en %s y los datos no "
            "persistirán entre invocaciones (se pierden cada vez que la función "
            "se enfría). Configura DATABASE_URL para que sobrevivan.",
            _EPHEMERAL_DB_PATH,
        )
        return EPHEMERAL_SQLITE_URL

    return LOCAL_SQLITE_URL


DATABASE_URL = _resolve_database_url()
IS_SQLITE = DATABASE_URL.startswith("sqlite")

_engine_kwargs: dict = {}
if IS_SQLITE:
    # SQLite no admite varias conexiones desde hilos distintos por defecto.
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # En serverless cada invocación puede ser un proceso nuevo: mantener un
    # pool de conexiones vivas entre peticiones solo produce conexiones
    # muertas ("server closed the connection unexpectedly"). NullPool abre y
    # cierra una conexión por petición, y el pooler de Neon se encarga de
    # multiplexar del lado del servidor.
    _engine_kwargs["poolclass"] = NullPool

engine = create_engine(DATABASE_URL, **_engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependencia de FastAPI: entrega una sesión de DB y la cierra al terminar."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Columnas agregadas después de la primera versión del esquema. `create_all`
# solo crea tablas que no existen: no toca una tabla ya creada, así que quien
# tenga un `app.db` de una versión anterior necesitaría borrar el archivo a
# mano. Este es el caso del taller (commit 7 agregó `category` y `method`).
#
# En Postgres no hace falta porque `create_all` crea el esquema desde cero, y
# en local sí, así que aplicamos el ALTER TABLE de forma idempotente y avisamos
# por el log. Es una migración deliberadamente mínima: para un esquema que
# evoluciona en serio haría falta Alembic.
_SQLITE_ADDED_COLUMNS = {
    "category": "VARCHAR(20) NOT NULL DEFAULT 'Unknown'",
    "method": "VARCHAR(30) NOT NULL DEFAULT 'unrecognized'",
}


def apply_sqlite_migrations() -> list[str]:
    """Agrega a `inspections` las columnas que falten. Devuelve las agregadas.

    Solo aplica a SQLite: en Postgres el esquema lo crea `create_all` completo
    y las columnas nuevas llegan con la tabla.
    """
    if not IS_SQLITE:
        return []

    inspector = sa_inspect(engine)
    if "inspections" not in inspector.get_table_names():
        return []

    existing = {column["name"] for column in inspector.get_columns("inspections")}
    missing = {
        name: ddl
        for name, ddl in _SQLITE_ADDED_COLUMNS.items()
        if name not in existing
    }
    if not missing:
        return []

    with engine.begin() as connection:
        for name, ddl in missing.items():
            connection.execute(text(f"ALTER TABLE inspections ADD COLUMN {name} {ddl}"))
            logger.info("Migración local: inspections.%s agregada", name)
    return sorted(missing)
