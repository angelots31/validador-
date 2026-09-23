import os
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure--)+79#$tllx0dw#d92nz(d&7fq+m4*pf$ars7d1va4f=qwa*mt",
)

DEBUG = os.environ.get("DJANGO_DEBUG", "True") == "True"

ALLOWED_HOSTS = os.environ.get(
    "DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1"
).split(",")

# Vercel expone varias URLs del despliegue en variables de entorno y las
# agregamos todas a ALLOWED_HOSTS para no tener que configurar el host a mano
# en cada despliegue: sin esto, con DEBUG=False la app responde 400.
#
# Ojo: `VERCEL_URL` es la URL *de este despliegue* (ej. mi-app-abc123.vercel.app),
# que NO es el dominio estable por el que entra el usuario (mi-app.vercel.app).
# Si solo se permitiera VERCEL_URL, entrar por el dominio de producción daría
# "DisallowedHost". Por eso se usa también VERCEL_PROJECT_PRODUCTION_URL (el
# dominio de producción del proyecto) y el comodín `.vercel.app`, que cubre
# cualquier subdominio de Vercel incluidos los despliegues de preview.
VERCEL_URL = os.environ.get("VERCEL_URL", "").strip()
VERCEL_PROJECT_PRODUCTION_URL = os.environ.get(
    "VERCEL_PROJECT_PRODUCTION_URL", ""
).strip()
VERCEL_BRANCH_URL = os.environ.get("VERCEL_BRANCH_URL", "").strip()

_VERCEL_HOSTS = [VERCEL_URL, VERCEL_PROJECT_PRODUCTION_URL, VERCEL_BRANCH_URL]
for _host in _VERCEL_HOSTS:
    if _host:
        ALLOWED_HOSTS.append(_host)
# El comodín con punto inicial hace que Django acepte ese dominio y todos sus
# subdominios (docs: ALLOWED_HOSTS permite ".example.com").
ALLOWED_HOSTS.append(".vercel.app")

# Django exige que el origen esté en CSRF_TRUSTED_ORIGINS para aceptar POST
# por HTTPS. Es lo que rompe el login en Vercel si se olvida: el formulario de
# inicio de sesión es un POST y devolvería 403 "CSRF verification failed".
CSRF_TRUSTED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",")
    if origin.strip()
]
for _host in _VERCEL_HOSTS:
    if _host:
        CSRF_TRUSTED_ORIGINS.append(f"https://{_host}")
CSRF_TRUSTED_ORIGINS.append("https://*.vercel.app")

# Vercel termina TLS en su proxy y reenvía la petición por HTTP con la cabecera
# X-Forwarded-Proto. Sin esto Django cree que la petición llegó por HTTP y la
# verificación de origen del CSRF compara "http://host" contra el
# "Origin: https://host" que envía el navegador: el POST del login fallaba con
# 403 aunque el dominio estuviera en CSRF_TRUSTED_ORIGINS.
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

FASTAPI_BASE_URL = os.environ.get("FASTAPI_BASE_URL", "http://127.0.0.1:8000")


def postgres_config(url: str) -> dict:
    """Traduce una cadena `postgresql://...` al diccionario de Django.

    Se parsea a mano en vez de sumar `dj-database-url` porque el proyecto
    mantiene sus dependencias al mínimo y el formato relevante es pequeño:
    esquema, credenciales, host, puerto, nombre de base y parámetros de query
    (donde viaja `sslmode`).
    """
    parsed = urlsplit(url)
    options = dict(parse_qsl(parsed.query))
    # Todo Postgres administrado (Neon incluido) exige TLS. Si la URL no lo dice
    # explícitamente, se pide: es más seguro fallar la conexión que viajar en
    # claro con las credenciales.
    options.setdefault("sslmode", "require")
    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": unquote(parsed.path.lstrip("/")) or "postgres",
        "USER": unquote(parsed.username or ""),
        "PASSWORD": unquote(parsed.password or ""),
        "HOST": parsed.hostname or "",
        "PORT": str(parsed.port or ""),
        "OPTIONS": options,
        # En serverless no se reutilizan conexiones entre invocaciones: el
        # proceso se recicla y una conexión "viva" en caché estaría muerta al
        # volver a usarla.
        "CONN_MAX_AGE": 0,
    }


DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

if DATABASE_URL:
    DATABASES = {"default": postgres_config(DATABASE_URL)}
elif os.environ.get("VERCEL"):
    # Último recurso si se olvidó configurar DATABASE_URL: /tmp es lo único
    # escribible en una Serverless Function, pero es efímero y las tablas de
    # Django no existen ahí, así que el login fallará. Ver README.
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": "/tmp/db.sqlite3",
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

# --- Sesiones ----------------------------------------------------------------
# El frontend NO necesita una base de datos para funcionar. Lo único que guarda
# en la sesión es el JWT que devuelve FastAPI, así que la sesión se guarda en una
# **cookie firmada** con SECRET_KEY y no se escribe nada en disco.
#
# Con el motor en base de datos, cada login escribía una fila en `django_session`
# y en Vercel eso es imposible: el sistema de archivos es de solo lectura, así que
# el login fallaba con "readonly database". Y apuntarlo a `/tmp` tampoco servía,
# porque las tablas no existirían ahí.
#
# Contrapartida honesta para la sustentación: la sesión deja de estar del lado del
# servidor. El token va en una cookie **HttpOnly** (el JavaScript de la página no
# puede leerla con `document.cookie`, y un XSS no se la lleva), pero el dueño de
# la sesión sí la ve en las herramientas del navegador.
SESSION_ENGINE = "django.contrib.sessions.backends.signed_cookies"

# En Vercel todo se sirve por HTTPS, así que las cookies no deben viajar por
# HTTP. En local (http://127.0.0.1) tienen que poder, o el login no funciona.
_ON_VERCEL = bool(os.environ.get("VERCEL"))
SESSION_COOKIE_SECURE = _ON_VERCEL
CSRF_COOKIE_SECURE = _ON_VERCEL

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "inspector",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                # Expone la URL del Swagger del backend a todas las plantillas.
                "inspector.context_processors.api_docs",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "es-co"
TIME_ZONE = "America/Bogota"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
