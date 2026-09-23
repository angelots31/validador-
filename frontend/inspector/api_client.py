"""
Cliente HTTP hacia el backend FastAPI.

Django NO tiene su propia tabla de usuarios para este proyecto: la fuente
de verdad de las credenciales es FastAPI (tabla `users`, en el Postgres administrado). Django
solo guarda el JWT recibido en la sesión y lo reenvía al backend en cada
petición que lo necesite: login/registro (commit 2) y, desde el commit 6,
también para analizar fotos y leer el historial.

La sesión vive en una **cookie firmada** (ver `SESSION_ENGINE` en
`config/settings.py`), no en una base de datos, porque en Vercel el sistema de
archivos es de solo lectura y no hay dónde escribirla. La cookie es HttpOnly: el
JavaScript de la página no puede leer el token, ni un XSS puede llevárselo. Lo que
sí cambia respecto a un motor de sesiones en base de datos es que la sesión ya no
vive del lado del servidor: su dueño puede ver el token en las herramientas del
navegador.
"""

import requests
from django.conf import settings

TIMEOUT = 5  # segundos
INSPECT_TIMEOUT = 20  # la inferencia del modelo tarda más que un login


class ApiError(Exception):
    """Error de comunicación con el backend o credenciales/datos rechazados."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _extract_detail(response, fallback: str) -> str:
    try:
        data = response.json()
    except ValueError:
        return fallback
    return data.get("detail", fallback)


def register(username: str, password: str) -> dict:
    url = f"{settings.FASTAPI_BASE_URL}/api/v1/auth/register"
    try:
        resp = requests.post(
            url, json={"username": username, "password": password}, timeout=TIMEOUT
        )
    except requests.RequestException as exc:
        raise ApiError(f"No se pudo conectar con el backend: {exc}") from exc

    if resp.status_code != 201:
        raise ApiError(_extract_detail(resp, "No se pudo crear la cuenta."), resp.status_code)
    return resp.json()


def login(username: str, password: str) -> dict:
    url = f"{settings.FASTAPI_BASE_URL}/api/v1/auth/login"
    try:
        # OAuth2PasswordRequestForm en FastAPI espera form-data, no JSON.
        resp = requests.post(
            url, data={"username": username, "password": password}, timeout=TIMEOUT
        )
    except requests.RequestException as exc:
        raise ApiError(f"No se pudo conectar con el backend: {exc}") from exc

    if resp.status_code != 200:
        raise ApiError(
            _extract_detail(resp, "Usuario o contraseña incorrectos."), resp.status_code
        )
    return resp.json()  # {"access_token": ..., "token_type": "bearer"}


def inspect_fruit(token: str, filename: str, content: bytes, content_type: str) -> dict:
    """Envía la foto capturada a FastAPI y devuelve {item, quality, ripeness, confidence}."""
    url = f"{settings.FASTAPI_BASE_URL}/api/v1/inspect-fruit"
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": (filename, content, content_type)}
    try:
        resp = requests.post(url, headers=headers, files=files, timeout=INSPECT_TIMEOUT)
    except requests.RequestException as exc:
        raise ApiError(f"No se pudo conectar con el backend: {exc}") from exc

    if resp.status_code != 200:
        raise ApiError(_extract_detail(resp, "No se pudo analizar la imagen."), resp.status_code)
    return resp.json()


def get_inspection_history(token: str) -> list[dict]:
    """Trae el historial de inspecciones del usuario dueño del token."""
    url = f"{settings.FASTAPI_BASE_URL}/api/v1/inspection-history"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(url, headers=headers, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise ApiError(f"No se pudo conectar con el backend: {exc}") from exc

    if resp.status_code != 200:
        raise ApiError(_extract_detail(resp, "No se pudo obtener el historial."), resp.status_code)
    return resp.json()
