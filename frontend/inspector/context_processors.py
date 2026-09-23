"""
Context processors de la app `inspector`.

Un context processor inyecta variables en **todas** las plantillas sin tener
que pasarlas vista por vista. Aquí se usa para una sola cosa: que el pie de
página pueda enlazar al Swagger del backend en cualquier pantalla (incluidas
login y registro, que no reciben ese dato en su contexto).
"""

from django.conf import settings


def api_docs(request) -> dict[str, str]:
    """URL del Swagger de FastAPI, para el enlace del pie de página."""
    return {"api_docs_url": f"{settings.FASTAPI_BASE_URL.rstrip('/')}/docs"}
