"""
Control de acceso del lado Django.

No usamos @login_required de django.contrib.auth porque la sesión de
autenticación real vive en FastAPI (el JWT). Este decorador simplemente
verifica que la sesión de Django tenga un access_token guardado; si no lo
tiene, redirige a la página de login.
"""

from functools import wraps

from django.contrib import messages
from django.shortcuts import redirect


def login_required_jwt(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.session.get("access_token"):
            messages.info(request, "Debes iniciar sesión para continuar.")
            return redirect("inspector:login")
        return view_func(request, *args, **kwargs)

    return wrapper
