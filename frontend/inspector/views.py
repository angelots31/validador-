import logging

from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_POST

from . import api_client
from .decorators import login_required_jwt

logger = logging.getLogger(__name__)


@login_required_jwt
def home(request):
    """Vista de inicio, ya protegida por el JWT en sesión."""
    return render(request, "inspector/home.html")


@login_required_jwt
@require_POST
def analyze_view(request):
    """Recibe la foto capturada desde camera.js (fetch/FormData) y la
    reenvía a FastAPI con el JWT de la sesión. Devuelve JSON, no HTML:
    esta vista la consume JavaScript, no el navegador directamente."""
    photo = request.FILES.get("photo")
    if not photo:
        return JsonResponse({"detail": "No se recibió ninguna foto."}, status=400)

    token = request.session.get("access_token")
    try:
        result = api_client.inspect_fruit(
            token,
            photo.name or "captura.jpg",
            photo.read(),
            photo.content_type or "image/jpeg",
        )
    except api_client.ApiError as exc:
        return JsonResponse({"detail": str(exc)}, status=exc.status_code or 502)
    except Exception:
        # Manejo de errores (commit 7): un fallo inesperado (ej. FastAPI
        # devolvió algo que no pudimos interpretar) no debe tumbar la vista
        # con un traceback — lo dejamos en el log del servidor y avisamos
        # al usuario con un mensaje genérico.
        logger.exception("Error inesperado analizando la imagen")
        return JsonResponse(
            {"detail": "Ocurrió un error inesperado analizando la imagen."}, status=500
        )

    return JsonResponse(result)


@login_required_jwt
@require_GET
def history_view(request):
    """Historial del usuario logueado, en JSON, para que camera.js lo pinte
    sin recargar la página."""
    token = request.session.get("access_token")
    try:
        history = api_client.get_inspection_history(token)
    except api_client.ApiError as exc:
        return JsonResponse({"detail": str(exc)}, status=exc.status_code or 502)
    except Exception:
        logger.exception("Error inesperado obteniendo el historial")
        return JsonResponse(
            {"detail": "Ocurrió un error inesperado obteniendo el historial."}, status=500
        )

    return JsonResponse({"history": history})


def register_view(request):
    if request.session.get("access_token"):
        return redirect("inspector:home")

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        password_confirm = request.POST.get("password_confirm", "")

        if not username or not password:
            messages.error(request, "Usuario y contraseña son obligatorios.")
        elif password != password_confirm:
            messages.error(request, "Las contraseñas no coinciden.")
        elif len(password) < 6:
            messages.error(request, "La contraseña debe tener al menos 6 caracteres.")
        else:
            try:
                api_client.register(username, password)
            except api_client.ApiError as exc:
                messages.error(request, str(exc))
            else:
                messages.success(request, "Cuenta creada correctamente. Ahora inicia sesión.")
                return redirect("inspector:login")

    return render(request, "inspector/register.html")


def login_view(request):
    if request.session.get("access_token"):
        return redirect("inspector:home")

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")

        try:
            token_data = api_client.login(username, password)
        except api_client.ApiError as exc:
            messages.error(request, str(exc))
        else:
            request.session["access_token"] = token_data["access_token"]
            request.session["username"] = username
            return redirect("inspector:home")

    return render(request, "inspector/login.html")


def logout_view(request):
    request.session.flush()
    messages.success(request, "Sesión cerrada correctamente.")
    return redirect("inspector:login")
