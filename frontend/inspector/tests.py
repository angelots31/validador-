"""
Pruebas del frontend Django (commit 7).

Se corren con:

    cd frontend
    python manage.py test

Cubren lo que puede romperse sin un backend levantado:

- el control de acceso propio (`login_required_jwt`), que no usa
  `django.contrib.auth` sino el token JWT guardado en la sesión;
- que `analyze_view` reenvíe la foto y respete el código de estado que
  devuelva FastAPI (incluidos los errores);
- que el cliente HTTP arme bien las peticiones hacia FastAPI;
- que las plantillas se rendericen con la estructura que el JavaScript
  espera (identificadores que `camera.js` busca por `id`).
"""

import os
from unittest.mock import patch

import requests
from django.test import TestCase

from . import api_client


def fake_response(status_code=200, payload=None, json_error=False):
    """Doble de `requests.Response` para no depender del backend real."""

    class FakeResponse:
        def __init__(self):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            if json_error:
                raise ValueError("no es JSON")
            return self._payload

    return FakeResponse()


class SessionLoginMixin:
    """Deja al cliente de pruebas con una sesión ya iniciada.

    Con `SESSION_ENGINE` en **cookie firmada** ya no sirve hacer
    `client.session["access_token"] = ...; session.save()`: el `Client` de
    Django obtiene la sesión de la cookie que viene en una *respuesta*, así que
    preparar el estado por esa vía deja una cookie vacía y la vista redirige al
    login (que es justo lo que rompió estas pruebas al cambiar el motor).

    Por eso aquí se inicia sesión de verdad contra la vista, con
    `api_client.login` mockeado. De paso, la prueba recorre el mismo camino que
    el usuario en lugar de fabricar el estado por detrás.
    """

    def login_as(self, username="angelo", token="token-de-prueba"):
        with patch("inspector.api_client.login") as login:
            login.return_value = {"access_token": token, "token_type": "bearer"}
            response = self.client.post(
                "/login/", {"username": username, "password": "secreto123"}
            )
        self.assertEqual(response.status_code, 302, "el login de prueba no redirigió")
        return response


class LoginRequiredJwtTests(SessionLoginMixin, TestCase):
    def test_home_redirects_to_login_without_token(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response["Location"])

    def test_home_renders_with_token_in_session(self):
        self.login_as()

        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cámara")
        self.assertContains(response, "Historial de inspecciones")

    def test_analyze_requires_login(self):
        response = self.client.post("/analizar/")
        self.assertEqual(response.status_code, 302)

    def test_history_requires_login(self):
        response = self.client.get("/historial/")
        self.assertEqual(response.status_code, 302)


class AnalyzeViewTests(SessionLoginMixin, TestCase):
    def setUp(self):
        self.login_as()

    def test_returns_400_when_no_photo_is_sent(self):
        response = self.client.post("/analizar/")
        self.assertEqual(response.status_code, 400)
        self.assertIn("foto", response.json()["detail"].lower())

    @patch("inspector.api_client.inspect_fruit")
    def test_forwards_the_photo_and_returns_the_result(self, inspect_fruit):
        inspect_fruit.return_value = {
            "item": "Papaya",
            "category": "Fruit",
            "quality": "Good",
            "ripeness": "Ripe",
            "confidence": 0.81,
            "method": "color-shape-heuristics",
            "candidates": [],
            "notes": [],
        }

        from django.core.files.uploadedfile import SimpleUploadedFile

        photo = SimpleUploadedFile("captura.jpg", b"contenido-falso", content_type="image/jpeg")
        response = self.client.post("/analizar/", {"photo": photo})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["item"], "Papaya")

        # El token de la sesión debe viajar al cliente HTTP, no el del navegador.
        args = inspect_fruit.call_args[0]
        self.assertEqual(args[0], "token-de-prueba")
        self.assertEqual(args[3], "image/jpeg")

    @patch("inspector.api_client.inspect_fruit")
    def test_propagates_backend_error_status(self, inspect_fruit):
        inspect_fruit.side_effect = api_client.ApiError("La imagen es demasiado grande (máximo 8 MB).", 413)

        from django.core.files.uploadedfile import SimpleUploadedFile

        photo = SimpleUploadedFile("captura.jpg", b"x", content_type="image/jpeg")
        response = self.client.post("/analizar/", {"photo": photo})

        self.assertEqual(response.status_code, 413)
        self.assertIn("8 MB", response.json()["detail"])

    @patch("inspector.api_client.inspect_fruit")
    def test_unexpected_failure_returns_500_without_traceback(self, inspect_fruit):
        inspect_fruit.side_effect = RuntimeError("algo se rompió")

        from django.core.files.uploadedfile import SimpleUploadedFile

        photo = SimpleUploadedFile("captura.jpg", b"x", content_type="image/jpeg")
        with self.assertLogs("inspector.views", level="ERROR"):
            response = self.client.post("/analizar/", {"photo": photo})

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("Traceback", response.json()["detail"])

    @patch("inspector.api_client.get_inspection_history")
    def test_history_view_returns_items(self, get_history):
        get_history.return_value = [{"id": 1, "item": "Mango"}]
        response = self.client.get("/historial/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["history"][0]["item"], "Mango")


class RegisterAndLoginViewTests(TestCase):
    def test_password_mismatch_is_reported(self):
        response = self.client.post(
            "/registro/",
            {"username": "angelo", "password": "secreto123", "password_confirm": "otro123"},
            follow=True,
        )
        self.assertContains(response, "no coinciden")

    def test_short_password_is_reported(self):
        response = self.client.post(
            "/registro/",
            {"username": "angelo", "password": "123", "password_confirm": "123"},
            follow=True,
        )
        self.assertContains(response, "al menos 6 caracteres")

    @patch("inspector.api_client.register")
    def test_successful_registration_redirects_to_login(self, register):
        response = self.client.post(
            "/registro/",
            {"username": "angelo", "password": "secreto123", "password_confirm": "secreto123"},
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response["Location"])
        register.assert_called_once_with("angelo", "secreto123")

    @patch("inspector.api_client.login")
    def test_login_stores_the_token_in_the_session(self, login):
        login.return_value = {"access_token": "jwt-de-prueba", "token_type": "bearer"}

        response = self.client.post("/login/", {"username": "angelo", "password": "secreto123"})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.client.session["access_token"], "jwt-de-prueba")
        self.assertEqual(self.client.session["username"], "angelo")

    @patch("inspector.api_client.login")
    def test_bad_credentials_show_an_error(self, login):
        login.side_effect = api_client.ApiError("Usuario o contraseña incorrectos.", 401)

        response = self.client.post("/login/", {"username": "angelo", "password": "mal"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "incorrectos")


class TemplateContractTests(SessionLoginMixin, TestCase):
    """El JavaScript busca elementos por `id`. Si alguien renombra uno de
    estos, la captura o el análisis dejan de funcionar en silencio."""

    REQUIRED_IDS = [
        "camera-video",
        "preview-image",
        "capture-canvas",
        "viewport",
        "viewport-guide",
        "camera-status",
        "camera-meta",
        "start-camera-btn",
        "capture-btn",
        "retake-btn",
        "analyze-btn",
        "switch-camera-btn",
        "file-input",
        "result-panel",
        "result-item",
        "result-quality",
        "result-ripeness",
        "result-method",
        "gauge-value",
        "result-confidence",
        "color-bar",
        "color-legend",
        "candidates",
        "result-notes",
        "history-list",
        "history-empty",
        "stat-total",
        "stat-confidence",
        "stat-mix",
        "stat-last",
        "toast-stack",
        "ambient-canvas",
    ]

    def setUp(self):
        self.login_as()

    def test_home_template_exposes_every_id_used_by_javascript(self):
        response = self.client.get("/")
        html = response.content.decode("utf-8")
        missing = [element_id for element_id in self.REQUIRED_IDS if f'id="{element_id}"' not in html]
        self.assertEqual(missing, [], f"Faltan elementos que camera.js/visual.js esperan: {missing}")

    def test_home_exposes_the_javascript_config(self):
        response = self.client.get("/")
        self.assertContains(response, "VALIDADOR_CONFIG")
        self.assertContains(response, "/analizar/")
        self.assertContains(response, "/historial/")

    def test_api_docs_link_is_built_from_settings(self):
        # Sin sesión, para caer en la pantalla de login y no en el redirect a la
        # página principal. Se cierra sesión con la vista (y no con
        # `session.flush()`) porque es la respuesta la que borra la cookie: con
        # sesiones en cookie firmada, tocar el `SessionStore` no basta.
        self.client.get("/logout/")

        response = self.client.get("/login/")

        self.assertContains(response, "/docs")


class ApiClientTests(TestCase):
    @patch("inspector.api_client.requests.post")
    def test_login_sends_form_data_not_json(self, post):
        post.return_value = fake_response(200, {"access_token": "abc"})
        api_client.login("angelo", "secreto123")

        self.assertIn("data", post.call_args.kwargs)
        self.assertNotIn("json", post.call_args.kwargs)
        self.assertEqual(post.call_args.kwargs["data"]["username"], "angelo")

    @patch("inspector.api_client.requests.post")
    def test_register_sends_json(self, post):
        post.return_value = fake_response(201, {"id": 1, "username": "angelo"})
        api_client.register("angelo", "secreto123")

        self.assertIn("json", post.call_args.kwargs)
        self.assertEqual(post.call_args.kwargs["json"]["username"], "angelo")

    @patch("inspector.api_client.requests.post")
    def test_inspect_sends_the_bearer_token(self, post):
        post.return_value = fake_response(200, {"item": "Papaya"})
        api_client.inspect_fruit("abc", "captura.jpg", b"datos", "image/jpeg")

        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer abc")
        self.assertIn("file", post.call_args.kwargs["files"])

    @patch("inspector.api_client.requests.get")
    def test_history_uses_the_bearer_token(self, get):
        get.return_value = fake_response(200, [])
        api_client.get_inspection_history("abc")
        self.assertEqual(get.call_args.kwargs["headers"]["Authorization"], "Bearer abc")

    @patch("inspector.api_client.requests.post")
    def test_backend_detail_is_surfaced_to_the_user(self, post):
        post.return_value = fake_response(401, {"detail": "Usuario o contraseña incorrectos."})
        with self.assertRaises(api_client.ApiError) as context:
            api_client.login("angelo", "mal")
        self.assertIn("incorrectos", str(context.exception))
        self.assertEqual(context.exception.status_code, 401)

    @patch("inspector.api_client.requests.post")
    def test_connection_failure_becomes_an_api_error(self, post):
        post.side_effect = requests.ConnectionError("sin red")
        with self.assertRaises(api_client.ApiError) as context:
            api_client.login("angelo", "secreto123")
        self.assertIn("No se pudo conectar", str(context.exception))

    @patch("inspector.api_client.requests.post")
    def test_non_json_error_response_uses_the_fallback_message(self, post):
        post.return_value = fake_response(500, None, json_error=True)
        with self.assertRaises(api_client.ApiError) as context:
            api_client.inspect_fruit("abc", "captura.jpg", b"datos", "image/jpeg")
        self.assertIn("analizar", str(context.exception).lower())


class DeploymentSettingsTests(TestCase):
    """Ajustes que solo importan al desplegar en Vercel (commit 8).

    Sin ellos el despliegue no queda "con un 404 raro": queda con 400
    (DisallowedHost) y con 403 al enviar el formulario de login.
    """

    def test_vercel_subdomains_are_accepted_as_hosts(self):
        """VERCEL_URL es la URL del despliegue, no el dominio de producción:
        el comodín es lo que hace que entrar por el dominio estable funcione."""
        from config import settings as settings_module

        self.assertIn(".vercel.app", settings_module.ALLOWED_HOSTS)

    def test_the_proxy_header_is_trusted(self):
        """Vercel termina TLS y reenvía por HTTP; sin esta cabecera Django
        cree que la petición no es segura y el CSRF la rechaza."""
        from django.conf import settings

        self.assertEqual(
            settings.SECURE_PROXY_SSL_HEADER, ("HTTP_X_FORWARDED_PROTO", "https")
        )

    def test_csrf_trusts_vercel_preview_deployments(self):
        from django.conf import settings

        self.assertIn("https://*.vercel.app", settings.CSRF_TRUSTED_ORIGINS)

    def test_postgres_config_parses_a_neon_style_url(self):
        from config import settings as settings_module

        config = settings_module.postgres_config(
            "postgresql://user:se%40cret@ep-cool-1.us-east-2.aws.neon.tech/neondb"
        )

        self.assertEqual(config["ENGINE"], "django.db.backends.postgresql")
        self.assertEqual(config["NAME"], "neondb")
        self.assertEqual(config["USER"], "user")
        # La contraseña viene con caracteres escapados en la URL.
        self.assertEqual(config["PASSWORD"], "se@cret")
        self.assertEqual(config["HOST"], "ep-cool-1.us-east-2.aws.neon.tech")
        # Postgres administrado siempre exige TLS.
        self.assertEqual(config["OPTIONS"]["sslmode"], "require")
        # En serverless no se reutilizan conexiones entre invocaciones.
        self.assertEqual(config["CONN_MAX_AGE"], 0)

    def test_postgres_config_respects_an_explicit_sslmode_and_port(self):
        from config import settings as settings_module

        config = settings_module.postgres_config(
            "postgresql://u:p@localhost:5432/dev?sslmode=disable"
        )

        self.assertEqual(config["PORT"], "5432")
        self.assertEqual(config["OPTIONS"]["sslmode"], "disable")

    def test_the_engine_matches_the_configured_database_url(self):
        from django.conf import settings

        engine = settings.DATABASES["default"]["ENGINE"]
        if os.environ.get("DATABASE_URL", "").strip():
            self.assertEqual(engine, "django.db.backends.postgresql")
        else:
            # Sin DATABASE_URL se trabaja en local con SQLite.
            self.assertEqual(engine, "django.db.backends.sqlite3")


class NoDatabaseDeploymentTests(SessionLoginMixin, TestCase):
    """El frontend tiene que funcionar SIN base de datos.

    Es la configuración con la que se despliega en Vercel: el sistema de
    archivos de las Serverless Functions es de solo lectura salvo `/tmp`, así
    que la sesión (el JWT) se guarda en una cookie firmada. Estas pruebas lo
    comprueban contando consultas: 0 significa que ese flujo no toca la base de
    datos. Si alguien cambia `SESSION_ENGINE` o mete una consulta al ORM en el
    camino de login/home, esto falla — y en producción el login dejaría de
    funcionar en silencio.
    """

    def test_sessions_are_stored_in_a_signed_cookie(self):
        from django.conf import settings

        self.assertEqual(
            settings.SESSION_ENGINE,
            "django.contrib.sessions.backends.signed_cookies",
        )

    def test_login_page_renders_without_touching_the_database(self):
        with self.assertNumQueries(0):
            response = self.client.get("/login/")
        self.assertEqual(response.status_code, 200)

    def test_registration_page_renders_without_touching_the_database(self):
        with self.assertNumQueries(0):
            response = self.client.get("/registro/")
        self.assertEqual(response.status_code, 200)

    @patch("inspector.api_client.login")
    def test_login_stores_the_token_without_writing_to_the_database(self, login):
        login.return_value = {"access_token": "jwt-de-prueba", "token_type": "bearer"}

        with self.assertNumQueries(0):
            response = self.client.post(
                "/login/", {"username": "angelo", "password": "secreto123"}
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.client.session["access_token"], "jwt-de-prueba")

    def test_home_renders_for_a_logged_in_user_without_queries(self):
        self.login_as()

        with self.assertNumQueries(0):
            response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cámara")

    def test_logout_clears_the_session_without_touching_the_database(self):
        self.login_as()

        with self.assertNumQueries(0):
            response = self.client.get("/logout/")

        self.assertEqual(response.status_code, 302)
        self.assertNotIn("access_token", self.client.session)

    def test_secure_cookies_depend_on_running_on_vercel(self):
        """En local se sirve por http: marcar las cookies como Secure ahí sería
        el equivalente a no enviarlas y el login no funcionaría."""
        import os

        from django.conf import settings

        on_vercel = bool(os.environ.get("VERCEL"))
        self.assertEqual(settings.SESSION_COOKIE_SECURE, on_vercel)
        self.assertEqual(settings.CSRF_COOKIE_SECURE, on_vercel)
