from django.urls import path

from . import views

app_name = "inspector"

urlpatterns = [
    path("", views.home, name="home"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("registro/", views.register_view, name="register"),
    path("analizar/", views.analyze_view, name="analyze"),
    path("historial/", views.history_view, name="history"),
]
