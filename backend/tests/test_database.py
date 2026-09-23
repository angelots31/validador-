"""
Pruebas de la configuración de base de datos (commit 8 / despliegue).

Lo que se rompió y aquí queda cubierto: en Vercel el sistema de archivos es de
solo lectura, así que la aplicación no puede depender de un SQLite dentro del
proyecto. Estas pruebas verifican que la cadena de conexión que entregan los
proveedores administrados se traduce al driver instalado (psycopg 3) y que en
local, sin `DATABASE_URL`, se sigue usando SQLite.
"""

import tempfile
import unittest
from pathlib import Path

from app.core import database


class NormalizeDatabaseUrlTests(unittest.TestCase):
    def test_postgres_scheme_is_rewritten_to_psycopg(self):
        self.assertEqual(
            database.normalize_database_url("postgres://u:p@host/db"),
            "postgresql+psycopg://u:p@host/db",
        )

    def test_postgresql_scheme_is_rewritten_to_psycopg(self):
        self.assertEqual(
            database.normalize_database_url("postgresql://u:p@host/db?sslmode=require"),
            "postgresql+psycopg://u:p@host/db?sslmode=require",
        )

    def test_an_already_normalised_url_is_left_alone(self):
        url = "postgresql+psycopg://u:p@host/db"
        self.assertEqual(database.normalize_database_url(url), url)

    def test_sqlite_urls_are_left_alone(self):
        self.assertEqual(
            database.normalize_database_url("sqlite:///./app.db"), "sqlite:///./app.db"
        )

    def test_surrounding_whitespace_is_ignored(self):
        """El valor llega de una variable de entorno y suele traer saltos o espacios."""
        self.assertEqual(
            database.normalize_database_url("  postgres://u:p@host/db\n"),
            "postgresql+psycopg://u:p@host/db",
        )


class EphemeralFallbackTests(unittest.TestCase):
    def test_the_fallback_points_to_the_system_temp_directory(self):
        """En Vercel solo el directorio temporal es escribible, así que el
        último recurso tiene que apuntar ahí y no a la carpeta del proyecto
        (que es de solo lectura).

        Se resuelve con `tempfile` en vez de la cadena "/tmp" para que la misma
        lógica funcione en Windows, donde el directorio temporal es otro.
        """
        url = database.EPHEMERAL_SQLITE_URL
        self.assertTrue(url.startswith("sqlite:///"))
        self.assertIn(Path(tempfile.gettempdir()).as_posix(), url)
        self.assertTrue(url.endswith("/app.db"))

    def test_the_ephemeral_url_is_a_sqlite_url(self):
        """El fallback no puede acabar usando el driver de Postgres."""
        self.assertTrue(database.EPHEMERAL_SQLITE_URL.startswith("sqlite://"))


class LocalFallbackTests(unittest.TestCase):
    def test_uses_sqlite_when_there_is_no_database_url(self):
        """En el equipo de desarrollo, sin DATABASE_URL, se trabaja con SQLite."""
        self.assertTrue(database.IS_SQLITE)
        self.assertTrue(database.engine.dialect.name == "sqlite")

    def test_sqlite_migrations_are_a_noop_here(self):
        """La migración mínima de columnas solo aplica a SQLite y es idempotente."""
        self.assertEqual(database.apply_sqlite_migrations(), [])
        self.assertEqual(database.apply_sqlite_migrations(), [])


if __name__ == "__main__":
    unittest.main()
