"""
Pruebas del pipeline de visión (commit 7).

Se corren con unittest, sin dependencias extra:

    cd backend
    python -m unittest discover -s tests -t . -v

Las imágenes se generan sintéticamente con Pillow (formas planas de color),
así que **no** sirven para medir precisión real del modelo: una manzana
perfectamente plana y sin textura es indistinguible de un tomate para
cualquier clasificador. Lo que sí prueban es lo que se rompió y arreglamos:

- que el pipeline nunca devuelva una etiqueta sin sentido (el bug original
  devolvía `"Rubber Eraser"` al fotografiar una papaya),
- que el clasificador de respaldo nombre productos que ImageNet desconoce,
- que la forma se mida bien en objetos alargados y delgados,
- que la calidad no castigue a los productos oscuros por ser oscuros,
- que el verde no marque como inmaduro a un pepino o una lechuga.
"""

import io
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from app.ml import fallback
from app.ml.features import extract_features
from app.ml.inference import (
    METHOD_HEURISTIC,
    METHOD_UNKNOWN,
    MODEL_PATH,
    analyze,
)
from app.ml.taxonomy import (
    ALL_PRODUCE_NAMES,
    CLIP_PENDING_NAMES,
    EXTRA_PRODUCE,
    IMAGENET_PRODUCE,
    PRODUCE_BY_NAME,
    category_of,
    is_known,
)
from app.ml.vision import assess_quality_and_ripeness

BACKGROUND = (214, 208, 196)
MODEL_AVAILABLE = MODEL_PATH.exists()


# ---------------------------------------------------------------------------
# Utilidades para dibujar "productos" sintéticos
# ---------------------------------------------------------------------------
def _canvas(size=(640, 480)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Fondo beige con unas líneas suaves, para que no sea un color plano."""
    image = Image.new("RGB", size, BACKGROUND)
    draw = ImageDraw.Draw(image)
    for x in range(0, size[0], 16):
        draw.line([(x, 0), (x, size[1])], fill=(206, 200, 188))
    return image, draw


def papaya_image() -> Image.Image:
    """Óvalo naranja con vetas verdes: la fruta que antes fallaba."""
    image, draw = _canvas()
    draw.ellipse([190, 120, 450, 360], fill=(232, 150, 58))
    draw.ellipse([175, 170, 465, 310], fill=(240, 168, 72))
    for x, y, r in [
        (230, 160, 22),
        (300, 150, 18),
        (370, 175, 26),
        (410, 230, 20),
        (250, 300, 24),
        (340, 315, 20),
    ]:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(150, 170, 70))
    return image.filter(ImageFilter.GaussianBlur(1.2))


def carrot_image() -> Image.Image:
    """Triángulo alargado y delgado (prueba la medición de forma)."""
    image, draw = _canvas()
    draw.polygon([(200, 360), (300, 90), (360, 95), (300, 365)], fill=(226, 120, 32))
    return image.filter(ImageFilter.GaussianBlur(1.0))


def cucumber_image() -> Image.Image:
    image, draw = _canvas()
    draw.ellipse([150, 190, 490, 300], fill=(58, 110, 52))
    return image.filter(ImageFilter.GaussianBlur(1.0))


def dark_eggplant_image(blemishes: bool = False) -> Image.Image:
    """Berenjena morada oscura, opcionalmente con manchas."""
    image, draw = _canvas()
    draw.ellipse([210, 130, 430, 350], fill=(58, 36, 74))
    if blemishes:
        draw.ellipse([250, 170, 300, 220], fill=(24, 14, 30))
        draw.ellipse([340, 250, 395, 305], fill=(20, 12, 26))
        draw.ellipse([260, 280, 320, 330], fill=(28, 16, 34))
    return image.filter(ImageFilter.GaussianBlur(1.0))


def empty_scene_image() -> Image.Image:
    """Escena sin producto: un fondo liso."""
    return Image.new("RGB", (640, 480), (200, 202, 205))


def jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Taxonomía: consistencia de los datos
# ---------------------------------------------------------------------------
class TaxonomyTests(unittest.TestCase):
    def test_every_profile_maps_to_a_known_produce(self):
        """Ningún perfil del respaldo puede quedar fuera de la taxonomía."""
        unknown = [name for name in fallback.PROFILES if name not in PRODUCE_BY_NAME]
        self.assertEqual(unknown, [], f"Perfiles sin taxonomía: {unknown}")

    def test_papaya_is_covered_by_the_fallback_and_not_by_imagenet(self):
        """El caso concreto que reportó el usuario."""
        self.assertIn("Papaya", fallback.PROFILES)
        self.assertIn("Papaya", EXTRA_PRODUCE)
        self.assertNotIn("Papaya", {p.name for p in IMAGENET_PRODUCE.values()})

    def test_fallback_covers_the_products_imagenet_misses(self):
        for name in ["Papaya", "Mango", "Guava", "Watermelon", "Avocado", "Tomato", "Carrot"]:
            with self.subTest(produce=name):
                self.assertIn(name, fallback.PROFILES)

    def test_produce_names_are_unique_and_non_empty(self):
        self.assertEqual(len(ALL_PRODUCE_NAMES), len(set(ALL_PRODUCE_NAMES)))
        for name in ALL_PRODUCE_NAMES:
            self.assertTrue(name.strip())

    def test_every_extra_produce_has_a_fallback_profile(self):
        """Cada producto de plaza tiene que ser alcanzable por alguna vía.

        EXTRA_PRODUCE son justamente los que ImageNet no conoce, así que su
        única ruta es el clasificador de respaldo. Un nombre sin perfil es un
        producto fantasma: la taxonomía promete algo que el sistema nunca
        puede devolver.
        """
        self.assertEqual(
            set(EXTRA_PRODUCE),
            set(fallback.PROFILES),
            "EXTRA_PRODUCE y fallback.PROFILES deben coincidir 1 a 1",
        )

    def test_clip_pending_products_are_not_advertised_as_known(self):
        """Los 36 productos de la ampliación de CLIP no son alcanzables hoy.

        Están fuera de la taxonomía a propósito: los modelos ONNX de CLIP se
        cuantizaron con `quantize_dynamic` y devuelven embeddings inservibles,
        así que ninguna etapa puede nombrarlos. Solo deben entrar a
        EXTRA_PRODUCE cuando `scripts/export_clip_onnx.py` valide unos modelos
        re-exportados.
        """
        self.assertTrue(CLIP_PENDING_NAMES, "la lista de pendientes no puede estar vacía")
        self.assertEqual(set(CLIP_PENDING_NAMES) & set(ALL_PRODUCE_NAMES), set())
        for name in CLIP_PENDING_NAMES:
            with self.subTest(produce=name):
                self.assertFalse(is_known(name))
                # Y si algún día llegara por un camino inesperado, la categoría
                # se reporta como Unknown en vez de inventar una.
                self.assertEqual(category_of(name), "Unknown")


# ---------------------------------------------------------------------------
# Extracción de características
# ---------------------------------------------------------------------------
class FeatureTests(unittest.TestCase):
    def test_detects_object_larger_than_background(self):
        feats = extract_features(papaya_image())
        self.assertGreater(feats.foreground_ratio, 0.05)
        self.assertGreater(feats.segmentation_confidence, 0.5)

    def test_measures_elongation_on_thin_diagonal_object(self):
        """Regresión: el recorte por densidad de filas/columnas fallaba acá."""
        feats = extract_features(carrot_image())
        self.assertGreater(feats.elongation, 1.8)
        self.assertLess(feats.elongation, 3.5)

    def test_round_object_is_not_reported_as_elongated(self):
        feats = extract_features(dark_eggplant_image())
        self.assertLess(feats.elongation, 1.35)

    def test_papaya_reads_as_orange_yellow(self):
        feats = extract_features(papaya_image())
        self.assertGreater(feats.orange_ratio, 0.3)
        self.assertLess(feats.mean_hue, 70)

    def test_flat_scene_has_low_segmentation_confidence(self):
        feats = extract_features(empty_scene_image())
        self.assertLess(feats.segmentation_confidence, 0.5)

    def test_blemish_ratio_is_relative_to_the_product(self):
        """Un producto oscuro sin manchas no debe parecer deteriorado."""
        clean = extract_features(dark_eggplant_image(blemishes=False))
        damaged = extract_features(dark_eggplant_image(blemishes=True))
        self.assertLess(clean.blemish_ratio, damaged.blemish_ratio)


# ---------------------------------------------------------------------------
# Calidad y madurez
# ---------------------------------------------------------------------------
class QualityRipenessTests(unittest.TestCase):
    def test_green_is_normal_items_are_not_marked_unripe(self):
        """Un pepino verde está en su punto; un tomate verde no."""
        feats = extract_features(cucumber_image())
        cucumber = assess_quality_and_ripeness(feats, "Cucumber")
        tomato = assess_quality_and_ripeness(feats, "Tomato")

        self.assertEqual(cucumber.ripeness, "Ripe")
        self.assertEqual(tomato.ripeness, "Unripe")

    def test_dark_produce_without_blemishes_is_good_quality(self):
        feats = extract_features(dark_eggplant_image(blemishes=False))
        result = assess_quality_and_ripeness(feats, "Eggplant")
        self.assertEqual(result.quality, "Good")

    def test_visible_blemishes_lower_the_quality(self):
        feats = extract_features(dark_eggplant_image(blemishes=True))
        result = assess_quality_and_ripeness(feats, "Eggplant")
        self.assertIn(result.quality, {"Regular", "Bad"})

    def test_root_vegetables_are_never_called_unripe(self):
        """En una zanahoria o una papa el color no indica madurez."""
        feats = extract_features(carrot_image())
        result = assess_quality_and_ripeness(feats, "Carrot")
        self.assertEqual(result.ripeness, "Ripe")

    def test_every_assessment_comes_with_an_explanation(self):
        feats = extract_features(papaya_image())
        result = assess_quality_and_ripeness(feats, "Papaya")
        self.assertTrue(result.notes)
        self.assertIn(result.quality, {"Good", "Regular", "Bad"})
        self.assertIn(result.ripeness, {"Unripe", "Ripe", "Overripe"})


# ---------------------------------------------------------------------------
# Clasificador de respaldo
# ---------------------------------------------------------------------------
class FallbackClassifierTests(unittest.TestCase):
    def test_ranking_is_sorted_and_normalised(self):
        result = fallback.classify_by_features(extract_features(carrot_image()))
        scores = [score for _, score in result.ranking]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))

    def test_scores_are_deterministic(self):
        feats = extract_features(papaya_image())
        first = fallback.classify_by_features(feats)
        second = fallback.classify_by_features(feats)
        self.assertEqual(first.name, second.name)
        self.assertEqual(first.score, second.score)

    def test_explains_the_decision_with_signals(self):
        result = fallback.classify_by_features(extract_features(papaya_image()))
        self.assertTrue(result.signals)
        feature_names = {name for name, _ in result.signals}
        self.assertTrue(feature_names.issubset({"hue", "sat", "val", "elongation", "fg", "hue_alt"}))


# ---------------------------------------------------------------------------
# Pipeline completo (requiere el modelo ONNX de 45 MB)
# ---------------------------------------------------------------------------
@unittest.skipUnless(MODEL_AVAILABLE, "falta app/ml/resnet18.onnx")
class PipelineTests(unittest.TestCase):
    """Regresión del bug reportado: la API nunca debe devolver etiquetas que
    no sean un producto real de la taxonomía."""

    def _analyze(self, image):
        return analyze(image)

    def test_never_returns_a_nonsense_imagenet_label(self):
        """Antes, una papaya devolvía 'Rubber Eraser'."""
        for name, image in [
            ("papaya", papaya_image()),
            ("carrot", carrot_image()),
            ("cucumber", cucumber_image()),
            ("empty", empty_scene_image()),
        ]:
            with self.subTest(scene=name):
                match = self._analyze(image)
                self.assertTrue(
                    match.item == "Unknown" or match.item in PRODUCE_BY_NAME,
                    f"Etiqueta fuera de la taxonomía: {match.item!r}",
                )

    def test_identifies_papaya_which_imagenet_cannot(self):
        match = self._analyze(papaya_image())
        self.assertEqual(match.item, "Papaya")
        self.assertEqual(match.category, "Fruit")
        self.assertEqual(match.method, METHOD_HEURISTIC)

    def test_identifies_a_thin_root_vegetable(self):
        match = self._analyze(carrot_image())
        self.assertEqual(match.item, "Carrot")
        self.assertEqual(match.category, "Vegetable")

    def test_empty_scene_is_not_forced_into_a_product(self):
        match = self._analyze(empty_scene_image())
        self.assertIn(match.item, {"Unknown"})
        self.assertEqual(match.method, METHOD_UNKNOWN)

    def test_result_carries_candidates_and_notes(self):
        match = self._analyze(papaya_image())
        self.assertTrue(match.candidates)
        self.assertTrue(match.notes)
        self.assertLessEqual(len(match.candidates), 5)

    def test_confidence_is_a_probability(self):
        for image in [papaya_image(), carrot_image(), empty_scene_image()]:
            match = self._analyze(image)
            self.assertGreaterEqual(match.confidence, 0.0)
            self.assertLessEqual(match.confidence, 1.0)

    def test_pipeline_accepts_jpeg_round_trip(self):
        """El flujo real del frontend manda JPEG, no un objeto Pillow."""
        payload = jpeg_bytes(papaya_image())
        match = analyze(Image.open(io.BytesIO(payload)))
        self.assertEqual(match.item, "Papaya")


if __name__ == "__main__":
    unittest.main()
