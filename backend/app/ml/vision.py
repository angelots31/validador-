"""
Estimación de calidad y madurez del producto.

**Importante para la sustentación:** esto NO es una red neuronal. Es visión
por computador clásica sobre las características de `features.py`. ImageNet
(y por lo tanto ResNet18) no tiene clases de "calidad" ni de "madurez":
entrenar eso requeriría un dataset etiquetado de frutas en distintos estados
de maduración, que está fuera del alcance de las 15 horas del taller.

**Qué cambió en el commit 7** (y por qué los resultados antes eran malos):

1. *Umbrales relativos al propio producto.* Antes, un píxel con brillo < 0.35
   contaba como "mancha". Con eso, cualquier producto oscuro — una berenjena,
   unas uvas, un aguacate Hass — salía como `Bad`/`Overripe` siempre, sin
   importar su estado real. Ahora una mancha es un píxel claramente más
   oscuro **que la mediana de ese mismo producto** (`blemish_ratio`).
2. *El color se interpreta según el producto.* Antes cualquier tono verde
   dominante significaba `Unripe`. Pero un pepino, un brócoli o una lechuga
   son verdes cuando están en su punto: eran marcados como inmaduros. Ahora
   la taxonomía dice qué productos usan el verde como señal de madurez
   (`GREEN_MEANS_UNRIPE`) y cuáles no.
3. *Se explica la decisión.* Cada resultado incluye notas en texto de por qué
   salió así — sirve para la exposición y para que la UI lo muestre.
"""

from dataclasses import dataclass, field

from app.ml.features import ImageFeatures
from app.ml.taxonomy import (
    GREEN_IS_NORMAL,
    GREEN_MEANS_UNRIPE,
    NATURALLY_DARK,
    RIPENESS_NOT_COLOR_BASED,
)

# --- Umbrales de calidad (todos sobre proporciones de píxeles, 0-1) ---
# Calibración: ~12% de superficie manchada ya se nota a simple vista al
# revisar un producto en una distribuidora, así que ese es el corte entre
# "Good" y "Regular" (bajarlo más marcaba como perfectas frutas con golpes).
BAD_BLEMISH_RATIO = 0.30  # casi un tercio del producto dañado
REGULAR_BLEMISH_RATIO = 0.12  # manchas visibles pero leves
BAD_BROWN_RATIO = 0.45  # mucho marrón: golpes / oxidación
REGULAR_BROWN_RATIO = 0.25
BAD_DARK_RATIO = 0.40  # oscuridad absoluta (solo si no es oscuro por natura)

# --- Umbrales de madurez ---
OVERRIPE_BLEMISH_RATIO = 0.34  # manchas extensas: pasado de maduro
OVERRIPE_BROWN_RATIO = 0.40
UNRIPE_GREEN_RATIO = 0.40  # proporción de tono verde para decir "no maduro"
UNRIPE_HUE_RANGE = (70.0, 170.0)  # banda de tono considerada verde


@dataclass(frozen=True)
class QualityAssessment:
    """Resultado del análisis de calidad/madurez, con su explicación."""

    quality: str  # Good | Regular | Bad
    ripeness: str  # Unripe | Ripe | Overripe
    notes: list[str] = field(default_factory=list)


def _assess_quality(feats: ImageFeatures, item: str) -> tuple[str, list[str]]:
    """Calcula la calidad y devuelve las notas que la explican."""
    notes: list[str] = []
    naturally_dark = item in NATURALLY_DARK

    if feats.blemish_ratio > BAD_BLEMISH_RATIO:
        notes.append(
            f"Manchas oscuras en {feats.blemish_ratio:.0%} de la superficie "
            "(superficie dañada o muy madura)."
        )
        return "Bad", notes

    if feats.brown_ratio > BAD_BROWN_RATIO:
        notes.append(
            f"Zonas marrones en {feats.brown_ratio:.0%} del producto "
            "(golpes, oxidación o descomposición)."
        )
        return "Bad", notes

    if not naturally_dark and feats.dark_ratio > BAD_DARK_RATIO:
        notes.append(
            f"Superficie mayormente oscura ({feats.dark_ratio:.0%}), "
            "poco brillo para un producto de este tipo."
        )
        return "Bad", notes

    if feats.blemish_ratio > REGULAR_BLEMISH_RATIO:
        notes.append(
            f"Manchas leves en {feats.blemish_ratio:.0%} de la superficie "
            "(aceptable, pero no primera calidad)."
        )
        return "Regular", notes

    if feats.brown_ratio > REGULAR_BROWN_RATIO:
        notes.append(f"Algunas zonas marrones ({feats.brown_ratio:.0%}).")
        return "Regular", notes

    if naturally_dark:
        notes.append(
            "Producto de piel naturalmente oscura: se evaluaron solo las "
            "manchas relativas a su propio color."
        )
    notes.append("Superficie uniforme y sin daños visibles.")
    return "Good", notes


def _assess_ripeness(feats: ImageFeatures, item: str) -> tuple[str, list[str]]:
    """Calcula la madurez según el producto (no todos usan el color igual)."""
    notes: list[str] = []

    is_green_band = UNRIPE_HUE_RANGE[0] <= feats.mean_hue <= UNRIPE_HUE_RANGE[1]

    # 1) Daño extenso siempre gana: un producto con manchas serias está pasado.
    if feats.blemish_ratio > OVERRIPE_BLEMISH_RATIO or feats.brown_ratio > OVERRIPE_BROWN_RATIO:
        worst = max(feats.blemish_ratio, feats.brown_ratio)
        notes.append(
            f"Color/tono degradado en {worst:.0%} del producto: señales de "
            "madurez excesiva."
        )
        return "Overripe", notes

    # 2) Productos donde la madurez no se juzga por color (tubérculos,
    #    raíces, bulbos, hongos): si no hay daño, están listos.
    if item in RIPENESS_NOT_COLOR_BASED:
        notes.append(
            "En este producto el color no indica madurez (es un tubérculo/"
            "raíz/bulbo): se evaluó solo el deterioro."
        )
        return "Ripe", notes

    # 3) Productos que son verdes en su punto: el verde NO es inmadurez.
    if item in GREEN_IS_NORMAL:
        notes.append(
            "Producto de color verde por naturaleza: el verde no se interpreta "
            "como falta de madurez."
        )
        return "Ripe", notes

    # 4) Productos en los que el verde sí significa "no maduro".
    if item in GREEN_MEANS_UNRIPE:
        if is_green_band and feats.green_ratio >= UNRIPE_GREEN_RATIO:
            notes.append(
                f"Tono verde dominante ({feats.green_ratio:.0%} de la "
                "superficie): todavía no ha madurado."
            )
            return "Unripe", notes
        notes.append(
            f"Tono medio {feats.mean_hue:.0f}° con poca superficie verde "
            f"({feats.green_ratio:.0%}): color de producto maduro."
        )
        return "Ripe", notes

    # 5) Producto desconocido o sin regla propia: criterio general conservador.
    if is_green_band and feats.green_ratio >= UNRIPE_GREEN_RATIO + 0.1:
        notes.append("El producto se ve verde en la foto.")
        return "Unripe", notes
    notes.append("No se detectaron señales de madurez excesiva ni de inmadurez.")
    return "Ripe", notes


def assess_quality_and_ripeness(
    feats: ImageFeatures, item: str = ""
) -> QualityAssessment:
    """Devuelve calidad y madurez estimadas más las notas que las explican.

    `item` es el nombre del producto ya identificado: es necesario porque el
    mismo color significa cosas distintas según el producto (un tomate verde
    está inmaduro; un pepino verde está en su punto).
    """
    quality, quality_notes = _assess_quality(feats, item)
    ripeness, ripeness_notes = _assess_ripeness(feats, item)

    if feats.segmentation_confidence < 0.35:
        quality_notes.append(
            "Fondo de la foto poco uniforme: la estimación de calidad puede "
            "ser menos precisa. Prueba con un fondo liso."
        )

    return QualityAssessment(
        quality=quality,
        ripeness=ripeness,
        notes=quality_notes + ripeness_notes,
    )
