"""
Clasificador de respaldo por color y forma.

**El problema que resuelve (commit 7):**
ImageNet solo tiene ~23 clases de frutas/verduras y **no incluye papaya,
mango, guayaba, sandía, aguacate, tomate, zanahoria** ni casi ninguna otra
de las que se venden en una plaza de mercado. Cuando se fotografiaba una
papaya, el modelo no encontraba ninguna fruta entre sus predicciones y la
API terminaba devolviendo la clase cruda de ImageNet, que para una papaya
era literalmente `{"item": "Rubber Eraser"}`.

**La solución:**
Cuando el modelo no reconoce el producto con confianza suficiente, se usa
esta segunda etapa. Es un clasificador *nearest-centroid* sobre las
características de `features.py`: cada producto tiene un perfil (tono,
saturación, brillo, qué tan alargado es, etc.) y se elige el perfil que
mejor "encaja" con la foto.

**Sé honesto en la sustentación:** esto NO es una red neuronal entrenada.
Es visión por computador clásica con umbrales calibrados a mano. La razón
de fondo es que no existe un dataset etiquetado de "calidad y madurez de
frutas colombianas" dentro del alcance de las 15 horas del taller, y en
Vercel no cabe un segundo modelo de deep learning grande. Lo que sí hace
bien es resolver el caso que fallaba: nombrar productos que ImageNet
desconoce, y decir explícitamente que el resultado viene de un heurístico
(`method="color-shape-heuristics"`) para que nadie lo confunda con el modelo.

**Cómo ajustarlo:** cada número de `PROFILES` es un centro (valor típico) y
una tolerancia. Si en la sustentación algún producto sale mal, se ajusta su
centro/tolerancia y listo — no hay que reentrenar nada.
"""

from dataclasses import dataclass

import numpy as np

from app.ml.features import ImageFeatures
from app.ml.taxonomy import PRODUCE_BY_NAME, category_of

# Puntaje mínimo (0-1) para aceptar la sugerencia del clasificador de
# respaldo. Por debajo de esto preferimos decir honestamente "Unknown" antes
# que inventar un producto.
MIN_SCORE = 0.52

# Saturación media mínima para intentar un análisis por color. Una foto
# prácticamente en escala de grises no tiene información de color: sin este
# candado, un fondo liso gris "coincidía" con perfiles de productos pálidos
# (cebolla, ajo) y el sistema inventaba un producto donde no había nada.
MIN_SATURATION = 0.15

# Spec de una característica: (valor típico, tolerancia, peso).
Spec = tuple[float, float, float]


def _s(center: float, tol: float, weight: float = 1.0) -> Spec:
    return (center, tol, weight)


# ---------------------------------------------------------------------------
# Perfiles por producto.
#
# Convención de nombres de característica: los mismos de `ImageFeatures`.
#   hue       -> tono medio en grados (distancia circular, no lineal)
#   hue_alt   -> segunda banda de tono válida (productos de dos colores)
#   elong*    -> qué tan alargado es el objeto (1.0 = redondo)
#   fg        -> proporción de la imagen ocupada por el producto
# ---------------------------------------------------------------------------
PROFILES: dict[str, dict[str, Spec]] = {
    # ---- Frutas grandes y carnosas ----
    # Papaya y mango se parecen mucho en color (ambos naranja-amarillo), así
    # que se separan sobre todo por forma y saturación: la papaya es un óvalo
    # largo y de color más apagado, el mango es más redondo y mucho más vivo
    # (además suele traer un rubor rojizo).
    "Papaya": {
        "hue": _s(38, 18, 1.2),
        "hue_alt": _s(60, 20, 0.4),  # papaya con vetas verdes
        "sat": _s(0.55, 0.22, 1.0),
        "val": _s(0.72, 0.20, 0.7),
        "elongation": _s(1.60, 0.65, 1.3),
        "fg": _s(0.55, 0.30, 0.4),
    },
    "Mango": {
        "hue": _s(33, 15, 1.2),
        "sat": _s(0.75, 0.18, 1.3),
        "val": _s(0.66, 0.17, 0.8),
        "elongation": _s(1.25, 0.25, 1.1),
        "red_ratio": _s(0.15, 0.20, 0.6),
    },
    "Watermelon": {
        "hue": _s(110, 30, 1.3),
        "sat": _s(0.50, 0.24, 0.9),
        "val": _s(0.36, 0.18, 1.0),
        "elongation": _s(1.15, 0.28, 1.1),
        "fg": _s(0.60, 0.28, 0.5),
    },
    "Melon": {
        "hue": _s(55, 24, 1.0),
        "sat": _s(0.26, 0.17, 1.2),
        "val": _s(0.72, 0.15, 0.9),
        "elongation": _s(1.15, 0.25, 0.9),
        "fg": _s(0.55, 0.30, 0.4),
    },
    "Guava": {
        "hue": _s(58, 16, 1.3),
        "sat": _s(0.45, 0.20, 1.0),
        "val": _s(0.68, 0.15, 0.9),
        "elongation": _s(1.10, 0.22, 1.0),
    },
    "Avocado": {
        "hue": _s(85, 32, 1.0),
        "hue_alt": _s(285, 30, 0.9),  # Hass madura (piel morada)
        "sat": _s(0.40, 0.22, 1.0),
        "val": _s(0.28, 0.15, 1.2),
        "elongation": _s(1.35, 0.32, 1.0),
    },
    "Pear": {
        "hue": _s(68, 20, 1.2),
        "sat": _s(0.32, 0.17, 1.1),
        "val": _s(0.72, 0.15, 1.0),
        "elongation": _s(1.30, 0.32, 1.0),
    },
    "Kiwi": {
        "hue": _s(30, 14, 1.1),
        "sat": _s(0.35, 0.16, 1.1),
        "val": _s(0.45, 0.14, 1.1),
        "elongation": _s(1.15, 0.22, 0.9),
    },
    "Peach": {
        "hue": _s(22, 15, 1.2),
        "sat": _s(0.45, 0.20, 1.0),
        "val": _s(0.66, 0.15, 0.9),
        "elongation": _s(1.10, 0.22, 0.9),
    },
    "Plum": {
        "hue": _s(320, 35, 1.2),
        "sat": _s(0.38, 0.20, 0.9),
        "val": _s(0.32, 0.16, 1.0),
        "elongation": _s(1.10, 0.22, 0.8),
    },
    "Grape": {
        "hue": _s(290, 42, 1.1),
        "hue_alt": _s(80, 25, 0.8),
        "sat": _s(0.36, 0.20, 0.9),
        "val": _s(0.36, 0.17, 0.9),
    },
    "Passion Fruit": {
        "hue": _s(45, 18, 1.1),
        "hue_alt": _s(285, 28, 0.8),
        "sat": _s(0.60, 0.22, 0.9),
        "val": _s(0.55, 0.18, 0.8),
    },
    "Tomato": {
        "hue": _s(8, 13, 1.4),
        "sat": _s(0.72, 0.18, 1.1),
        "val": _s(0.52, 0.20, 0.9),
        "elongation": _s(1.05, 0.18, 0.9),
    },
    "Cherry": {
        "hue": _s(352, 14, 1.3),
        "sat": _s(0.62, 0.20, 1.0),
        "val": _s(0.40, 0.18, 0.9),
    },
    "Coconut": {
        "hue": _s(32, 18, 1.0),
        "sat": _s(0.24, 0.15, 1.1),
        "val": _s(0.48, 0.16, 0.9),
    },
    "Lime": {
        "hue": _s(85, 22, 1.2),
        "sat": _s(0.55, 0.20, 1.0),
        "val": _s(0.60, 0.16, 0.9),
    },
    "Plantain": {
        "hue": _s(78, 22, 1.1),
        "sat": _s(0.45, 0.20, 0.9),
        "val": _s(0.45, 0.18, 0.9),
        "elongation": _s(2.10, 0.60, 1.3),
    },
    # ---- Hortalizas y tubérculos ----
    "Carrot": {
        "hue": _s(28, 12, 1.3),
        "sat": _s(0.80, 0.15, 1.2),
        "val": _s(0.72, 0.15, 0.9),
        "elongation": _s(2.30, 0.75, 1.5),
    },
    "Sweet Potato": {
        "hue": _s(24, 14, 1.1),
        "sat": _s(0.55, 0.18, 1.0),
        "val": _s(0.55, 0.16, 0.9),
        "elongation": _s(1.60, 0.45, 1.1),
    },
    "Potato": {
        "hue": _s(33, 15, 1.0),
        "sat": _s(0.34, 0.17, 1.1),
        "val": _s(0.52, 0.15, 1.0),
        "elongation": _s(1.35, 0.35, 0.9),
    },
    "Cassava": {
        "hue": _s(40, 16, 1.0),
        "sat": _s(0.22, 0.14, 1.2),
        "val": _s(0.55, 0.16, 0.9),
        "elongation": _s(1.90, 0.55, 1.3),
    },
    "Onion": {
        "hue": _s(48, 22, 0.8),
        "sat": _s(0.18, 0.13, 1.2),
        "val": _s(0.70, 0.16, 1.0),
        "elongation": _s(1.05, 0.20, 0.9),
    },
    "Garlic": {
        "hue": _s(50, 25, 0.7),
        "sat": _s(0.14, 0.11, 1.3),
        "val": _s(0.78, 0.14, 1.0),
    },
    "Eggplant": {
        "hue": _s(285, 32, 1.2),
        "sat": _s(0.36, 0.18, 1.0),
        "val": _s(0.22, 0.12, 1.2),
        "elongation": _s(1.45, 0.38, 1.0),
    },
    "Beet": {
        "hue": _s(340, 25, 1.2),
        "sat": _s(0.45, 0.20, 1.0),
        "val": _s(0.30, 0.14, 1.1),
    },
    "Radish": {
        "hue": _s(350, 16, 1.2),
        "sat": _s(0.58, 0.20, 1.0),
        "val": _s(0.60, 0.16, 0.9),
    },
    "Lettuce": {
        "hue": _s(95, 28, 1.2),
        "sat": _s(0.45, 0.20, 1.0),
        "val": _s(0.58, 0.16, 0.9),
        "elongation": _s(1.10, 0.22, 0.8),
    },
    "Spinach": {
        "hue": _s(100, 26, 1.2),
        "sat": _s(0.55, 0.20, 1.0),
        "val": _s(0.34, 0.15, 1.0),
    },
    "Green Bean": {
        "hue": _s(90, 24, 1.1),
        "sat": _s(0.48, 0.20, 1.0),
        "val": _s(0.46, 0.16, 0.9),
        "elongation": _s(2.00, 0.60, 1.3),
    },
    "Chili Pepper": {
        "hue": _s(5, 14, 1.2),
        "hue_alt": _s(85, 20, 0.9),
        "sat": _s(0.70, 0.20, 1.1),
        "val": _s(0.48, 0.18, 0.9),
        "elongation": _s(1.85, 0.50, 1.3),
    },
}

# Solo se permite predecir perfiles que la taxonomía conozca: así no puede
# aparecer un producto "fantasma" que la UI no sepa pintar.
PROFILES = {name: spec for name, spec in PROFILES.items() if name in PRODUCE_BY_NAME}


@dataclass(frozen=True)
class FallbackResult:
    """Resultado del clasificador de respaldo."""

    name: str
    category: str
    score: float
    ranking: list[tuple[str, float]]
    signals: list[tuple[str, float]]  # características del ganador, ordenadas


def _membership(value: float, spec: Spec, circular: bool = False) -> float:
    """Qué tan bien encaja `value` con el centro del perfil (0-1).

    Se usa una campana gaussiana: 1.0 en el centro exacto y decae suave
    según la tolerancia. Para el tono la distancia es circular (359° y 1°
    están a 2°, no a 358°).
    """
    center, tol, _ = spec
    delta = abs(value - center)
    if circular:
        delta = min(delta, 360.0 - delta)
    return float(np.exp(-0.5 * (delta / max(tol, 1e-6)) ** 2))


def _score_profile(feats: ImageFeatures, profile: dict[str, Spec]):
    """Puntaje ponderado de un perfil y el detalle por característica."""
    lookup = {
        "hue": (feats.mean_hue, True),
        "sat": (feats.mean_sat, False),
        "val": (feats.mean_val, False),
        "elongation": (feats.elongation, False),
        "solidity": (feats.solidity, False),
        "fg": (feats.foreground_ratio, False),
        "green_ratio": (feats.green_ratio, False),
        "yellow_ratio": (feats.yellow_ratio, False),
        "orange_ratio": (feats.orange_ratio, False),
        "red_ratio": (feats.red_ratio, False),
    }

    total_weight = 0.0
    weighted = 0.0
    signals: list[tuple[str, float]] = []

    for feature, spec in profile.items():
        if feature == "hue_alt":
            value, circular = lookup["hue"]
            match = max(_membership(value, spec, circular=True), 0.0)
        else:
            value, circular = lookup[feature]
            match = _membership(value, spec, circular)

        weight = spec[2]
        total_weight += weight
        weighted += weight * match
        signals.append((feature, round(match, 3)))

    score = weighted / total_weight if total_weight else 0.0
    signals.sort(key=lambda pair: pair[1], reverse=True)
    return score, signals


def classify_by_features(feats: ImageFeatures, top_k: int = 5) -> FallbackResult:
    """Elige el producto cuyo perfil de color/forma mejor encaja con la foto.

    Si la foto no tiene información de color suficiente, devuelve un resultado
    vacío con puntaje 0: el llamador lo interpretará como "no identificado"
    (ver `MIN_SATURATION`).
    """
    if feats.mean_sat < MIN_SATURATION:
        return FallbackResult(
            name="Unknown",
            category="Unknown",
            score=0.0,
            ranking=[],
            signals=[],
        )

    scored: list[tuple[str, float, list[tuple[str, float]]]] = []
    for name, profile in PROFILES.items():
        score, signals = _score_profile(feats, profile)
        scored.append((name, score, signals))

    scored.sort(key=lambda row: row[1], reverse=True)
    best_name, best_score, best_signals = scored[0]

    return FallbackResult(
        name=best_name,
        category=category_of(best_name),
        score=round(float(best_score), 4),
        ranking=[(name, round(float(score), 4)) for name, score, _ in scored[:top_k]],
        signals=best_signals[:4],
    )
