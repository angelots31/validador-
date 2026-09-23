"""
Extracción de características de la imagen (color + forma).

Este módulo es la base del análisis "clásico" de visión por computador que
acompaña al modelo ResNet18. Hace, en un solo paso:

1. **Segmentación**: estima el color de fondo (mediana del borde de la
   imagen) y separa los píxeles que pertenecen al producto, con una apertura
   morfológica (erosión + dilatación) para quitar ruido.
2. **Color**: convierte a HSV y calcula medias y proporciones de familias de
   color (verde, amarillo, naranja, rojo, morado, marrón, pálido).
3. **Forma**: del recorte del objeto saca qué tan alargado es, qué tan lleno
   está su rectángulo envolvente y qué tan circular es (compacidad).

Todos los cálculos son vectorizados con numpy: sin OpenCV ni scipy, porque
el despliegue en Vercel (commit 8) tiene un límite de tamaño por función
serverless y no queremos sumar dependencias binarias pesadas.

Dos detalles que importan para que los resultados sean sensatos:

- Las proporciones de color se **ponderan por saturación**: un píxel gris
  claro no debería contar como "amarillo" solo porque su tono cae ahí.
- El brillo se mide **relativo al propio producto** (`blemish_ratio`): así
  una berenjena o unas uvas oscuras no se marcan como "deterioradas" solo
  por ser oscuras. Ese fue uno de los bugs del heurístico anterior.
"""

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

# Resolución de trabajo: suficiente para estadísticas de color/forma y
# barata de calcular (una foto de 1024px se reduce ~40x en píxeles).
FEATURE_SIZE = 160

# Distancia mínima de color al fondo para considerar un píxel "objeto".
_BG_DISTANCE = 0.20
# Un píxel muy saturado se considera objeto aunque su distancia al fondo
# sea baja (escena donde el producto y el fondo tienen brillo parecido).
_VIVID_SAT = 0.45
# Criterios para confiar en la máscara del objeto. Se evalúan sobre el
# rectángulo envolvente y no sobre el área de la máscara: un producto
# alargado y delgado (una zanahoria, un plátano) ocupa poco del encuadre
# aunque esté perfectamente segmentado.
_MIN_BBOX_RATIO = 0.06  # el rectángulo del objeto debe cubrir >= 6% del frame
_MIN_BBOX_FILL = 0.12  # y la máscara debe llenar al menos 12% de ese rectángulo


@dataclass(frozen=True)
class ImageFeatures:
    """Características numéricas de una imagen de producto."""

    # --- Color ---
    mean_hue: float  # tono medio circular, en grados [0, 360)
    hue_concentration: float  # 0-1: qué tan concentrado está el tono
    mean_sat: float
    mean_val: float
    sat_std: float
    val_std: float
    green_ratio: float
    yellow_ratio: float
    orange_ratio: float
    red_ratio: float
    purple_ratio: float
    brown_ratio: float
    pale_ratio: float
    dark_ratio: float  # píxeles muy oscuros (absoluto)
    blemish_ratio: float  # manchas oscuras relativas al propio producto

    # --- Forma ---
    elongation: float  # >= 1.0, largo/ancho del objeto
    solidity: float  # 0-1, área del objeto / área de su rectángulo
    compactness: float  # 0-1, 1.0 = perfectamente circular
    foreground_ratio: float  # 0-1, proporción de la imagen que es objeto

    # --- Calidad de la segmentación (para no confiar en basura) ---
    segmentation_confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


def rgb_to_hsv(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conversión RGB -> HSV vectorizada. `arr` debe ser (H, W, 3) en [0, 1].

    Devuelve `(h, s, v)` con el tono en [0, 1) (se multiplica por 360 para
    tenerlo en grados), la saturación y el valor en [0, 1].
    """
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    delta = maxc - minc

    v = maxc
    s = np.divide(delta, maxc, out=np.zeros_like(maxc), where=maxc > 0)

    delta_safe = np.where(delta == 0, 1.0, delta)
    rc = (maxc - r) / delta_safe
    gc = (maxc - g) / delta_safe
    bc = (maxc - b) / delta_safe

    h = np.zeros_like(maxc)
    h = np.where(maxc == r, bc - gc, h)
    h = np.where(maxc == g, 2.0 + rc - bc, h)
    h = np.where(maxc == b, 4.0 + gc - rc, h)
    h = np.where(delta == 0, 0.0, (h / 6.0) % 1.0)
    return h, s, v


def _neighbours(mask: np.ndarray) -> list[np.ndarray]:
    """Los 8 vecinos de cada píxel (sin envolver los bordes)."""
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    return [
        padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy, dx) != (0, 0)
    ]


def _erode(mask: np.ndarray) -> np.ndarray:
    return mask & np.logical_and.reduce(_neighbours(mask))


def _dilate(mask: np.ndarray) -> np.ndarray:
    return mask | np.logical_or.reduce(_neighbours(mask))


def _box_density(mask: np.ndarray, size: int = 5) -> np.ndarray:
    """Cantidad de píxeles de máscara en una ventana `size`x`size` por píxel.

    Se calcula con una imagen integral (suma acumulada) para que sea O(n):
    es la forma barata de saber dónde está la zona más "llena" del objeto.
    """
    h, w = mask.shape
    integral = np.zeros((h + 1, w + 1), dtype=np.int32)
    integral[1:, 1:] = mask.astype(np.int32).cumsum(axis=0).cumsum(axis=1)

    half = size // 2
    y0 = np.clip(np.arange(h) - half, 0, h)
    y1 = np.clip(np.arange(h) + half + 1, 0, h)
    x0 = np.clip(np.arange(w) - half, 0, w)
    x1 = np.clip(np.arange(w) + half + 1, 0, w)

    return (
        integral[np.ix_(y1, x1)]
        - integral[np.ix_(y0, x1)]
        - integral[np.ix_(y1, x0)]
        + integral[np.ix_(y0, x0)]
    )


def _main_region(mask: np.ndarray) -> np.ndarray | None:
    """Región conectada principal de la máscara (el producto).

    Sin scipy no hay `label()` disponible, así que se hace crecimiento de
    región vectorizado: se siembra en el píxel con mayor densidad local y se
    dilata repetidamente dentro de la máscara hasta que deja de crecer.

    Esto reemplaza al recorte por densidad de filas/columnas de la primera
    versión, que fallaba con productos alargados y delgados (una zanahoria o
    un plátano en diagonal quedaban recortados a casi nada).
    """
    if not mask.any():
        return None

    seed_flat = int(np.argmax(_box_density(mask)))
    seed_y, seed_x = np.unravel_index(seed_flat, mask.shape)

    region = np.zeros_like(mask)
    region[seed_y, seed_x] = True
    region &= mask
    if not region.any():  # el punto más denso cayó fuera de la máscara
        ys, xs = np.nonzero(mask)
        region[ys[0], xs[0]] = True

    for _ in range(80):
        grown = _dilate(region) & mask
        if np.array_equal(grown, region):
            break
        region = grown

    return region


def _bbox(region: np.ndarray):
    """Rectángulo envolvente (r0, r1, c0, c1) de una región, o None si vacía."""
    ys, xs = np.nonzero(region)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _background_color(arr: np.ndarray) -> np.ndarray:
    """Color de fondo estimado con la mediana de un anillo del borde."""
    h, w = arr.shape[:2]
    band = max(2, int(round(min(h, w) * 0.08)))
    border = np.concatenate(
        [
            arr[:band].reshape(-1, 3),
            arr[-band:].reshape(-1, 3),
            arr[:, :band].reshape(-1, 3),
            arr[:, -band:].reshape(-1, 3),
        ]
    )
    return np.median(border, axis=0)


def _color_ratios(h: np.ndarray, s: np.ndarray) -> dict[str, float]:
    """Proporción de píxeles en cada familia de color, ponderada por saturación.

    Se cuenta cada píxel como "de color" solo si su saturación supera 0.18;
    si no, es un tono gris que no debe votar por ninguna familia (de lo
    contrario un fondo beige vota como "amarillo").
    """
    hue_deg = h * 360.0
    colored = s > 0.18
    total = max(int(colored.sum()), 1)

    def ratio(lo: float, hi: float) -> float:
        return float((colored & (hue_deg >= lo) & (hue_deg < hi)).sum() / total)

    return {
        "red_ratio": float(((colored & ((hue_deg < 15) | (hue_deg >= 345))).sum()) / total),
        "orange_ratio": ratio(15, 40),
        "yellow_ratio": ratio(40, 70),
        "green_ratio": ratio(70, 165),
        "purple_ratio": ratio(250, 330),
    }


def extract_features(image: Image.Image) -> ImageFeatures:
    """Calcula todas las características de color y forma de una imagen."""
    img = image.convert("RGB").resize((FEATURE_SIZE, FEATURE_SIZE), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float64) / 255.0

    h, s, v = rgb_to_hsv(arr)
    hue_deg = h * 360.0

    # --- Segmentación del producto ---
    bg = _background_color(arr)
    distance = np.sqrt(((arr - bg) ** 2).sum(axis=-1))
    mask = (distance > _BG_DISTANCE) | (s > _VIVID_SAT)

    mask = _erode(mask)
    mask = _dilate(mask)
    mask = _dilate(mask)

    main = _main_region(mask)
    bbox = _bbox(main) if main is not None else None
    usable = False
    if main is not None and bbox is not None:
        r0, r1, c0, c1 = bbox
        bbox_ratio = ((r1 - r0) * (c1 - c0)) / float(mask.size)
        fill = float(main[r0:r1, c0:c1].mean())
        usable = bbox_ratio >= _MIN_BBOX_RATIO and fill >= _MIN_BBOX_FILL

        # Nos quedamos solo con el producto: así el color y la forma no se
        # contaminan con píxeles sueltos del fondo.
        mask = main

    foreground_ratio = float(mask.mean())

    # Si la máscara quedó inservible, analizamos la imagen completa y
    # marcamos baja confianza de segmentación.
    if not usable:
        mask = np.ones_like(mask)
        foreground_ratio = 1.0

    flat = mask.reshape(-1)
    hs, ss, vs = h.reshape(-1)[flat], s.reshape(-1)[flat], v.reshape(-1)[flat]
    hd = (hs * 360.0)

    # --- Color ---
    # Media circular del tono (el tono es un ángulo: 359° y 1° son vecinos).
    angles = np.radians(hd)
    sin_m, cos_m = float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles)))
    mean_hue = (np.degrees(np.arctan2(sin_m, cos_m)) + 360.0) % 360.0
    hue_concentration = float(np.clip(np.hypot(sin_m, cos_m), 0.0, 1.0))

    mean_val = float(np.mean(vs))
    median_val = float(np.median(vs))

    ratios = _color_ratios(hs, ss)

    # "Mancha": píxel claramente más oscuro que el propio producto. El
    # umbral es relativo para no castigar productos naturalmente oscuros.
    blemish_threshold = max(0.10, median_val * 0.62)
    blemish_ratio = float(np.mean(vs < blemish_threshold))
    dark_ratio = float(np.mean(vs < 0.22))

    brown_ratio = float(
        np.mean((hd >= 10) & (hd < 55) & (ss > 0.28) & (vs < 0.58))
    )
    pale_ratio = float(np.mean((ss < 0.20) & (vs > 0.70)))

    # --- Forma ---
    elongation, solidity, compactness = 1.0, 1.0, 0.0
    if bbox is not None and usable:
        r0, r1, c0, c1 = bbox
        height, width = max(r1 - r0, 1), max(c1 - c0, 1)
        elongation = max(height, width) / min(height, width)
        solidity = float(mask[r0:r1, c0:c1].mean())

        boundary = mask & ~_erode(mask)
        perimeter = float(boundary.sum())
        area = float(mask.sum())
        if perimeter > 0:
            compactness = float(np.clip(4.0 * np.pi * area / (perimeter**2), 0.0, 1.0))

    # Confianza de la segmentación: qué tan creíble es la máscara. Se apoya
    # en que la proporción de objeto sea razonable y el fondo uniforme.
    band = max(2, int(round(FEATURE_SIZE * 0.08)))
    border_pixels = np.concatenate(
        [
            arr[:band].reshape(-1, 3),
            arr[-band:].reshape(-1, 3),
            arr[:, :band].reshape(-1, 3),
            arr[:, -band:].reshape(-1, 3),
        ]
    )
    bg_noise = float(border_pixels.std(axis=0).mean())
    background_uniformity = float(np.clip(1.0 - bg_noise * 3.5, 0.0, 1.0))
    segmentation_confidence = (
        background_uniformity if usable else background_uniformity * 0.45
    )

    return ImageFeatures(
        mean_hue=round(mean_hue, 2),
        hue_concentration=round(hue_concentration, 4),
        mean_sat=round(float(np.mean(ss)), 4),
        mean_val=round(mean_val, 4),
        sat_std=round(float(np.std(ss)), 4),
        val_std=round(float(np.std(vs)), 4),
        green_ratio=round(ratios["green_ratio"], 4),
        yellow_ratio=round(ratios["yellow_ratio"], 4),
        orange_ratio=round(ratios["orange_ratio"], 4),
        red_ratio=round(ratios["red_ratio"], 4),
        purple_ratio=round(ratios["purple_ratio"], 4),
        brown_ratio=round(brown_ratio, 4),
        pale_ratio=round(pale_ratio, 4),
        dark_ratio=round(dark_ratio, 4),
        blemish_ratio=round(blemish_ratio, 4),
        elongation=round(elongation, 3),
        solidity=round(solidity, 4),
        compactness=round(compactness, 4),
        foreground_ratio=round(foreground_ratio, 4),
        segmentation_confidence=round(segmentation_confidence, 4),
    )
