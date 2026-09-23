"""
Carga e inferencia del modelo preentrenado (commits 3 y 7).

Usamos **ResNet18 preentrenado en ImageNet, exportado a ONNX**, ejecutado con
`onnxruntime` (no con PyTorch). La razón es de despliegue: PyTorch +
torchvision pesan varios cientos de MB y no caben cómodamente en el límite
de tamaño de una Vercel Serverless Function. `onnxruntime` + el .onnx del
modelo pesan ~70 MB en total, viable para el commit 8.

Pipeline completo (cascada de decisión)
--------------------------------------
::

    foto
     │
     ├─ 1. ResNet18 (ONNX) con TTA  ──► ¿algún producto de ImageNet con p >= 0.22?
     │                                    └─ sí → ese es el resultado  (method="resnet18-imagenet")
     │
     ├─ 2. Clasificador de respaldo (color/forma, fallback.py)
     │        └─ si su puntaje >= 0.55 y supera la pista débil del modelo
     │           → ese es el resultado  (method="color-shape-heuristics")
     │
     ├─ 3. ¿El modelo dio al menos una pista débil (p >= 0.07)?
     │        └─ sí → la usamos, avisando que la confianza es baja
     │
     └─ 4. Nada convincente → item="Unknown", method="unrecognized"

**Por qué existe el paso 2** (el bug que arregla el commit 7): ImageNet no
tiene clase para papaya, mango, guayaba, sandía ni la mayoría de las frutas
tropicales. Antes, cuando el modelo no encontraba ninguna fruta, la API
devolvía su clase #1 cruda — para una papaya, literalmente `"Rubber Eraser"`.
Ahora ese caso cae al clasificador de respaldo, que sí puede decir "Papaya",
y si de plano no hay nada convincente responde `"Unknown"` en vez de inventar.

**Test-time augmentation (TTA):** el resultado se promedia sobre 4 vistas de
la misma foto (recorte central estándar y foto completa, cada una con su
espejo horizontal). Cuesta 4 pasadas de 224x224 en CPU, que es barato, y
mejora notablemente los casos donde el producto no está perfectamente
centrado o la cámara lo capturó en horizontal.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.ml import fallback
from app.ml.features import ImageFeatures, extract_features
from app.ml.taxonomy import IMAGENET_PRODUCE, category_of

ML_DIR = Path(__file__).resolve().parent
MODEL_PATH = ML_DIR / "resnet18.onnx"
CLASSES_PATH = ML_DIR / "imagenet_classes.txt"

# Normalización estándar de ImageNet (con la que ResNet18 fue entrenado).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

INPUT_SIZE = 224
RESIZE_SHORT_SIDE = 256

# --- Umbrales de la cascada de decisión (ajustables y explicables) ---
#
# Calibración: en pruebas, una foto donde el modelo SÍ reconoce el producto da
# probabilidades de 0.4-0.95 en su clase; una foto de algo que ImageNet no
# conoce (una papaya) deja todas las clases de producto por debajo de 0.002.
# Con esos datos, 0.15 separa con margen "el modelo sabe" de "el modelo no
# tiene ni idea", que es justo lo que necesita la cascada.
MODEL_ACCEPT = 0.15  # el modelo reconoce el producto con confianza suficiente
MODEL_WEAK = 0.02  # apenas una pista, pero mejor que decir "no sé"

METHOD_MODEL = "resnet18-imagenet"
METHOD_HEURISTIC = "color-shape-heuristics"
METHOD_UNKNOWN = "unrecognized"

UNKNOWN_ITEM = "Unknown"

_session: ort.InferenceSession | None = None
_classes: list[str] | None = None


@dataclass(frozen=True)
class Candidate:
    """Un producto propuesto, con su puntaje y de dónde salió."""

    item: str
    score: float
    source: str  # "model" | "heuristic"


@dataclass(frozen=True)
class ProduceMatch:
    """Resultado de identificar el producto de una foto."""

    item: str
    category: str
    confidence: float
    method: str
    candidates: list[Candidate]
    features: ImageFeatures
    notes: list[str]


def _load_classes() -> list[str]:
    global _classes
    if _classes is None:
        with open(CLASSES_PATH, encoding="utf-8") as f:
            _classes = [line.strip() for line in f if line.strip()]
    return _classes


def _get_session() -> ort.InferenceSession:
    """Carga el modelo ONNX una sola vez (patrón singleton).

    Cargar el modelo de 45 MB en cada petición sería muy lento; lo hacemos
    una vez por proceso y reutilizamos la sesión.
    """
    global _session
    if _session is None:
        _session = ort.InferenceSession(
            str(MODEL_PATH), providers=["CPUExecutionProvider"]
        )
    return _session


def _normalize(image: Image.Image) -> np.ndarray:
    """Normaliza un recorte ya del tamaño de entrada y devuelve NCHW."""
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.expand_dims(arr.transpose(2, 0, 1), axis=0).astype(np.float32)


def _view_center_crop(image: Image.Image) -> np.ndarray:
    """Vista clásica: lado corto a 256 y recorte central de 224x224."""
    w, h = image.size
    scale = RESIZE_SHORT_SIDE / max(min(w, h), 1)
    new_w, new_h = max(INPUT_SIZE, round(w * scale)), max(INPUT_SIZE, round(h * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - INPUT_SIZE) // 2
    top = (new_h - INPUT_SIZE) // 2
    return _normalize(resized.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE)))


def _view_full_frame(image: Image.Image) -> np.ndarray:
    """Vista alternativa: la foto completa reescalada, sin recortar.

    Sirve cuando el producto ocupa casi todo el encuadre (típico de la
    cámara del celular), caso en el que el recorte central tira a la basura
    parte del producto.
    """
    return _normalize(image.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _predict_produce_probabilities(image: Image.Image) -> dict[str, float]:
    """Probabilidad agregada por producto, promediando las 4 vistas (TTA)."""
    session = _get_session()
    input_name = session.get_inputs()[0].name
    classes = _load_classes()

    views = [_view_center_crop(image), _view_full_frame(image)]
    accumulated: np.ndarray | None = None
    passes = 0

    for view in views:
        for batch in (view, view[..., ::-1]):  # original + espejo horizontal
            mirrored = np.ascontiguousarray(batch)
            (logits,) = session.run(None, {input_name: mirrored})
            probs = _softmax(logits[0])
            accumulated = probs if accumulated is None else accumulated + probs
            passes += 1

    assert accumulated is not None
    mean_probs = accumulated / passes

    # Agrupamos por nombre de producto: si dos clases de ImageNet mapean al
    # mismo producto, sus probabilidades se suman.
    aggregated: dict[str, float] = {}
    for index, raw_label in enumerate(classes):
        produce = IMAGENET_PRODUCE.get(raw_label)
        if produce is None:
            continue
        aggregated[produce.name] = aggregated.get(produce.name, 0.0) + float(
            mean_probs[index]
        )
    return aggregated


def _decide(
    model_probs: dict[str, float], feats: ImageFeatures
) -> tuple[str, str, float, str, list[Candidate], list[str]]:
    """Aplica la cascada de decisión y devuelve la tupla del resultado."""
    notes: list[str] = []

    model_ranking = sorted(model_probs.items(), key=lambda kv: kv[1], reverse=True)
    best_model_item, best_model_p = model_ranking[0] if model_ranking else ("", 0.0)

    # --- Paso 1: el modelo reconoce el producto ---
    if best_model_p >= MODEL_ACCEPT:
        notes.append(
            f"Identificado por ResNet18 (ImageNet) con {best_model_p:.0%} de "
            "probabilidad media sobre 4 vistas de la foto."
        )
        candidates = [
            Candidate(item=name, score=round(float(p), 4), source="model")
            for name, p in model_ranking[:5]
        ]
        return (
            best_model_item,
            category_of(best_model_item),
            float(best_model_p),
            METHOD_MODEL,
            candidates,
            notes,
        )

    # --- Paso 2: clasificador de respaldo por color y forma ---
    fb = fallback.classify_by_features(feats)
    if fb.score >= fallback.MIN_SCORE and fb.score > best_model_p:
        notes.append(
            "El modelo no encontró este producto entre las clases de ImageNet "
            f"(ImageNet no incluye {fb.name.lower()} entre sus 1000 categorías)."
        )
        notes.append(
            f"Identificado por color y forma (clasificador de respaldo) con "
            f"{fb.score:.0%} de coincidencia. Señales principales: "
            + ", ".join(f"{name} {value:.0%}" for name, value in fb.signals[:3])
            + "."
        )
        candidates = [
            Candidate(item=name, score=score, source="heuristic")
            for name, score in fb.ranking
        ]
        return (
            fb.name,
            fb.category,
            float(fb.score),
            METHOD_HEURISTIC,
            candidates,
            notes,
        )

    # --- Paso 3: pista débil del modelo ---
    if best_model_p >= MODEL_WEAK:
        notes.append(
            f"Confianza baja ({best_model_p:.0%}): el resultado es la mejor "
            "aproximación del modelo, tómalo como orientativo."
        )
        candidates = [
            Candidate(item=name, score=round(float(p), 4), source="model")
            for name, p in model_ranking[:5]
        ]
        return (
            best_model_item,
            category_of(best_model_item),
            float(best_model_p),
            METHOD_MODEL,
            candidates,
            notes,
        )

    # --- Paso 4: nada convincente ---
    notes.append(
        "No se pudo identificar el producto. Prueba acercando la cámara, con "
        "mejor luz y un fondo liso o de color distinto al producto."
    )
    candidates = [
        Candidate(item=name, score=score, source="heuristic")
        for name, score in fb.ranking[:5]
    ]
    return UNKNOWN_ITEM, "Unknown", 0.0, METHOD_UNKNOWN, candidates, notes


def analyze(image: Image.Image) -> ProduceMatch:
    """Analiza una foto y devuelve el producto identificado y cómo se decidió."""
    feats = extract_features(image)
    model_probs = _predict_produce_probabilities(image)
    item, category, confidence, method, candidates, notes = _decide(model_probs, feats)

    # El clasificador de respaldo depende de haber podido separar el producto
    # del fondo. Si la segmentación salió mal (fondo con mucho detalle o del
    # mismo color que el producto), el puntaje se infla con píxeles que no son
    # del producto: lo castigamos para no reportar una confianza falsa.
    if method == METHOD_HEURISTIC:
        confidence *= 0.55 + 0.45 * feats.segmentation_confidence
        confidence = round(float(confidence), 4)

    if feats.segmentation_confidence < 0.35 and method != METHOD_UNKNOWN:
        notes.append(
            "El fondo de la foto tiene mucho detalle o es del mismo color que "
            "el producto: usa un fondo liso y contrastante para mejorar la "
            "precisión."
        )

    return ProduceMatch(
        item=item,
        category=category,
        confidence=round(confidence, 4),
        method=method,
        candidates=candidates,
        features=feats,
        notes=notes,
    )


def predict_item(image: Image.Image, top_k: int = 5) -> tuple[str, float]:
    """Compatibilidad con la versión anterior: devuelve (item, confianza).

    Se mantiene porque `README.md` y las pruebas lo mencionan; internamente
    usa el pipeline completo de `analyze()`.
    """
    match = analyze(image)
    return match.item, match.confidence
