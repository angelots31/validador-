from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

# Tipos fijos: los devuelve siempre nuestro propio análisis, así que
# documentarlos como Literal (en vez de str) hace que Swagger muestre las
# opciones posibles como un enum, y de paso valida que el backend nunca
# devuelva un valor inesperado.
QualityLabel = Literal["Good", "Regular", "Bad"]
RipenessLabel = Literal["Unripe", "Ripe", "Overripe"]
CategoryLabel = Literal["Fruit", "Vegetable", "Unknown"]

# Cómo se llegó al resultado. Es deliberadamente explícito para no presentar
# un heurístico como si fuera el modelo de deep learning.
MethodLabel = Literal[
    "resnet18-imagenet",  # lo identificó el modelo preentrenado
    "color-shape-heuristics",  # lo identificó el clasificador de respaldo
    "unrecognized",  # no se pudo identificar
]


class Candidate(BaseModel):
    """Un producto propuesto para la foto, con su puntaje y su origen."""

    item: str = Field(description="Nombre del producto propuesto.", examples=["Papaya"])
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Puntaje de esa propuesta (probabilidad del modelo o coincidencia del heurístico).",
        examples=[0.81],
    )
    source: Literal["model", "heuristic"] = Field(
        description="'model' si viene de ResNet18, 'heuristic' si viene del clasificador de respaldo.",
        examples=["heuristic"],
    )


class ColorBreakdown(BaseModel):
    """Reparto de la superficie del producto entre familias de color (0-1)."""

    green: float = Field(description="Proporción de superficie verde.", examples=[0.19])
    yellow: float = Field(description="Proporción de superficie amarilla.", examples=[0.31])
    orange: float = Field(description="Proporción de superficie naranja.", examples=[0.42])
    red: float = Field(description="Proporción de superficie roja.", examples=[0.02])
    purple: float = Field(description="Proporción de superficie morada.", examples=[0.0])
    brown: float = Field(description="Proporción de superficie marrón (golpes u oxidación).", examples=[0.04])
    pale: float = Field(description="Proporción de superficie clara y poco saturada.", examples=[0.08])


class InspectionDiagnostics(BaseModel):
    """Medidas internas del análisis. Sirven para explicar y depurar, y la UI
    las usa para dibujar el desglose de color y la forma detectada."""

    mean_hue: float = Field(description="Tono medio de la superficie, en grados (0-360).", examples=[41.6])
    saturation: float = Field(description="Saturación media (0 = gris, 1 = color puro).", examples=[0.64])
    brightness: float = Field(description="Brillo medio (0 = negro, 1 = blanco).", examples=[0.86])
    elongation: float = Field(
        description="Qué tan alargado es el producto (1.0 = tan alto como ancho).",
        examples=[1.11],
    )
    blemish_ratio: float = Field(
        description="Proporción de manchas oscuras *relativas al propio producto* (no absolutas).",
        examples=[0.03],
    )
    segmentation_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Qué tan confiable fue separar el producto del fondo (0-1).",
        examples=[0.99],
    )
    colors: ColorBreakdown = Field(description="Reparto de la superficie entre familias de color.")


class InspectionResult(BaseModel):
    """Resultado de analizar una foto — incluye el formato del enunciado del
    taller ({"item": "Apple", "quality": "Good", "ripeness": "Ripe"}) y campos
    extra que explican de dónde salió cada valor."""

    item: str = Field(
        description=(
            "Producto identificado. `Unknown` si no se pudo identificar con "
            "suficiente confianza (es preferible a inventar un nombre)."
        ),
        examples=["Papaya"],
    )
    category: CategoryLabel = Field(
        description="Categoría del producto identificado.",
        examples=["Fruit"],
    )
    quality: QualityLabel = Field(
        description="Calidad estimada según manchas y daño visible (heurístico de color, no un modelo entrenado).",
        examples=["Good"],
    )
    ripeness: RipenessLabel = Field(
        description="Madurez estimada. Se interpreta distinto según el producto (un tomate verde está inmaduro, un pepino verde está en su punto).",
        examples=["Ripe"],
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confianza (0 a 1) en la identificación del producto. No aplica a quality/ripeness.",
        examples=[0.81],
    )
    method: MethodLabel = Field(
        description=(
            "Cómo se identificó el producto: por el modelo ResNet18, por el "
            "clasificador de respaldo de color/forma, o no identificado."
        ),
        examples=["color-shape-heuristics"],
    )
    candidates: list[Candidate] = Field(
        default_factory=list,
        description="Las mejores propuestas para esta foto, de mayor a menor puntaje.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Explicación en texto de por qué se llegó a este resultado.",
    )
    diagnostics: InspectionDiagnostics = Field(
        description="Medidas internas del análisis (color, forma, confianza de la segmentación)."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "item": "Papaya",
                    "category": "Fruit",
                    "quality": "Good",
                    "ripeness": "Ripe",
                    "confidence": 0.81,
                    "method": "color-shape-heuristics",
                    "candidates": [
                        {"item": "Papaya", "score": 0.81, "source": "heuristic"},
                        {"item": "Mango", "score": 0.78, "source": "heuristic"},
                    ],
                    "notes": [
                        "ImageNet no incluye papaya entre sus 1000 categorías.",
                        "Identificado por color y forma con 81% de coincidencia.",
                    ],
                    "diagnostics": {
                        "mean_hue": 41.6,
                        "saturation": 0.64,
                        "brightness": 0.86,
                        "elongation": 1.11,
                        "blemish_ratio": 0.03,
                        "segmentation_confidence": 0.99,
                        "colors": {
                            "green": 0.19,
                            "yellow": 0.31,
                            "orange": 0.42,
                            "red": 0.02,
                            "purple": 0.0,
                            "brown": 0.04,
                            "pale": 0.08,
                        },
                    },
                }
            ]
        }
    )


class InspectionHistoryItem(BaseModel):
    """Una inspección guardada, tal como aparece en el historial del usuario."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="Identificador de la inspección.", examples=[1])
    item: str = Field(description="Producto identificado en esa inspección.", examples=["Papaya"])
    category: CategoryLabel = Field(description="Categoría del producto.", examples=["Fruit"])
    quality: QualityLabel = Field(description="Calidad estimada en esa inspección.", examples=["Good"])
    ripeness: RipenessLabel = Field(description="Madurez estimada en esa inspección.", examples=["Ripe"])
    confidence: float = Field(description="Confianza en la identificación del producto.", examples=[0.81])
    method: MethodLabel = Field(
        description="Cómo se identificó el producto en esa inspección.",
        examples=["color-shape-heuristics"],
    )
    created_at: datetime = Field(description="Fecha y hora en que se realizó la inspección (UTC).")

    @field_serializer("created_at")
    def _serialize_created_at(self, value: datetime) -> str:
        """Marca explícitamente que la fecha es UTC.

        SQLite guarda `func.now()` en UTC pero SQLAlchemy lo devuelve *sin*
        zona horaria, así que si se serializara como
        "2026-09-22T21:44:10" el navegador lo interpretaría como hora local
        —y en Colombia (UTC-5) el historial aparecería cinco horas en el
        futuro—. Agregando el offset, `new Date(...)` en JavaScript lo
        convierte a la zona del usuario correctamente.
        """
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
