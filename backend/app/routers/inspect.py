import io

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from PIL import Image, ImageOps, UnidentifiedImageError
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.ml.features import ImageFeatures
from app.ml.inference import METHOD_UNKNOWN, analyze
from app.ml.vision import assess_quality_and_ripeness
from app.models.inspection import Inspection
from app.models.user import User
from app.schemas.common import ErrorResponse
from app.schemas.inspection import (
    Candidate,
    ColorBreakdown,
    InspectionDiagnostics,
    InspectionHistoryItem,
    InspectionResult,
)

router = APIRouter(tags=["inspection"])

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB, suficiente para una foto de cámara web
MIN_SIDE_PIXELS = 48  # por debajo de esto no hay información suficiente

# Código 413 (payload demasiado grande) escrito literal a propósito: la
# constante de Starlette cambió de nombre entre versiones
# (HTTP_413_REQUEST_ENTITY_TOO_LARGE -> HTTP_413_CONTENT_TOO_LARGE) y usar la
# nueva rompería con una versión antigua del framework.
HTTP_413_PAYLOAD_TOO_LARGE = 413


def _load_image(raw_bytes: bytes) -> Image.Image:
    """Convierte los bytes subidos en una imagen RGB lista para analizar.

    Se aplica `exif_transpose` porque las fotos tomadas con celular suelen
    traer la orientación en los metadatos EXIF en vez de en los píxeles: sin
    esto, una foto vertical podía llegar girada 90° y el análisis de forma
    (que es una de las señales del clasificador) veía un producto acostado.
    """
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo enviado no es una imagen válida.",
        )

    image = ImageOps.exif_transpose(image) or image
    if min(image.size) < MIN_SIDE_PIXELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"La imagen es demasiado pequeña ({image.size[0]}x{image.size[1]}). "
                f"El lado más corto debe medir al menos {MIN_SIDE_PIXELS} píxeles."
            ),
        )
    return image.convert("RGB")


def _build_diagnostics(features: ImageFeatures) -> InspectionDiagnostics:
    """Traduce las características internas a la parte pública del contrato."""
    return InspectionDiagnostics(
        mean_hue=features.mean_hue,
        saturation=features.mean_sat,
        brightness=features.mean_val,
        elongation=features.elongation,
        blemish_ratio=features.blemish_ratio,
        segmentation_confidence=features.segmentation_confidence,
        colors=ColorBreakdown(
            green=features.green_ratio,
            yellow=features.yellow_ratio,
            orange=features.orange_ratio,
            red=features.red_ratio,
            purple=features.purple_ratio,
            brown=features.brown_ratio,
            pale=features.pale_ratio,
        ),
    )


@router.post(
    "/inspect-fruit",
    response_model=InspectionResult,
    summary="Inspeccionar una fruta o verdura",
    description=(
        "Sube una foto (JPG/PNG/WEBP, máx. 8 MB) y devuelve el producto "
        "identificado, su calidad y su madurez.\n\n"
        "**Cómo identifica el producto** (cascada de decisión):\n"
        "1. `resnet18-imagenet` — el modelo preentrenado reconoce el producto "
        "con confianza suficiente. Cubre ~23 clases de frutas y verduras de "
        "ImageNet (manzana, banano, naranja, brócoli, pepino...).\n"
        "2. `color-shape-heuristics` — ImageNet **no** tiene clases para "
        "papaya, mango, guayaba, sandía, aguacate, tomate, zanahoria y muchas "
        "otras, así que cuando el modelo no reconoce nada se usa un "
        "clasificador de respaldo por color y forma.\n"
        "3. `unrecognized` — cuando nada es convincente se devuelve "
        "`item=\"Unknown\"` en vez de inventar un nombre.\n\n"
        "El campo `method` dice siempre cuál de los tres caminos se usó, y "
        "`notes` explica la decisión en texto. Guarda el resultado en el "
        "historial del usuario autenticado. Requiere `Authorization: Bearer <token>`."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "El archivo no es una imagen válida o es demasiado pequeño."},
        401: {"model": ErrorResponse, "description": "Falta el token o no es válido."},
        413: {"model": ErrorResponse, "description": "La imagen supera los 8 MB."},
        415: {"model": ErrorResponse, "description": "Content-Type no soportado (solo JPG/PNG/WEBP)."},
    },
)
async def inspect_fruit(
    file: UploadFile = File(..., description="Foto del producto (jpg/png/webp)."),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Formato de imagen no soportado. Usa JPG, PNG o WEBP.",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=HTTP_413_PAYLOAD_TOO_LARGE,
            detail="La imagen es demasiado grande (máximo 8 MB).",
        )
    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se recibió ningún archivo.",
        )

    image = _load_image(raw_bytes)

    # Una sola extracción de características: se reutiliza para identificar el
    # producto, para medir calidad/madurez y para el diagnóstico que se
    # devuelve al cliente (antes se analizaba la imagen varias veces).
    match = analyze(image)
    assessment = assess_quality_and_ripeness(match.features, match.item)

    record = Inspection(
        user_id=current_user.id,
        item=match.item,
        category=match.category,
        quality=assessment.quality,
        ripeness=assessment.ripeness,
        confidence=match.confidence,
        method=match.method,
    )
    db.add(record)
    db.commit()

    notes = match.notes + assessment.notes
    if match.method == METHOD_UNKNOWN:
        notes.append("No se guardó un producto reconocible en esta inspección.")

    return InspectionResult(
        item=match.item,
        category=match.category,
        quality=assessment.quality,
        ripeness=assessment.ripeness,
        confidence=match.confidence,
        method=match.method,
        candidates=[Candidate(item=c.item, score=c.score, source=c.source) for c in match.candidates],
        notes=notes,
        diagnostics=_build_diagnostics(match.features),
    )


@router.get(
    "/inspection-history",
    response_model=list[InspectionHistoryItem],
    summary="Historial de inspecciones",
    description=(
        "Devuelve todas las inspecciones hechas por el usuario autenticado, "
        "ordenadas de la más reciente a la más antigua. Cada registro dice con "
        "qué método se identificó el producto (`method`) y su categoría."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Falta el token o no es válido."},
    },
)
def inspection_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Historial de inspecciones del usuario autenticado, más recientes primero."""
    records = (
        db.query(Inspection)
        .filter(Inspection.user_id == current_user.id)
        .order_by(Inspection.created_at.desc(), Inspection.id.desc())
        .all()
    )
    return records
