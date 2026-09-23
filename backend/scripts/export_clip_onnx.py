"""
Re-exporta los modelos de CLIP a ONNX y **valida que sirvan**.

Por qué existe este script
--------------------------
Los `app/ml/clip/*.onnx` originales se generaron con
`onnxruntime.quantization.quantize_dynamic`, que es incompatible con CLIP:
descompone las LayerNormalization y cuantiza mal las proyecciones. El
resultado no es "un poco peor": el encoder de texto colapsa (dos prompts
distintos quedan a 0.996 de distancia coseno, cuando lo normal es 0.7-0.85) y
el encoder de visión devuelve vectores con normas de 10⁵ en vez de ~20. Con
esos archivos, cualquier clasificación zero-shot es ruido.

Este script hace la exportación **sin cuantizar** y, sobre todo, **comprueba la
calidad del resultado antes de instalarlo**. Ese paso es el que faltaba: sin
él, un modelo roto entra al repositorio y nadie se entera hasta que la API
empieza a responder barbaridades con total seguridad.

Uso
---
En un equipo con PyTorch (no en Vercel; esto es una tarea de un solo uso):

    pip install "optimum[exporters]" torch transformers
    cd backend
    python scripts/export_clip_onnx.py

Requiere descargar el checkpoint (~600 MB) desde Hugging Face. Al terminar,
`app/ml/clip/` queda con `vision_model.onnx`, `text_model.onnx` y
`tokenizer.json`, y el script imprime el siguiente paso.

Aviso de tamaño (importante)
----------------------------
El CLIP FP32 completo pesa ~590 MB (visión ~340 MB + texto ~250 MB) y el
límite del bundle de una Vercel Function es de 500 MB, de los que ya se usan
~45 MB del ResNet18. Es decir: **aunque el export sea correcto, CLIP no cabe
junto a ResNet18**. Antes de conectarlo hay que decidir una de estas:

* cuantizar de forma *estática* y por canal, y volver a pasar esta validación;
* usar un modelo más pequeño (p. ej. MobileCLIP-S0);
* mover la inferencia a un servicio aparte con GPU.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
CLIP_DIR = BACKEND_DIR / "app" / "ml" / "clip"
MODEL_ID = "openai/clip-vit-base-patch32"

# Umbrales de la validación. No son caprichos: con los modelos cuantizados
# estos valores se disparan, y con un export sano quedan holgadamente dentro.
MAX_TEXT_COSINE_BETWEEN_CONCEPTS = 0.95
MAX_IMAGE_EMBED_NORM = 200.0
# Las imágenes de prueba (rojo/verde/azul) son muy distintas: si el encoder
# funciona, sus embeddings no pueden parecerse casi por completo.
MIN_IMAGE_EMBED_DISTANCE = 0.05


def fail(message: str) -> None:
    print(f"\n[ERROR] {message}\n", file=sys.stderr)
    raise SystemExit(1)


def export_with_optimum(output_dir: Path) -> None:
    """Delega la exportación en optimum (no reimplementamos el trazado).

    Se prueban los dos puntos de entrada de optimum porque según cómo se haya
    instalado existe el ejecutable `optimum-cli` o solo el módulo.
    """
    arguments = [
        "export",
        "onnx",
        "--model",
        MODEL_ID,
        "--task",
        "zero-shot-image-classification",
        str(output_dir),
    ]
    if shutil.which("optimum-cli"):
        candidates = [["optimum-cli", *arguments]]
    else:
        candidates = [[sys.executable, "-m", "optimum.exporters.onnx", *arguments[2:]]]

    for command in candidates:
        print("Exportando con:", " ".join(command), "\n")
        try:
            subprocess.run(command, check=True)
            return
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as exc:
            fail(
                "La exportación falló. Revisa que tengas torch y transformers "
                f"instalados (código de salida {exc.returncode})."
            )

    fail(
        "No se encontró optimum. Instálalo con:\n"
        '    pip install "optimum[exporters]" torch transformers'
    )


def install_models(export_dir: Path) -> tuple[Path, Path, Path]:
    """Copia los archivos exportados a app/ml/clip/ con los nombres esperados.

    Se buscan por patrón en vez de por nombre exacto: optimum ha cambiado los
    nombres de los archivos entre versiones y no queremos que el script se
    rompa por eso, sino que diga claramente qué encontró.
    """
    CLIP_DIR.mkdir(parents=True, exist_ok=True)

    vision = next(iter(sorted(export_dir.glob("vision*.onnx"))), None)
    text = next(iter(sorted(export_dir.glob("text*.onnx"))), None)
    tokenizer = next(iter(sorted(export_dir.glob("tokenizer.json"))), None)

    missing = [
        name
        for name, found in (("vision*.onnx", vision), ("text*.onnx", text), ("tokenizer.json", tokenizer))
        if found is None
    ]
    if missing:
        fail(
            "La exportación no produjo estos archivos: "
            + ", ".join(missing)
            + f"\nContenido de {export_dir}:\n  "
            + "\n  ".join(sorted(p.name for p in export_dir.iterdir()))
        )

    destination_vision = CLIP_DIR / "vision_model.onnx"
    destination_text = CLIP_DIR / "text_model.onnx"
    destination_tokenizer = CLIP_DIR / "tokenizer.json"
    shutil.copy2(vision, destination_vision)
    shutil.copy2(text, destination_text)
    shutil.copy2(tokenizer, destination_tokenizer)
    print(f"\nInstalados en {CLIP_DIR}")
    return destination_vision, destination_text, destination_tokenizer


def validate_models(vision_path: Path, text_path: Path, tokenizer_path: Path) -> None:
    """Comprueba que los modelos exportados sean realmente utilizables.

    Tres señales, elegidas porque son exactamente las que delataron a los
    modelos cuantizados:
      1. dos conceptos distintos deben quedar claramente separados en el
         espacio de texto;
      2. las normas de los embeddings de imagen deben ser de orden normal
         (~20), no astronómicas;
      3. distintas imágenes no pueden producir el mismo embedding.
    """
    import numpy as np
    import onnxruntime as ort

    sys.path.insert(0, str(BACKEND_DIR))
    from app.ml.clip_tokenizer import load_tokenizer

    from PIL import Image

    tokenizer = load_tokenizer(tokenizer_path)
    text_session = ort.InferenceSession(str(text_path), providers=["CPUExecutionProvider"])
    vision_session = ort.InferenceSession(str(vision_path), providers=["CPUExecutionProvider"])

    def embed_text(prompt: str) -> np.ndarray:
        ids = np.array([tokenizer.encode(prompt)], dtype=np.int64)
        return text_session.run(None, {"input_ids": ids})[0][0]

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    print("\nValidando el modelo exportado...")

    # 1. El tokenizador: ejemplo canonico de CLIP (la secuencia completa, con
    #    BOS y EOS; `encode` ademas rellena hasta 77).
    expected = [49406, 320, 1125, 539, 320, 2368, 49407]
    got = tokenizer.encode("a photo of a cat")[: len(expected)]
    if got != expected:
        fail(f"El tokenizador no reproduce los ids de CLIP: {got!r} != {expected!r}")
    print("  [ok]    tokenizador: 'a photo of a cat' -> ids canonicos de CLIP")

    # 2. Separacion semantica del texto.
    pairs = [
        ("a photo of a papaya", "a photo of a carrot"),
        ("a photo of a cat", "a photo of a car"),
        ("a photo of a banana", "a photo of a cucumber"),
    ]
    worst = 0.0
    for left, right in pairs:
        similarity = cosine(embed_text(left), embed_text(right))
        worst = max(worst, similarity)
        print(f"  cos={similarity:.4f}  {left!r} vs {right!r}")
    if worst >= MAX_TEXT_COSINE_BETWEEN_CONCEPTS:
        fail(
            "El encoder de texto no distingue conceptos distintos "
            f"(cos = {worst:.4f} >= {MAX_TEXT_COSINE_BETWEEN_CONCEPTS}). Suele ser "
            "sintoma de una cuantizacion agresiva. Exporta en FP32."
        )
    print(f"  [ok]    texto discriminativo (peor caso {worst:.4f})")

    # 3. Escala y variedad de los embeddings de imagen.
    height, width = vision_session.get_inputs()[0].shape[2:]
    samples = {
        "rojo": Image.new("RGB", (width, height), (220, 40, 40)),
        "verde": Image.new("RGB", (width, height), (40, 160, 40)),
        "azul": Image.new("RGB", (width, height), (40, 40, 220)),
    }
    embeddings = {}
    for label, image in samples.items():
        array = np.asarray(image).astype(np.float32) / 255.0
        array = (array - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array(
            [0.26862954, 0.26130258, 0.27577711]
        )
        embeddings[label] = vision_session.run(
            None, {"pixel_values": array.transpose(2, 0, 1)[None]}
        )[0][0]

    norms = {label: float(np.linalg.norm(vector)) for label, vector in embeddings.items()}
    for label, norm in norms.items():
        print(f"  norma image_embeds ({label}) = {norm:.2f}")
    if max(norms.values()) > MAX_IMAGE_EMBED_NORM:
        fail(
            f"Las normas de los embeddings de imagen son absurdas ({max(norms.values()):.0f}). "
            "El modelo exportado no está devolviendo embeddings proyectados."
        )

    distinct = cosine(embeddings["rojo"], embeddings["verde"])
    if 1.0 - distinct < MIN_IMAGE_EMBED_DISTANCE:
        fail("Todas las imágenes producen el mismo embedding: el encoder de visión no funciona.")
    print(f"  [ok]    visión: embeddings con escala normal (cos rojo/verde = {distinct:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BACKEND_DIR / ".clip_export",
        help="Carpeta temporal donde optimum deja los archivos exportados.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Solo valida los archivos que ya están en app/ml/clip/.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="No copia los archivos exportados a app/ml/clip/ (solo valida).",
    )
    arguments = parser.parse_args()

    if arguments.skip_export:
        paths = (
            CLIP_DIR / "vision_model.onnx",
            CLIP_DIR / "text_model.onnx",
            CLIP_DIR / "tokenizer.json",
        )
        missing = [path for path in paths if not path.exists()]
        if missing:
            fail("Faltan archivos: " + ", ".join(str(path) for path in missing))
    else:
        export_with_optimum(arguments.output_dir)
        paths = (
            (
                arguments.output_dir / "vision_model.onnx",
                arguments.output_dir / "text_model.onnx",
                arguments.output_dir / "tokenizer.json",
            )
            if arguments.skip_install
            else install_models(arguments.output_dir)
        )

    validate_models(*paths)

    print(
        "\nListo. Siguiente paso: conectar la etapa de CLIP en app/ml/inference.py\n"
        "entre el paso 1 (ResNet18) y el paso 2 (clasificador de color y forma),\n"
        "y devolver los productos de taxonomy.CLIP_PENDING_PRODUCE a EXTRA_PRODUCE.\n"
        "No olvides revisar el tamaño del bundle antes de desplegar."
    )


if __name__ == "__main__":
    main()
