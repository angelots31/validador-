"""
Taxonomía de frutas y verduras que maneja el sistema.

Este módulo es la **única fuente de verdad** sobre qué productos conocemos.
Lo usan `inference.py` (para traducir las clases de ImageNet a nombres
legibles), `fallback.py` (clasificador de respaldo) y `vision.py` (para
saber cómo interpretar el color de cada producto).

Por qué existe este archivo (contexto del commit 7):
La versión anterior solo mapeaba 10 frutas y cuando el modelo no acertaba
devolvía directamente la clase cruda de ImageNet. Eso producía respuestas
sin sentido como `{"item": "Rubber Eraser"}` al fotografiar una papaya,
porque **ImageNet no tiene la clase "papaya"** (ni mango, ni guayaba, ni
muchas otras). Ahora la taxonomía distingue dos grupos:

- `IMAGENET_PRODUCE`: productos que sí existen en las 1000 clases de
  ImageNet y por lo tanto puede reconocer el modelo ResNet18.
- `EXTRA_PRODUCE`: productos frecuentes en plaza de mercado que NO existen
  en ImageNet (papaya, mango, guayaba...). Esos los cubre el clasificador
  de respaldo por color/forma de `fallback.py`.
"""

from dataclasses import dataclass

# Categorías. Se devuelven en la API tal cual (inglés, igual que el ejemplo
# del taller: {"item": "Apple", "quality": "Good", "ripeness": "Ripe"}).
FRUIT = "Fruit"
VEGETABLE = "Vegetable"


@dataclass(frozen=True)
class Produce:
    """Un producto reconocido, con su nombre legible y su categoría."""

    name: str
    category: str


def _f(name: str) -> Produce:
    return Produce(name=name, category=FRUIT)


def _v(name: str) -> Produce:
    return Produce(name=name, category=VEGETABLE)


# ---------------------------------------------------------------------------
# 1. Productos que SÍ están en ImageNet (los que puede ver el modelo ONNX).
#
#    La clave es la etiqueta EXACTA como aparece en `imagenet_classes.txt`
#    (en minúscula / tal cual viene el archivo de clases estándar).
# ---------------------------------------------------------------------------
IMAGENET_PRODUCE: dict[str, Produce] = {
    # --- Frutas ---
    "Granny Smith": _f("Apple"),
    "strawberry": _f("Strawberry"),
    "orange": _f("Orange"),
    "lemon": _f("Lemon"),
    "fig": _f("Fig"),
    "pineapple": _f("Pineapple"),
    "banana": _f("Banana"),
    "jackfruit": _f("Jackfruit"),
    "custard apple": _f("Custard Apple"),
    "pomegranate": _f("Pomegranate"),
    # --- Verduras y hortalizas ---
    "head cabbage": _v("Cabbage"),
    "broccoli": _v("Broccoli"),
    "cauliflower": _v("Cauliflower"),
    "zucchini": _v("Zucchini"),
    "spaghetti squash": _v("Spaghetti Squash"),
    "acorn squash": _v("Acorn Squash"),
    "butternut squash": _v("Butternut Squash"),
    "cucumber": _v("Cucumber"),
    "artichoke": _v("Artichoke"),
    "bell pepper": _v("Bell Pepper"),
    "cardoon": _v("Cardoon"),
    "mushroom": _v("Mushroom"),
    "corn": _v("Corn"),
}


# ---------------------------------------------------------------------------
# 2. Productos que NO están en ImageNet y cubre el clasificador de respaldo.
#
#    La lista se eligió pensando en una plaza de mercado colombiana: son los
#    productos que más probablemente aparezcan en una foto y que el modelo
#    por sí solo NO puede nombrar.
# ---------------------------------------------------------------------------
EXTRA_PRODUCE: dict[str, Produce] = {
    "Papaya": _f("Papaya"),
    "Mango": _f("Mango"),
    "Guava": _f("Guava"),
    "Watermelon": _f("Watermelon"),
    "Melon": _f("Melon"),
    "Avocado": _f("Avocado"),
    "Tomato": _f("Tomato"),
    "Kiwi": _f("Kiwi"),
    "Peach": _f("Peach"),
    "Pear": _f("Pear"),
    "Grape": _f("Grape"),
    "Plum": _f("Plum"),
    "Passion Fruit": _f("Passion Fruit"),
    "Cherry": _f("Cherry"),
    "Coconut": _f("Coconut"),
    "Lime": _f("Lime"),
    "Plantain": _f("Plantain"),
    "Carrot": _v("Carrot"),
    "Potato": _v("Potato"),
    "Onion": _v("Onion"),
    "Garlic": _v("Garlic"),
    "Eggplant": _v("Eggplant"),
    "Beet": _v("Beet"),
    "Lettuce": _v("Lettuce"),
    "Spinach": _v("Spinach"),
    "Chili Pepper": _v("Chili Pepper"),
    "Cassava": _v("Cassava"),
    "Green Bean": _v("Green Bean"),
    "Sweet Potato": _v("Sweet Potato"),
    "Radish": _v("Radish"),
}

# NOTA: hay una prueba (tests/test_ml.py) que exige que EXTRA_PRODUCE y
# fallback.PROFILES coincidan 1 a 1. Si agregas un producto aquí, agrega su
# perfil de color/forma en fallback.py, o no será alcanzable por ninguna vía.


# ---------------------------------------------------------------------------
# 2b. Ampliación prevista para CLIP zero-shot (todavía NO alcanzable).
#
#     Estos 36 productos se eligieron para una etapa de CLIP zero-shot con la
#     que cubrir cientos de productos sin calibrar perfiles a mano. El
#     tokenizador ya está en `clip_tokenizer.py`, pero **los modelos ONNX de
#     `app/ml/clip/` resultaron inservibles**: se generaron con
#     `onnxruntime.quantization.quantize_dynamic`, que descompone las
#     LayerNormalization y corrompe las proyecciones (el encoder de texto
#     colapsa: dos prompts distintos quedan a 0.996 de distancia coseno).
#     Ver `scripts/export_clip_onnx.py` para re-exportarlos correctamente.
#
#     Por eso estos productos NO están en EXTRA_PRODUCE: `ALL_PRODUCE_NAMES`
#     debe contener exactamente lo que el sistema puede devolver hoy. Si
#     estuvieran ahí, la taxonomía prometería productos que ninguna ruta
#     puede alcanzar. Las reglas de color y madurez de más abajo
#     (GREEN_IS_NORMAL, GREEN_MEANS_UNRIPE, NATURALLY_DARK,
#     RIPENESS_NOT_COLOR_BASED) ya los mencionan, listas para cuando CLIP
#     esté re-exportado.
# ---------------------------------------------------------------------------
CLIP_PENDING_PRODUCE: dict[str, Produce] = {
    # --- Frutas ---
    "Lulo": _f("Lulo"),
    "Soursop": _f("Soursop"),
    "Tamarillo": _f("Tamarillo"),
    "Granadilla": _f("Granadilla"),
    "Curuba": _f("Curuba"),
    "Dragon Fruit": _f("Dragon Fruit"),
    "Cantaloupe": _f("Cantaloupe"),
    "Tangerine": _f("Tangerine"),
    "Grapefruit": _f("Grapefruit"),
    "Nectarine": _f("Nectarine"),
    "Apricot": _f("Apricot"),
    "Persimmon": _f("Persimmon"),
    "Starfruit": _f("Starfruit"),
    "Tamarind": _f("Tamarind"),
    "Blackberry": _f("Blackberry"),
    "Blueberry": _f("Blueberry"),
    "Raspberry": _f("Raspberry"),
    "Cape Gooseberry": _f("Cape Gooseberry"),
    # --- Hortalizas ---
    "Pumpkin": _v("Pumpkin"),
    "Chayote": _v("Chayote"),
    "Celery": _v("Celery"),
    "Leek": _v("Leek"),
    "Green Onion": _v("Green Onion"),
    "Red Cabbage": _v("Red Cabbage"),
    "Kale": _v("Kale"),
    "Chard": _v("Chard"),
    "Arugula": _v("Arugula"),
    "Parsley": _v("Parsley"),
    "Cilantro": _v("Cilantro"),
    "Asparagus": _v("Asparagus"),
    "Peas": _v("Peas"),
    "Okra": _v("Okra"),
    "Turnip": _v("Turnip"),
    "Ginger": _v("Ginger"),
    "Yam": _v("Yam"),
    "Taro": _v("Taro"),
}

# Nombre legible -> Produce, para buscar por nombre en cualquier dirección.
PRODUCE_BY_NAME: dict[str, Produce] = {
    **{p.name: p for p in IMAGENET_PRODUCE.values()},
    **{p.name: p for p in EXTRA_PRODUCE.values()},
}

ALL_PRODUCE_NAMES: tuple[str, ...] = tuple(sorted(PRODUCE_BY_NAME))

#: Productos de la ampliación de CLIP. Hoy NINGUNA ruta del sistema puede
#: devolverlos (ver la nota de `CLIP_PENDING_PRODUCE`); se exponen aquí para
#: que la documentación y las pruebas puedan referirse a ellos.
CLIP_PENDING_NAMES: tuple[str, ...] = tuple(
    sorted(produce.name for produce in CLIP_PENDING_PRODUCE.values())
)


# ---------------------------------------------------------------------------
# 3. Reglas de madurez por producto.
#
#    El mismo color significa cosas distintas según el producto: un tomate
#    verde está inmaduro, pero un pepino verde está en su punto. Sin esta
#    distinción el heurístico marcaba "Unripe" a medio mercado.
# ---------------------------------------------------------------------------

# Productos en los que el color verde indica claramente "no maduro".
GREEN_MEANS_UNRIPE: frozenset[str] = frozenset(
    {
        "Banana",
        "Plantain",
        "Mango",
        "Papaya",
        "Tomato",
        "Peach",
        "Pear",
        "Avocado",
        "Pineapple",
        "Fig",
        "Guava",
        "Passion Fruit",
        "Plum",
        "Cherry",
        "Custard Apple",
        "Jackfruit",
        "Kiwi",
        "Granadilla",
        "Curuba",
        "Tamarillo",
        "Tangerine",
        "Nectarine",
        "Apricot",
        "Persimmon",
        "Starfruit",
        "Soursop",
    }
)

# Productos que son verdes cuando están en su punto: acá el verde NO es
# señal de inmadurez y la madurez se juzga solo por daño/descomposición.
GREEN_IS_NORMAL: frozenset[str] = frozenset(
    {
        "Cucumber",
        "Zucchini",
        "Broccoli",
        "Cabbage",
        "Lettuce",
        "Spinach",
        "Artichoke",
        "Cardoon",
        "Bell Pepper",
        "Chili Pepper",
        "Green Bean",
        "Lime",
        # Sandía y melón son verdes/tonos de cáscara por fuera sin importar su
        # madurez: la cáscara no dice nada sobre el punto de la fruta.
        "Watermelon",
        "Melon",
        "Kale",
        "Chard",
        "Arugula",
        "Parsley",
        "Cilantro",
        "Celery",
        "Leek",
        "Green Onion",
        "Asparagus",
        "Peas",
        "Okra",
        "Chayote",
        "Red Cabbage",
    }
)

# Productos donde la madurez simplemente no se juzga por el color: son
# tubérculos, raíces, bulbos u hongos. Si no tienen daño, están listos.
RIPENESS_NOT_COLOR_BASED: frozenset[str] = frozenset(
    {
        "Potato",
        "Sweet Potato",
        "Cassava",
        "Carrot",
        "Beet",
        "Radish",
        "Onion",
        "Garlic",
        "Mushroom",
        "Corn",
        "Cauliflower",
        "Artichoke",
        "Cardoon",
        "Coconut",
        "Ginger",
        "Turnip",
        "Yam",
        "Taro",
    }
)

# Productos de piel/tema oscuro: el heurístico no debe castigarlos por ser
# oscuros (una berenjena o unas uvas moradas son oscuras por naturaleza).
NATURALLY_DARK: frozenset[str] = frozenset(
    {
        "Eggplant",
        "Grape",
        "Plum",
        "Beet",
        "Avocado",
        "Fig",
        "Pomegranate",
        "Cherry",
        "Cassava",
        "Blackberry",
        "Blueberry",
        "Raspberry",
        "Tamarind",
    }
)


def is_known(name: str) -> bool:
    """True si el nombre corresponde a un producto de nuestra taxonomía."""
    return name in PRODUCE_BY_NAME


def category_of(name: str) -> str:
    """Categoría ('Fruit' / 'Vegetable') de un producto, o 'Unknown'."""
    produce = PRODUCE_BY_NAME.get(name)
    return produce.category if produce else "Unknown"
