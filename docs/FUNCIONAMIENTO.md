# Funcionamiento completo del proyecto

**Validador de Calidad de Frutas y Verduras** — Taller 4 (SENA, Análisis y
Desarrollo de Software).

Este documento explica **qué hace el sistema de punta a punta** y **qué hace
cada archivo del repositorio**. Está escrito sobre el estado real del código
(no sobre el ideal): cuando algo está a medias o no sirve, lo dice.

---

## Índice

1. [Qué problema resuelve](#1-qué-problema-resuelve)
2. [Arquitectura general](#2-arquitectura-general)
3. [El viaje completo de una foto](#3-el-viaje-completo-de-una-foto)
4. [Cómo identifica el producto (la cascada)](#4-cómo-identifica-el-producto-la-cascada)
5. [Cómo estima calidad y madurez](#5-cómo-estima-calidad-y-madurez)
6. [Catálogo de archivos](#6-catálogo-de-archivos)
7. [Autenticación y sesiones](#7-autenticación-y-sesiones)
8. [Base de datos y despliegue](#8-base-de-datos-y-despliegue)
9. [Pruebas automáticas](#9-pruebas-automáticas)
10. [Estado de CLIP](#10-estado-de-clip)
11. [Limitaciones conocidas](#11-limitaciones-conocidas)
12. [Chuleta de comandos](#12-chuleta-de-comandos)

---

## 1. Qué problema resuelve

Una persona quiere saber **qué fruta o verdura** tiene en la mano, **en qué
estado está** (calidad) y **si ya maduró**. En una plaza de mercado eso se
resuelve a ojo y con experiencia. El proyecto automatiza esas tres preguntas con
una foto:

| Pregunta | Campo de la respuesta | Valores posibles |
|---|---|---|
| ¿Qué es? | `item` / `category` | nombre del producto · `Fruit` / `Vegetable` / `Unknown` |
| ¿Está bueno? | `quality` | `Good` / `Regular` / `Bad` |
| ¿Está maduro? | `ripeness` | `Unripe` / `Ripe` / `Overripe` |
| ¿Cómo lo supiste? | `method`, `candidates`, `notes`, `diagnostics` | campos añadidos que explican la decisión |

El sistema **no** es “una red neuronal que adivina”. Es una **cascada de
decisión** en la que cada etapa dice si actuó o no, y la respuesta siempre
incluye la explicación en texto.

---

## 2. Arquitectura general

Son **dos aplicaciones** que se despliegan por separado y hablan por HTTP.

```
┌───────────────────────────┐
│        NAVEGADOR          │
│  camera.js + visual.js    │
│  getUserMedia (cámara)    │
└────────────┬──────────────┘
             │ 1. POST /analizar/  (FormData con la foto + cookie de sesión)
             │    GET  /historial/
             ▼
┌───────────────────────────┐
│   FRONTEND — DJANGO       │
│   (Serverless Function)   │
│  · No tiene modelo de     │
│    usuario ni base de     │
│    datos para funcionar   │
│  · Guarda el JWT en una   │
│    cookie firmada         │
└────────────┬──────────────┘
             │ 2. POST /api/v1/inspect-fruit   (header Authorization: Bearer)
             │    GET  /api/v1/inspection-history
             ▼
┌───────────────────────────┐
│   BACKEND — FASTAPI       │
│   (Serverless Function)   │
│  · Valida el JWT          │
│  · Ejecuta el análisis    │
│  · Persiste el resultado  │
└────────────┬──────────────┘
             │ 3. Análisis
             ▼
┌───────────────────────────┐
│   PIPELINE DE VISIÓN      │
│  ResNet18 (ONNX) +        │
│  visión clásica numpy     │
└───────────────────────────┘
```

**Por qué está partido en dos:** el enunciado pide Django como cliente y FastAPI
como servidor de la API. Eso además permite que el análisis (que consume CPU y
memoria) viva en una función aparte, y que el frontend no sepa nada de modelos
de machine learning: solo pinta lo que la API devuelve.

**Por qué el navegador nunca habla directo con FastAPI:** el JWT vive en la
sesión del frontend. Así el JavaScript de la página no tiene que manejar
credenciales, y el navegador no necesita CORS. Es Django quien reenvía el token.

---

## 3. El viaje completo de una foto

Este es el camino exacto, archivo por archivo.

| # | Paso | Archivo que actúa |
|---|---|---|
| 1 | El navegador pide permiso de cámara con `getUserMedia`. | `static/js/camera.js` |
| 2 | El visor ajusta su marco a la proporción real del video. | `camera.js` (`applyViewportRatio`) |
| 3 | Al capturar, se calcula **qué región del sensor corresponde a la guía** de encuadre (se deshace el escalado de `object-fit: cover`). | `camera.js` (`guideRegionInVideo`) |
| 4 | La foto se recorta a esa región, se limita a 1024 px de lado mayor y se comprime a JPEG calidad 0.9. | `camera.js` (`capturePhoto`) |
| 5 | Se envía con `fetch` a `/analizar/` como `FormData`, con el token CSRF. | `camera.js` (`analyze`) |
| 6 | La vista valida que haya foto y que la sesión tenga token. | `frontend/inspector/views.py` (`analyze_view`) |
| 7 | Se reenvía la foto a FastAPI con el `Authorization: Bearer <JWT>`. | `frontend/inspector/api_client.py` |
| 8 | FastAPI recibe el multipart y valida formato y tamaño. | `backend/app/routers/inspect.py` |
| 9 | Se identifica al usuario dueño del token. | `backend/app/core/security.py` |
| 10 | Se abre la imagen, se corrige su rotación EXIF y se pasa a RGB. | `inspect.py` (`_load_image`) |
| 11 | **Un solo** análisis de características de la imagen (color y forma). | `backend/app/ml/features.py` |
| 12 | Se identifica el producto (cascada de 4 pasos). | `backend/app/ml/inference.py` + `fallback.py` + `taxonomy.py` |
| 13 | Se estiman calidad y madurez **según el producto** identificado. | `backend/app/ml/vision.py` |
| 14 | Se guarda la inspección y se arma la respuesta con explicaciones. | `inspect.py` |
| 15 | La respuesta vuelve a Django, que la reenvía al navegador. | `api_client.py` → `views.py` |
| 16 | El JavaScript pinta el resultado: nombre, medidor de confianza, barras, candidatos y notas. | `camera.js` (`renderResult`) + `static/js/visual.js` |
| 17 | Se recarga el historial y las estadísticas del encabezado. | `camera.js` (`loadHistory`, `renderStats`) |

---

## 4. Cómo identifica el producto (la cascada)

Todo el mérito de este diseño está en que **la API dice siempre qué camino
usó**, en el campo `method`.

### Paso 0 — Qué sabe reconocer el sistema

| Grupo | Cuántos | Quién lo reconoce |
|---|---|---|
| Productos que existen en ImageNet | **23** | ResNet18 |
| Productos que ImageNet **no** tiene | **30** | clasificador de respaldo por color/forma |
| **Total alcanzable** | **53** | — |
| Preparados para CLIP (no alcanzables hoy) | 36 | nadie todavía (ver §10) |

Los 23 de ImageNet: manzana, fresa, naranja, limón, higo, piña, banano, jaca,
anón, granada, repollo, brócoli, coliflor, calabacín, ahuyama espagueti, ahuyama
bellota, ahuyama mantequilla, pepino, alcachofa, pimentón, cardo, champiñón y
mazorca.

Los 30 del respaldo: papaya, mango, guayaba, sandía, melón, aguacate, tomate,
kiwi, durazno, pera, uva, ciruela, maracuyá, cereza, coco, lima, plátano,
zanahoria, papa, cebolla, ajo, berenjena, remolacha, lechuga, espinaca, ají,
yuca, habichuela, batata y rábano.

> Hay una prueba (`test_every_extra_produce_has_a_fallback_profile`) que exige
> que esos 30 coincidan **1 a 1** con los perfiles del clasificador de respaldo.
> Si alguien agrega un producto al catálogo y no le crea su perfil, la prueba
> falla: sería un producto que el sistema nunca podría devolver.

### Paso 1 — El modelo preentrenado

`ResNet18` preentrenado en ImageNet, exportado a **ONNX** y ejecutado con
`onnxruntime` (no con PyTorch: pesa ~70 MB contra varios cientos de MB, y eso
importa para caber en una Serverless Function).

Se aplica **TTA (test-time augmentation)**: se promedian 4 pasadas sobre la
misma foto — recorte central y foto completa, cada una con su espejo horizontal.
Cuesta 4 inferencias de 224×224 en CPU (barato) y mejora los casos en que el
producto no está centrado o la cámara lo tomó en horizontal.

Las probabilidades de las clases que mapean al **mismo** producto se suman. Si
algún producto de ImageNet alcanza **p ≥ 0.15**, ese es el resultado y
`method = "resnet18-imagenet"`.

### Paso 2 — El clasificador de respaldo (la parte que arregló el bug)

**El bug original:** ImageNet no tiene clase para papaya, mango, guayaba, sandía,
aguacate, tomate ni zanahoria. En la primera versión, cuando el modelo no
encontraba ninguna fruta, la API devolvía su clase número 1 tal cual. El
resultado, al fotografiar una papaya, era literalmente:

```json
{ "item": "Rubber Eraser", "quality": "Bad", "ripeness": "Overripe" }
```

**La solución:** un clasificador *nearest-centroid* sobre las características de
color y forma. Cada producto tiene un **perfil**: un valor típico y una
tolerancia por característica (tono, saturación, brillo, elongación, proporción
de objeto…), más un peso. Se compara la foto contra los 30 perfiles y se elige el
que mejor encaja.

Cómo se calcula el encaje: por cada característica se usa una **campana
gaussiana** sobre la diferencia al centro del perfil:

```
encaje = exp( -0.5 · (|valor - centro| / tolerancia)² )
```

Para el tono la distancia es **circular** (359° y 1° están a 2°, no a 358°). El
puntaje final es el promedio de los encajes **ponderado** por el peso de cada
característica.

Se acepta si el puntaje es **≥ 0.52** y **supera** la pista del modelo. Entonces
`method = "color-shape-heuristics"`.

**Honestidad:** esto **no** es una red neuronal entrenada. Es visión clásica con
umbrales calibrados a mano. La respuesta lo dice explícitamente en `method`, para
que nadie lo confunda con el modelo.

Además hay un castigo por mala segmentación: si el fondo no se pudo separar bien,
el puntaje se multiplica por `0.55 + 0.45 · confianza_de_segmentación`, para no
reportar una confianza falsa.

### Paso 3 — Pista débil

Si el respaldo no convenció pero el modelo dio al menos **p ≥ 0.02**, se usa esa
aproximación y se avisa en `notes` que la confianza es baja.

### Paso 4 — Nada convincente

Se devuelve `item = "Unknown"` y `method = "unrecognized"`. Es preferible decir
“no sé” antes que inventar un nombre. Los candidatos del respaldo se adjuntan
igual, para que el usuario vea qué se parecía más.

### Cómo se miden el color y la forma (`features.py`)

La imagen se reduce a **160×160** (suficiente para estadísticas y ~40 veces menos
píxeles que una foto de 1024) y todo se calcula vectorizado con numpy, sin OpenCV
ni scipy.

1. **Segmentación.** Se estima el color de fondo con la **mediana del borde** de
   la imagen, y se marca como “producto” todo píxel que esté a más de 0.20 de
   distancia de ese color, o que tenga saturación muy alta (> 0.45). Luego una
   apertura morfológica (erosión + dilatación) quita ruido.
2. **Región principal.** Sin scipy no hay etiquetado de componentes conectados,
   así que se hace **crecimiento de región**: se siembra en el píxel con mayor
   densidad local y se dilata dentro de la máscara hasta que deja de crecer
   (máximo 80 iteraciones). Esto reemplazó a un recorte por densidad de
   filas/columnas que fallaba con productos alargados y delgados (una zanahoria
   o un plátano en diagonal quedaban recortados a casi nada).
3. **¿La máscara es confiable?** El rectángulo envolvente del producto debe
   cubrir al menos el 6 % del encuadre y la máscara debe llenar al menos el 12 %
   de ese rectángulo. Se mide sobre el rectángulo y no sobre el área porque un
   producto delgado ocupa poco del encuadre aunque esté perfectamente
   segmentado.
4. **Color.** Conversión a HSV propia (vectorizada) y proporciones de familias de
   color: verde, amarillo, naranja, rojo, morado, marrón y claro. Las
   proporciones se cuentan **solo sobre píxeles con saturación > 0.18**: un
   píxel gris no debe votar como “amarillo”. Además se usa la **media circular**
   del tono (el tono es un ángulo).
5. **Forma.** Qué tan alargado es (`elongation`), qué tan lleno está su
   rectángulo (`solidity`) y qué tan circular es (`compactness`, con la fórmula
   `4π·área / perímetro²`).
6. **Manchas relativas.** Una “mancha” es un píxel claramente más oscuro **que la
   mediana de ese mismo producto** (`brillo < max(0.10, mediana·0.62)`). Antes el
   umbral era absoluto y por eso una berenjena, unas uvas o un aguacate Hass
   salían siempre como `Bad` / `Overripe` solo por ser oscuros.
7. **Confianza de segmentación.** Se basa en cuánto ruido tiene el borde de la
   imagen (`1 − ruido·3.5`, acotado a 0–1) y se castiga si la máscara no fue
   utilizable.

---

## 5. Cómo estima calidad y madurez

Archivo: `backend/app/ml/vision.py`. Todo con **umbrales relativos al propio
producto**, y el color se **interpreta según el producto**.

### Calidad

Se evalúa en orden; el primer criterio que se cumple decide:

| Condición | Resultado | Mensaje en `notes` |
|---|---|---|
| Manchas > **30 %** de la superficie | `Bad` | “superficie dañada o muy madura” |
| Zonas marrones > **45 %** | `Bad` | “golpes, oxidación o descomposición” |
| Oscuridad absoluta > **40 %** y el producto no es naturalmente oscuro | `Bad` | “poco brillo para un producto de este tipo” |
| Manchas > **12 %** | `Regular` | “aceptable, pero no primera calidad” |
| Zonas marrones > **25 %** | `Regular` | “algunas zonas marrones” |
| Ninguna de las anteriores | `Good` | “superficie uniforme y sin daños visibles” |

### Madurez

| Orden | Regla | Resultado |
|---|---|---|
| 1 | Manchas > **34 %** o marrón > **40 %** | `Overripe` (el daño extenso siempre gana) |
| 2 | Producto de la lista “la madurez no se juzga por color” (tubérculos, raíces, bulbos, hongos) | `Ripe` |
| 3 | Producto de la lista “son verdes en su punto” (pepino, brócoli, lechuga, lima, sandía…) | `Ripe` |
| 4 | Producto de la lista “el verde significa inmaduro” (banano, mango, papaya, tomate, aguacate…): si el tono medio cae en la banda verde (70°–170°) y hay ≥ 40 % de superficie verde | `Unripe`; si no, `Ripe` |
| 5 | Producto desconocido: criterio general conservador | `Unripe` si hay mucho verde; si no, `Ripe` |

La clave es que **el mismo color significa cosas distintas**: un tomate verde
está inmaduro, pero un pepino verde está en su punto. Eso lo decide
`taxonomy.py` con cuatro listas (`GREEN_MEANS_UNRIPE`, `GREEN_IS_NORMAL`,
`RIPENESS_NOT_COLOR_BASED`, `NATURALLY_DARK`).

> **Para la sustentación:** esta parte **no es machine learning**. ImageNet no
> tiene etiquetas de calidad ni de madurez, y entrenar eso pediría un dataset
> etiquetado de frutas en distintos estados que no existe dentro del alcance del
> taller.

---

## 6. Catálogo de archivos

Leyenda: **⚙️** = se ejecuta desplegado · **🧪** = pruebas · **📄** = config o
documentación.

### Raíz del repositorio

| Archivo | Qué hace |
|---|---|
| `README.md` | Documentación principal: qué es, stack, endpoints, despliegue, limitaciones. |
| `.gitignore` | Qué no se versiona (entornos virtuales, bases de datos locales, `.env`, builds). |
| `docs/FUNCIONAMIENTO.md` | Este documento. |
| `docs/EXPOSICION.md` | Guion de la exposición de ~20 minutos para dos personas. |

### `backend/` — API FastAPI

| Archivo | Qué hace |
|---|---|
| ⚙️ `api/index.py` | **Entry point para Vercel.** Mete la carpeta `backend` en el `sys.path` e importa `app` desde `main.py`. Vercel solo sabe ejecutar archivos dentro de `api/`, así que este es el puente. |
| ⚙️ `main.py` | **La aplicación FastAPI.** Crea las tablas, registra los routers, configura CORS y gzip, define el manejo de errores (422 de Pydantic en una línea legible, y un *catch-all* que nunca devuelve un traceback) y el endpoint raíz `/`. También define la documentación de Swagger por secciones. |
| 📄 `vercel.json` | Configuración del despliegue: construye `api/index.py` con `@vercel/python`, le da 30 s de duración máxima y le dice que incluya la carpeta `app/**` (donde van el modelo y los módulos) en el paquete de la función. Las rutas mandan todo al entry point. |
| 📄 `requirements.txt` | Dependencias: FastAPI, Uvicorn, Pydantic v2, SQLAlchemy 2, driver Postgres (`psycopg`), JWT (`python-jose`), `bcrypt`, `python-multipart`, `onnxruntime`, `numpy`, `pillow`. **No** incluye PyTorch a propósito. |
| 📄 `.env.example` | Plantilla de variables de entorno: `DATABASE_URL` (opcional), `JWT_SECRET_KEY`, `JWT_ALGORITHM`, `JWT_EXPIRE_MINUTES`, `CORS_ORIGINS`. |
| 📄 `.gitignore` / `.vercelignore` | Primero: qué no se versiona. Segundo: qué **no se sube a Vercel** (el `venv/`, que pesa cientos de MB, y los modelos CLIP inservibles). |
| 📄 `.python-version` | Fija **Python 3.12** para el despliegue, que es la versión con la que se probaron las dependencias binarias. |

#### `backend/app/core/` — el núcleo

| Archivo | Qué hace |
|---|---|
| 📄 `__init__.py` | Archivo vacío: marca la carpeta como paquete de Python. |
| ⚙️ `config.py` | **Configuración central.** Define la clase `Settings` (Pydantic) que lee las variables de entorno y el `.env`: nombre del proyecto, prefijo `/api/v1`, orígenes de CORS (acepta lista separada por comas), la cadena `DATABASE_URL` y los parámetros del JWT. Al final instancia `settings`, que importan todos los demás módulos. |
| ⚙️ `database.py` | **La conexión a la base de datos.** Elige Postgres si existe `DATABASE_URL` (normalizando `postgres://` → `postgresql+psycopg://`), o SQLite local si no; en Vercel, y como último recurso, apunta al directorio temporal (que es lo único escribible). Crea el `engine` (con `NullPool` en Postgres, porque en serverless las conexiones cacheadas quedan muertas), la `SessionLocal` y la `Base` de SQLAlchemy. Expone `get_db()`, la dependencia que FastAPI usa para inyectar una sesión. También tiene `apply_sqlite_migrations()`, una migración mínima e idempotente que agrega columnas nuevas a un SQLite viejo. |
| ⚙️ `security.py` | **Seguridad.** Hashea contraseñas con **bcrypt** directamente (el comentario explica por qué no usa `passlib`: las versiones 4.x/5.x de bcrypt rompen su detección de versión). Valida el límite de 72 bytes de bcrypt para dar un error claro. Crea los JWT (`create_access_token`) y expone `get_current_user()`, la dependencia que exige un token válido y devuelve el usuario dueño. |

#### `backend/app/models/` — las tablas

| Archivo | Qué hace |
|---|---|
| 📄 `__init__.py` | Archivo vacío (paquete). |
| ⚙️ `user.py` | Modelo SQLAlchemy `User` → tabla `users`: `id`, `username` (único, indexado), `hashed_password`, `created_at`. |
| ⚙️ `inspection.py` | Modelo SQLAlchemy `Inspection` → tabla `inspections`: a qué usuario pertenece, qué producto era, su categoría, calidad, madurez, confianza, con qué método se identificó y cuándo. Tiene la relación con `User`. |

#### `backend/app/schemas/` — el contrato de la API

| Archivo | Qué hace |
|---|---|
| 📄 `__init__.py` | Archivo vacío (paquete). |
| ⚙️ `common.py` | `ErrorResponse`: la forma estándar de los errores (`{"detail": "..."}`), la misma que genera FastAPI. |
| ⚙️ `user.py` | `UserCreate` (usuario + contraseña con validaciones de longitud), `UserOut` (datos públicos: **nunca** la contraseña) y `Token` (el JWT que devuelve el login). |
| ⚙️ `inspection.py` | **El contrato más importante del proyecto.** Define con tipos `Literal` los valores permitidos de `quality`, `ripeness`, `category` y `method` (así Swagger los muestra como listas cerradas y el backend no puede devolver un valor inesperado). Define `Candidate`, `ColorBreakdown`, `InspectionDiagnostics`, `InspectionResult` (con un ejemplo completo para Swagger) y `InspectionHistoryItem`. Este último tiene un serializador que **marca explícitamente las fechas como UTC**: SQLite/Postgres las devuelven sin zona y sin eso, en Colombia (UTC−5), el historial aparecería cinco horas en el futuro. |

#### `backend/app/routers/` — las rutas

| Archivo | Qué hace |
|---|---|
| 📄 `__init__.py` | Archivo vacío (paquete). |
| ⚙️ `health.py` | `GET /health`: devuelve `{"status": "ok"}`. Sirve para monitoreo y para comprobar que el despliegue está vivo. |
| ⚙️ `auth.py` | Registro (`POST /auth/register`, rechaza usuarios repetidos), login (`POST /auth/login`, recibe formulario estilo OAuth2 y devuelve el JWT) y `GET /auth/me` (endpoint protegido de ejemplo: devuelve el usuario dueño del token). |
| ⚙️ `inspect.py` | **El corazón de la API.** `POST /inspect-fruit`: valida el tipo MIME (JPG/PNG/WEBP), el tamaño (máx. 8 MB) y el lado mínimo (48 px); corrige la rotación EXIF de las fotos de celular; llama **una sola vez** a la extracción de características y la reutiliza para identificar, medir calidad y armar el diagnóstico; guarda la inspección; y devuelve el resultado con candidatos y notas. También `GET /inspection-history`, que devuelve las inspecciones del usuario autenticado, de la más reciente a la más antigua. |

#### `backend/app/ml/` — el pipeline de visión

| Archivo | Qué hace |
|---|---|
| 📄 `__init__.py` | Archivo vacío (paquete). |
| ⚙️ `taxonomy.py` | **La única fuente de verdad sobre qué productos conocemos.** Define `IMAGENET_PRODUCE` (23 productos que el modelo sí puede ver, con la etiqueta exacta de ImageNet), `EXTRA_PRODUCE` (30 que ImageNet no tiene y cubre el respaldo), `CLIP_PENDING_PRODUCE` (36 preparados para CLIP, **fuera** del catálogo activo), `PRODUCE_BY_NAME`, `ALL_PRODUCE_NAMES` y las cuatro listas que le dicen a `vision.py` cómo interpretar el color de cada producto. |
| ⚙️ `features.py` | **Extracción de características.** Segmenta el producto del fondo, calcula color (HSV propio, media circular del tono, proporciones ponderadas por saturación) y forma (elongación, solidez, compacidad), más las manchas relativas y una confianza de segmentación. Todo vectorizado con numpy, sin OpenCV ni scipy. Ver §4 para el detalle. |
| ⚙️ `fallback.py` | **El clasificador de respaldo.** Tiene los 30 perfiles (centro, tolerancia y peso por característica) y el cálculo de encaje con campana gaussiana y distancia circular para el tono. Devuelve el ganador, el ranking y las señales principales. Si la foto casi no tiene color (saturación < 0.15), devuelve vacío: sin información de color, no inventa. |
| ⚙️ `inference.py` | **La cascada de decisión y el modelo.** Carga el ResNet18 ONNX una sola vez (patrón *singleton*), preprocesa con la normalización estándar de ImageNet, aplica TTA de 4 vistas, suma las probabilidades por producto y ejecuta los 4 pasos: modelo → respaldo → pista débil → `Unknown`. Es el único módulo que importa `onnxruntime`. |
| ⚙️ `vision.py` | **Calidad y madurez.** Los umbrales relativos de §5 y las notas que explican cada decisión. |
| 📄 `imagenet_classes.txt` | Las 1000 etiquetas de ImageNet, en orden (la línea *i* corresponde a la salida *i* del modelo). |
| 📄 `resnet18.onnx` | El modelo preentrenado (~45 MB) ya exportado a ONNX. |
| ⚙️ `clip_tokenizer.py` | Tokenizador de **CLIP** reimplementado en Python puro (~195 líneas): normalización NFC, minúsculas, la expresión regular oficial de CLIP, BPE byte-level con sufijo de fin de palabra, envoltura BOS/EOS y relleno hasta 77 posiciones. Evita la dependencia de `transformers`, que pesa cientos de MB. **Funciona y está probado** (§10). |
| 📄 `clip/tokenizer.json` | El vocabulario de CLIP que carga el tokenizador de arriba. |
| 📄 `clip/vision_model.onnx` | Modelo visual de CLIP. **Inservible** (cuantizado con `quantize_dynamic`); excluido del despliegue. |
| 📄 `clip/text_model.onnx` | Modelo de texto de CLIP. **Inservible** por la misma razón. |

#### `backend/scripts/`

| Archivo | Qué hace |
|---|---|
| 📄 `export_clip_onnx.py` | **Re-exporta y valida los modelos de CLIP.** Llama a `optimum` para exportarlos en FP32, los copia a `app/ml/clip/` y después **comprueba que sirvan**: que el tokenizador reproduzca los ids canónicos de CLIP, que dos conceptos distintos no queden a menos de 0.95 de distancia coseno en el espacio de texto, y que los embeddings de imagen tengan una escala normal. Si alguna falla, aborta sin instalar nada. Es la verificación que faltaba y que habría detectado el problema. |

#### `backend/tests/`

| Archivo | Qué hace |
|---|---|
| 📄 `__init__.py` | Archivo vacío (paquete). |
| 🧪 `test_ml.py` | Pruebas del pipeline: consistencia de la taxonomía (incluida la invariante de que cada producto de plaza tenga su perfil), extracción de características (producto detectado, elongación en objetos delgados, manchas relativas), calidad y madurez (un pepino verde no es inmaduro, un producto oscuro sin manchas es bueno), el clasificador de respaldo y el **pipeline completo**, con la regresión del bug: la API nunca puede devolver una etiqueta que no sea de la taxonomía. |
| 🧪 `test_database.py` | Pruebas de la configuración de base de datos: que la cadena de Postgres de los proveedores se traduzca al driver instalado, que sin `DATABASE_URL` se use SQLite local y que el fallback de Vercel apunte al directorio temporal. |
| 🧪 `test_clip_tokenizer.py` | Pruebas del tokenizador de CLIP: reproduce los ids canónicos de la librería, normaliza mayúsculas y espacios, envuelve con BOS/EOS, rellena hasta 77, trunca textos largos y cachea la carga. |

### `frontend/` — cliente Django

| Archivo | Qué hace |
|---|---|
| ⚙️ `manage.py` | Utilidad de línea de comandos de Django. Se usa para migrar, correr el servidor de desarrollo y ejecutar las pruebas. |
| ⚙️ `api/index.py` | **Entry point para Vercel.** Mete la carpeta `frontend` en el `sys.path`, define `DJANGO_SETTINGS_MODULE` e importa `application` desde `config/wsgi.py`. `application` es exactamente el nombre que Vercel busca en un proyecto Django. |
| 📄 `vercel.json` | Construye la función con `@vercel/python` durante 30 s como máximo, e incluye `config/**`, `inspector/**`, las plantillas y `requirements.txt`. La capa `/static/(.*)` la sirve `@vercel/static` (por eso **no** hace falta `collectstatic`). |
| 📄 `requirements.txt` | Django (5 o 6), `python-dotenv`, `requests` y el driver `psycopg` (solo por si se decide usar Postgres). |
| 📄 `.env.example` | Plantilla: `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `DJANGO_ALLOWED_HOSTS`, `DJANGO_CSRF_TRUSTED_ORIGINS`, `FASTAPI_BASE_URL` y `DATABASE_URL` (opcional). |
| 📄 `.gitignore` / `.vercelignore` | Qué no se versiona y qué no se sube a Vercel (`venv/`, `db.sqlite3`). |
| 📄 `.python-version` | Fija **Python 3.12** para el despliegue. |

#### `frontend/config/` — configuración del proyecto Django

| Archivo | Qué hace |
|---|---|
| 📄 `__init__.py` | Archivo vacío (paquete). |
| ⚙️ `settings.py` | **La configuración completa.** Carga el `.env`; define `SECRET_KEY`, `DEBUG`, hosts y orígenes CSRF (agregando automáticamente las variables de Vercel y el comodín `.vercel.app`); define `SECURE_PROXY_SSL_HEADER` para reconocer que Vercel terminó TLS; elige la base de datos (Postgres si hay `DATABASE_URL`, si no SQLite); fija `SESSION_ENGINE` en **cookie firmada**; marca las cookies como `Secure` solo en Vercel; lista apps, middleware, plantillas y ajustes de idioma (`es-co`) y zona horaria (`America/Bogota`). Ver §7 y §8. |
| ⚙️ `urls.py` | Tabla de rutas raíz: `/admin/` (el admin de Django) y todo lo demás a la aplicación `inspector`. |
| ⚙️ `wsgi.py` | Expone `application`, el objeto que Vercel ejecuta. Es lo que hace que Django corra como función serverless. |
| 📄 `asgi.py` | Equivalente asíncrono. No se usa en Vercel, pero viene con el proyecto Django estándar. |

#### `frontend/inspector/` — la aplicación

| Archivo | Qué hace |
|---|---|
| 📄 `__init__.py` | Archivo vacío (paquete). |
| ⚙️ `apps.py` | Registra la aplicación `inspector` ante Django. |
| 📄 `admin.py` | Lugar reservado para registrar modelos en el admin. Está vacío: la app no tiene modelos propios. |
| 📄 `models.py` | Lugar reservado para modelos. Está vacío a propósito: **Django no tiene tabla de usuarios ni de inspecciones**; la fuente de verdad es FastAPI. |
| 📄 `migrations/__init__.py` | Archivo vacío: sin modelos propios no hay migraciones que generar. |
| ⚙️ `api_client.py` | **El cliente HTTP hacia FastAPI.** Funciones `register`, `login`, `inspect_fruit` y `get_inspection_history`. Arma cada petición, pone el `Authorization: Bearer`, traduce las respuestas de error en una excepción propia (`ApiError`) con su código de estado, y distingue *timeouts* (5 s para login, 20 s para el análisis, que es más lento). |
| ⚙️ `decorators.py` | `@login_required_jwt`: el control de acceso de Django. No usa `@login_required` de Django porque la autenticación real vive en FastAPI; lo que hace es comprobar que la sesión tenga un `access_token` y, si no, redirigir al login con un mensaje. |
| ⚙️ `context_processors.py` | Inyecta la URL del Swagger de FastAPI en **todas** las plantillas, para que el pie de página tenga el enlace incluso en login y registro. |
| ⚙️ `views.py` | **Las vistas.** `home` (página principal, protegida), `analyze_view` (recibe la foto por `fetch`, la reenvía a FastAPI y devuelve JSON), `history_view` (historial en JSON para pintarlo sin recargar), `register_view`, `login_view` y `logout_view`. Todas las que llaman a FastAPI capturan los errores y devuelven JSON legible, sin trazas. |
| ⚙️ `urls.py` | Las rutas de la app: `` (inicio), `login/`, `logout/`, `registro/`, `analizar/` y `historial/`. |
| 🧪 `tests.py` | Pruebas del frontend: control de acceso, reenvío de la foto, propagación de los códigos de error del backend, construcción de las peticiones HTTP, un **contrato de plantilla** que verifica que existan todos los `id` que el JavaScript busca por nombre, los ajustes del despliegue y las pruebas de que los flujos principales **no tocan la base de datos**. |

#### `frontend/templates/`

| Archivo | Qué hace |
|---|---|
| 📄 `base.html` | Plantilla madre: encabezado, navegación, pie, el **sprite de iconos SVG** (se define una vez y se reutiliza con `<use>`), los mensajes de Django, el apilador de avisos flotantes y la carga de `visual.js`. |
| 📄 `inspector/home.html` | La pantalla principal: encabezado con estadísticas, el panel de cámara (con visor, guía de encuadre, etapas del análisis, botones y consejos), el panel de análisis (medidor de confianza, métricas, composición de color, candidatos y notas), el historial y el bloque `window.VALIDADOR_CONFIG` que le pasa al JavaScript las URLs y el token CSRF. |
| 📄 `inspector/login.html` | Formulario de inicio de sesión, con la argumentación del proyecto en la columna lateral. |
| 📄 `inspector/register.html` | Formulario de creación de cuenta, con validación en el navegador (`minlength`) que refleja las reglas del backend. |

#### `frontend/static/`

| Archivo | Qué hace |
|---|---|
| 📄 `css/styles.css` | Todo el diseño, con la estética “editorial clara”: tipografías, colores, disposición de dos columnas, estados del visor, medidores, tarjetas, animaciones y reglas de accesibilidad (`focus` visible, `prefers-reduced-motion`). |
| ⚙️ `js/camera.js` | **La lógica de la cámara y de la conversación con la API.** Enciende el visor con `getUserMedia`, ajusta el marco a la proporción del sensor, recorta la captura a la zona de la guía, limita la imagen a 1024 px y la comprime a JPEG 0.9, la envía a `/analizar/`, pinta el resultado, carga el historial y las estadísticas, permite subir un archivo o arrastrarlo, y maneja los atajos de teclado (Espacio, Enter, R). |
| ⚙️ `js/visual.js` | **La capa visual, sin lógica de negocio.** Expone el espacio de nombres global `VF`: fondo animado en canvas con manchas de color y partículas, el arco de confianza con contador, los contadores de las estadísticas, la barra de composición de color, las barras de candidatos, la línea de escaneo, los avisos flotantes, el reloj, la aparición al hacer *scroll*, las ondas al presionar botones y la inclinación suave del panel. Cuida el rendimiento (30 fps, se pausa en segundo plano) y no anima nada con `prefers-reduced-motion`. |

---

## 7. Autenticación y sesiones

La fuente de verdad de los usuarios es **FastAPI**. Django no tiene modelo de
usuario: sus vistas de login y registro llaman por HTTP a `/api/v1/auth/*` y
guardan el `access_token` en su sesión.

```
1. Usuario → POST /registro/  → Django → POST /api/v1/auth/register → FastAPI
                                   (bcrypt guarda la contraseña hasheada)
2. Usuario → POST /login/     → Django → POST /api/v1/auth/login    → FastAPI
                                   (devuelve un JWT firmado, válido 60 min)
                               Django guarda el JWT EN LA SESIÓN
3. /analizar/                 → Django lee el JWT de la sesión y lo reenvía:
                                   Authorization: Bearer <JWT> → FastAPI
```

Detalles que importan:

- **Contraseñas:** bcrypt, con validación explícita del límite de 72 bytes.
- **JWT:** algoritmo `HS256`, firmado con `JWT_SECRET_KEY`, expira en 60 minutos.
- **La sesión vive en una cookie firmada** (`signed_cookies`), no en una base de
  datos. Es lo que permite desplegar el frontend en Vercel **sin ninguna base de
  datos**: la cookie se firma con `SECRET_KEY` y no hay ninguna escritura en
  disco.
- **La cookie es `HttpOnly`:** el JavaScript de la página no puede leer el token,
  así que un XSS no puede robarlo.
- **Contrapartida honesta:** la sesión ya no vive del lado del servidor. Su dueño
  puede ver el token en las herramientas de desarrollo del navegador. Con un
  motor de sesiones en base de datos eso no pasaba, pero ese motor necesita disco
  escribible.
- **`SECURE_PROXY_SSL_HEADER`:** Vercel termina TLS y reenvía por HTTP con la
  cabecera `X-Forwarded-Proto`. Sin esto Django cree que la petición no es segura
  y la verificación de origen del CSRF rechaza el POST del login con 403.

---

## 8. Base de datos y despliegue

### Por qué el proyecto no puede usar SQLite en Vercel

El sistema de archivos de una Serverless Function es **de solo lectura** salvo el
directorio temporal. Como `main.py` creaba las tablas al importar, la función
fallaba en el arranque en frío y **todas** las rutas devolvían 500, incluido
`/health`. Y apuntar SQLite al directorio temporal tampoco resuelve la
persistencia: es efímero y vive por instancia.

### Cómo queda, entonces

| Dato | Dónde vive | ¿Sobrevive? |
|---|---|---|
| La sesión del frontend (el JWT) | Cookie firmada con `SECRET_KEY` | Sí, mientras no caduque |
| Usuarios e inspecciones | SQLite en el directorio temporal del backend | **No**: se reinician cuando la función se enfría (~5 min sin uso) |

Alcanza para la entrega y para demostrar el flujo completo, y no requiere crear
ninguna cuenta.

### Si algún día se quiere persistencia real

Basta con definir `DATABASE_URL` en el proyecto del backend apuntando a un
Postgres administrado (Neon tiene plan gratuito). **No hay que tocar código**: el
backend detecta la variable, usa ese Postgres en lugar de SQLite y crea sus
tablas al arrancar. El frontend no cambia nada, porque su sesión viaja en la
cookie.

### Los tres detalles que rompen el despliegue si se olvidan

| Síntoma | Causa |
|---|---|
| 500 en todas las rutas | SQLite en un sistema de archivos de solo lectura |
| **400** al abrir la app | `ALLOWED_HOSTS` con solo `VERCEL_URL`: esa es la URL *de ese despliegue*, no el dominio estable por el que entra el usuario. Lo cubre el comodín `.vercel.app`. |
| **403 CSRF** al iniciar sesión | Falta confiar en `X-Forwarded-Proto` (Vercel termina TLS y reenvía por HTTP) |

### Nota sobre el repositorio

El proyecto necesita ser un **repositorio git propio** con su primer commit.
Vercel construye lo que haya en la raíz del repositorio que se le indique.

### El admin de Django

`/admin/` existe en las rutas, pero necesita tablas en una base de datos. Sin
`DATABASE_URL` configurada, esa pantalla fallará. No la usa ni la enlaza ninguna
parte de la aplicación.

---

## 9. Pruebas automáticas

| Suite | Comando | Cuántas |
|---|---|---|
| Backend | `cd backend && python -m unittest discover -s tests -t .` | **47** |
| Frontend | `cd frontend && python manage.py test` | **37** |

### Qué cubren de verdad (y qué no)

Las imágenes de prueba se **generan sintéticamente** con Pillow: son formas
planas de color. Eso significa que **no miden la precisión real del modelo** —
una manzana plana y sin textura es indistinguible de un tomate para cualquier
clasificador. Lo que miden es la **lógica del pipeline**:

- que la API nunca devuelva una etiqueta fuera de la taxonomía (la regresión del
  bug `"Rubber Eraser"`);
- que la cascada elija el camino correcto en cada caso;
- que la segmentación encuentre el producto y mida bien la forma en objetos
  alargados y delgados;
- que los umbrales de calidad sean relativos al producto y no lo castiguen por ser
  oscuro;
- que las reglas de madurez por producto funcionen (un pepino verde no es
  inmaduro, un tomate verde sí);
- que la taxonomía y el clasificador de respaldo no se desincronicen;
- que el frontend pueda funcionar **sin base de datos** (se verifica contando
  consultas: cero).

### Dos herramientas de prueba que conviene conocer

- **`SessionLoginMixin.login_as()`** (en `frontend/inspector/tests.py`): con
  sesiones en cookie firmada, `client.session["x"] = ...; save()` **no** deja la
  sesión lista, porque el cliente de pruebas de Django solo lee la cookie que
  viene en una *respuesta*. Este helper inicia sesión de verdad contra la vista.
- **`assertNumQueries(0)`**: la forma de exigir que un flujo no toque la base de
  datos. Si alguien mete una consulta al ORM en el camino de login o de la
  pantalla principal, la prueba falla antes de romper producción.

---

## 10. Estado de CLIP

La idea era sumar una etapa de **CLIP zero-shot** para cubrir cientos de
productos sin calibrar perfiles a mano. Qué quedó hecho:

| Pieza | Estado |
|---|---|
| Tokenizador (`clip_tokenizer.py` + `clip/tokenizer.json`) | **Listo y probado.** BPE byte-level en Python puro, sin `transformers`, reproduce los ids canónicos de CLIP. |
| Modelos (`clip/*.onnx`) | **Inservibles.** |
| Productos (`taxonomy.CLIP_PENDING_PRODUCE`, 36) | Fuera del catálogo activo a propósito. |
| Etapa dentro de la cascada | No conectada. |

### Por qué los modelos no sirven

Se generaron con `onnxruntime.quantization.quantize_dynamic`, que es incompatible
con CLIP: descompone las LayerNormalization y corrompe las proyecciones. Las
mediciones:

| Medición | Obtenido | Esperado |
|---|---|---|
| `cos("a photo of a papaya", "a photo of a carrot")` | **0.996** | 0.70–0.85 |
| `cos("a photo of a cat", "a photo of a car")` | **0.998** | claramente menor |
| norma de los embeddings de imagen | **10⁵ – 10⁶** | ~20 |
| nodos `LayerNormalization` en el grafo | **0** | presentes |

Es decir: el codificador de texto colapsa (todos los prompts quedan casi
idénticos) y el de visión devuelve vectores sin sentido. Con esos archivos la
clasificación es ruido, así que **no se usan**.

### Cómo se retoma

`scripts/export_clip_onnx.py` los re-exporta en FP32 y **valida antes de
instalar**. Cuando la validación pase, hay que conectar la etapa en
`inference.py` entre el paso 1 y el paso 2, y devolver los 36 productos a
`EXTRA_PRODUCE`.

> **Ojo con el tamaño:** el CLIP FP32 completo pesa ~590 MB (visión ~340 MB +
> texto ~250 MB) y el límite del paquete de una Vercel Function es de 500 MB, de
> los que ya se usan ~45 MB del ResNet18. Es decir, aunque la exportación sea
> correcta, CLIP **no cabe junto a ResNet18**. Hay que cuantizar de forma estática
> y por canal (y volver a validar), usar un modelo más pequeño (tipo MobileCLIP-S0)
> o mover la inferencia a un servicio aparte.

---

## 11. Limitaciones conocidas

### Qué es machine learning y qué no

| Parte | Cómo se resuelve | ¿Es ML? |
|---|---|---|
| Identificar productos de ImageNet | ResNet18 preentrenado (ONNX) | **Sí**, modelo preentrenado |
| Identificar papaya, mango, guayaba… | Perfiles de color y forma calibrados a mano | **No**, visión clásica |
| Calidad y madurez | Heurístico de color con umbrales relativos | **No**, visión clásica |

### Consecuencias prácticas para la demostración

- **Un producto muy parecido a otro puede confundirse:** un tomate y una manzana
  roja son casi idénticos en color y forma. Por eso la respuesta incluye
  `candidates` y no pretende certeza absoluta.
- **Fondo del mismo color que el producto:** la segmentación confunde fondo y
  producto y la confianza baja. El sistema lo detecta y lo avisa en `notes` y en
  la interfaz.
- **Luz y cámara influyen mucho:** luz directa fuerte, sombras duras o contraluz
  generan manchas falsas.
- **Arranque en frío:** el backend carga un modelo de 45 MB; la primera petición
  tras un rato sin tráfico tarda más. La interfaz espera hasta 45 segundos.
  Medido en un portátil: **~0,9 s la primera llamada** (incluye cargar el modelo
  en memoria) y **~0,4–0,6 s** las siguientes. En el despliegue hay que sumarle
  el arranque completo de la función.
- **Los datos no persisten** entre invocaciones (ver §8).
- **El admin de Django** no funciona sin base de datos configurada.

---

## 12. Chuleta de comandos

### Correr en local (dos procesos)

```bash
# Terminal 1 — backend en el puerto 8000
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
# Swagger: http://127.0.0.1:8000/docs
```

```bash
# Terminal 2 — frontend en el puerto 8080
cd frontend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python manage.py migrate
python manage.py runserver 8080
# App: http://127.0.0.1:8080
```

> La cámara (`getUserMedia`) solo funciona en `localhost` o por HTTPS. Si abres
> la app por IP de red local, el navegador bloqueará la cámara y la app ofrecerá
> subir un archivo.

### Pruebas

```bash
cd backend  && python -m unittest discover -s tests -t . -v   # 47 pruebas
cd frontend && python manage.py test                          # 37 pruebas
```

### Inspeccionar el pipeline desde Python

```python
from PIL import Image
from app.ml.inference import analyze
from app.ml.vision import assess_quality_and_ripeness

foto = Image.open("mi_foto.jpg")
match = analyze(foto)
print(match.item, match.method, match.confidence)
print(assess_quality_and_ripeness(match.features, match.item))
```

### Re-exportar los modelos de CLIP (cuando se retome)

```bash
pip install "optimum[exporters]" torch transformers
cd backend
python scripts/export_clip_onnx.py          # exporta y valida
python scripts/export_clip_onnx.py --skip-export   # solo valida lo que hay
```
