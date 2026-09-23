# Validador de Calidad de Frutas y Verduras

> **Taller 4 — Python + ML + FastAPI** (SENA, Análisis y Desarrollo de Software)
> Proyecto 7: *Validador de Calidad de Control de Calidad en Frutas/Alimentos*.

Aplicación de visión por computador que permite **capturar la foto de una fruta
o verdura desde la cámara del navegador** y obtener automáticamente su
**tipo**, su **calidad** y su **estado de madurez**, con la explicación de cómo
se llegó a ese resultado.

---

## Integrantes

| Nombre completo | GitHub |
|---|---|
| Angelo Martínez Díaz | @angelots31 |
| Simón Carmona Betancur | @stronghoold |

## Enlaces de despliegue

| Servicio | Enlace |
|---|---|
| Backend (FastAPI · Swagger en `/docs`) | https://validador-lyart.vercel.app/ |
| Frontend (Django) | https://validador-a2vb.vercel.app/ |

---

## Stack

| Capa | Tecnología |
|---|---|
| Frontend | Django 6 · plantillas HTML5 · JavaScript sin frameworks · `getUserMedia` para la cámara |
| Backend | FastAPI · Pydantic v2 · SQLAlchemy · Swagger en `/docs` |
| Modelo | ResNet18 preentrenado en ImageNet, exportado a **ONNX** y ejecutado con `onnxruntime` |
| Visión clásica | numpy + Pillow (segmentación, color y forma) — sin OpenCV |
| Autenticación | JWT emitido por FastAPI; Django guarda el token en su sesión |
| Persistencia | **Ninguna por defecto en producción**: la sesión va en una cookie firmada y el backend usa SQLite en el directorio temporal (efímero). Definiendo `DATABASE_URL`, el backend usa un Postgres administrado y los datos sobreviven. |
| Base de datos en local | SQLite · SQLAlchemy 2 + psycopg 3 en el backend, ORM de Django en el frontend |
| Despliegue | Vercel · Serverless Functions Python vía `vercel.json` |

---

## Cómo identifica el producto (parte central del proyecto)

El sistema **no** es "un modelo que adivina". Es una **cascada de decisión** con
tres caminos posibles, y la respuesta siempre dice cuál se usó en el campo
`method`:

```
foto subida
    │
    ├─ 1. ResNet18 (ONNX) con TTA  ──► ¿producto de ImageNet con p ≥ 0.15?
    │                                     └─ sí → method = "resnet18-imagenet"
    │
    ├─ 2. Clasificador de respaldo por color y forma
    │        └─ si su puntaje ≥ 0.52 y supera la pista del modelo
    │           → method = "color-shape-heuristics"
    │
    ├─ 3. ¿El modelo dio al menos una pista débil (p ≥ 0.02)?
    │        └─ sí → se usa, avisando que la confianza es baja
    │
    └─ 4. Nada convincente → item = "Unknown", method = "unrecognized"
```

### Por qué existe el paso 2

**ImageNet solo tiene ~23 clases de frutas y verduras**, y no incluye papaya,
mango, guayaba, sandía, melón, aguacate, tomate, zanahoria, papa, cebolla ni
casi ninguna de las que se venden en una plaza de mercado.

En la primera versión, cuando el modelo no encontraba ninguna fruta entre sus
predicciones, la API devolvía su clase número 1 tal cual. El resultado, al
fotografiar una papaya, era literalmente:

```json
{ "item": "Rubber Eraser", "quality": "Bad", "ripeness": "Overripe" }
```

Eso quedó corregido con dos cambios: un **clasificador de respaldo** que sí
nombra esos productos, y un caso explícito de `"Unknown"` para no inventar
nombres cuando no hay información suficiente.

### Paso 1 — Qué reconoce el modelo preentrenado

ResNet18 sobre ImageNet, con *test-time augmentation* (promedia 4 vistas de la
foto: recorte central y foto completa, cada una con su espejo horizontal).

**23 productos:** manzana, fresa, naranja, limón, higo, piña, banano, jaca,
anón, granada, repollo, brócoli, coliflor, calabacín, ahuyama espagueti,
ahuyama bellota, ahuyama mantequilla, pepino, alcachofa, pimentón, cardo,
champiñón y mazorca.

### Paso 2 — Qué cubre el clasificador de respaldo

Es un clasificador *nearest-centroid* sobre características de color y forma.
**No es una red neuronal entrenada** (ver *Limitaciones*): cada producto tiene
un perfil con su tono, saturación, brillo y forma típicos, y se elige el perfil
que mejor encaja con la foto. Cubre **30 productos**:

papaya, mango, guayaba, sandía, melón, aguacate, tomate, kiwi, durazno, pera,
uva, ciruela, maracuyá, cereza, coco, lima, plátano, zanahoria, papa, cebolla,
ajo, berenjena, remolacha, lechuga, espinaca, ají, yuca, habichuela, batata y
rábano.

### Cómo se mide el color y la forma

Todo sale de un solo paso de análisis (`app/ml/features.py`):

1. **Segmentación:** se estima el color de fondo (mediana del borde de la
   imagen), se separan los píxeles del producto y se toma la región conectada
   principal con crecimiento de región vectorizado. Así el color y la forma no
   se contaminan con el fondo.
2. **Color:** conversión a HSV y proporciones de verde, amarillo, naranja, rojo,
   morado, marrón y claro. Las proporciones se **ponderan por saturación** para
   que un fondo beige no vote como "amarillo".
3. **Forma:** qué tan alargado es el producto, qué tan lleno está su rectángulo
   envolvente y qué tan circular es (compacidad).

### Calidad y madurez

También es visión clásica, no un modelo entrenado (`app/ml/vision.py`). Dos
decisiones de diseño importantes:

- **Los umbrales son relativos al propio producto.** Una "mancha" es un píxel
  claramente más oscuro *que la mediana de ese mismo producto*. Antes el umbral
  era absoluto, y por eso una berenjena, unas uvas o un aguacate Hass salían
  siempre como `Bad` / `Overripe` solo por ser oscuros.
- **El color se interpreta según el producto.** Un tomate verde está inmaduro,
  pero un pepino, un brócoli o una lechuga son verdes cuando están en su punto.
  La taxonomía (`app/ml/taxonomy.py`) define qué productos usan el verde como
  señal de madurez y cuáles no.

Cada respuesta incluye `notes`: la explicación en texto de por qué se decidió
esa calidad y esa madurez.

---

## Endpoints (Swagger interactivo en `/docs`)

| Método | Ruta | Auth | Descripción |
|---|---|---|---|
| GET | `/api/v1/health` | No | Estado de la API |
| POST | `/api/v1/auth/register` | No | Crea un usuario nuevo |
| POST | `/api/v1/auth/login` | No | Login, devuelve `access_token` (JWT) |
| GET | `/api/v1/auth/me` | Sí | Usuario dueño del token |
| POST | `/api/v1/inspect-fruit` | Sí | Sube la foto y devuelve el análisis |
| GET | `/api/v1/inspection-history` | Sí | Historial del usuario autenticado |

Los endpoints con **Auth: Sí** requieren `Authorization: Bearer <token>`. En
`/docs` se prueban con el botón **Authorize** (usuario creado en
`/auth/register`).

### Ejemplo de respuesta de `POST /api/v1/inspect-fruit`

```json
{
  "item": "Papaya",
  "category": "Fruit",
  "quality": "Good",
  "ripeness": "Ripe",
  "confidence": 0.8018,
  "method": "color-shape-heuristics",
  "candidates": [
    { "item": "Papaya", "score": 0.8043, "source": "heuristic" },
    { "item": "Mango", "score": 0.7847, "source": "heuristic" },
    { "item": "Guava", "score": 0.6359, "source": "heuristic" }
  ],
  "notes": [
    "El modelo no encontró este producto entre las clases de ImageNet (ImageNet no incluye papaya entre sus 1000 categorías).",
    "Identificado por color y forma (clasificador de respaldo) con 80% de coincidencia. Señales principales: hue 100%, sat 91%, elongation 75%.",
    "Superficie uniforme y sin daños visibles.",
    "Tono medio 40° con poca superficie verde (12%): color de producto maduro."
  ],
  "diagnostics": {
    "mean_hue": 39.57,
    "saturation": 0.6442,
    "brightness": 0.8777,
    "elongation": 1.107,
    "blemish_ratio": 0.0,
    "segmentation_confidence": 0.9931,
    "colors": {
      "green": 0.1224, "yellow": 0.0682, "orange": 0.8095,
      "red": 0.0, "purple": 0.0, "brown": 0.0, "pale": 0.0475
    }
  }
}
```

Los tres campos que pide el enunciado del taller (`item`, `quality`,
`ripeness`) se mantienen tal cual; el resto son campos añadidos que explican el
resultado y alimentan la interfaz.

`quality` solo puede ser `Good` / `Regular` / `Bad`, `ripeness` solo `Unripe` /
`Ripe` / `Overripe`, `category` `Fruit` / `Vegetable` / `Unknown` y `method`
`resnet18-imagenet` / `color-shape-heuristics` / `unrecognized` — tipados como
`Literal` en los esquemas Pydantic (`app/schemas/inspection.py`).

---

## Autenticación

La fuente de verdad de los usuarios es **FastAPI** (tabla `users`, contraseñas
hasheadas con `bcrypt`). Django **no** tiene su propio modelo de usuario: sus
vistas de login y registro llaman por HTTP a `/api/v1/auth/*` y guardan el
`access_token` en la sesión de Django. Las vistas protegidas usan el decorador
`@login_required_jwt`.

La sesión se guarda en una **cookie firmada**
(`SESSION_ENGINE = "django.contrib.sessions.backends.signed_cookies"`), no en
una base de datos. Es lo que permite desplegar el frontend en Vercel sin
ninguna base de datos: no hay ni una escritura en disco. Hay una prueba que lo
vigila contando consultas (`NoDatabaseDeploymentTests`): si alguien mete una
consulta al ORM en el camino de login o de la página principal, falla antes de
romper producción.

El JWT no es accesible desde el JavaScript de la página: la cookie es
**HttpOnly**, así que no se puede leer con `document.cookie` ni robar con un
XSS. Lo que sí cambia respecto a un motor de sesiones en base de datos es que la
sesión ya **no** vive del lado del servidor: quien la posee puede ver el token en
las herramientas del navegador. Con `DATABASE_URL` configurada se puede volver
al motor en base de datos cambiando `SESSION_ENGINE`.

---

## Interfaz de la cámara

- **Layout de dos columnas:** la cámara a la izquierda y el análisis a la
  derecha (en pantallas grandes; en móvil se apilan).
- **Guía de encuadre:** la foto que se envía se **recorta a la zona de la
  guía**, no se manda el encuadre completo. Esto mejora mucho la precisión,
  porque el producto ocupa la mayor parte de los píxeles analizados.
- **Proporción real del sensor:** el marco del visor se ajusta a la proporción
  del video, y la captura deshace el escalado de `object-fit: cover`, así que el
  recorte coincide exactamente con lo que el usuario ve dentro de las esquinas.
- **Alternativas:** cambiar entre cámara frontal y trasera, subir un archivo y
  arrastrar y soltar una imagen sobre el visor.
- **Atajos de teclado:** `Espacio` captura, `Enter` analiza, `R` repite.
- **Etapas del análisis:** mientras el backend responde se muestran las cuatro
  etapas del pipeline (separar el producto, identificar, medir, estimar).
- **Toasts** para errores y confirmaciones, y un botón para **copiar el JSON**
  del resultado (útil para la sustentación).
- **Accesibilidad:** `aria-live` en los estados, foco visible, y todo el
  movimiento se desactiva con `prefers-reduced-motion`.

### `visual.js`

Archivo dedicado a lo visual, independiente de la cámara y de la API (expone el
namespace global `VF`):

- Canvas de fondo con manchas de color que derivan lentamente y **ráfaga de
  partículas** al terminar un análisis, teñida con el color del resultado.
- **Arco de confianza** animado con contador numérico.
- Contadores animados en las estadísticas del encabezado.
- Barra de composición de color y barras de candidatos que se llenan animadas.
- **Línea de escaneo** sobre el visor y esquinas que pulsan mientras se analiza.
- Apartado de aparición al hacer *scroll*, ondas al presionar botones e
  inclinación suave del panel con el puntero.
- Reloj de la barra superior.
- Rendimiento: el canvas limita a 30 fps, se pausa cuando la pestaña está en
  segundo plano y no anima nada con `prefers-reduced-motion`.

---

## Estructura del repositorio

```
.
├── docs/
│   ├── FUNCIONAMIENTO.md         # Cómo funciona todo y qué hace cada archivo
│   └── EXPOSICION.md             # Guion de la exposición (~20 min, 2 personas)
├── backend/                      # API FastAPI
│   ├── api/index.py              # Entry point para Vercel Serverless
│   ├── app/
│   │   ├── core/                 # config, database (Postgres/SQLite), security (JWT/bcrypt)
│   │   ├── models/               # Modelos SQLAlchemy (User, Inspection)
│   │   ├── schemas/              # Esquemas Pydantic (contrato de la API)
│   │   ├── routers/              # health, auth, inspect
│   │   └── ml/                   # Pipeline de visión
│   │       ├── taxonomy.py       # Productos conocidos y reglas de madurez
│   │       ├── features.py       # Segmentación, color y forma
│   │       ├── fallback.py       # Clasificador de respaldo (color/forma)
│   │       ├── inference.py      # ResNet18 ONNX + TTA + cascada de decisión
│   │       ├── vision.py         # Calidad y madurez
│   │       ├── clip_tokenizer.py # Tokenizador CLIP en Python puro (listo y probado)
│   │       ├── clip/             # Modelos CLIP INUTILIZABLES (cuantizados), ver scripts/
│   │       ├── resnet18.onnx     # Modelo preentrenado (~45 MB)
│   │       └── imagenet_classes.txt
│   ├── scripts/
│   │   └── export_clip_onnx.py   # Re-exporta y valida los ONNX de CLIP
│   ├── tests/                    # test_ml · test_database · test_clip_tokenizer
│   ├── main.py                   # App FastAPI (CORS, gzip, manejo de errores, Swagger)
│   ├── .python-version           # Python fijado para Vercel (3.12)
│   ├── .vercelignore             # Lo que no se sube al desplegar (venv, modelos rotos)
│   └── vercel.json
│
└── frontend/                     # Cliente Django
    ├── api/index.py              # Entry point para Vercel Serverless
    ├── config/                   # settings, urls, wsgi
    ├── inspector/
    │   ├── api_client.py         # Cliente HTTP hacia FastAPI
    │   ├── decorators.py         # @login_required_jwt (control de acceso)
    │   ├── context_processors.py # URL del Swagger en todas las plantillas
    │   ├── views.py              # home, analyze, history, login, register, logout
    │   └── tests.py              # Pruebas de vistas, decorador y cliente HTTP
    ├── templates/                # base, home, login, register
    ├── static/css/styles.css     # Diseño "editorial claro"
    ├── static/js/camera.js       # Cámara, captura y llamadas a la API
    ├── static/js/visual.js       # Capa visual (namespace VF)
    ├── .vercelignore             # Lo que no se sube al desplegar (venv, db.sqlite3)
    └── vercel.json
```

---

## Cómo correr en local

Se necesitan **dos procesos**: el backend en el puerto 8000 y el frontend en el
8080 (Django no puede usar el 8000 si FastAPI ya lo tiene).

### 1. Backend (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
# Swagger: http://127.0.0.1:8000/docs
```

> `DATABASE_URL` puede quedar vacía en local: en ese caso se usa un SQLite en
> `backend/app.db`. Solo hay que llenarla para apuntar al Postgres de
> producción (ver *Despliegue*).

### 2. Frontend (Django)

```bash
cd frontend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python manage.py migrate
python manage.py runserver 8080
# App: http://127.0.0.1:8080
```

> Igual que en el backend, `DATABASE_URL` vacía significa SQLite local
> (`frontend/db.sqlite3`), que es donde Django guarda la sesión.

> `getUserMedia` (la cámara) solo funciona en `localhost` o por **HTTPS**. Si
> abres la app por IP de red local, el navegador bloqueará la cámara y la app te
> ofrecerá subir un archivo en su lugar.

---

## Pruebas

### Backend — pipeline de visión

```bash
cd backend
python -m unittest discover -s tests -t . -v
```

47 pruebas. Además de la taxonomía y los heurísticos, incluyen la **regresión
del bug reportado**: verifican que la API nunca devuelva una etiqueta que no sea
un producto de la taxonomía (es decir, que no vuelva a responder
`"Rubber Eraser"`), y que identifique correctamente una papaya y una zanahoria.

Las que se agregaron al preparar el despliegue cubren dos huecos reales:

- `tests/test_database.py` — la cadena de conexión que entregan los proveedores
  (`postgres://...`) se traduce al driver instalado, y sin `DATABASE_URL` se
  sigue usando SQLite local.
- `tests/test_clip_tokenizer.py` — el tokenizador de CLIP reproduce los ids
  canónicos de la librería (el docstring afirmaba que había tests, pero no
existían). También hay una **invariante de la taxonomía**: `EXTRA_PRODUCE` y
  `fallback.PROFILES` deben coincidir 1 a 1, porque un nombre sin perfil es un
  producto que ninguna ruta puede devolver.

Las imágenes de prueba se generan sintéticamente con Pillow, así que **no miden
precisión real del modelo** (una manzana plana y sin textura es indistinguible
de un tomate para cualquier clasificador). Miden la lógica del pipeline:
cascada de decisión, segmentación, forma en objetos alargados, umbrales
relativos de calidad y reglas de madurez por producto.

### Frontend — Django

```bash
cd frontend
python manage.py test
```

37 pruebas: control de acceso JWT, reenvío de la foto, propagación de errores
del backend, construcción de las peticiones HTTP hacia FastAPI y un **contrato
de plantilla** que verifica que existan todos los `id` que el JavaScript
necesita (si alguien renombra uno, la prueba falla antes de que se rompa la
captura en silencio).

Las que se agregaron al preparar el despliegue cubren lo que solo importa ahí,
y que es justo lo que no avisa a tiempo:

- `DeploymentSettingsTests` — que `.vercel.app` esté en `ALLOWED_HOSTS`, que la
  cabecera de proxy se confíe (sin eso el login da 403) y que la cadena de
  Postgres se traduzca bien a la configuración de Django.
- `NoDatabaseDeploymentTests` — que los caminos de login, página principal y
  logout hagan **cero** consultas a la base de datos (`assertNumQueries(0)`).
  Es la garantía de que el frontend puede desplegarse sin ninguna base de datos.

> Ojo al escribir pruebas nuevas: con sesiones en cookie firmada,
> `client.session["x"] = ...; session.save()` **no** deja la sesión lista, porque
> el `Client` de Django solo obtiene la sesión de la cookie que viene en una
> *respuesta*. Para preparar el estado hay que iniciar sesión de verdad contra la
> vista; para eso está el helper `SessionLoginMixin.login_as()`.

---

## Despliegue en Vercel

El repositorio es un **monorepo**: se crean **dos proyectos** en Vercel, cada uno
apuntando a su carpeta.

Antes de nada, esta carpeta tiene que ser un **repositorio git propio** con su
primer commit: Vercel construye lo que haya en la raíz del repositorio, así que
si el `.git` está en una carpeta superior se sube todo lo de arriba.

### Cómo queda la persistencia

Vercel **no** puede hospedar la base de datos: el sistema de archivos de las
Serverless Functions es de **solo lectura** salvo el directorio temporal, así que
un SQLite dentro del proyecto ni siquiera se puede crear (la función fallaba en el
arranque en frío con `unable to open database file`). Por eso el despliegue por
defecto **no usa ninguna base de datos**:

| Dato | Dónde vive | ¿Sobrevive? |
|---|---|---|
| La sesión del frontend (el JWT) | Cookie **firmada** con `SECRET_KEY` | Sí, mientras no caduque |
| Usuarios e inspecciones | SQLite en el directorio temporal del backend | **No**: se pierden cada vez que la función se enfría (~5 min sin tráfico) |

Alcanza para la entrega y para mostrar el flujo completo, y no requiere crear
ninguna cuenta. La contrapartida hay que decirla en la sustentación: los usuarios
y el historial se reinician solos.

**Si en algún momento se quiere persistencia real** (que los usuarios y las
inspecciones no se borren), basta con definir `DATABASE_URL` en el proyecto del
**backend**, apuntando a un Postgres administrado ([Neon](https://neon.tech)
tiene plan gratuito, ~3 minutos de configuración). No hay que tocar código: el
backend detecta la variable, usa ese Postgres en lugar de SQLite y crea sus
tablas al arrancar.

El frontend no cambia nada, porque su sesión viaja en la cookie y no consulta
ninguna tabla. Solo si además se quiere usar el `/admin/` de Django hay que crear
sus tablas una vez, desde el equipo:

```bash
cd frontend
DATABASE_URL="postgresql://usuario:clave@host/neondb?sslmode=require" python manage.py migrate
```

### Backend

1. Nuevo proyecto → Root Directory: `backend`
2. Variables de entorno:
   - `JWT_SECRET_KEY` — cadena larga y aleatoria
   - `JWT_ALGORITHM` — `HS256`
   - `JWT_EXPIRE_MINUTES` — `60`
   - `DATABASE_URL` — **opcional**, solo para persistencia real
3. `vercel.json` incluye `app/**` y el modelo `.onnx` en el paquete de la
   función (por eso se usa `onnxruntime` y no PyTorch: pesa mucho menos y cabe
   en el límite de la Serverless Function).
4. `.python-version` fija **Python 3.12**, que es la versión con la que se
   probaron las dependencias binarias (`onnxruntime`, `psycopg`).
5. `.vercelignore` deja fuera del despliegue el `venv/` local (que pesa cientos
   de MB y el CLI de Vercel subiría tal cual, porque no mira `.gitignore`) y los
   modelos CLIP que no sirven.

### Frontend

1. Nuevo proyecto → Root Directory: `frontend`
2. Variables de entorno:
   - `DJANGO_SECRET_KEY` — cadena larga y aleatoria
   - `DJANGO_DEBUG` — `False`
   - `FASTAPI_BASE_URL` — la URL del backend desplegado
   - `DATABASE_URL` — **opcional**, solo si se va a usar el `/admin/`
   - `DJANGO_ALLOWED_HOSTS` / `DJANGO_CSRF_TRUSTED_ORIGINS` — **no hacen falta**:
     `config/settings.py` agrega `VERCEL_URL`,
     `VERCEL_PROJECT_PRODUCTION_URL`, `VERCEL_BRANCH_URL` y el comodín
     `.vercel.app` por su cuenta.
3. Los archivos estáticos se sirven directamente desde `static/` con
   `@vercel/static`, así que **no** hace falta ejecutar `collectstatic`.
4. `.vercelignore` deja fuera el `venv/` local y `db.sqlite3`.

### Los tres detalles que rompen el despliegue si se olvidan

Son los tres que se corrigieron aquí, y ninguno da un error obvio:

| Síntoma | Causa |
|---|---|
| 500 en **todas** las rutas, incluida `/health` | SQLite en un sistema de archivos de solo lectura. La función falla al importar. |
| **400** al abrir la app | `ALLOWED_HOSTS` con solo `VERCEL_URL`. Esa es la URL *de ese despliegue* (`mi-app-abc123.vercel.app`), no el dominio estable por el que entra el usuario (`mi-app.vercel.app`). El comodín `.vercel.app` es lo que lo cubre. |
| **403 CSRF** al iniciar sesión | Vercel termina TLS y reenvía por HTTP con `X-Forwarded-Proto`; sin `SECURE_PROXY_SSL_HEADER` Django cree que el POST llegó por HTTP y la verificación de origen lo rechaza. |

> **Nota sobre el arranque en frío:** el backend carga un ResNet18 de 45 MB desde
> el paquete de la función. La primera petición tras un rato sin tráfico tarda
> más que las siguientes; por eso la interfaz espera hasta 45 s.

---

## Historial de commits

| # | Commit |
|---|---|
| 1 | `init: estructura base django frontend y fastapi backend` |
| 2 | `feat: modulo de autenticacion de usuario y control de acceso` |
| 3 | `feat: carga e inferencia del modelo preentrenado en fastapi` |
| 4 | `docs: esquemas pydantic y documentacion de endpoints en swagger` |
| 5 | `feat: interfaz ui en django y captura de stream de cámara en js` |
| 6 | `feat: integracion http entre cliente django y servidor fastapi` |
| 7 | `fix: optimizacion de respuesta, manejo de errores y ui polish` |
| 8 | `deploy: configuracion vercel.json y pruebas finales de produccion` |

---

## Limitaciones (honestidad para la sustentación)

Vale la pena decir explícitamente qué es Machine Learning y qué no:

| Parte | Cómo se resuelve | ¿Es ML? |
|---|---|---|
| Identificar el producto de ImageNet | ResNet18 preentrenado (ONNX) | **Sí**, modelo preentrenado |
| Identificar papaya, mango, guayaba... | Clasificador de color/forma con perfiles calibrados a mano | **No**, es visión clásica |
| Calidad y madurez | Heurístico de color con umbrales relativos | **No**, es visión clásica |

Por qué no hay un modelo entrenado para calidad/madurez: ImageNet no tiene esas
etiquetas, y entrenar un clasificador de madurez necesitaría un dataset
etiquetado de frutas en distintos estados que no existe dentro del alcance de
las 15 horas del taller ni cabría en una Serverless Function de Vercel.

Consecuencias prácticas a tener en cuenta en la demo:

- Un producto **muy parecido a otro** puede confundirse: un tomate y una manzana
  roja son casi idénticos en color y forma. Por eso la respuesta incluye
  `candidates` (las mejores propuestas) y no un único valor con pretensión de
  certeza absoluta.
- Con **fondo del mismo color que el producto** la segmentación confunde el
  fondo con el producto, y la confianza baja: la app lo detecta, avisa en
  `notes` y en la interfaz.
- La cámara y el brillo influyen mucho: luz directa fuerte, sombras duras o
  contraluz generan manchas falsas.

## Estado del trabajo de CLIP (a medio camino)

La idea era sumar una etapa de **CLIP zero-shot** para cubrir cientos de
productos sin calibrar perfiles a mano, y así nombrar los 36 productos de plaza
que ImageNet no conoce. Qué quedó hecho y qué no:

| Pieza | Estado |
|---|---|
| Tokenizador (`app/ml/clip_tokenizer.py` + `clip/tokenizer.json`) | **Listo y probado.** BPE byte-level en Python puro, sin `transformers`, reproduce los ids canónicos de CLIP. |
| Modelos (`app/ml/clip/*.onnx`) | **Inservibles.** Ver abajo. |
| Productos (`taxonomy.CLIP_PENDING_PRODUCE`) | Fuera de la taxonomía activa a propósito. |
| Etapa en la cascada | No conectada. |

**Por qué los `.onnx` no sirven.** Se generaron con
`onnxruntime.quantization.quantize_dynamic`, que es incompatible con CLIP:
descompone las LayerNormalization y corrompe las proyecciones. Las señales
medidas:

| Medición | Obtenido | Esperado |
|---|---|---|
| `cos("a photo of a papaya", "a photo of a carrot")` | **0.996** | 0.70–0.85 |
| `cos("a photo of a cat", "a photo of a car")` | **0.998** | claramente menor |
| norma de `image_embeds` | **10⁵–10⁶** | ~20 |
| LayerNormalization en el grafo | **0 nodos** | presente |

O sea: el encoder de texto colapsa (todos los prompts quedan casi idénticos) y el
de visión devuelve vectores sin sentido. La clasificación con esos archivos es
ruido, así que **no se usan**: `.vercelignore` los deja fuera del despliegue y
`taxonomy.CLIP_PENDING_PRODUCE` mantiene sus 36 productos fuera de
`ALL_PRODUCE_NAMES`, que así contiene exactamente lo que el sistema puede
devolver hoy.

**Cómo se retoma.** `scripts/export_clip_onnx.py` re-exporta los modelos en FP32
con `optimum` y, antes de instalarlos, **valida que sirvan** (tokenizador,
separación semántica del texto y escala de los embeddings de visión). Esa
validación es justamente lo que faltaba: sin ella, un modelo roto entra al
repositorio y nadie se entera hasta que la API responde con total seguridad
algo que no es.

> **Ojo con el tamaño antes de conectarlo:** el CLIP FP32 completo pesa ~590 MB
> (visión ~340 MB + texto ~250 MB) y el límite del bundle de una Vercel Function
> es de 500 MB, de los que ya se usan ~45 MB del ResNet18. Es decir, aunque el
> export sea correcto, **CLIP no cabe junto a ResNet18**. Hay que cuantizar de
> forma estática y por canal (y volver a pasar la validación), usar un modelo más
> pequeño (tipo MobileCLIP-S0) o sacar la inferencia a un servicio aparte.

---

## Posibles mejoras

- Terminar la etapa de CLIP (ver la sección anterior).
- Mover la inferencia a un servicio aparte (o a una GPU) para reducir la
  latencia de la primera petición.
- Cambiar `create_all` y la migración mínima de columnas por **Alembic**: ahora
  que el esquema vive en un Postgres real, toca tener migraciones de verdad.
- Añadir un pequeño dataset etiquetado de calidad/madurez por producto.
