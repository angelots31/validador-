# Exposición — Validador de Calidad de Frutas y Verduras

Guion de **20 minutos** para **dos expositores**. Está escrito para leerse o
adaptarse, no para improvisar: cada bloque dice quién habla, en qué minuto, qué
se ve en pantalla y qué idea tiene que quedar.

En el documento, **A** y **B** son los dos integrantes (cambien los nombres).
La regla de oro: **mientras uno habla, el otro no interrumpe**: prepara la
siguiente diapositiva o la demo.

---

## 0. Antes de empezar

### Reparto de roles

| | **Expositor A** | **Expositor B** |
|---|---|---|
| Tema | Problema, arquitectura, el modelo y la cascada, demo, cierre | Cómo se mide la imagen, calidad y madurez, autenticación, pruebas y despliegue |
| Fortaleza que explota | Cuenta la historia y maneja la demo | Explica el “cómo” técnico con precisión |
| Durante la demo | Conduce y habla | Muestra el JSON y el código si preguntan |

Los dos tienen que poder responder **cualquier** pregunta: si el jurado pregunta
algo del tema del otro, el otro responde. Ensayen intercambiando roles al menos
una vez.

### Lista de chequeo 15 minutos antes

- [ ] Dos pestañas abiertas: **frontend desplegado** y **`/docs` del backend**
      (Swagger).
- [ ] Una sesión de prueba ya creada y con **una inspección en el historial**,
      para que la pantalla no se vea vacía si algo falla.
- [ ] **Despertar el backend**: hacer un análisis cualquiera ~3 minutos antes.
      La primera petición tras un rato sin tráfico es lenta (arranque en frío).
- [ ] **Producto real** listo: una manzana roja o un tomate, con un fondo liso y
      de color distinto (una hoja blanca) y luz sin sombras duras.
- [ ] **Tres fotos guardadas** en el escritorio, listas para usar “Subir
      archivo” si la cámara falla (una fruta, una verdura, una foto ambigua o
      con fondo del mismo color).
- [ ] El repositorio abierto, por si piden ver el código.
- [ ] Cronómetro visible (el reloj del celular sirve) y los minutos clave
      anotados.

### Cronograma

| Minuto | Bloque | Habla |
|---|---|---|
| 0:00 – 1:30 | Apertura y problema | A |
| 1:30 – 4:00 | Arquitectura y stack | B |
| 4:00 – 7:30 | Cómo identifica: la cascada y el bug del “Rubber Eraser” | A |
| 7:30 – 10:00 | Cómo se mide color y forma | B |
| 10:00 – 13:00 | **Demo en vivo** | A (conduce) + B |
| 13:00 – 15:30 | Calidad y madurez · qué es ML y qué no | B |
| 15:30 – 17:30 | Autenticación, seguridad y por qué dos aplicaciones | A |
| 17:30 – 19:00 | Pruebas y despliegue | B |
| 19:00 – 20:00 | Limitaciones, trabajo real y cierre | A |

> Si van con retraso, el bloque que se recorta es el **7:30–10:00** (se resume a
> “el color y la forma se miden con numpy, sin OpenCV” y se sigue). Nunca se
> recorta la demo.

---

## 1 · 0:00 – 1:30 · Apertura y problema  (A)

**En pantalla:** diapositiva con el título del proyecto y tres fotos de frutas.

> **A:** “Buenos días. Nuestro proyecto se llama **Validador de Calidad de Frutas
> y Verduras**, y responde tres preguntas que en una plaza de mercado se
> contestan a ojo: **¿qué fruta es esta?**, **¿está en buen estado?** y **¿ya
> maduró?**
>
> Lo que hicimos fue automatizar esas tres respuestas con una foto tomada desde
> el navegador. La persona abre la aplicación, encuadra la fruta, captura y en
> unos segundos ve el nombre del producto, su calidad, su madurez — **y la
> explicación de cómo se llegó a ese resultado**.
>
> Ese pedazo final es el que le da sentido al proyecto: el sistema no solo
> dice un resultado, dice **en qué se apoyó y con qué método lo obtuvo**. Y va a
> ver que hay un campo que se llama `method` que aparece en cada respuesta y que
> es la columna vertebral de todo: dice si respondió un modelo preentrenado, un
> clasificador clásico, o si honestamente dijo **‘no lo pude identificar’**.”

**Idea que debe quedar:** son tres preguntas, y la respuesta siempre se explica.

**Transición:** “Para que exista esa explicación, la aplicación está partida en
dos. Simón les cuenta cómo.”

---

## 2 · 1:30 – 4:00 · Arquitectura y stack  (B)

**En pantalla:** el diagrama de arquitectura del `README` (navegador → Django →
FastAPI → pipeline de visión).

> **B:** “El proyecto son **dos aplicaciones** que se despliegan por separado y
> hablan por HTTP.
>
> Del lado del usuario está un **cliente en Django**, que es el que sirve las
> páginas, toma la foto con la cámara del navegador y pinta los resultados. Del
> otro lado está el **servidor en FastAPI**, que es donde vive la inteligencia:
> recibe la imagen, la analiza y devuelve el resultado en JSON.
>
> ¿Por qué partido? Por dos razones. La primera es de diseño: el análisis
> consume CPU y memoria, y no queremos que el servidor que sirve páginas cargue
> con eso. La segunda es práctica: FastAPI nos da **Swagger automático** en
> `/docs`, y eso significa que cualquiera puede probar la API sin escribir una
> línea de código — se lo vamos a mostrar en la demo.
>
> El frontend no sabe nada de modelos. Recibe un JSON y lo dibuja. Eso también es
> una decisión: si mañana cambiamos el modelo, el frontend no se entera.
>
> La comunicación entre los dos tiene una regla: **el navegador nunca habla
> directo con FastAPI.** El token de sesión vive en Django, y es Django quien lo
> reenvía a la API. Con eso el JavaScript de la página nunca toca credenciales, y
> el navegador no necesita CORS. Más adelante volvemos sobre esto.
>
> En cuanto a herramientas: **Django 6** con plantillas y JavaScript sin
> frameworks en el frontend, **FastAPI con Pydantic** en el backend, y para el
> análisis **ResNet18 preentrenado en ImageNet exportado a ONNX** — que se
> ejecuta con **onnxruntime**, no con PyTorch — más **visión clásica con numpy y
> Pillow**, sin OpenCV.
>
> Ese último detalle es a propósito y lo quiero subrayar: no usamos PyTorch
> porque pesa varios cientos de megas y nosotros desplegamos en funciones
> serverless, que tienen límite de tamaño. Con ONNX y onnxruntime el modelo más
> la librería pesan alrededor de 70 megas. Cabe, y arranca más rápido.”

**Idea que debe quedar:** dos aplicaciones, separadas por una razón; ONNX en vez
de PyTorch por una razón de despliegue, no por capricho.

**Transición:** “Ahora sí: ¿cómo hace para saber qué fruta es? Ahí está la parte
central del proyecto, y también nuestro mayor error.”

---

## 3 · 4:00 – 7:30 · Cómo identifica: la cascada y el bug  (A)

**En pantalla:** el diagrama de la cascada (4 pasos) del `README`.

> **A:** “Lo primero que hay que decir es que **el sistema no es ‘un modelo que
> adivina’**. Es una **cascada de decisión**: una serie de intentos ordenados, y
> cada intento anota si tuvo éxito. Hay cuatro caminos posibles.
>
> **Paso uno:** el **ResNet18** preentrenado en ImageNet. Es un modelo de verdad,
> preentrenado, entrenado por otros y que nosotros usamos tal cual. Le aplicamos
> además **TTA**, *test-time augmentation*: en vez de mirar la foto una sola vez,
> la miramos cuatro veces —recorte central y foto completa, cada una con su
> espejo horizontal— y promediamos. Cuesta cuatro inferencias, que en CPU es
> barato, y mejora mucho los casos en que el producto no está centrado. Si el
> modelo reconoce el producto con probabilidad suficiente, ese es el resultado y
> `method` queda como `resnet18-imagenet`.
>
> **Y acá viene el problema.** ImageNet tiene **mil categorías**, pero entre esas
> mil solo hay unas **veintitrés** frutas y verduras. Y no incluye papaya, mango,
> guayaba, sandía, melón, aguacate, tomate ni ninguna de las que uno compra en
> una plaza colombiana.
>
> En la primera versión, cuando el modelo no encontraba ninguna fruta, la API
> devolvía su predicción número uno tal cual. Y el resultado, al fotografiar una
> papaya, era literalmente este: **`{ "item": "Rubber Eraser" }`.** Un borrador
> de goma. Con calidad ‘mala’ y madurez ‘pasada de madura’. Y hay que ver la
> gravedad del asunto: **la API no estaba fallando, estaba respondiendo con
> total seguridad algo completamente falso.** Ese es el peor tipo de error en un
> sistema así.
>
> **La solución tiene dos partes.** La primera: un **clasificador de respaldo**
> que sí conoce esos productos — treinta de ellos — y al que se cae cuando el
> modelo no reconoce nada. Ese es el paso dos, y cuando actúa, `method` dice
> `color-shape-heuristics`.
>
> La segunda parte es igual de importante: **si de plano no hay nada
> convincente, el sistema responde `"Unknown"`.** Aprendimos que decir ‘no sé’ es
> mejor que inventar un nombre.
>
> Y quedan dos pasos más: si el respaldo no convenció pero el modelo dio una
> pista apenas perceptible, se usa avisando que la confianza es baja; y si no,
> `unrecognized`.
>
> Lo que hace que esto sea auditable es que **cada respuesta dice siempre cuál de
> los cuatro caminos se usó**, y ahora se lo mostramos funcionando.”

**Idea que debe quedar:** cascada explícita, cuatro caminos, y el `method` siempre
declarado. El bug del “Rubber Eraser” es la historia que justifica todo el diseño.

**Transición:** “Ese clasificador de respaldo, ¿cómo funciona exactamente?
Simón.”

---

## 4 · 7:30 – 10:00 · Cómo se mide color y forma  (B)

**En pantalla:** diapositiva con la segmentación de una foto: original, máscara
del producto y el desglose de color.

> **B:** “Todo el análisis sale de **un solo paso**: la imagen se reduce a 160 por
> 160 píxeles y se calculan dos familias de medidas.
>
> **Primero, la segmentación.** Necesitamos saber **qué píxeles son el producto y
> cuáles el fondo**, porque si mezclamos los dos, el color y la forma mienten. Lo
> que hacemos es estimar el color del fondo con la **mediana del borde** de la
> imagen y marcar como producto todo píxel que esté suficientemente lejos de ese
> color. Después aplicamos una apertura morfológica para quitar ruido, y nos
> quedamos con la **región conectada más grande**, que debería ser el producto.
>
> Ese último paso tiene una historia: la primera versión recortaba por densidad
> de filas y columnas, y con un producto alargado y delgado — una zanahoria, un
> plátano en diagonal — el recorte se comía casi todo. Lo reemplazamos por
> **crecimiento de región desde el punto más denso**, y hay una prueba que
> verifica que la elongación de una zanahoria se mida bien.
>
> **Segundo, el color.** Convertimos a HSV y sacamos la proporción de superficie
> verde, amarilla, naranja, roja, morada, marrón y clara. Dos detalles: las
> proporciones se cuentan **solo sobre píxeles con color real** — un píxel gris
> no debe votar como amarillo — y el tono medio se calcula de forma **circular**,
> porque el tono es un ángulo: 359 grados y 1 grado están a 2 grados de
> distancia, no a 358.
>
> **Tercero, la forma:** qué tan alargado es, qué tan lleno está su rectángulo
> envolvente y qué tan circular es.
>
> Y ahora, el clasificador de respaldo. **No es una red neuronal**, y quiero ser
> explícito en eso porque es lo más fácil de malinterpretar. Es un clasificador
> *nearest-centroid*: cada producto tiene un **perfil** con su tono típico, su
> saturación, su brillo, qué tan alargado suele ser, y una **tolerancia** para
> cada uno de esos valores. Se compara la foto con los treinta perfiles, y cada
> característica se puntúa con una **campana gaussiana**: uno en el valor exacto,
> y va bajando suavemente según la tolerancia. El puntaje final es el promedio
> ponderado.
>
> Dos protecciones: si la foto casi no tiene color — está prácticamente en
> escala de grises — el clasificador **no responde nada**, porque no hay
> información para decidir; y si la segmentación salió mal, el puntaje se
> **castiga**, para no reportar una confianza inflada cuando en realidad
> separamos mal el producto del fondo.
>
> Y para que quede claro el nivel de detalle: los números que están en los
> perfiles están **calibrados a mano**. Si en la sustentación un producto sale
> mal, se ajusta su valor típico o su tolerancia y listo; no hay que reentrenar
> nada.”

**Idea que debe quedar:** segmentación → color → forma → comparación con perfiles
calibrados; y que esta parte **no** es machine learning.

**Transición:** “Suficiente teoría. Vamos a verlo funcionando.” (A toma el
control de la demo.)

---

## 5 · 10:00 – 13:00 · Demo en vivo  (A conduce, B apoya)

**En pantalla:** la aplicación desplegada, en la pestaña del frontend.

### Guion de la demo (seguir en este orden)

**1. Registro y login (30 s).** Crear una cuenta nueva en vivo.

> **A:** “Creo la cuenta aquí. Fíjense que la contraseña viaja al backend, que la
> guarda **cifrada con bcrypt** — no en texto plano — y devuelve un token que
> queda guardado en la sesión de Django.”

**2. Analizar una fruta real (1 min).** Encuadrar el producto en la guía y
capturar. Señalar mientras responde el sistema:

> **A:** “Fíjense en la columna de la izquierda: el sistema muestra las **cuatro
> etapas** del análisis mientras espera. Y ojo con este detalle de diseño: la
> foto que se envía **no es el encuadre completo**, se recorta exactamente a la
> zona que está dentro de la guía. Eso mejora mucho la precisión, porque el
> producto ocupa la mayor parte de los píxeles analizados.”

**3. Leer el resultado (1 min).** Señalar, uno por uno, estos elementos:

- el **nombre** del producto, en inglés y en español;
- el **chip de método** — “aquí está el campo del que les hablé”;
- el **medidor de confianza** — “y aquí la confianza, que solo aplica a la
  identificación, no a la calidad”;
- **calidad** y **madurez**;
- **“Cómo se decidió”** — leer en voz alta **una** de las notas (es la joya de la
  demo);
- **“Otras propuestas”** — “por qué mostramos esto y no un solo valor: porque un
  tomate y una manzana roja son casi idénticos en color y forma, y preferimos ser
  honestos sobre la ambigüedad”.

**4. Copiar el JSON (20 s).** Usar el botón “Copiar JSON” y pegarlo en un
bloc de notas a la vista.

> **A:** “Todo esto que ven en pantalla es exactamente este JSON. La interfaz no
> inventa nada: es una vista de la respuesta de la API.”

**5. El historial (20 s).** Bajar y mostrar que la inspección quedó registrada
con su método y su fecha.

**6. Swagger (30 s).** Cambiar a la pestaña de `/docs`.

> **A:** “Y aquí está la API sola, sin el frontend. Swagger se genera
> automáticamente del código: los endpoints están agrupados, cada campo está
> documentado y los valores posibles son listas cerradas. Cualquier persona
> puede probar la API desde aquí.”

### Si algo falla (tener esto en mente, no improvisar)

| Falla | Qué hacer |
|---|---|
| La cámara no da permiso | Usar **“Subir archivo”** con una de las tres fotos preparadas. Decirlo en voz alta: “uso la alternativa de subir archivo, que está prevista en la app para esto mismo”. |
| El análisis tarda mucho | Es el arranque en frío. Decirlo: “la primera petición carga el modelo de 45 megas; la aplicación espera hasta 45 segundos justamente por esto”. Volver a analizar: la segunda es rápida. |
| El resultado es `Unknown` | **No disimular.** “Este es el caso que aprendimos a manejar: preferimos decir que no sabemos antes que inventar un nombre. Prueben con más luz y un fondo liso.” Y probar con la otra foto. |
| Falla la red | Tener **capturas de pantalla** de la demo en la presentación, y una grabación corta de respaldo. |
| Se acabó el tiempo | Saltar directo al punto 4 (copiar el JSON) y cerrar. |

---

## 6 · 13:00 – 15:30 · Calidad y madurez · qué es ML y qué no  (B)

**En pantalla:** tabla comparativa “¿es machine learning?” y el diagrama de la
respuesta JSON con `notes`.

> **B:** “Ya sabemos qué es el producto. Las otras dos preguntas son **calidad** y
> **madurez**, y acá quiero ser especialmente honesto porque es donde más fácil
> se exagera.
>
> **Esto no es machine learning.** Es visión clásica con umbrales. ¿Por qué? Por
> una razón concreta: **ImageNet no tiene etiquetas de calidad ni de madurez**. Y
> entrenar un clasificador de madurez pediría un conjunto de datos de frutas
> etiquetadas en distintos estados de maduración, que no existe dentro del
> alcance de este taller. Así que aquí la respuesta honesta es: heurísticas
> explicables, dicho en la propia interfaz.
>
> Y decimos ‘esta calidad’ con dos decisiones de diseño muy concretas.
>
> **La primera: los umbrales son relativos al propio producto.** Antes, un píxel
> oscuro contaba como mancha usando un umbral absoluto. ¿Resultado? Una
> berenjena, unas uvas o un aguacate Hass salían **siempre** como ‘mala’ y
> ‘pasada de madura’ — no por su estado, sino por ser oscuros por naturaleza.
> Ahora una mancha es un píxel claramente más oscuro **que la mediana de ese
> mismo producto**. Eso arregló el problema de raíz.
>
> **La segunda: el color se interpreta según el producto.** Un tomate verde está
> inmaduro. Pero un pepino, un brócoli o una lechuga son verdes **cuando están en
> su punto**. Con un criterio único, el sistema marcaba medio mercado como
> inmaduro. Así que la taxonomía define qué productos usan el verde como señal de
> madurez y cuáles no. Hay pruebas para esto: un pepino verde tiene que dar
> ‘maduro’ y un tomate verde tiene que dar ‘inmaduro’, con la misma imagen.
>
> Y todo esto termina en el campo `notes`: cada resultado trae **la explicación
> en texto** de por qué se decidió esa calidad y esa madurez. Por ejemplo:
> *‘manchas oscuras en el 34 % de la superficie’* o *‘tono medio 40 grados con
> poca superficie verde: color de producto maduro’*.
>
> Ese campo es el que hace que el sistema sea **auditable**. Si la respuesta está
> mal, no es un misterio: sabemos exactamente qué umbral se cumplió.”

**Idea que debe quedar:** calidad y madurez son heurísticas explicables, con
umbrales relativos, y el sistema dice por qué decidió lo que decidió.

**Transición:** “¿Y cómo protegimos las cuentas y los datos? Eso lo retoma
Angelo.”

---

## 7 · 15:30 – 17:30 · Autenticación, seguridad y por qué dos aplicaciones  (A)

**En pantalla:** esquema del flujo de autenticación.

> **A:** “La fuente de verdad de los usuarios es **FastAPI**. Django no tiene
> tabla de usuarios: sus vistas de login y registro llaman por HTTP a la API y
> guardan el token que esta devuelve.
>
> Cuando alguien se registra, la contraseña se cifra con **bcrypt** antes de
> guardarse. Nunca se guarda en texto plano, y hay un cuidado específico: bcrypt
> trunca en 72 bytes, así que validamos ese límite para dar un error claro en vez
> de un comportamiento silencioso raro.
>
> El login devuelve un **JWT firmado** con algoritmo HS256, válido por 60
> minutos. Ese token es el que autoriza las peticiones al análisis y al
> historial.
>
> Y acá está la pieza que quiero destacar: **el navegador nunca ve ese token.**
> El token vive en la sesión de Django, y es Django, desde el servidor, quien lo
> reenvía a FastAPI en cada petición. Las vistas protegidas usan un decorador
> propio que verifica que la sesión lo tenga. Eso significa que el JavaScript de
> la página no maneja credenciales: no hay forma de que un error en el frontend
> exponga el token al usuario.
>
> Para que eso funcione, la sesión se guarda en una **cookie firmada** con la
> clave secreta de Django, y esa cookie es **HttpOnly**: el JavaScript de la
> página no puede leerla, así que un ataque de tipo XSS no se la puede llevar.
>
> Y quiero mencionar una decisión de despliegue que tiene que ver con esto.
> Nuestro primer despliegue **no funcionaba**: la aplicación devolvía error en
> todas las rutas. La causa: las funciones serverless de Vercel **no tienen disco
> escribible**. Nuestro código intentaba crear una base de datos SQLite dentro
> del proyecto, y eso es imposible en producción. Lo resolvimos de dos maneras:
> el backend ahora usa un Postgres administrado si se le configura, o el
> directorio temporal si no; y el frontend **ya no necesita base de datos**,
> porque la sesión va en la cookie. Es decir: el mismo diseño de sesión resolvió
> a la vez un problema de seguridad y un problema de despliegue.”

**Idea que debe quedar:** el token nunca llega al navegador; la sesión va en
cookie HttpOnly; y eso además es lo que permite desplegar sin base de datos.

---

## 8 · 17:30 – 19:00 · Pruebas y despliegue  (B)

**En pantalla:** terminal con la salida de las dos suites de pruebas.

> **B:** “Tenemos **47 pruebas automatizadas en el backend y 37 en el frontend**.
> Y quiero contarles **qué** prueban, porque es fácil tener pruebas que no sirven
> para nada.
>
> **Lo primero que quiero aclarar es lo que NO prueban.** Las imágenes de prueba
> se generan sintéticamente con Pillow: son formas planas de color. Una manzana
> plana y sin textura es indistinguible de un tomate para cualquier clasificador.
> Así que **no medimos precisión real del modelo, y no vamos a decir un
> porcentaje que no medimos.** Lo que sí medimos es la **lógica** del sistema.
>
> Y ahí sí, tenemos pruebas para cada cosa que se rompió:
>
> - que la API **nunca** devuelva una etiqueta que no sea de la taxonomía — esa
>   es la regresión directa del bug del ‘Rubber Eraser’, y es la prueba que no
>   puede fallar nunca;
> - que la elongación se mida bien en un objeto alargado y delgado, que fue el
>   bug de la segmentación;
> - que un producto oscuro sin manchas dé calidad buena, que fue el bug de los
>   umbrales absolutos;
> - que un pepino verde no sea ‘inmaduro’ y un tomate verde sí;
> - que la taxonomía y el clasificador de respaldo **no se desincronicen**: hay
>   una prueba que exige que cada producto del catálogo tenga su perfil. Si
>   alguien agrega un producto y se olvida del perfil, la prueba falla, en vez de
>   crear un producto fantasma que el sistema nunca podría devolver.
>
> En el frontend probamos el control de acceso, el reenvío de la foto, que los
> errores del backend se propaguen con su código correcto, y un **contrato de
> plantilla**: verifica que existan todos los identificadores que el JavaScript
> busca. Si alguien renombra uno, la prueba falla **antes** de que la captura se
> rompa en silencio.
>
> Y una que me parece la más útil de todas: comprobamos que el frontend haga
> **cero consultas a la base de datos** en el login y en la pantalla principal.
> Contamos las consultas y exigimos cero. Si alguien mete una consulta ahí, la
> prueba falla — y en producción el login dejaría de funcionar, porque no hay
> disco. Esa prueba es la que nos protege del error que ya cometimos.
>
> Pueden correrlas ustedes mismos: son dos comandos, uno por aplicación.”

**Idea que debe quedar:** las pruebas miden lógica, no precisión; cada prueba
nació de un bug real.

---

## 9 · 19:00 – 20:00 · Limitaciones, trabajo real y cierre  (A)

**En pantalla:** tabla de limitaciones y tabla “¿es ML?”.

> **A:** “Para cerrar, lo que **no** funciona bien, porque un sistema que dice
> que todo funciona bien no es creíble.
>
> **Un producto muy parecido a otro se puede confundir.** Un tomate y una manzana
> roja son casi idénticos en color y forma. Por eso la respuesta trae varias
> propuestas en lugar de una sola.
>
> **Con el fondo del mismo color que el producto**, la segmentación confunde
> fondo y producto, y la confianza baja. El sistema lo detecta y lo avisa; pero
> es una limitación real.
>
> **La luz influye mucho:** luz directa fuerte, sombras duras o contraluz generan
> manchas falsas.
>
> **Y los datos no persisten entre reinicios** en el despliegue sin base de
> datos: los usuarios y el historial se reinician cuando el servidor se enfría.
> Es una decisión que tomamos a conciencia para la entrega, y está documentada.
> El día que se quiera persistencia real, es configurar una variable de entorno:
> el código ya está preparado.
>
> Hubo además una parte que **no pudimos terminar y lo decimos**: intentamos
> sumar **CLIP**, un modelo que reconoce imágenes por descripción de texto, para
> cubrir treinta y seis productos más sin calibrar perfiles a mano. El
> tokenizador quedó listo y probado, pero **los modelos que conseguimos estaban
> dañados**: se habían comprimido con una técnica de cuantización que no es
> compatible con CLIP, y el resultado era ruido. Lo detectamos con tres
> mediciones concretas — dos frases distintas quedaban a 0.996 de similitud,
> cuando lo normal es 0.7 a 0.85 — y **preferimos no conectarlo antes que
> entregar algo que responde mal con seguridad.** Dejamos un script que los
> re-exporta y que **valida antes de instalarlos**, que es justamente la
> verificación que faltaba.
>
> Resumiendo lo que construimos: **una cascada de decisión que nunca inventa una
> respuesta, que siempre dice con qué método llegó a su conclusión y por qué, con
> umbrales relativos al producto, autenticación con JWT que nunca llega al
> navegador, y 84 pruebas automatizadas que vigilan cada error que ya cometimos.**
>
> Y sobre nuestro trabajo, lo más honesto que podemos decir es esto: lo más
> valioso de este proyecto no fue que funcionara la primera vez — no funcionó.
> Fue que cada vez que algo devolvía una respuesta falsa con seguridad, lo
> convertimos en una regla, y después en una prueba, para que no volviera a
> pasar. Gracias.”

---

## Anexo 1 · Datos clave para memorizar

| Dato | Valor |
|---|---|
| Productos que conoce ImageNet | **23** |
| Productos del clasificador de respaldo | **30** |
| Total alcanzable | **53** |
| Productos preparados para CLIP (sin conectar) | 36 |
| Umbral para aceptar al modelo | probabilidad ≥ **0.15** |
| Umbral para la “pista débil” | probabilidad ≥ **0.02** |
| Puntaje mínimo del clasificador de respaldo | **0.52** |
| Vistas del TTA | **4** (recorte + foto completa, cada una espejada) |
| Tamaño del modelo ResNet18 | ~45 MB |
| Tamaño máximo de la foto | 8 MB |
| Confianza mínima de segmentación para avisar | 0.35 |
| Umbrales de calidad | manchas 12 % → Regular · 30 % → Bad |
| Umbrales de madurez | manchas 34 % → Overripe · verde ≥ 40 % → Unripe |
| Validez del JWT | 60 minutos, HS256 |
| Pruebas | 47 backend · 37 frontend |
| Tiempo de análisis | ~0,4–0,6 s con el modelo ya cargado, y ~0,9 s la primera vez (medido en portátil). En el despliegue hay que sumarle el arranque en frío de la función, y por eso la interfaz espera hasta 45 s. |

## Anexo 2 · Cómo responder a lo que van a preguntar

**“¿Entrenaron ustedes el modelo?”**
> “No. Usamos **ResNet18 preentrenado en ImageNet**, que es un modelo entrenado
> por otros. Lo que sí construimos es todo lo demás: la cascada de decisión, el
> clasificador de respaldo por color y forma, y las reglas de calidad y madurez.
> Y lo decimos explícitamente en la respuesta, con el campo `method`: cuando
> responde el modelo dice `resnet18-imagenet`, y cuando responde nuestro
> clasificador dice `color-shape-heuristics`. No los mezclamos.”

**“Entonces, ¿dónde está el machine learning?”**
> “En un solo lugar: identificar productos que están en ImageNet, con el ResNet18.
> La clasificación de papaya, mango o guayaba, y las estimaciones de calidad y
> madurez, **no** son machine learning: son visión clásica con umbrales
> explicables. Preferimos decirlo así antes que presentar un heurístico como si
> fuera una red neuronal.”

**“¿Por qué no entrenaron su propio modelo?”**
> “Por dos razones. ImageNet no tiene etiquetas de calidad ni de madurez, así que
> habría hecho falta construir un conjunto de datos etiquetado de frutas en
> distintos estados, que no existe dentro del alcance del taller. Y un segundo
> modelo grande no cabe en el límite de tamaño de una función serverless.”

**“¿Cuál es la precisión del sistema?”**
> “No tenemos un número de precisión real, y no vamos a inventarlo. Nuestras
> pruebas usan **imágenes sintéticas** — formas planas de color — y por eso miden
> la lógica del pipeline, no la precisión. Una manzana plana y sin textura es
> indistinguible de un tomate para cualquier clasificador. Para decir un
> porcentaje habría que armar un conjunto de fotos reales etiquetadas.”

**“¿Por qué usan Django y FastAPI a la vez?”**
> “El frontend sirve las páginas y toma la foto; el backend hace el análisis.
> Separarlos permite que el análisis viva en su propia función y que el frontend
> no sepa nada de modelos. Además FastAPI nos da **Swagger** automático, así que
> la API es probable sin escribir código.”

**“¿Por qué no usaron OpenCV?”**
> “Porque desplegamos en funciones serverless con límite de tamaño, y OpenCV es
> una dependencia binaria pesada. Con numpy y Pillow resolvemos la segmentación,
> el color y la forma, y además el código queda explícito: se ve exactamente qué
> cálculo hace. La limitación es que no tenemos etiquetado de componentes
> conectados de scipy, y por eso escribimos el crecimiento de región a mano —
> que dicho sea de paso fue lo que arregló el recorte de productos alargados.”

**“¿Qué pasa si le muestro una piedra?”**
> “Responde `"Unknown"`, y `method` dice `unrecognized`. Es el caso que
> corregimos: antes devolvía el nombre de una clase de ImageNet tal cual, aunque
> fuera un borrador de goma.”

**“¿Por qué las respuestas traen tantos campos si el enunciado pedía tres?”**
> “`item`, `quality` y `ripeness` están tal cual los pide el enunciado. Los demás
> los agregamos para que el resultado sea **auditable**: en qué se basó
> (`candidates`), con qué método (`method`), qué se midió (`diagnostics`) y por
> qué (`notes`). Sin eso, la respuesta sería un dato que hay que creer.”

**“¿Cómo protegen las contraseñas?”**
> “Con **bcrypt**, nunca en texto plano. Y validamos explícitamente el límite de
> 72 bytes que tiene bcrypt, para dar un error claro en vez de un comportamiento
> raro.”

**“¿Y el token de sesión?”**
> “Es un JWT firmado, válido 60 minutos. Y no llega nunca al navegador: vive en la
> sesión de Django y es Django, desde el servidor, quien lo reenvía a la API. La
> cookie de sesión es HttpOnly, así que el JavaScript de la página no la puede
> leer.”

**“¿Qué pasa con los datos si el servidor se reinicia?”**
> “En el despliegue sin base de datos configurada, se reinician: es una decisión
> consciente, porque las funciones serverless no tienen disco. El código ya está
> preparado para usar un Postgres administrado en cuanto se configure
> `DATABASE_URL`, y eso lo documentamos en el README.”

**“¿Qué es eso de CLIP que aparece en el repositorio?”**
> “Es la etapa que **no** pudimos conectar y no escondemos. Los modelos que
> conseguimos estaban guardados con una cuantización incompatible con CLIP, y
> devolvían ruido. Lo detectamos midiendo: dos frases distintas quedaban a 0.996
> de similitud. El tokenizador sí quedó funcionando y probado, y hay un script
> que re-exporta los modelos y **valida antes de instalarlos**. Preferimos no
> conectarlo antes que entregar una etapa que responde mal.”

## Anexo 3 · Frases que conviene evitar

| ❌ No decir | ✅ Decir en su lugar |
|---|---|
| “La IA reconoce el producto.” | “Un modelo preentrenado reconoce 23 productos; para los otros 30 usamos un clasificador de color y forma, y lo distinguimos en el campo `method`.” |
| “Nuestro modelo tiene un 90 % de precisión.” | “No medimos precisión real; las pruebas usan imágenes sintéticas y verifican la lógica.” |
| “Entrenamos una red neuronal para la calidad.” | “La calidad y la madurez son heurísticas de color con umbrales relativos, y lo decimos en la respuesta.” |
| “Funciona con cualquier fruta.” | “Cubre 53 productos; fuera de esa lista responde `Unknown` a propósito.” |
| “Los datos se guardan.” | “Persisten si se configura una base de datos; en el despliegue por defecto se reinician, y está documentado.” |
| “El proyecto quedó perfecto.” | “Estas son sus limitaciones, y esto es lo que dejamos preparado para resolverlas.” |

## Anexo 4 · Si sobra o falta tiempo

**Si van ~2 minutos adelantados:** en el bloque de la demo, hacer además un
análisis **con la foto de fondo del mismo color** y explicar cómo el sistema
detecta la mala segmentación y **baja la confianza** en vez de mentir. Es el
ejemplo más contundente de que el sistema se autocritica.

**Si van ~2 minutos atrasados:** en el bloque 4 recortar la parte del
crecimiento de región y decir solo: “la segmentación separa el producto del
fondo con la mediana del borde, y el color y la forma se miden con numpy, sin
OpenCV”. Y en el bloque 8 mencionar solo las tres pruebas más fuertes: la
regresión del `"Rubber Eraser"`, la invariante de la taxonomía y la de cero
consultas a la base de datos.

**Si se cae la demo por completo:** no improvisar. Pasar a las capturas de
pantalla incluidas en la presentación, mostrar el JSON de ejemplo del `README` y
decir: “la demo está preparada para mostrarse en vivo; mientras resolvemos, les
muestro exactamente la misma respuesta que acabamos de obtener en los ensayos”.
Nunca culpar a la herramienta: describir el problema y seguir.
