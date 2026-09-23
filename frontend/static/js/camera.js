/**
 * camera.js — Captura por cámara y conversación con la API.
 *
 * Responsabilidades:
 *   1. Encender la cámara con `getUserMedia` (prefiriendo la trasera) y
 *      mantener el marco del visor con la proporción real del sensor.
 *   2. Capturar la foto **recortada a la guía de encuadre**. Esto es clave:
 *      el modelo recibe solo la zona que el usuario encuadró, en vez de una
 *      foto panorámica donde el producto ocupa un 10% de los píxeles.
 *   3. Enviarla al backend de Django (que a su vez la reenvía a FastAPI) y
 *      pintar el resultado en el panel de análisis.
 *   4. Cargar el historial y las estadísticas del usuario.
 *   5. Alternativas cuando no hay cámara: subir un archivo.
 *
 * El trabajo visual vive en `visual.js` (namespace global `VF`).
 */
(function () {
    "use strict";

    /* ==================================================================
       Referencias del DOM
       ================================================================== */
    var video = document.getElementById("camera-video");
    var previewImg = document.getElementById("preview-image");
    var canvas = document.getElementById("capture-canvas");
    var viewport = document.getElementById("viewport");
    var guide = document.getElementById("viewport-guide");
    var statusEl = document.getElementById("camera-status");
    var metaEl = document.getElementById("camera-meta");
    var hudResolution = document.getElementById("hud-resolution");
    var hudState = document.getElementById("hud-state");

    var startBtn = document.getElementById("start-camera-btn");
    var captureBtn = document.getElementById("capture-btn");
    var retakeBtn = document.getElementById("retake-btn");
    var analyzeBtn = document.getElementById("analyze-btn");
    var switchBtn = document.getElementById("switch-camera-btn");
    var fileInput = document.getElementById("file-input");

    var emptyPanel = document.getElementById("analysis-empty");
    var resultPanel = document.getElementById("result-panel");
    var resultItem = document.getElementById("result-item");
    var resultCategory = document.getElementById("result-category");
    var resultCategoryText = document.getElementById("result-category-text");
    var resultQuality = document.getElementById("result-quality");
    var resultRipeness = document.getElementById("result-ripeness");
    var resultMethod = document.getElementById("result-method");
    var resultMethodNote = document.getElementById("result-method-note");
    var resultShape = document.getElementById("result-shape");
    var resultSegmentation = document.getElementById("result-segmentation");
    var resultBlemish = document.getElementById("result-blemish");
    var shapeFill = document.getElementById("shape-fill");
    var segmentationFill = document.getElementById("segmentation-fill");
    var blemishFill = document.getElementById("blemish-fill");
    var copyBtn = document.getElementById("copy-result-btn");

    var historyList = document.getElementById("history-list");
    var historyHead = document.getElementById("history-head");
    var historyCount = document.getElementById("history-count");
    var historyPlaceholder = document.getElementById("history-placeholder");
    var historyEmpty = document.getElementById("history-empty");

    var config = window.VALIDADOR_CONFIG || {};
    var VF = window.VF || {};

    /* ==================================================================
       Constantes y estado
       ================================================================== */
    var MAX_OUTPUT_DIMENSION = 1024; // lado mayor de la imagen que se envía
    var ANALYZE_TIMEOUT_MS = 45000; // el arranque en frío de Vercel es lento
    var JPEG_QUALITY = 0.9;
    var MAX_FILE_BYTES = 8 * 1024 * 1024; // mismo límite que valida el backend

    var stream = null;
    var capturedBlob = null;
    var previewUrl = null;
    var facingMode = "environment"; // "environment" (trasera) | "user" (frontal)
    var isAnalyzing = false;
    var lastResult = null;

    var CATEGORY_LABEL = { Fruit: "Fruta", Vegetable: "Verdura", Unknown: "Sin identificar" };
    // Nombres de los productos en español, para mostrar junto al inglés que
    // devuelve la API (el contrato se mantiene en inglés, como pide el taller).
    var ITEM_ES = {
        Apple: "Manzana", Strawberry: "Fresa", Orange: "Naranja", Lemon: "Limón",
        Fig: "Higo", Pineapple: "Piña", Banana: "Banano", Jackfruit: "Jaca",
        "Custard Apple": "Anón", Pomegranate: "Granada", Cabbage: "Repollo",
        Broccoli: "Brócoli", Cauliflower: "Coliflor", Zucchini: "Calabacín",
        "Spaghetti Squash": "Ahuyama espagueti", "Acorn Squash": "Ahuyama bellota",
        "Butternut Squash": "Ahuyama mantequilla", Cucumber: "Pepino",
        Artichoke: "Alcachofa", "Bell Pepper": "Pimentón", Cardoon: "Cardo",
        Mushroom: "Champiñón", Corn: "Mazorca", Papaya: "Papaya", Mango: "Mango",
        Guava: "Guayaba", Watermelon: "Sandía", Melon: "Melón", Avocado: "Aguacate",
        Tomato: "Tomate", Kiwi: "Kiwi", Peach: "Durazno", Pear: "Pera",
        Grape: "Uva", Plum: "Ciruela", "Passion Fruit": "Maracuyá",
        Cherry: "Cereza", Coconut: "Coco", Lime: "Lima", Plantain: "Plátano",
        Carrot: "Zanahoria", Potato: "Papa", Onion: "Cebolla", Garlic: "Ajo",
        Eggplant: "Berenjena", Beet: "Remolacha", Lettuce: "Lechuga",
        Spinach: "Espinaca", "Chili Pepper": "Ají", Cassava: "Yuca",
        "Green Bean": "Habichuela", "Sweet Potato": "Batata", Radish: "Rábano",
        Unknown: "Sin identificar",
    };

    /* ==================================================================
       Utilidades de interfaz
       ================================================================== */
    function setStatus(message, tone) {
        if (!statusEl) return;
        statusEl.textContent = message || "";
        if (tone) statusEl.setAttribute("data-tone", tone);
        else statusEl.removeAttribute("data-tone");
    }

    function setMeta(text, tone) {
        if (!metaEl) return;
        metaEl.textContent = text;
        metaEl.setAttribute("data-live", tone || "false");
    }

    function setState(state) {
        if (viewport) viewport.setAttribute("data-state", state);
    }

    function setHud(state, resolution) {
        if (hudState) hudState.textContent = state;
        if (resolution && hudResolution) hudResolution.textContent = resolution;
    }

    function show(node) {
        if (node) node.hidden = false;
    }

    function hide(node) {
        if (node) node.hidden = true;
    }

    function toast(message, tone) {
        if (VF.toast) VF.toast.show(message, tone);
    }

    function stopStream() {
        if (stream) {
            stream.getTracks().forEach(function (track) {
                track.stop();
            });
            stream = null;
        }
    }

    function releasePreview() {
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
            previewUrl = null;
        }
    }

    function isFrontCamera() {
        return facingMode === "user";
    }

    /* ==================================================================
       Cámara
       ================================================================== */
    function supportsCamera() {
        return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    }

    async function startCamera() {
        setStatus("");

        if (!supportsCamera()) {
            setStatus(
                "Este navegador no expone la cámara. Usa Chrome, Edge o Firefox actualizados, o sube una foto con el botón «Subir archivo».",
                "error"
            );
            toast("Tu navegador no soporta captura por cámara. Puedes subir un archivo.", "error");
            return;
        }

        if (!window.isSecureContext) {
            setStatus(
                "La cámara solo funciona en HTTPS o en localhost. Abre la app por HTTPS o usa «Subir archivo».",
                "error"
            );
            return;
        }

        startBtn.disabled = true;
        setStatus("Pidiendo permiso para usar la cámara…");
        setMeta("conectando", false);

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { ideal: facingMode },
                    width: { ideal: 1920 },
                    height: { ideal: 1080 },
                },
                audio: false,
            });

            video.srcObject = stream;
            video.classList.toggle("is-mirrored", isFrontCamera());

            // Esperamos a que el navegador sepa el tamaño real del video para
            // ajustar el marco y poder calcular el recorte correctamente.
            await waitForVideoMetadata();

            applyViewportRatio(video.videoWidth, video.videoHeight);

            show(video);
            hide(previewImg);
            hide(startBtn);
            show(captureBtn);
            show(switchBtn);
            hide(retakeBtn);
            hide(analyzeBtn);
            setState("live");
            setHud("EN VIVO", video.videoWidth + "×" + video.videoHeight);
            setMeta("activa", true);
            setStatus("Cámara lista. Encuadra el producto dentro de la guía y toma la foto.");
        } catch (error) {
            setMeta("error", "error");
            setStatus(describeCameraError(error), "error");
            toast(describeCameraError(error), "error");
        } finally {
            startBtn.disabled = false;
        }
    }

    function waitForVideoMetadata() {
        return new Promise(function (resolve) {
            if (video.readyState >= 1 && video.videoWidth) {
                resolve();
                return;
            }
            var done = function () {
                video.removeEventListener("loadedmetadata", done);
                resolve();
            };
            video.addEventListener("loadedmetadata", done, { once: true });
        });
    }

    function describeCameraError(error) {
        var name = error && error.name;
        if (name === "NotAllowedError" || name === "SecurityError") {
            return "Permiso de cámara denegado. Habilítalo en el candado de la barra de direcciones y vuelve a intentar.";
        }
        if (name === "NotFoundError" || name === "OverconstrainedError") {
            return "No se encontró ninguna cámara disponible en este dispositivo.";
        }
        if (name === "NotReadableError" || name === "TrackStartError") {
            return "La cámara está siendo usada por otra aplicación. Ciérrala e intenta otra vez.";
        }
        return "No se pudo abrir la cámara: " + (error && error.message ? error.message : "error desconocido") + ".";
    }

    /** Ajusta el marco del visor a la proporción real del sensor. */
    function applyViewportRatio(width, height) {
        if (!viewport || !width || !height) return;
        var ratio = width / height;
        var label = "1:1";
        if (ratio > 1.6) label = "16:9";
        else if (ratio > 1.15) label = "4:3";
        else if (ratio < 0.85) label = "3:4";
        viewport.setAttribute("data-ratio", label);
    }

    function switchCamera() {
        facingMode = isFrontCamera() ? "environment" : "user";
        stopStream();
        startCamera();
    }

    /* ==================================================================
       Captura recortada a la guía
       ================================================================== */
    /**
     * Calcula la región del video que corresponde a la guía de encuadre.
     *
     * El visor muestra el video con `object-fit: cover`, así que hay que
     * deshacer ese escalado y centrado para pasar de coordenadas de pantalla
     * a coordenadas del sensor. Sin esto, el recorte no coincide con lo que
     * el usuario ve dentro de las esquinas.
     */
    function guideRegionInVideo() {
        var videoW = video.videoWidth;
        var videoH = video.videoHeight;
        var viewRect = viewport.getBoundingClientRect();
        var guideRect = guide ? guide.getBoundingClientRect() : null;

        var fallback = { sx: 0, sy: 0, sw: videoW, sh: videoH };
        if (!videoW || !videoH || !viewRect.width || !viewRect.height) return fallback;

        // Área de la guía en coordenadas del visor (o el visor completo).
        var gx = 0;
        var gy = 0;
        var gw = viewRect.width;
        var gh = viewRect.height;
        if (guideRect && guideRect.width > 4 && guideRect.height > 4) {
            gx = guideRect.left - viewRect.left;
            gy = guideRect.top - viewRect.top;
            gw = guideRect.width;
            gh = guideRect.height;
        }

        var scale = Math.max(viewRect.width / videoW, viewRect.height / videoH);
        var offsetX = (viewRect.width - videoW * scale) / 2;
        var offsetY = (viewRect.height - videoH * scale) / 2;

        var sx = (gx - offsetX) / scale;
        var sy = (gy - offsetY) / scale;
        var sw = gw / scale;
        var sh = gh / scale;

        // Recortamos a los límites del sensor y dejamos un pequeño margen
        // para no cortar el producto justo en el borde de la guía.
        var margin = 0.03;
        sx = clamp(sx - sw * margin, 0, videoW);
        sy = clamp(sy - sh * margin, 0, videoH);
        sw = clamp(sw * (1 + margin * 2), 1, videoW - sx);
        sh = clamp(sh * (1 + margin * 2), 1, videoH - sy);

        return { sx: sx, sy: sy, sw: sw, sh: sh };
    }

    function clamp(value, min, max) {
        return Math.min(Math.max(value, min), max);
    }

    function flashEffect() {
        if (!viewport || VF.REDUCED_MOTION) return;
        var flash = document.createElement("div");
        flash.className = "capture-flash";
        viewport.appendChild(flash);
        window.setTimeout(function () {
            if (flash.parentNode) flash.parentNode.removeChild(flash);
        }, 500);
    }

    function capturePhoto() {
        if (viewport && viewport.getAttribute("data-state") === "idle") {
            startCamera();
            return;
        }

        var nativeWidth = video.videoWidth;
        var nativeHeight = video.videoHeight;
        if (!nativeWidth || !nativeHeight) {
            setStatus("La cámara todavía no está lista. Espera un instante e intenta de nuevo.", "error");
            return;
        }

        flashEffect();

        var region = guideRegionInVideo();
        var scale = Math.min(1, MAX_OUTPUT_DIMENSION / Math.max(region.sw, region.sh));
        var outWidth = Math.max(1, Math.round(region.sw * scale));
        var outHeight = Math.max(1, Math.round(region.sh * scale));

        canvas.width = outWidth;
        canvas.height = outHeight;
        var ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, outWidth, outHeight);

        if (isFrontCamera()) {
            // La vista previa es un espejo; la foto también debe serlo, para
            // que se vea igual a lo que el usuario encuadró.
            ctx.translate(outWidth, 0);
            ctx.scale(-1, 1);
        }

        ctx.drawImage(video, region.sx, region.sy, region.sw, region.sh, 0, 0, outWidth, outHeight);
        ctx.setTransform(1, 0, 0, 1, 0, 0);

        canvas.toBlob(
            function (blob) {
                if (!blob) {
                    setStatus("No se pudo procesar la captura. Intenta otra vez.", "error");
                    return;
                }
                setCapturedBlob(blob, outWidth, outHeight, "capturada");
            },
            "image/jpeg",
            JPEG_QUALITY
        );
    }

    function setCapturedBlob(blob, width, height, origin) {
        releasePreview();
        capturedBlob = blob;
        previewUrl = URL.createObjectURL(blob);
        previewImg.src = previewUrl;

        show(previewImg);
        hide(video);
        stopStream();

        hide(captureBtn);
        hide(switchBtn);
        show(retakeBtn);
        show(analyzeBtn);
        analyzeBtn.disabled = false;

        setState("captured");
        var size = width && height ? width + "×" + height : "—";
        setHud(origin.toUpperCase() + " · " + size, size);
        setMeta("foto lista", "true");
        setStatus("Foto lista (" + size + " px). Revísala y presiona «Analizar».");

        // El resultado anterior ya no corresponde a esta foto.
        if (resultPanel) resultPanel.hidden = true;
        if (emptyPanel) emptyPanel.hidden = false;
    }

    function retake() {
        releasePreview();
        capturedBlob = null;
        lastResult = null;
        if (resultPanel) resultPanel.hidden = true;
        if (emptyPanel) emptyPanel.hidden = false;
        hide(retakeBtn);
        hide(analyzeBtn);
        hide(previewImg);
        startCamera();
    }

    /* ==================================================================
       Subir archivo (alternativa a la cámara)
       ================================================================== */
    function handleFile(file) {
        if (!file) return;

        if (!/^image\/(jpeg|png|webp)$/.test(file.type)) {
            setStatus("Formato no soportado. Usa una imagen JPG, PNG o WEBP.", "error");
            toast("Formato no soportado: usa JPG, PNG o WEBP.", "error");
            return;
        }

        if (file.size > MAX_FILE_BYTES) {
            setStatus("La imagen pesa más de 8 MB. Elige una más liviana.", "error");
            toast("La imagen supera el límite de 8 MB.", "error");
            return;
        }

        stopStream();
        hide(video);
        hide(startBtn);
        hide(captureBtn);
        hide(switchBtn);

        // El archivo se envía tal cual: no lo recortamos porque no conocemos
        // su encuadre. El backend lo analiza completo.
        readImageSize(file).then(function (size) {
            if (size) setCapturedBlob(file, size.width, size.height, "archivo");
            else setCapturedBlob(file, 0, 0, "archivo");
        });
    }

    /** Lee las dimensiones reales de una imagen sin subirla a ningún lado. */
    function readImageSize(file) {
        return new Promise(function (resolve) {
            var url = URL.createObjectURL(file);
            var probe = new Image();
            var finish = function (size) {
                URL.revokeObjectURL(url);
                resolve(size);
            };
            probe.onload = function () {
                finish({ width: probe.naturalWidth, height: probe.naturalHeight });
            };
            probe.onerror = function () {
                finish(null);
            };
            probe.src = url;
        });
    }

    /* ==================================================================
       Análisis
       ================================================================== */
    async function analyze() {
        if (isAnalyzing) return;
        if (!capturedBlob) {
            setStatus("Primero toma una foto o sube un archivo.", "error");
            toast("Necesitas una foto para analizar.", "error");
            return;
        }
        if (!config.analyzeUrl) {
            setStatus("Falta configurar la URL de análisis en la página.", "error");
            return;
        }

        isAnalyzing = true;
        analyzeBtn.disabled = true;
        retakeBtn.disabled = true;
        setState("analyzing");
        setMeta("analizando", false);
        setHud("ANALIZANDO", hudResolution ? hudResolution.textContent : "");
        setStatus("Analizando la foto. Puede tardar unos segundos si el servidor estaba dormido…");
        if (VF.stages) VF.stages.start();

        var formData = new FormData();
        formData.append("photo", capturedBlob, "captura.jpg");

        var controller = new AbortController();
        var timeoutId = window.setTimeout(function () {
            controller.abort();
        }, ANALYZE_TIMEOUT_MS);

        try {
            var response = await fetch(config.analyzeUrl, {
                method: "POST",
                headers: { "X-CSRFToken": config.csrfToken || "" },
                body: formData,
                signal: controller.signal,
                credentials: "same-origin",
            });

            var data;
            try {
                data = await response.json();
            } catch (parseError) {
                throw new Error("El servidor respondió algo que no pudimos interpretar. Intenta otra vez.");
            }

            if (!response.ok) {
                throw new Error(data.detail || "No se pudo analizar la imagen.");
            }

            renderResult(data);
            setStatus("Análisis terminado. Revisa el resultado a la derecha.", "ok");
            setMeta("listo", "true");
            setState("result");
            setHud("RESULTADO", hudResolution ? hudResolution.textContent : "");
            loadHistory();
        } catch (error) {
            setState("captured");
            setMeta("error", "error");
            setHud("ERROR", hudResolution ? hudResolution.textContent : "");

            if (error.name === "AbortError") {
                setStatus(
                    "El análisis tardó demasiado. Suele pasar cuando el servidor está arrancando en frío: intenta de nuevo.",
                    "error"
                );
                toast("El análisis expiró. Vuelve a intentarlo.", "error");
            } else {
                setStatus(error.message, "error");
                toast(error.message, "error");
            }
        } finally {
            window.clearTimeout(timeoutId);
            if (VF.stages) VF.stages.stop();
            analyzeBtn.disabled = false;
            retakeBtn.disabled = false;
            isAnalyzing = false;
        }
    }

    function renderResult(result) {
        lastResult = result;

        if (emptyPanel) emptyPanel.hidden = true;
        if (resultPanel) resultPanel.hidden = false;
        if (copyBtn) copyBtn.hidden = false;

        var unknown = !result.item || result.item === "Unknown";
        var spanish = ITEM_ES[result.item] || "";

        if (resultItem) {
            resultItem.textContent = unknown
                ? "No identificado"
                : spanish
                  ? result.item + " · " + spanish
                  : result.item;
        }
        if (resultCategory) {
            resultCategory.textContent =
                (CATEGORY_LABEL[result.category] || "Producto") + " · " + result.method;
        }
        if (resultCategoryText) {
            resultCategoryText.textContent = CATEGORY_LABEL[result.category] || "—";
        }

        var accent = VF.COLOR_HEX ? VF.COLOR_HEX.orange : "#b5522f";
        if (result.quality && VF.QUALITY_TONE) {
            var tone = VF.QUALITY_TONE[result.quality];
            if (tone === "good") accent = "#5c6b33";
            else if (tone === "bad") accent = "#9c3524";
            else if (tone === "regular") accent = "#a9761b";
        }

        if (VF.setGauge) VF.setGauge(result.confidence, accent);
        if (VF.ambient) {
            VF.ambient.setAccent(accent);
            VF.ambient.burst(accent, unknown ? 12 : 30);
        }

        if (resultQuality) {
            resultQuality.textContent = result.quality || "—";
            resultQuality.setAttribute(
                "data-tone",
                (VF.QUALITY_TONE && VF.QUALITY_TONE[result.quality]) || "unknown"
            );
        }
        if (resultRipeness) {
            resultRipeness.textContent = result.ripeness || "—";
            resultRipeness.setAttribute(
                "data-tone",
                (VF.RIPENESS_TONE && VF.RIPENESS_TONE[result.ripeness]) || "unknown"
            );
        }
        if (resultMethod) {
            resultMethod.textContent = (VF.METHOD_LABEL && VF.METHOD_LABEL[result.method]) || result.method || "—";
            resultMethod.setAttribute("data-tone", methodTone(result.method));
        }
        if (resultMethodNote) {
            resultMethodNote.textContent = (VF.METHOD_NOTE && VF.METHOD_NOTE[result.method]) || "";
        }

        var diagnostics = result.diagnostics || {};
        var elongation = typeof diagnostics.elongation === "number" ? diagnostics.elongation : 0;
        var segmentation = typeof diagnostics.segmentation_confidence === "number" ? diagnostics.segmentation_confidence : 0;
        var blemish = typeof diagnostics.blemish_ratio === "number" ? diagnostics.blemish_ratio : 0;

        if (resultShape) resultShape.textContent = elongation.toFixed(2) + "×";
        if (resultSegmentation) resultSegmentation.textContent = Math.round(segmentation * 100) + "%";
        if (resultBlemish) resultBlemish.textContent = Math.round(blemish * 100) + "%";

        if (VF.animateBar) {
            // La forma se normaliza con 2.5× (un plátano ronda 2.0-2.5).
            VF.animateBar(shapeFill, elongation / 2.5);
            VF.animateBar(segmentationFill, segmentation);
            VF.animateBar(blemishFill, blemish / 0.5);
        }
        if (segmentationFill) {
            segmentationFill.style.background = segmentation < 0.4 ? "#9c3524" : "#5c6b33";
        }
        if (blemishFill) {
            blemishFill.style.background = blemish > 0.3 ? "#9c3524" : blemish > 0.12 ? "#a9761b" : "#5c6b33";
        }

        if (VF.renderColors) VF.renderColors(diagnostics.colors || {});
        if (VF.renderCandidates) VF.renderCandidates(result.candidates || []);
        if (VF.renderNotes) VF.renderNotes(result.notes || []);

        if (unknown) {
            toast("No pudimos identificar el producto. Prueba con más luz y un fondo liso.", "error");
        } else {
            toast(result.item + " identificado como " + result.quality + " / " + result.ripeness + ".", "ok");
        }
    }

    function methodTone(method) {
        if (method === "resnet18-imagenet") return "model";
        if (method === "color-shape-heuristics") return "heuristic";
        return "unrecognized";
    }

    /* ==================================================================
       Historial y estadísticas
       ================================================================== */
    function renderHistory(items) {
        if (!historyList) return;
        historyList.innerHTML = "";

        var hasItems = items && items.length;
        if (historyPlaceholder) historyPlaceholder.hidden = hasItems;
        if (historyEmpty) historyEmpty.hidden = Boolean(hasItems);
        if (historyHead) historyHead.hidden = !hasItems;
        if (historyList) historyList.hidden = !hasItems;

        renderStats(items || []);

        if (!hasItems) return;

        items.forEach(function (entry, index) {
            var li = document.createElement("li");
            li.className = "history__row";

            var number = document.createElement("span");
            number.className = "history__index";
            number.textContent = String(index + 1).padStart(2, "0");

            var name = document.createElement("span");
            name.className = "history__item";
            var spanish = ITEM_ES[entry.item];
            name.textContent = entry.item + (spanish && spanish !== entry.item ? " · " + spanish : "");
            var sub = document.createElement("small");
            sub.textContent = CATEGORY_LABEL[entry.category] || "Producto";
            name.appendChild(sub);

            var chips = document.createElement("span");
            chips.className = "history__chips";
            chips.appendChild(
                buildChip(entry.quality, (VF.QUALITY_TONE && VF.QUALITY_TONE[entry.quality]) || "unknown")
            );
            chips.appendChild(
                buildChip(entry.ripeness, (VF.RIPENESS_TONE && VF.RIPENESS_TONE[entry.ripeness]) || "unknown")
            );

            var method = document.createElement("span");
            method.className = "history__method";
            method.textContent =
                ((VF.METHOD_LABEL && VF.METHOD_LABEL[entry.method]) || entry.method || "—") +
                " · " + Math.round((entry.confidence || 0) * 100) + "%";

            var date = document.createElement("time");
            date.className = "history__date";
            date.dateTime = entry.created_at || "";
            date.textContent = VF.formatDate ? VF.formatDate(entry.created_at) : entry.created_at;

            li.appendChild(number);
            li.appendChild(name);
            li.appendChild(chips);
            li.appendChild(method);
            li.appendChild(date);
            historyList.appendChild(li);
        });
    }

    function buildChip(text, tone) {
        var span = document.createElement("span");
        span.className = "chip";
        span.setAttribute("data-tone", tone);
        span.textContent = text || "—";
        return span;
    }

    function renderStats(items) {
        var total = items.length;

        var totalEl = document.getElementById("stat-total");
        if (totalEl) {
            if (VF.countUp) {
                VF.countUp(totalEl, total, {
                    duration: 800,
                    render: function (value) {
                        return String(Math.round(value));
                    },
                });
            } else {
                totalEl.textContent = String(total);
            }
        }

        var confidenceEl = document.getElementById("stat-confidence");
        var known = items.filter(function (entry) {
            return entry.item && entry.item !== "Unknown";
        });
        if (confidenceEl) {
            if (known.length) {
                var average =
                    known.reduce(function (sum, entry) {
                        return sum + (entry.confidence || 0);
                    }, 0) / known.length;
                confidenceEl.textContent = Math.round(average * 100) + "%";
            } else {
                confidenceEl.textContent = "—";
            }
        }

        var mixEl = document.getElementById("stat-mix");
        if (mixEl) {
            var fruits = items.filter(function (entry) {
                return entry.category === "Fruit";
            }).length;
            var vegetables = items.filter(function (entry) {
                return entry.category === "Vegetable";
            }).length;
            mixEl.innerHTML = total
                ? fruits + '<small>frutas</small> · ' + vegetables + "<small>verduras</small>"
                : "—";
        }

        var lastEl = document.getElementById("stat-last");
        if (lastEl) {
            var last = items[0];
            lastEl.textContent = last && last.created_at
                ? (VF.formatDate ? VF.formatDate(last.created_at) : last.created_at)
                : "—";
        }

        if (historyCount) {
            historyCount.textContent = total ? total + (total === 1 ? " registro" : " registros") : "sin registros";
        }
    }

    async function loadHistory() {
        if (!config.historyUrl) return;

        try {
            var response = await fetch(config.historyUrl, {
                headers: { Accept: "application/json" },
                credentials: "same-origin",
            });
            if (!response.ok) throw new Error("HTTP " + response.status);
            var data = await response.json();
            renderHistory(data.history || []);
        } catch (error) {
            if (historyPlaceholder) {
                historyPlaceholder.textContent =
                    "No se pudo cargar el historial en este momento. Recarga la página para reintentar.";
            }
        }
    }

    /* ==================================================================
       Copiar el resultado
       ================================================================== */
    async function copyResult() {
        if (!lastResult) return;
        var payload = JSON.stringify(lastResult, null, 2);
        try {
            await navigator.clipboard.writeText(payload);
            toast("Resultado copiado como JSON.", "ok");
        } catch (error) {
            // clipboard API requiere contexto seguro; dejamos una salida.
            var area = document.createElement("textarea");
            area.value = payload;
            area.style.position = "fixed";
            area.style.opacity = "0";
            document.body.appendChild(area);
            area.select();
            try {
                document.execCommand("copy");
                toast("Resultado copiado como JSON.", "ok");
            } catch (fallbackError) {
                toast("No se pudo copiar automáticamente.", "error");
            }
            document.body.removeChild(area);
        }
    }

    /* ==================================================================
       Eventos
       ================================================================== */
    function bind() {
        if (startBtn) startBtn.addEventListener("click", startCamera);
        if (captureBtn) captureBtn.addEventListener("click", capturePhoto);
        if (retakeBtn) retakeBtn.addEventListener("click", retake);
        if (analyzeBtn) analyzeBtn.addEventListener("click", analyze);
        if (switchBtn) switchBtn.addEventListener("click", switchCamera);
        if (copyBtn) copyBtn.addEventListener("click", copyResult);

        if (fileInput) {
            fileInput.addEventListener("change", function (event) {
                var file = event.target.files && event.target.files[0];
                handleFile(file);
                fileInput.value = ""; // permite volver a subir el mismo archivo
            });
        }

        // Arrastrar y soltar una imagen sobre el visor.
        if (viewport) {
            ["dragenter", "dragover"].forEach(function (type) {
                viewport.addEventListener(type, function (event) {
                    event.preventDefault();
                    viewport.style.outline = "2px dashed var(--terracotta)";
                    viewport.style.outlineOffset = "-6px";
                });
            });
            ["dragleave", "drop"].forEach(function (type) {
                viewport.addEventListener(type, function (event) {
                    event.preventDefault();
                    viewport.style.outline = "";
                });
            });
            viewport.addEventListener("drop", function (event) {
                var file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
                handleFile(file);
            });
        }

        document.addEventListener("keydown", handleShortcut);
        window.addEventListener("beforeunload", function () {
            stopStream();
            releasePreview();
        });
    }

    function handleShortcut(event) {
        var target = event.target;
        var typing =
            target &&
            (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable);
        if (typing || event.ctrlKey || event.metaKey || event.altKey) return;

        if (event.code === "Space") {
            event.preventDefault();
            // Espacio alterna: enciende la cámara o dispara la captura.
            if (captureBtn && !captureBtn.hidden) capturePhoto();
            else if (startBtn && !startBtn.hidden) startCamera();
            return;
        }

        if (event.key === "Enter") {
            if (analyzeBtn && !analyzeBtn.hidden && !analyzeBtn.disabled) {
                event.preventDefault();
                analyze();
            }
            return;
        }

        if (event.key === "r" || event.key === "R") {
            if (retakeBtn && !retakeBtn.hidden) {
                event.preventDefault();
                retake();
            }
        }
    }

    /* ==================================================================
       Arranque
       ================================================================== */
    bind();

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", loadHistory);
    } else {
        loadHistory();
    }

    // Expuesto para depurar desde la consola del navegador.
    window.ValidadorFrutas = {
        getCapturedBlob: function () {
            return capturedBlob;
        },
        getLastResult: function () {
            return lastResult;
        },
    };
})();
