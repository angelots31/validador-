/**
 * visual.js — Capa visual del Validador de Frutas y Verduras.
 *
 * Este archivo NO habla con la cámara ni con la API: se encarga de todo lo
 * que se ve. Expone un namespace global `VF` que `camera.js` consume.
 *
 * Contenido:
 *   VF.ambient    Fondo animado en <canvas>: manchas de color que derivan
 *                 lentamente + ráfaga de partículas al terminar un análisis.
 *   VF.toast      Notificaciones flotantes.
 *   VF.clock      Reloj de la barra superior.
 *   VF.gauge      Arco de confianza y contador animado.
 *   VF.countUp    Animación numérica reutilizable.
 *   VF.colors     Barra de composición de color del producto.
 *   VF.candidates Otras propuestas del clasificador.
 *   VF.notes      Lista "cómo se decidió".
 *   VF.stages     Indicador de etapas del análisis.
 *   VF.reveal     Aparición al hacer scroll.
 *   VF.ripple     Onda al presionar botones.
 *   VF.tilt       Inclinación suave del panel con el puntero.
 *   VF.format*    Utilidades de formato (fecha, porcentaje).
 *
 * Reglas que se respetan en todo el archivo:
 *   - Si un elemento no existe en la página, la función que lo usa no falla.
 *   - Con `prefers-reduced-motion` se dibuja un cuadro fijo, sin animación.
 *   - Nada de animaciones cuando la pestaña está en segundo plano.
 */
(function () {
    "use strict";

    var VF = {};

    var REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    /* ==================================================================
       Paleta compartida
       ================================================================== */
    var COLOR_HEX = {
        green: "#6f8f3f",
        yellow: "#d3a52b",
        orange: "#d4703a",
        red: "#b1402c",
        purple: "#7a5c94",
        brown: "#8a6a4a",
        pale: "#e2d7bd",
    };

    var COLOR_LABEL = {
        green: "verde",
        yellow: "amarillo",
        orange: "naranja",
        red: "rojo",
        purple: "morado",
        brown: "marrón",
        pale: "claro",
    };

    var QUALITY_TONE = { Good: "good", Regular: "regular", Bad: "bad" };
    var RIPENESS_TONE = { Unripe: "unripe", Ripe: "good", Overripe: "bad" };

    var METHOD_LABEL = {
        "resnet18-imagenet": "ResNet18 · ImageNet",
        "color-shape-heuristics": "Color y forma",
        "unrecognized": "No identificado",
    };

    var METHOD_NOTE = {
        "resnet18-imagenet": "identificado por el modelo preentrenado",
        "color-shape-heuristics": "producto fuera de ImageNet: clasificador de respaldo",
        "unrecognized": "sin coincidencias suficientes",
    };

    /* ==================================================================
       Utilidades
       ================================================================== */
    function $(id) {
        return document.getElementById(id);
    }

    function clamp(value, min, max) {
        return Math.min(Math.max(value, min), max);
    }

    function formatPercent(value) {
        if (typeof value !== "number" || isNaN(value)) return "—";
        return Math.round(value * 100) + "%";
    }

    function formatDate(isoString) {
        var date = new Date(isoString);
        if (isNaN(date.getTime())) return isoString || "—";
        return date.toLocaleString("es-CO", {
            day: "2-digit",
            month: "short",
            year: "numeric",
            hour: "2-digit",
            minute: "2-digit",
        });
    }

    function formatShortDate(isoString) {
        var date = new Date(isoString);
        if (isNaN(date.getTime())) return "—";
        return date.toLocaleDateString("es-CO", { day: "2-digit", month: "short" });
    }

    /** easing suave para contadores y barras */
    function easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    /**
     * Anima un número desde `from` hasta `to` escribiéndolo en `el`.
     * `render` permite formatear (porcentajes, decimales...).
     */
    function countUp(el, to, options) {
        if (!el) return;
        options = options || {};
        var from = typeof options.from === "number" ? options.from : 0;
        var duration = options.duration || 900;
        var render = options.render || function (v) { return String(Math.round(v)); };

        if (REDUCED_MOTION) {
            el.textContent = render(to);
            return;
        }

        var start = null;
        function step(timestamp) {
            if (start === null) start = timestamp;
            var progress = clamp((timestamp - start) / duration, 0, 1);
            var value = from + (to - from) * easeOutCubic(progress);
            el.textContent = render(value);
            if (progress < 1) window.requestAnimationFrame(step);
        }
        window.requestAnimationFrame(step);
    }

    /** Anima el ancho de una barra (elemento con `width` en CSS). */
    function animateBar(el, ratio) {
        if (!el) return;
        var target = clamp(ratio, 0, 1) * 100;
        if (REDUCED_MOTION) {
            el.style.width = target + "%";
            return;
        }
        // Forzamos un reflow para que la transición CSS se dispare siempre.
        el.style.width = "0%";
        void el.offsetWidth;
        el.style.width = target + "%";
    }

    VF.formatPercent = formatPercent;
    VF.formatDate = formatDate;

    /* ==================================================================
       Ambiente: canvas de fondo con manchas de color y partículas
       ================================================================== */
    var ambient = (function () {
        var canvas = document.getElementById("ambient-canvas");
        if (!canvas) {
            return { burst: function () {}, setAccent: function () {} };
        }

        var ctx = canvas.getContext("2d");
        var width = 0;
        var height = 0;
        var dpr = 1;
        var blobs = [];
        var particles = [];
        var running = false;
        var rafId = null;
        var lastFrame = 0;
        var FRAME_MS = 1000 / 30; // 30 fps alcanza para un fondo lento

        var PALETTE = ["#b5522f", "#5c6b33", "#d3a52b", "#6f8f3f", "#8a6a4a"];
        var accent = null;

        function resize() {
            dpr = Math.min(window.devicePixelRatio || 1, 2);
            width = canvas.clientWidth || window.innerWidth;
            height = canvas.clientHeight || window.innerHeight;
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            seed();
            if (!running) drawFrame(0, true);
        }

        function seed() {
            var count = width < 760 ? 3 : 5;
            blobs = [];
            for (var i = 0; i < count; i++) {
                var radius = (Math.min(width, height) * (0.28 + Math.random() * 0.22));
                blobs.push({
                    x: width * (0.1 + Math.random() * 0.8),
                    y: height * (0.1 + Math.random() * 0.8),
                    r: radius,
                    vx: (Math.random() - 0.5) * 0.16,
                    vy: (Math.random() - 0.5) * 0.16,
                    color: PALETTE[i % PALETTE.length],
                });
            }
        }

        function drawBlob(blob, alpha) {
            var fill = accent || blob.color;
            var gradient = ctx.createRadialGradient(blob.x, blob.y, 0, blob.x, blob.y, blob.r);
            gradient.addColorStop(0, hexToRgba(fill, alpha));
            gradient.addColorStop(1, hexToRgba(fill, 0));
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(blob.x, blob.y, blob.r, 0, Math.PI * 2);
            ctx.fill();
        }

        function hexToRgba(hex, alpha) {
            var value = hex.replace("#", "");
            if (value.length === 3) {
                value = value[0] + value[0] + value[1] + value[1] + value[2] + value[2];
            }
            var int = parseInt(value, 16);
            var r = (int >> 16) & 255;
            var g = (int >> 8) & 255;
            var b = int & 255;
            return "rgba(" + r + ", " + g + ", " + b + ", " + alpha + ")";
        }

        function drawParticles() {
            for (var i = particles.length - 1; i >= 0; i--) {
                var p = particles[i];
                p.x += p.vx;
                p.y += p.vy;
                p.vy += 0.045; // gravedad suave
                p.life -= 1;
                if (p.life <= 0) {
                    particles.splice(i, 1);
                    continue;
                }
                var alpha = (p.life / p.maxLife) * 0.5;
                ctx.fillStyle = hexToRgba(p.color, alpha);
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        function drawFrame(timestamp, once) {
            ctx.clearRect(0, 0, width, height);

            for (var i = 0; i < blobs.length; i++) {
                var blob = blobs[i];
                if (!once) {
                    blob.x += blob.vx;
                    blob.y += blob.vy;
                    // Rebote suave dentro del lienzo.
                    if (blob.x < -blob.r * 0.4 || blob.x > width + blob.r * 0.4) blob.vx *= -1;
                    if (blob.y < -blob.r * 0.4 || blob.y > height + blob.r * 0.4) blob.vy *= -1;
                }
                drawBlob(blob, 0.13);
            }

            drawParticles();
        }

        function loop(timestamp) {
            if (!running) return;
            rafId = window.requestAnimationFrame(loop);
            if (timestamp - lastFrame < FRAME_MS) return;
            lastFrame = timestamp;
            drawFrame(timestamp, false);
        }

        function start() {
            if (running || REDUCED_MOTION) return;
            running = true;
            lastFrame = 0;
            rafId = window.requestAnimationFrame(loop);
        }

        function stop() {
            running = false;
            if (rafId) window.cancelAnimationFrame(rafId);
            rafId = null;
        }

        /** Ráfaga de partículas, usada al terminar un análisis. */
        function burst(hexColor, amount) {
            var color = hexColor || "#b5522f";
            var total = amount || 26;
            var originX = width * 0.5;
            var originY = height * 0.42;
            for (var i = 0; i < total; i++) {
                var angle = (Math.PI * 2 * i) / total + Math.random() * 0.3;
                var speed = 1.6 + Math.random() * 3.4;
                var life = 40 + Math.random() * 40;
                particles.push({
                    x: originX,
                    y: originY,
                    vx: Math.cos(angle) * speed,
                    vy: Math.sin(angle) * speed - 1.2,
                    size: 1.4 + Math.random() * 3,
                    color: color,
                    life: life,
                    maxLife: life,
                });
            }
            start();
        }

        /** Tiñe el fondo con el color del último resultado. */
        function setAccent(hexColor) {
            accent = hexColor || null;
        }

        window.addEventListener("resize", debounce(resize, 180));

        document.addEventListener("visibilitychange", function () {
            if (document.hidden) stop();
            else start();
        });

        resize();

        return { burst: burst, setAccent: setAccent, start: start, stop: stop };
    })();

    VF.ambient = ambient;

    function debounce(fn, wait) {
        var timer = null;
        return function () {
            var args = arguments;
            var self = this;
            window.clearTimeout(timer);
            timer = window.setTimeout(function () {
                fn.apply(self, args);
            }, wait);
        };
    }

    /* ==================================================================
       Toasts
       ================================================================== */
    var toast = (function () {
        var stack = document.getElementById("toast-stack");
        var ICONS = { error: "#i-alert", ok: "#i-check", info: "#i-info" };

        function show(message, tone, timeout) {
            if (!stack || !message) return;
            var resolved = tone || "info";

            var node = document.createElement("div");
            node.className = "toast";
            node.setAttribute("data-tone", resolved);
            node.setAttribute("role", resolved === "error" ? "alert" : "status");

            var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
            svg.setAttribute("aria-hidden", "true");
            var use = document.createElementNS("http://www.w3.org/2000/svg", "use");
            use.setAttribute("href", ICONS[resolved] || ICONS.info);
            svg.appendChild(use);

            var text = document.createElement("span");
            text.textContent = message;

            node.appendChild(svg);
            node.appendChild(text);
            stack.appendChild(node);

            window.setTimeout(function () {
                node.classList.add("is-leaving");
                window.setTimeout(function () {
                    if (node.parentNode) node.parentNode.removeChild(node);
                }, 320);
            }, timeout || 5200);
        }

        return { show: show };
    })();

    VF.toast = toast;

    /* ==================================================================
       Reloj de la barra superior
       ================================================================== */
    function startClock() {
        var el = $("topbar-clock");
        if (!el) return;

        function tick() {
            el.textContent = new Date().toLocaleTimeString("es-CO", {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
                hour12: false,
            });
        }
        tick();
        window.setInterval(tick, 1000);
    }

    /* ==================================================================
       Gauge de confianza
       ================================================================== */
    var GAUGE_CIRCUMFERENCE = 2 * Math.PI * 52; // r=52 en el SVG

    function setGauge(value, hexColor) {
        var circle = $("gauge-value");
        var number = $("result-confidence");
        var ratio = clamp(typeof value === "number" ? value : 0, 0, 1);

        if (circle) {
            circle.setAttribute("stroke-dasharray", GAUGE_CIRCUMFERENCE.toFixed(1));
            if (hexColor) circle.style.stroke = hexColor;

            if (REDUCED_MOTION) {
                circle.setAttribute("stroke-dashoffset", (GAUGE_CIRCUMFERENCE * (1 - ratio)).toFixed(1));
            } else {
                circle.style.transition = "stroke-dashoffset 1.1s cubic-bezier(0.16, 1, 0.3, 1)";
                circle.setAttribute("stroke-dashoffset", GAUGE_CIRCUMFERENCE.toFixed(1));
                // Doble rAF: garantiza que el navegador aplique el estado inicial.
                window.requestAnimationFrame(function () {
                    window.requestAnimationFrame(function () {
                        circle.setAttribute("stroke-dashoffset", (GAUGE_CIRCUMFERENCE * (1 - ratio)).toFixed(1));
                    });
                });
            }
        }

        countUp(number, ratio, { render: formatPercent, duration: 1000 });
    }

    /* ==================================================================
       Composición de color
       ================================================================== */
    function renderColors(colors) {
        var bar = $("color-bar");
        var legend = $("color-legend");
        if (!bar || !legend) return;

        bar.innerHTML = "";
        legend.innerHTML = "";

        var entries = [];
        Object.keys(COLOR_HEX).forEach(function (key) {
            var value = colors && typeof colors[key] === "number" ? colors[key] : 0;
            if (value > 0.005) entries.push([key, value]);
        });

        if (!entries.length) {
            entries = [["pale", 1]];
        }

        entries.forEach(function (entry) {
            var key = entry[0];
            var value = entry[1];

            var seg = document.createElement("span");
            seg.className = "color-bar__seg";
            seg.style.background = COLOR_HEX[key];
            bar.appendChild(seg);
            animateBar(seg, value);

            var li = document.createElement("li");
            var swatch = document.createElement("i");
            swatch.style.background = COLOR_HEX[key];
            li.appendChild(swatch);
            li.appendChild(document.createTextNode(COLOR_LABEL[key] + " " + Math.round(value * 100) + "%"));
            legend.appendChild(li);
        });
    }

    /* ==================================================================
       Candidatos y notas
       ================================================================== */
    function renderCandidates(candidates) {
        var list = $("candidates");
        if (!list) return;
        list.innerHTML = "";

        if (!candidates || !candidates.length) {
            var empty = document.createElement("li");
            empty.className = "candidate__name";
            empty.textContent = "Sin propuestas.";
            list.appendChild(empty);
            return;
        }

        candidates.slice(0, 5).forEach(function (candidate) {
            var li = document.createElement("li");
            li.className = "candidate";

            var name = document.createElement("span");
            name.className = "candidate__name";
            name.textContent = candidate.item;

            var track = document.createElement("span");
            track.className = "candidate__track";
            var fill = document.createElement("span");
            fill.className = "candidate__fill";
            fill.setAttribute("data-source", candidate.source || "heuristic");
            track.appendChild(fill);

            var score = document.createElement("span");
            score.className = "candidate__score";
            score.textContent = formatPercent(candidate.score);

            li.appendChild(name);
            li.appendChild(track);
            li.appendChild(score);
            list.appendChild(li);

            animateBar(fill, candidate.score);
        });
    }

    function renderNotes(notes) {
        var list = $("result-notes");
        if (!list) return;
        list.innerHTML = "";

        (notes || []).forEach(function (text) {
            var li = document.createElement("li");
            li.textContent = text;
            list.appendChild(li);
        });

        if (!notes || !notes.length) {
            var li = document.createElement("li");
            li.textContent = "Sin observaciones adicionales.";
            list.appendChild(li);
        }
    }

    /* ==================================================================
       Etapas del análisis
       ================================================================== */
    var stagesTimer = null;

    function stagesReset() {
        var container = $("stages");
        if (!container) return;
        container.hidden = false;
        container.querySelectorAll(".stage").forEach(function (node) {
            node.removeAttribute("data-active");
            node.removeAttribute("data-done");
        });
    }

    function stagesSet(index) {
        var container = $("stages");
        if (!container) return;
        container.querySelectorAll(".stage").forEach(function (node) {
            var position = parseInt(node.getAttribute("data-stage"), 10);
            if (position < index) {
                node.setAttribute("data-done", "true");
                node.removeAttribute("data-active");
            } else if (position === index) {
                node.setAttribute("data-active", "true");
                node.removeAttribute("data-done");
            } else {
                node.removeAttribute("data-active");
                node.removeAttribute("data-done");
            }
        });
    }

    function stagesComplete() {
        var container = $("stages");
        if (!container) return;
        container.querySelectorAll(".stage").forEach(function (node) {
            node.setAttribute("data-done", "true");
            node.removeAttribute("data-active");
        });
    }

    /**
     * Arranca la secuencia de etapas mientras el backend responde.
     *
     * Ojo con la honestidad: estas etapas describen el pipeline real (separar
     * el producto, identificar, medir, estimar), pero el avance se estima por
     * tiempo, no por mensajes del servidor, porque la API responde en una sola
     * petición. Es un indicador de progreso, no una traza.
     */
    function stagesStart() {
        stagesStop();
        stagesReset();
        var index = 0;
        stagesSet(0);
        stagesTimer = window.setInterval(function () {
            index += 1;
            if (index > 3) {
                window.clearInterval(stagesTimer);
                stagesTimer = null;
                stagesComplete();
                return;
            }
            stagesSet(index);
        }, 900);
    }

    function stagesStop() {
        if (stagesTimer) {
            window.clearInterval(stagesTimer);
            stagesTimer = null;
        }
    }

    VF.stages = {
        start: stagesStart,
        stop: stagesStop,
        reset: function () {
            var container = $("stages");
            if (container) container.hidden = true;
        },
    };

    /* ==================================================================
       Reveal al hacer scroll
       ================================================================== */
    function initReveal() {
        var nodes = document.querySelectorAll(".reveal");
        if (!nodes.length) return;

        if (REDUCED_MOTION || !("IntersectionObserver" in window)) {
            nodes.forEach(function (node) {
                node.classList.add("is-visible");
            });
            return;
        }

        var observer = new IntersectionObserver(
            function (entries) {
                entries.forEach(function (entry) {
                    if (entry.isIntersecting) {
                        entry.target.classList.add("is-visible");
                        observer.unobserve(entry.target);
                    }
                });
            },
            { rootMargin: "0px 0px -8% 0px", threshold: 0.05 }
        );

        nodes.forEach(function (node, index) {
            node.style.transitionDelay = Math.min(index * 80, 240) + "ms";
            observer.observe(node);
        });
    }

    /* ==================================================================
       Onda en botones
       ================================================================== */
    function initRipple() {
        if (REDUCED_MOTION) return;

        document.addEventListener("pointerdown", function (event) {
            var button = event.target.closest(".btn");
            if (!button || button.disabled) return;

            var rect = button.getBoundingClientRect();
            var size = Math.max(rect.width, rect.height);
            var ripple = document.createElement("span");
            ripple.className = "ripple";
            ripple.style.width = ripple.style.height = size + "px";
            ripple.style.left = event.clientX - rect.left - size / 2 + "px";
            ripple.style.top = event.clientY - rect.top - size / 2 + "px";

            button.appendChild(ripple);
            window.setTimeout(function () {
                if (ripple.parentNode) ripple.parentNode.removeChild(ripple);
            }, 560);
        });
    }

    /* ==================================================================
       Inclinación suave del panel de análisis
       ================================================================== */
    function initTilt() {
        var panel = $("analysis-panel");
        if (!panel || REDUCED_MOTION) return;
        // En pantallas táctiles no hay puntero que seguir, y el efecto molesta.
        if (!window.matchMedia("(hover: hover) and (pointer: fine)").matches) return;

        var MAX_DEGREES = 1.6;

        panel.addEventListener("pointermove", function (event) {
            var rect = panel.getBoundingClientRect();
            var relativeX = (event.clientX - rect.left) / rect.width - 0.5;
            var relativeY = (event.clientY - rect.top) / rect.height - 0.5;
            panel.style.transform =
                "perspective(1400px) rotateY(" + (relativeX * MAX_DEGREES).toFixed(2) + "deg)" +
                " rotateX(" + (-relativeY * MAX_DEGREES).toFixed(2) + "deg)";
            panel.style.transition = "transform 0.1s linear";
        });

        panel.addEventListener("pointerleave", function () {
            panel.style.transition = "transform 0.5s cubic-bezier(0.16, 1, 0.3, 1)";
            panel.style.transform = "";
        });
    }

    /* ==================================================================
       Arranque
       ================================================================== */
    function init() {
        startClock();
        initReveal();
        initRipple();
        initTilt();
        ambient.start();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }

    /* API pública para camera.js */
    VF.setGauge = setGauge;
    VF.renderColors = renderColors;
    VF.renderCandidates = renderCandidates;
    VF.renderNotes = renderNotes;
    VF.countUp = countUp;
    VF.animateBar = animateBar;
    VF.COLOR_HEX = COLOR_HEX;
    VF.QUALITY_TONE = QUALITY_TONE;
    VF.RIPENESS_TONE = RIPENESS_TONE;
    VF.METHOD_LABEL = METHOD_LABEL;
    VF.METHOD_NOTE = METHOD_NOTE;
    VF.REDUCED_MOTION = REDUCED_MOTION;

    window.VF = VF;
})();
