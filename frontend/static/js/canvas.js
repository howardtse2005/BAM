/**
 * Canvas editor.
 *
 * Layer order (bottom to top): image, stroke trail, mask preview. The mask data
 * canvas is invisible; it is the source of truth and is what gets saved. The
 * preview canvas is a tinted, contoured view of it.
 *
 * All pointer coordinates go through `screenToImage`, which uses the image
 * canvas's bounding rect. That automatically accounts for the wrapper's
 * translate/scale transform, so there is no separate coordinate bookkeeping.
 */

import { imageUrl, loadImage } from "./file_system.js";
import { CropCollector, sendFill, sendSegment } from "./magic_brush.js";

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

/** Stage children that are controls, not canvas surface. */
const STAGE_OVERLAYS = ".stage__progress, .stage__hint, .stage__empty";

export class EditorCanvas {
  constructor(options) {
    this.stage = options.stage;
    this.wrap = options.wrap;

    this.imageCanvas = document.getElementById("image-canvas");
    this.strokeCanvas = document.getElementById("stroke-canvas");
    this.previewCanvas = document.getElementById("preview-canvas");
    this.maskCanvas = document.getElementById("mask-canvas");

    this.imageCtx = this.imageCanvas.getContext("2d");
    this.strokeCtx = this.strokeCanvas.getContext("2d");
    this.previewCtx = this.previewCanvas.getContext("2d");
    this.maskCtx = this.maskCanvas.getContext("2d", { willReadFrequently: true });

    this.dataset = options.dataset;
    this.split = options.split || "train";
    this.imageName = options.image || "";
    this.cropSize = options.cropSize || 256;
    this.cropMin = options.cropMin || 64;
    this.cropMax = options.cropMax || this.cropSize;
    this.cropRatio = options.cropRatio || 0.4;
    this.shortSide = options.shortSide || 640;
    this.stride = options.stride || 128;
    this.maxCrops = options.maxCrops || 64;
    this.lineWidth = options.lineWidth || 3;
    this.fillClose = options.fillClose || 5;

    // Magic brush geometry. The band is what the user draws and sees; the
    // prompt is the thinner stroke actually handed to the model. The gap
    // between them is the tolerance the predicted line may move in, so it can
    // wander a few px off the centreline and still sit under the stroke.
    this.magicWidth = options.magicWidth || 32;
    this.promptRatio = options.promptRatio || 0.625;
    this.promptMin = options.promptMin || 3;
    this.promptMax = options.promptMax || 20;

    this.width = 0;
    this.height = 0;
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    this.viewDirty = false;

    this.tool = "magicBrush";
    this.brushSize = 8;
    this.spaceDown = false;
    this.drawing = false;
    this.panning = null;
    this.lastPos = null;
    this.magicLast = null;
    this.subtract = false;

    this.maskVisible = true;
    this.maskOpacity = 0.35;
    this.showEdge = false;
    this.maskNonEmpty = false;

    this.history = [];
    this.future = [];
    this.historyLimit = 20;

    this.collector = new CropCollector({
      cropSize: this.cropSize,
      stride: this.stride,
      maxCrops: this.maxCrops,
    });

    this.strokeData = document.createElement("canvas");
    this.strokeDataCtx = this.strokeData.getContext("2d");
    // The band is painted opaquely here and composited translucent, because
    // painting it translucent directly would double-darken every joint between
    // two pointer samples.
    this.bandCanvas = document.createElement("canvas");
    this.bandCtx = this.bandCanvas.getContext("2d");
    this.tint = document.createElement("canvas");
    this.tintCtx = this.tint.getContext("2d");
    this.rim = document.createElement("canvas");
    this.rimCtx = this.rim.getContext("2d");
    this.edgeCanvas = document.createElement("canvas");
    this.edgeCtx = this.edgeCanvas.getContext("2d");
    this.edgeReady = false;

    this.previewQueued = false;
    this.strokeQueued = false;

    this.cursor = document.createElement("div");
    this.cursor.className = "cursor-ring";
    this.cursor.hidden = true;
    this.cursorCore = document.createElement("div");
    this.cursorCore.className = "cursor-ring__core";
    this.cursorCore.hidden = true;
    this.cursor.appendChild(this.cursorCore);
    this.stage.appendChild(this.cursor);

    this._readThemeColors();
    this._bind();
  }

  // ------------------------------------------------------------------ setup

  _readThemeColors() {
    const styles = getComputedStyle(document.documentElement);
    this.fillColor = (styles.getPropertyValue("--mask-fill") || "#4f46e5").trim();
    this.rimColor = (styles.getPropertyValue("--mask-rim") || "#312a96").trim();
    this.strokeColor = (styles.getPropertyValue("--stroke") || "#f59e0b").trim();
  }

  refreshTheme() {
    this._readThemeColors();
    this.updatePreview();
  }

  _bind() {
    this.stage.addEventListener("pointerdown", (event) => this._down(event));
    this.stage.addEventListener("pointermove", (event) => this._move(event));
    this.stage.addEventListener("pointerup", (event) => this._up(event));
    this.stage.addEventListener("pointercancel", (event) => this._up(event));
    this.stage.addEventListener("pointerleave", () => {
      if (!this.drawing) this.cursor.hidden = true;
    });
    this.stage.addEventListener("wheel", (event) => this._wheel(event), {
      passive: false,
    });
    this.stage.addEventListener("contextmenu", (event) => event.preventDefault());
  }

  // ------------------------------------------------------------------ load

  async load(name) {
    if (!name) return false;
    const url = imageUrl(this.dataset, this.split, name);
    const image = await loadImage(url);

    // Work at the canonical short side, never at native resolution. A 4K source
    // would otherwise exceed the server's canvas limit and, more importantly,
    // would feed the model an object scale it was never trained on.
    const native = Math.min(image.naturalWidth, image.naturalHeight);
    const scale = Math.min(1, this.shortSide / Math.max(1, native));
    const width = Math.max(1, Math.round(image.naturalWidth * scale));
    const height = Math.max(1, Math.round(image.naturalHeight * scale));

    this.imageName = name;
    this.resize(width, height);
    this.imageCtx.clearRect(0, 0, this.width, this.height);
    this.imageCtx.drawImage(image, 0, 0, width, height);
    this.clearMask(false);
    this.fit();
    this.updatePreview();
    this._status();
    return true;
  }

  resize(width, height) {
    this.width = Math.max(1, Math.round(width));
    this.height = Math.max(1, Math.round(height));

    for (const canvas of [
      this.imageCanvas,
      this.strokeCanvas,
      this.previewCanvas,
      this.maskCanvas,
      this.strokeData,
      this.bandCanvas,
      this.tint,
      this.rim,
      this.edgeCanvas,
    ]) {
      canvas.width = this.width;
      canvas.height = this.height;
    }

    this.wrap.style.width = `${this.width}px`;
    this.wrap.style.height = `${this.height}px`;

    // Canvas state resets when the backing store is resized.
    // The mask is white with alpha, not opaque black: alpha is what makes the
    // preview tint and the saved PNG agree.
    this.maskCtx.clearRect(0, 0, this.width, this.height);
    this.edgeReady = false;
    this.maskNonEmpty = false;
  }

  // ------------------------------------------------------------------ view

  applyTransform() {
    this.wrap.style.transform = `translate(${this.panX}px, ${this.panY}px) scale(${this.zoom})`;
  }

  fit() {
    if (!this.width) return;
    const pad = 36;
    const availableW = Math.max(64, this.stage.clientWidth - pad * 2);
    const availableH = Math.max(64, this.stage.clientHeight - pad * 2);
    const zoom = Math.min(availableW / this.width, availableH / this.height, 4);
    this.zoom = clamp(zoom, 0.05, 12);
    this.panX = (this.stage.clientWidth - this.width * this.zoom) / 2;
    this.panY = (this.stage.clientHeight - this.height * this.zoom) / 2;
    this.viewDirty = false;
    this.applyTransform();
    this._status();
  }

  actualSize() {
    const cx = this.stage.clientWidth / 2;
    const cy = this.stage.clientHeight / 2;
    this.setZoom(1, cx, cy);
  }

  /** Re-fit after a viewport change unless the user set their own zoom/pan. */
  handleResize() {
    if (!this.width || this.viewDirty) return;
    this.fit();
  }

  setZoom(next, anchorX, anchorY) {
    const target = clamp(next, 0.05, 16);
    const k = target / this.zoom;
    this.panX = anchorX - (anchorX - this.panX) * k;
    this.panY = anchorY - (anchorY - this.panY) * k;
    this.zoom = target;
    this.viewDirty = true;
    this.applyTransform();
    this._status();
  }

  zoomIn() {
    this.setZoom(this.zoom * 1.25, this.stage.clientWidth / 2, this.stage.clientHeight / 2);
  }

  zoomOut() {
    this.setZoom(this.zoom / 1.25, this.stage.clientWidth / 2, this.stage.clientHeight / 2);
  }

  _wheel(event) {
    event.preventDefault();
    const rect = this.stage.getBoundingClientRect();
    const factor = Math.exp(-event.deltaY * 0.0015);
    this.setZoom(
      this.zoom * factor,
      event.clientX - rect.left,
      event.clientY - rect.top
    );
  }

  screenToImage(event) {
    if (!this.width) return null;
    const rect = this.imageCanvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    return [
      ((event.clientX - rect.left) / rect.width) * this.width,
      ((event.clientY - rect.top) / rect.height) * this.height,
    ];
  }

  // ------------------------------------------------------------------ tools

  setTool(tool) {
    this.tool = tool;
    this.stage.classList.toggle("is-panning", tool === "pan");
    this.cursor.hidden = true;
    this._status();
  }

  setBrushSize(size) {
    this.brushSize = clamp(Number(size) || 8, 1, 80);
    this._updateCursor(this.lastClient);
  }

  /** Visible magic-brush band width, in canvas px. The slider drives this. */
  setMagicWidth(width) {
    this.magicWidth = clamp(Number(width) || 32, 4, 160);
    this._updateCursor(this.lastClient);
    this._status();
  }

  /**
   * Width of the stroke handed to the model for a given band width.
   *
   * Mirrors ``Config.prompt_width_for``: a fixed fraction of the band, clamped
   * to the width range the model was trained on and never wider than the band
   * it has to sit inside.
   */
  promptWidthFor(bandWidth) {
    const band = Math.max(1, Math.round(Number(bandWidth) || 1));
    const raw = Math.round(band * this.promptRatio);
    const clamped = Math.min(this.promptMax, Math.max(this.promptMin, raw));
    return Math.max(1, Math.min(band, clamped));
  }

  get promptWidth() {
    return this.promptWidthFor(this.magicWidth);
  }

  setMaskVisible(visible) {
    this.maskVisible = Boolean(visible);
    this.updatePreview();
  }

  setMaskOpacity(value) {
    this.maskOpacity = clamp(Number(value) / 100, 0.05, 1);
    this.updatePreview();
  }

  setShowEdge(visible) {
    this.showEdge = Boolean(visible);
    this.updatePreview();
  }

  // ---------------------------------------------------------------- pointer

  _isPan(event) {
    return event.button === 1 || this.tool === "pan" || this.spaceDown;
  }

  _down(event) {
    if (event.button === 2) return;
    if (!this.width) return;

    // The progress pill and hint are children of the stage, so their clicks
    // would otherwise start a stroke underneath them.
    if (event.target instanceof Element && event.target.closest(STAGE_OVERLAYS)) {
      return;
    }

    try {
      this.stage.setPointerCapture(event.pointerId);
    } catch {
      /* the pointer can already be gone on a fast tap; capture is an optimisation */
    }
    this.lastClient = { x: event.clientX, y: event.clientY };

    if (this._isPan(event)) {
      this.panning = {
        x: event.clientX,
        y: event.clientY,
        panX: this.panX,
        panY: this.panY,
      };
      this.stage.classList.add("is-panning");
      return;
    }

    const point = this.screenToImage(event);
    if (!point) return;
    const [x, y] = point;

    if (this.tool === "magicBrush") {
      this.subtract = event.altKey;
      this._magicStart(x, y);
      return;
    }

    if (this.tool === "fill") {
      this.fillMask();
      return;
    }

    this.pushHistory();
    this.drawing = true;
    this.lastPos = null;
    this._paint(x, y);
  }

  _move(event) {
    this.lastClient = { x: event.clientX, y: event.clientY };
    this._updateCursor(this.lastClient);

    if (this.panning) {
      this.panX = this.panning.panX + (event.clientX - this.panning.x);
      this.panY = this.panning.panY + (event.clientY - this.panning.y);
      this.viewDirty = true;
      this.applyTransform();
      return;
    }

    if (!this.width) return;
    const point = this.screenToImage(event);
    if (!point) return;
    const [x, y] = point;

    if (this.drawing) this._paint(x, y);
    else if (this.tool === "magicBrush" && this.magicLast) this._magicDraw(x, y);
  }

  _up(event) {
    if (this.panning) {
      this.panning = null;
      this.stage.classList.remove("is-panning");
      return;
    }
    if (this.drawing) {
      this.drawing = false;
      this.lastPos = null;
      this.binarize();
      this.updatePreview();
      this._status();
      return;
    }
    if (this.tool === "magicBrush" && this.magicLast) {
      this._magicEnd();
    }
  }

  _updateCursor(position) {
    if (!position || !this.width) {
      this.cursor.hidden = true;
      return;
    }
    const rect = this.stage.getBoundingClientRect();
    const magic = this.tool === "magicBrush";
    const width = magic ? this.magicWidth : this.brushSize;
    const size = Math.max(6, width * this.zoom);
    this.cursor.style.width = `${size}px`;
    this.cursor.style.height = `${size}px`;
    this.cursor.style.left = `${position.x - rect.left}px`;
    this.cursor.style.top = `${position.y - rect.top}px`;
    this.cursor.hidden = this.tool === "pan";

    // The inner ring is the prompt: the only part of the band the model sees.
    const core = magic ? Math.max(4, this.promptWidth * this.zoom) : 0;
    this.cursorCore.hidden = !core;
    if (core) {
      this.cursorCore.style.width = `${core}px`;
      this.cursorCore.style.height = `${core}px`;
    }
  }

  // ---------------------------------------------------------------- drawing

  _paint(x, y) {
    const ctx = this.maskCtx;
    const radius = Math.max(0.5, this.brushSize / 2);

    ctx.globalCompositeOperation =
      this.tool === "eraser" ? "destination-out" : "source-over";
    ctx.strokeStyle = "#fff";
    ctx.fillStyle = "#fff";
    ctx.lineWidth = this.brushSize;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    if (this.lastPos) {
      ctx.beginPath();
      ctx.moveTo(this.lastPos[0], this.lastPos[1]);
      ctx.lineTo(x, y);
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.globalCompositeOperation = "source-over";
    this.lastPos = [x, y];
    if (this.tool !== "eraser") this.maskNonEmpty = true;
    this._queuePreview();
  }

  // ------------------------------------------------------------ magic brush

  /**
   * Crop size for this canvas: a fixed fraction of the short side, clamped and
   * snapped to a multiple of 32. At the canonical 640 short side this is 256;
   * a 400 px upload gets 160, which keeps the object at the scale the model was
   * trained on instead of shrinking the object to a dot inside a fixed crop.
   */
  computeCropSize() {
    const short = Math.min(this.width || this.shortSide, this.height || this.shortSide);
    // The crop has to hold the whole band with context around it; a band wider
    // than the crop would be clipped before the model ever saw its outer edge.
    const raw = Math.max(Math.round(short * this.cropRatio), this.magicWidth * 2);
    const bounded = Math.min(this.cropMax, Math.max(this.cropMin, raw));
    return Math.max(32, Math.round(bounded / 32) * 32);
  }

  _magicStart(x, y) {
    // A new stroke starts from a clean trail; the previous line is already in
    // the mask, so it is left alone.
    this.strokeCtx.clearRect(0, 0, this.width, this.height);
    this.strokeDataCtx.clearRect(0, 0, this.width, this.height);
    this.bandCtx.clearRect(0, 0, this.width, this.height);
    this.collector.reset(this.computeCropSize());
    this.magicLast = null;
    this._magicDraw(x, y);
  }

  _magicDraw(x, y) {
    const band = this.bandCtx;
    const data = this.strokeDataCtx;
    const promptWidth = this.promptWidth;

    // The band is painted in the stroke colour, not white: it is only ever used
    // for display, and compositing it at low alpha is what makes it read as a
    // tolerance region around the solid core.
    band.strokeStyle = this.strokeColor;
    band.fillStyle = this.strokeColor;
    band.lineWidth = this.magicWidth;
    band.lineCap = "round";
    band.lineJoin = "round";

    data.strokeStyle = "#fff";
    data.fillStyle = "#fff";
    data.lineWidth = Math.max(2, promptWidth);
    data.lineCap = "round";
    data.lineJoin = "round";

    if (this.magicLast) {
      for (const ctx of [band, data]) {
        ctx.beginPath();
        ctx.moveTo(this.magicLast[0], this.magicLast[1]);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
    } else {
      for (const ctx of [band, data]) {
        ctx.beginPath();
        ctx.arc(x, y, Math.max(1, ctx.lineWidth / 2), 0, Math.PI * 2);
        ctx.fill();
      }
    }

    this.magicLast = [x, y];
    this._queueStroke();
    if (this.collector.add(x, y)) this._status();
  }

  /** Coalesce repaints: pointermove fires faster than the compositor redraws. */
  _queueStroke() {
    if (this.strokeQueued) return;
    this.strokeQueued = true;
    requestAnimationFrame(() => {
      this.strokeQueued = false;
      // The stroke may have ended between the sample and this frame.
      if (this.magicLast) this._composeStroke();
    });
  }

  /**
   * Rebuild the visible stroke from its two opaque layers.
   *
   * The band is the tolerance region — where the predicted line is allowed to
   * land — so it is drawn translucent. The prompt core is drawn solid, because
   * it is the only part the model is conditioned on. Redrawing both layers on
   * every sample is what keeps the band free of the darker joints that
   * per-segment alpha compositing would leave.
   */
  _composeStroke() {
    const ctx = this.strokeCtx;
    ctx.clearRect(0, 0, this.width, this.height);

    ctx.save();
    ctx.globalAlpha = 0.3;
    ctx.drawImage(this.bandCanvas, 0, 0);
    ctx.restore();

    ctx.drawImage(this._drawTint(this.strokeData, this.strokeColor), 0, 0);

    // Crop boxes come from the collector rather than being painted once, so a
    // full repaint does not lose them.
    ctx.save();
    ctx.strokeStyle = "rgba(245, 158, 11, 0.28)";
    ctx.lineWidth = 1;
    for (const crop of this.collector.crops) {
      ctx.strokeRect(crop.x0, crop.y0, crop.size, crop.size);
    }
    ctx.restore();
  }

  async _magicEnd() {
    const crops = this.collector.crops.slice();
    const cropSize = this.collector.cropSize;
    // Read the modifier now: the response arrives after the pointer is gone and
    // a new stroke may already have reset it.
    const subtract = this.subtract;
    this.strokeCtx.clearRect(0, 0, this.width, this.height);
    this.bandCtx.clearRect(0, 0, this.width, this.height);
    this.magicLast = null;
    if (!crops.length) return;

    this.onProgress?.(true, `snapping ${crops.length} crop${crops.length === 1 ? "" : "s"}…`);
    const started = performance.now();

    try {
      const payload = {
        dataset: this.dataset,
        split: this.split,
        image: this.imageName,
        canvas: { width: this.width, height: this.height },
        crops,
        stroke_mask: this.strokeData.toDataURL("image/png"),
        params: {
          ...(this.getParams ? this.getParams() : {}),
          mode: "line",
          line_width: this.lineWidth,
          fill_close: this.fillClose,
          brush_width: this.magicWidth,
          prompt_width: this.promptWidth,
        },
      };
      const result = await sendSegment(payload);
      const committed = await this.commitLine(result.line_base64, { subtract });
      this.onSegmented?.({
        ...result,
        closed: Boolean(result.closed),
        cropSize: result.crop_size || cropSize,
        bandWidth: result.band_width || this.magicWidth,
        promptWidth: result.prompt_width || this.promptWidth,
        elapsed: performance.now() - started,
        crops: crops.length,
        committed,
      });
    } catch (error) {
      this.onError?.(error);
    } finally {
      this.onProgress?.(false);
      this.collector.reset();
      this._status();
    }
  }

  // ---------------------------------------------------------------- commit

  /**
   * Paint the predicted line straight into the mask.
   *
   * There is no review step: the server's line *is* the answer, so it lands in
   * the mask as soon as it arrives. Undo is the escape hatch, which is cheaper
   * than a modal decision after every stroke.
   *
   * The response is an opaque grayscale PNG, so it is re-read into the same
   * white-with-alpha convention the mask uses; drawing it straight in would
   * paint the black background over the mask.
   */
  async commitLine(lineBase64, { subtract = false } = {}) {
    if (!lineBase64 || !this.width) return false;
    const image = await loadImage(lineBase64);

    const temp = document.createElement("canvas");
    temp.width = this.width;
    temp.height = this.height;
    const tempCtx = temp.getContext("2d", { willReadFrequently: true });
    tempCtx.drawImage(image, 0, 0, this.width, this.height);
    const source = tempCtx.getImageData(0, 0, this.width, this.height);
    const data = source.data;
    let seen = false;
    for (let i = 0; i < data.length; i += 4) {
      const on = data[i] > 127;
      data[i] = 255;
      data[i + 1] = 255;
      data[i + 2] = 255;
      data[i + 3] = on ? 255 : 0;
      if (on) seen = true;
    }
    // Nothing predicted: do not spend a history entry or a repaint on it.
    if (!seen) return false;
    tempCtx.putImageData(source, 0, 0);

    this.pushHistory();
    const ctx = this.maskCtx;
    ctx.save();
    ctx.globalCompositeOperation = subtract ? "destination-out" : "source-over";
    ctx.drawImage(temp, 0, 0);
    ctx.restore();
    if (!subtract) this.maskNonEmpty = true;
    this.updatePreview();
    this._status();
    return true;
  }

  // ------------------------------------------------------------------- mask

  pushHistory() {
    if (!this.width) return;
    try {
      this.history.push(this.maskCtx.getImageData(0, 0, this.width, this.height));
      if (this.history.length > this.historyLimit) this.history.shift();
      this.future.length = 0;
    } catch {
      /* getImageData can fail on a tainted canvas; ignore */
    }
  }

  undo() {
    if (!this.history.length) return false;
    this.future.push(this.maskCtx.getImageData(0, 0, this.width, this.height));
    const previous = this.history.pop();
    this.maskCtx.putImageData(previous, 0, 0);
    this._recountMask();
    this.updatePreview();
    this._status();
    return true;
  }

  redo() {
    if (!this.future.length) return false;
    this.history.push(this.maskCtx.getImageData(0, 0, this.width, this.height));
    const next = this.future.pop();
    this.maskCtx.putImageData(next, 0, 0);
    this._recountMask();
    this.updatePreview();
    this._status();
    return true;
  }

  _recountMask() {
    const data = this.maskCtx.getImageData(0, 0, this.width, this.height).data;
    let seen = false;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] > 127) {
        seen = true;
        break;
      }
    }
    this.maskNonEmpty = seen;
  }

  binarize() {
    if (!this.width) return;
    const image = this.maskCtx.getImageData(0, 0, this.width, this.height);
    const data = image.data;
    for (let i = 0; i < data.length; i += 4) {
      const on = data[i + 3] > 127;
      data[i] = 255;
      data[i + 1] = 255;
      data[i + 2] = 255;
      data[i + 3] = on ? 255 : 0;
    }
    this.maskCtx.putImageData(image, 0, 0);
  }

  clearMask(record = true) {
    if (!this.width) return;
    if (record) this.pushHistory();
    this.maskCtx.clearRect(0, 0, this.width, this.height);
    this.maskNonEmpty = false;
    this.edgeReady = false;
    this.updatePreview();
    this._status();
  }

  invertMask() {
    if (!this.width) return;
    this.pushHistory();
    const image = this.maskCtx.getImageData(0, 0, this.width, this.height);
    const data = image.data;
    for (let i = 0; i < data.length; i += 4) {
      const on = data[i + 3] <= 127;
      data[i] = 255;
      data[i + 1] = 255;
      data[i + 2] = 255;
      data[i + 3] = on ? 255 : 0;
    }
    this.maskCtx.putImageData(image, 0, 0);
    this._recountMask();
    this.updatePreview();
    this._status();
  }

  /**
   * Segment the region enclosed by the current line.
   *
   * Server-side first: it closes hairline gaps, floods the outside, keeps holes
   * and snaps the boundary to image edges. The local flood is the offline
   * fallback, so a stopped server degrades instead of breaking the tool.
   */
  async fillMask() {
    if (!this.width) return false;

    if (!this.maskNonEmpty) {
      this.onError?.(new Error("Draw a line first"));
      return false;
    }

    this.onProgress?.(true, "filling…");
    try {
      const payload = {
        dataset: this.dataset,
        split: this.split,
        image: this.imageName,
        canvas: { width: this.width, height: this.height },
        line_mask: this.maskCanvas.toDataURL("image/png"),
        params: {
          ...(this.getParams ? this.getParams() : {}),
          fill_close: this.fillClose,
        },
      };
      const result = await sendFill(payload);
      if (!result.grew) {
        this.onError?.(
          new Error("The line does not enclose a region — close the gap and try again")
        );
        return false;
      }
      await this.applyMask(result.mask_base64, { replace: true });
      this.onFilled?.(result);
      return true;
    } catch (error) {
      const grew = this.fillHoles();
      if (!grew) this.onError?.(error);
      return grew;
    } finally {
      this.onProgress?.(false);
    }
  }

  /** Fill background regions enclosed by the mask. Returns whether it grew. */
  fillHoles() {
    if (!this.width) return false;
    this.pushHistory();
    const image = this.maskCtx.getImageData(0, 0, this.width, this.height);
    const data = image.data;
    const w = this.width;
    const h = this.height;
    const visited = new Uint8Array(w * h);
    const stack = [];

    const isBackground = (index) => data[index * 4 + 3] < 128;

    const push = (index) => {
      if (!visited[index] && isBackground(index)) {
        visited[index] = 1;
        stack.push(index);
      }
    };

    for (let x = 0; x < w; x += 1) {
      push(x);
      push((h - 1) * w + x);
    }
    for (let y = 0; y < h; y += 1) {
      push(y * w);
      push(y * w + w - 1);
    }

    while (stack.length) {
      const index = stack.pop();
      const x = index % w;
      const y = (index / w) | 0;
      if (x > 0) push(index - 1);
      if (x < w - 1) push(index + 1);
      if (y > 0) push(index - w);
      if (y < h - 1) push(index + w);
    }

    let changed = false;
    for (let index = 0; index < w * h; index += 1) {
      if (!visited[index] && isBackground(index)) {
        const offset = index * 4;
        data[offset] = 255;
        data[offset + 1] = 255;
        data[offset + 2] = 255;
        data[offset + 3] = 255;
        changed = true;
      }
    }

    if (!changed) {
      this.history.pop();
      return false;
    }
    this.maskCtx.putImageData(image, 0, 0);
    this.maskNonEmpty = true;
    this.updatePreview();
    this._status();
    return true;
  }

  _drawTint(source, color) {
    const ctx = this.tintCtx;
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.drawImage(source, 0, 0);
    ctx.globalCompositeOperation = "source-in";
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, this.width, this.height);
    ctx.globalCompositeOperation = "source-over";
    return this.tint;
  }

  /**
   * 1-px outline: tint the mask, then erase the mask shifted by (dx, dy).
   * Drawing the filled mask at four offsets (the obvious approach) paints the
   * whole interior, not just the rim.
   */
  _drawOutline(source, color, dx, dy) {
    const ctx = this.rimCtx;
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.drawImage(source, 0, 0);
    ctx.globalCompositeOperation = "source-in";
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, this.width, this.height);
    ctx.globalCompositeOperation = "destination-out";
    ctx.drawImage(source, dx, dy);
    ctx.globalCompositeOperation = "source-over";
    return this.rim;
  }

  updatePreview() {
    if (!this.width) return;
    const ctx = this.previewCtx;
    ctx.clearRect(0, 0, this.width, this.height);

    if (this.showEdge && this.edgeReady) {
      const edgeTint = this._drawTint(this.edgeCanvas, this.strokeColor);
      ctx.save();
      ctx.globalAlpha = 0.85;
      ctx.drawImage(edgeTint, 0, 0);
      ctx.restore();
    }

    if (this.maskVisible) {
      ctx.save();
      ctx.globalAlpha = 0.85;
      ctx.drawImage(this._drawOutline(this.maskCanvas, this.rimColor, 1, 1), 0, 0);
      ctx.drawImage(this._drawOutline(this.maskCanvas, this.rimColor, -1, -1), 0, 0);
      ctx.restore();

      const fill = this._drawTint(this.maskCanvas, this.fillColor);
      ctx.save();
      ctx.globalAlpha = this.maskOpacity;
      ctx.drawImage(fill, 0, 0);
      ctx.restore();
    }
  }

  _queuePreview() {
    if (this.previewQueued) return;
    this.previewQueued = true;
    requestAnimationFrame(() => {
      this.previewQueued = false;
      this.updatePreview();
    });
  }

  async applyMask(maskBase64, { subtract = false, replace = false } = {}) {
    if (!maskBase64 || !this.width) return;
    const image = await loadImage(maskBase64);

    const temp = document.createElement("canvas");
    temp.width = this.width;
    temp.height = this.height;
    const tempCtx = temp.getContext("2d", { willReadFrequently: true });
    tempCtx.drawImage(image, 0, 0, this.width, this.height);
    const incoming = tempCtx.getImageData(0, 0, this.width, this.height).data;

    this.pushHistory();
    const current = this.maskCtx.getImageData(0, 0, this.width, this.height);
    const data = current.data;

    for (let i = 0; i < data.length; i += 4) {
      const on = incoming[i] > 127;
      data[i] = 255;
      data[i + 1] = 255;
      data[i + 2] = 255;
      if (replace) {
        // A filled region replaces the outline it was grown from, so the thin
        // line does not survive as a ring around the object.
        data[i + 3] = on ? 255 : 0;
      } else if (subtract) {
        if (on) data[i + 3] = 0;
      } else if (on) {
        data[i + 3] = 255;
      }
    }

    this.maskCtx.putImageData(current, 0, 0);
    this.maskNonEmpty = true;
    this.updatePreview();
    this._status();
  }

  setEdge(edgeBase64) {
    if (!edgeBase64) return;
    loadImage(edgeBase64)
      .then((image) => {
        this.edgeCtx.clearRect(0, 0, this.width, this.height);
        this.edgeCtx.drawImage(image, 0, 0, this.width, this.height);
        this.edgeReady = true;
        if (this.showEdge) this.updatePreview();
      })
      .catch(() => {
        this.edgeReady = false;
      });
  }

  getMaskBase64() {
    return this.maskCanvas.toDataURL("image/png");
  }

  get cropCount() {
    return this.collector.count;
  }

  _status() {
    this.onStatus?.({
      zoom: this.zoom,
      crops: this.collector.count,
      cropSize: this.collector.cropSize,
      brushWidth: this.magicWidth,
      promptWidth: this.promptWidth,
      hasMask: this.maskNonEmpty,
      tool: this.tool,
    });
  }
}
