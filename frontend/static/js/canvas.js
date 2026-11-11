/**
 * Canvas editor.
 *
 * Layer order (bottom to top): image, stroke trail, mask preview, mask data.
 * The mask data canvas is invisible; it is the source of truth and is what gets
 * saved. The preview canvas is a tinted, contoured view of it.
 *
 * All pointer coordinates go through `screenToImage`, which uses the image
 * canvas's bounding rect. That automatically accounts for the wrapper's
 * translate/scale transform, so there is no separate coordinate bookkeeping.
 */

import { imageUrl, loadImage } from "./file_system.js";
import { CropCollector, sendSegment } from "./magic_pen.js";

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

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
    this.shortSide = options.shortSide || 640;
    this.stride = options.stride || 128;
    this.maxCrops = options.maxCrops || 64;

    this.width = 0;
    this.height = 0;
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    this.viewDirty = false;

    this.tool = "brush";
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
    this.tint = document.createElement("canvas");
    this.tintCtx = this.tint.getContext("2d");
    this.rim = document.createElement("canvas");
    this.rimCtx = this.rim.getContext("2d");
    this.edgeCanvas = document.createElement("canvas");
    this.edgeCtx = this.edgeCanvas.getContext("2d");
    this.edgeReady = false;

    this.previewQueued = false;

    this.cursor = document.createElement("div");
    this.cursor.className = "cursor-ring";
    this.cursor.hidden = true;
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

  setSplit(split) {
    this.split = split;
    this.imageName = "";
  }

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

    this.stage.setPointerCapture(event.pointerId);
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

    if (this.tool === "magic") {
      this.subtract = event.altKey;
      this._magicStart(x, y);
      return;
    }

    if (this.tool === "fill") {
      this.fillHoles();
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
    else if (this.tool === "magic" && this.magicLast) this._magicDraw(x, y);
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
    if (this.tool === "magic" && this.magicLast) {
      this._magicEnd();
    }
  }

  _updateCursor(position) {
    if (!position || !this.width) {
      this.cursor.hidden = true;
      return;
    }
    const rect = this.stage.getBoundingClientRect();
    const size = Math.max(6, this.brushSize * this.zoom);
    this.cursor.style.width = `${size}px`;
    this.cursor.style.height = `${size}px`;
    this.cursor.style.left = `${position.x - rect.left}px`;
    this.cursor.style.top = `${position.y - rect.top}px`;
    this.cursor.hidden = this.tool === "pan";
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

  // ------------------------------------------------------------------ magic

  _magicStart(x, y) {
    this.pushHistory();
    this.strokeCtx.clearRect(0, 0, this.width, this.height);
    this.strokeDataCtx.clearRect(0, 0, this.width, this.height);
    this.collector.reset();
    this.magicLast = null;
    this._magicDraw(x, y);
  }

  _magicDraw(x, y) {
    const visual = this.strokeCtx;
    const data = this.strokeDataCtx;
    const radius = Math.max(1, this.brushSize / 2);

    visual.strokeStyle = this.strokeColor;
    visual.fillStyle = this.strokeColor;
    visual.lineWidth = 2;
    visual.lineCap = "round";

    data.strokeStyle = "#fff";
    data.fillStyle = "#fff";
    data.lineWidth = Math.max(2, this.brushSize);
    data.lineCap = "round";

    if (this.magicLast) {
      for (const ctx of [visual, data]) {
        ctx.beginPath();
        ctx.moveTo(this.magicLast[0], this.magicLast[1]);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
    } else {
      for (const ctx of [visual, data]) {
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    this.magicLast = [x, y];

    if (this.collector.add(x, y)) {
      const half = this.cropSize / 2;
      visual.save();
      visual.strokeStyle = "rgba(245, 158, 11, 0.28)";
      visual.lineWidth = 1;
      visual.strokeRect(x - half, y - half, this.cropSize, this.cropSize);
      visual.restore();
      this._status();
    }
  }

  async _magicEnd() {
    const crops = this.collector.crops.slice();
    this.strokeCtx.clearRect(0, 0, this.width, this.height);
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
        params: this.getParams ? this.getParams() : {},
      };
      const result = await sendSegment(payload);
      await this.applyMask(result.mask_base64, { subtract: this.subtract });
      this.onResult?.({ ...result, elapsed: performance.now() - started, crops: crops.length });
    } catch (error) {
      this.onError?.(error);
    } finally {
      this.onProgress?.(false);
      this.collector.reset();
      this._status();
    }
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

  /** Fill background regions that are enclosed by the mask. */
  fillHoles() {
    if (!this.width) return;
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
      return;
    }
    this.maskCtx.putImageData(image, 0, 0);
    this.maskNonEmpty = true;
    this.updatePreview();
    this._status();
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

    if (!this.maskVisible) return;

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

  _queuePreview() {
    if (this.previewQueued) return;
    this.previewQueued = true;
    requestAnimationFrame(() => {
      this.previewQueued = false;
      this.updatePreview();
    });
  }

  async applyMask(maskBase64, { subtract = false } = {}) {
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
      if (subtract) {
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

  async loadMask(maskBase64) {
    if (!maskBase64 || !this.width) return;
    const image = await loadImage(maskBase64);

    const temp = document.createElement("canvas");
    temp.width = this.width;
    temp.height = this.height;
    const tempCtx = temp.getContext("2d", { willReadFrequently: true });
    tempCtx.drawImage(image, 0, 0, this.width, this.height);
    const source = tempCtx.getImageData(0, 0, this.width, this.height);

    const data = source.data;

    // A saved mask is either opaque grayscale (white = object) or white with
    // alpha (alpha = object). Decide once per image; testing alpha per pixel
    // would mark an opaque grayscale mask as fully foreground.
    let hasAlpha = false;
    for (let i = 3; i < data.length; i += 4) {
      if (data[i] < 250) {
        hasAlpha = true;
        break;
      }
    }

    for (let i = 0; i < data.length; i += 4) {
      const on = hasAlpha ? data[i + 3] > 127 : data[i] > 127;
      data[i] = 255;
      data[i + 1] = 255;
      data[i + 2] = 255;
      data[i + 3] = on ? 255 : 0;
    }

    this.maskCtx.putImageData(source, 0, 0);
    this._recountMask();
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
      hasMask: this.maskNonEmpty,
      tool: this.tool,
    });
  }
}
