/**
 * Inspector bindings.
 *
 * Values are read into a plain object that is sent with every magic-brush and
 * fill request, and persisted to localStorage so the tool feels stable between
 * sessions.
 */

const STORAGE_KEY = "bam.params";

const DEFAULTS = {
  sensitivity: 1.0,
  threshold: 0.5,
  edge_threshold: 0.45,
  expand: 0,
  smooth: 2,
  merge_weighting: "uniform",
  magicWidth: 32,
  brush: 8,
  opacity: 35,
  showMask: true,
  showEdge: false,
};

export class Settings {
  constructor(canvas) {
    this.canvas = canvas;
    this.values = { ...DEFAULTS };
    this.el = {
      sensitivity: document.getElementById("p-sensitivity"),
      edge: document.getElementById("p-edge"),
      mask: document.getElementById("p-mask"),
      expand: document.getElementById("p-expand"),
      smooth: document.getElementById("p-smooth"),
      weight: document.getElementById("p-weight"),
      magicWidth: document.getElementById("p-magic-width"),
      brush: document.getElementById("p-brush"),
      opacity: document.getElementById("p-opacity"),
      showMask: document.getElementById("c-show-mask"),
      showEdge: document.getElementById("c-show-edge"),
    };
    this.out = {
      sensitivity: document.getElementById("o-sensitivity"),
      edge: document.getElementById("o-edge"),
      mask: document.getElementById("o-mask"),
      expand: document.getElementById("o-expand"),
      smooth: document.getElementById("o-smooth"),
      magicWidth: document.getElementById("o-magic-width"),
      brush: document.getElementById("o-brush"),
      opacity: document.getElementById("o-opacity"),
    };
    this.bind();
    this.load();
  }

  bind() {
    this.range("sensitivity", (value) => value.toFixed(1));
    this.range("edge", (value) => value.toFixed(2));
    this.range("mask", (value) => value.toFixed(2));
    this.range("expand", (value) => `${value > 0 ? "+" : ""}${value} px`);
    this.range("smooth", (value) => String(value));
    this.range("magicWidth", (value) => this.magicLabel(value));
    this.range("brush", (value) => `${value} px`);
    this.range("opacity", (value) => `${value}%`);

    this.el.weight?.addEventListener("change", (event) => {
      this.values.merge_weighting = event.target.value;
      this.save();
    });
    this.el.showMask?.addEventListener("change", (event) => {
      this.values.showMask = event.target.checked;
      this.canvas.setMaskVisible(this.values.showMask);
      this.save();
    });
    this.el.showEdge?.addEventListener("change", (event) => {
      this.values.showEdge = event.target.checked;
      this.canvas.setShowEdge(this.values.showEdge);
      this.save();
    });
  }

  /**
   * "32 px · core 20": the band the user draws, then the thinner prompt stroke
   * the model is actually given. Showing both makes the tolerance explicit
   * instead of leaving it as an invisible implementation detail.
   */
  magicLabel(band) {
    return `${band} px · core ${this.canvas.promptWidthFor(band)}`;
  }

  range(key, format) {
    const input = this.el[key];
    if (!input) return;
    input.addEventListener("input", () => {
      const value = Number(input.value);
      this.values[key] = value;
      // The canvas has to be updated before the label is formatted: the label
      // reports the prompt width derived from the new band.
      if (key === "brush") this.canvas.setBrushSize(value);
      if (key === "magicWidth") this.canvas.setMagicWidth(value);
      if (key === "opacity") this.canvas.setMaskOpacity(value);
      const out = this.out[key];
      if (out) out.textContent = format(value);
      this.save();
    });
  }

  applyToCanvas() {
    this.canvas.setBrushSize(this.values.brush);
    this.canvas.setMagicWidth(this.values.magicWidth);
    this.canvas.setMaskOpacity(this.values.opacity);
    this.canvas.setMaskVisible(this.values.showMask);
    this.canvas.setShowEdge(this.values.showEdge);
  }

  params() {
    return {
      sensitivity: this.values.sensitivity,
      threshold: this.values.threshold,
      edge_threshold: this.values.edge_threshold,
      expand: this.values.expand,
      smooth: this.values.smooth,
      merge_weighting: this.values.merge_weighting,
    };
  }

  load() {
    let stored = null;
    try {
      stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    } catch {
      stored = null;
    }
    if (stored && typeof stored === "object") {
      for (const key of Object.keys(DEFAULTS)) {
        if (key in stored) this.values[key] = stored[key];
      }
    }
    this.render();
  }

  render() {
    const set = (input, value) => {
      if (input) input.value = value;
    };
    set(this.el.sensitivity, this.values.sensitivity);
    set(this.el.edge, this.values.edge_threshold);
    set(this.el.mask, this.values.threshold);
    set(this.el.expand, this.values.expand);
    set(this.el.smooth, this.values.smooth);
    set(this.el.magicWidth, this.values.magicWidth);
    set(this.el.brush, this.values.brush);
    set(this.el.opacity, this.values.opacity);
    if (this.el.weight) this.el.weight.value = this.values.merge_weighting;
    if (this.el.showMask) this.el.showMask.checked = this.values.showMask;
    if (this.el.showEdge) this.el.showEdge.checked = this.values.showEdge;

    if (this.out.sensitivity) this.out.sensitivity.textContent = Number(this.values.sensitivity).toFixed(1);
    if (this.out.edge) this.out.edge.textContent = Number(this.values.edge_threshold).toFixed(2);
    if (this.out.mask) this.out.mask.textContent = Number(this.values.threshold).toFixed(2);
    if (this.out.expand) {
      const v = Number(this.values.expand);
      this.out.expand.textContent = `${v > 0 ? "+" : ""}${v} px`;
    }
    if (this.out.smooth) this.out.smooth.textContent = String(this.values.smooth);
    if (this.out.magicWidth) {
      this.out.magicWidth.textContent = this.magicLabel(this.values.magicWidth);
    }
    if (this.out.brush) this.out.brush.textContent = `${this.values.brush} px`;
    if (this.out.opacity) this.out.opacity.textContent = `${this.values.opacity}%`;
  }

  save() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.values));
    } catch {
      /* storage may be disabled */
    }
  }
}
