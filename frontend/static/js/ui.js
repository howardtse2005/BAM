import { errorMessage, toast } from "./toast.js";
import { initTheme } from "./theme.js";

function el(id) {
  return document.getElementById(id);
}

function formatMs(value) {
  if (value === null || value === undefined) return "—";
  return `${Math.round(value)} ms`;
}

export class UI {
  constructor({ canvas, fs, settings, bootstrap }) {
    this.canvas = canvas;
    this.fs = fs;
    this.settings = settings;
    this.bootstrap = bootstrap;
    this.images = [];
    this.modal = el("modal");
    this.list = el("image-list");
    this.search = el("modal-search");
    this.splitSelect = el("split-select");

    this.bindToolbar();
    this.bindTopbar();
    this.bindModal();
    this.bindSplit();
    this.bindProposal();

    initTheme(el("theme-toggle"), () => this.canvas.refreshTheme());

    let resizeTimer = null;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => this.canvas.handleResize(), 150);
    });
    this.canvas.refreshTheme();
    this.settings.applyToCanvas();
    this.canvas.onStatus = (status) => this.updateStatus(status);
    this.canvas.onProgress = (on, text) => this.setProgress(on, text);
    this.canvas.onProposal = (result) => this.onProposal(result);
    this.canvas.onProposalCleared = () => this.hideProposalBar();
    this.canvas.onFilled = (result) => this.onFilled(result);
    this.canvas.onError = (error) => this.onError(error);
    this.canvas.getParams = () => this.settings.params();
  }

  // ------------------------------------------------------------------ bind

  bindToolbar() {
    document.querySelectorAll("[data-tool]").forEach((button) => {
      button.addEventListener("click", () => this.setTool(button.dataset.tool));
    });
    el("btn-undo")?.addEventListener("click", () => this.canvas.undo());
    el("btn-redo")?.addEventListener("click", () => this.canvas.redo());
    el("btn-zoom-in")?.addEventListener("click", () => this.canvas.zoomIn());
    el("btn-zoom-out")?.addEventListener("click", () => this.canvas.zoomOut());
    el("btn-fit")?.addEventListener("click", () => this.canvas.fit());
  }

  bindTopbar() {
    el("btn-open")?.addEventListener("click", () => this.openPicker());
    el("btn-open-2")?.addEventListener("click", () => this.openPicker());
    el("btn-save")?.addEventListener("click", () => this.save());
    el("btn-upload")?.addEventListener("click", () => el("file-input")?.click());
    el("file-input")?.addEventListener("change", (event) => {
      const file = event.target.files?.[0];
      if (file) this.upload(file);
      event.target.value = "";
    });
  }

  bindModal() {
    el("modal-close")?.addEventListener("click", () => this.closePicker());
    this.modal?.addEventListener("click", (event) => {
      if (event.target === this.modal) this.closePicker();
    });
    this.search?.addEventListener("input", () => {
      const query = this.search.value.trim().toLowerCase();
      const filtered = query
        ? this.images.filter((name) => name.toLowerCase().includes(query))
        : this.images;
      this.renderList(filtered);
    });
  }

  bindProposal() {
    el("btn-accept-line")?.addEventListener("click", () => this.acceptProposal());
    el("btn-fill-line")?.addEventListener("click", () => this.fillProposal());
    el("btn-discard-line")?.addEventListener("click", () => this.discardProposal());
  }

  bindSplit() {
    if (!this.splitSelect) return;
    this.splitSelect.value = this.bootstrap.split || "train";
    this.splitSelect.addEventListener("change", () => this.setSplit(this.splitSelect.value));
  }

  async setSplit(split) {
    this.fs.setSplit(split);
    this.canvas.setSplit(split);
    if (this.splitSelect) this.splitSelect.value = split;

    const url = new URL(window.location.href);
    url.searchParams.set("split", split);
    window.history.replaceState({}, "", url);

    try {
      const images = await this.fs.images();
      if (images.length) {
        await this.loadImage(images[0]);
      } else {
        this.canvas.clearMask(false);
        el("image-name").textContent = "no images";
        this.hint(`No images in the ${split} split`);
      }
    } catch (error) {
      this.onError(error);
    }
  }

  setTool(tool) {
    this.canvas.setTool(tool);
    document.querySelectorAll("[data-tool]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.tool === tool);
    });
    const hints = {
      magicBrush:
        "Drag roughly along the object edge. Release to see the predicted line, then accept it or re-stroke.",
      pan: "Drag to pan. Wheel to zoom.",
      fill: "Fill the region enclosed by the accepted line.",
    };
    this.hint(hints[tool] || null);
  }

  hint(text) {
    const node = el("stage-hint");
    if (!node) return;
    if (!text) {
      node.hidden = true;
      return;
    }
    node.textContent = text;
    node.hidden = false;
    clearTimeout(this._hintTimer);
    this._hintTimer = setTimeout(() => {
      node.hidden = true;
    }, 4200);
  }

  // ------------------------------------------------------------------ image

  async openPicker() {
    if (!this.modal) return;
    this.modal.hidden = false;
    this.list.innerHTML = '<li class="muted" style="padding:8px 10px">loading…</li>';
    if (this.search) this.search.value = "";
    try {
      this.images = await this.fs.images();
      this.renderList(this.images);
      this.search?.focus();
    } catch (error) {
      this.list.innerHTML = "";
      toast(errorMessage(error), "error");
    }
  }

  closePicker() {
    if (this.modal) this.modal.hidden = true;
  }

  renderList(items) {
    if (!this.list) return;
    this.list.innerHTML = "";
    if (!items.length) {
      const empty = document.createElement("li");
      empty.className = "muted";
      empty.style.padding = "8px 10px";
      empty.textContent = "no images";
      this.list.appendChild(empty);
      return;
    }
    const fragment = document.createDocumentFragment();
    for (const name of items) {
      const li = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = name;
      if (name === this.canvas.imageName) button.classList.add("is-current");
      button.addEventListener("click", () => {
        this.closePicker();
        this.loadImage(name);
      });
      li.appendChild(button);
      fragment.appendChild(li);
    }
    this.list.appendChild(fragment);
  }

  async loadImage(name) {
    try {
      // Opening an image starts from an empty mask. A stored mask is never
      // fetched here: the dataset's gt is the answer to the question the editor
      // asks, so loading it would segment the object before the user drew
      // anything. The flow is stroke -> clean line -> accept -> fill.
      await this.canvas.load(name);
      el("image-name").textContent = name;
      document.title = `${name} — BAM`;
      el("stage-empty").hidden = true;
      this.hideProposalBar();
      this.updateImageInfo();
      this.hint(
        "Magic brush: drag roughly along the object edge, accept the clean line, then fill."
      );
    } catch (error) {
      toast(`Could not open ${name}: ${errorMessage(error)}`, "error");
    }
  }

  async upload(file) {
    try {
      const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error("could not read file"));
        reader.readAsDataURL(file);
      });
      const result = await this.fs.upload(file.name, dataUrl);
      toast(`Uploaded ${file.name}`, "ok");
      await this.loadImage(result.image);
    } catch (error) {
      toast(errorMessage(error), "error");
    }
  }

  async save() {
    if (!this.canvas.imageName) {
      toast("Open an image first", "warn");
      return;
    }
    // Saving an empty mask would replace the dataset's gt with a blank image.
    // The editor starts empty by design, so this is the state a stray Ctrl+S
    // lands in.
    if (!this.canvas.maskNonEmpty) {
      toast("Nothing to save — draw an object first", "warn");
      return;
    }
    try {
      const result = await this.fs.saveMask(
        this.canvas.imageName,
        this.canvas.getMaskBase64()
      );
      toast(`Mask saved (${result.width}×${result.height})`, "ok");
      this.updateImageInfo();
    } catch (error) {
      toast(errorMessage(error), "error");
    }
  }

  // -------------------------------------------------------------- proposal

  showProposalBar(meta) {
    const bar = el("proposal-bar");
    if (!bar) return;
    bar.hidden = false;

    const label = el("proposal-label");
    if (label) {
      const closed = meta.closed ? "closed loop" : "open line";
      const crops = meta.crops ? `${meta.crops} crop${meta.crops === 1 ? "" : "s"}` : "";
      const size = meta.cropSize ? `${meta.cropSize}px` : "";
      label.textContent = ["Predicted edge", closed, crops, size]
        .filter(Boolean)
        .join(" · ");
    }

    // Filling an open line has no interior to fill, so the action is disabled
    // rather than offered and then refused.
    const fillButton = el("btn-fill-line");
    if (fillButton) {
      fillButton.disabled = !meta.closed;
      fillButton.title = meta.closed
        ? "Fill the region enclosed by this line"
        : "The line does not enclose a region yet";
    }
  }

  hideProposalBar() {
    const bar = el("proposal-bar");
    if (bar) bar.hidden = true;
  }

  acceptProposal() {
    if (!this.canvas.acceptProposal()) return false;
    this.hideProposalBar();
    this.updateImageInfo();
    this.hint("Line accepted. Press F to fill the enclosed region.");
    return true;
  }

  discardProposal() {
    this.canvas.clearProposal();
    this.hideProposalBar();
    this.hint("Line discarded. Drag again to re-stroke.");
  }

  async fillProposal() {
    if (!this.canvas.proposalReady) return;
    this.canvas.acceptProposal();
    this.hideProposalBar();
    await this.canvas.fillMask();
    this.updateImageInfo();
  }

  // ----------------------------------------------------------------- status

  updateStatus(status) {
    const zoom = Math.round(status.zoom * 100);
    const crops = status.crops;
    if (el("s-zoom")) el("s-zoom").textContent = `${zoom}%`;
    // Keep the last meaningful crop count: the collector resets after a stroke,
    // and "0 crops" is less useful than "the last stroke used 9".
    if (crops > 0) {
      if (el("s-crops")) el("s-crops").textContent = `${crops} crop${crops === 1 ? "" : "s"}`;
      if (el("i-crops")) el("i-crops").textContent = String(crops);
    }
    if (el("i-crop-size") && status.cropSize) {
      el("i-crop-size").textContent = `${status.cropSize} px`;
    }
    if (el("s-size") && this.canvas.width) {
      el("s-size").textContent = `${this.canvas.width} × ${this.canvas.height}`;
    }
    if (el("i-size") && this.canvas.width) {
      el("i-size").textContent = `${this.canvas.width} × ${this.canvas.height}`;
    }
    if (el("i-mask")) {
      el("i-mask").textContent = status.hasMask ? "edited" : "empty";
    }
    if (el("i-line")) {
      el("i-line").textContent = status.proposal ? "pending" : "none";
    }
  }

  updateImageInfo() {
    if (el("i-size") && this.canvas.width) {
      el("i-size").textContent = `${this.canvas.width} × ${this.canvas.height}`;
    }
    if (el("s-size") && this.canvas.width) {
      el("s-size").textContent = `${this.canvas.width} × ${this.canvas.height}`;
    }
    if (el("i-mask")) {
      el("i-mask").textContent = this.canvas.maskNonEmpty ? "edited" : "empty";
    }
  }

  setProgress(on, text) {
    const node = el("stage-progress");
    if (!node) return;
    node.hidden = !on;
    if (text && el("progress-text")) el("progress-text").textContent = text;
  }

  onProposal(result) {
    const stats = result.stats || {};
    if (el("m-latency")) el("m-latency").textContent = formatMs(stats.ms_total);
    if (el("s-latency")) {
      el("s-latency").textContent = `${Math.round(stats.ms_infer || 0)} ms infer · ${Math.round(
        stats.ms_post || 0
      )} ms post`;
    }
    if (el("i-crops")) el("i-crops").textContent = String(stats.num_crops || 0);
    this.canvas.setEdge(result.edge_base64);

    this.showProposalBar(result);
    if (!result.closed) {
      this.hint("The line is open. Accept it as-is, re-stroke, or close the gap to fill.");
    }
    this.updateImageInfo();
  }

  onFilled(result) {
    const stats = result.stats || {};
    if (el("s-latency")) {
      el("s-latency").textContent = `${Math.round(stats.ms_fill || 0)} ms fill`;
    }
    toast(`Filled ${Number(result.added_px || 0).toLocaleString()} px`, "ok");
    this.updateImageInfo();
  }

  onError(error) {
    toast(errorMessage(error), "error", 5200);
  }

  updateModel(info) {
    if (el("m-backend")) {
      el("m-backend").textContent = info.is_mock ? "mock" : info.model || "—";
    }
    if (el("m-device")) el("m-device").textContent = info.device || "—";
    if (el("m-tier")) el("m-tier").textContent = info.data_tier || "—";
    if (el("s-backend")) {
      el("s-backend").textContent = `${info.model || "none"} · ${info.device || "cpu"}`;
    }

    const warning = el("m-warning");
    if (!warning) return;
    if (!info.available) {
      warning.hidden = false;
      warning.textContent =
        "No model loaded. The magic brush will fail until a checkpoint exists in model/.";
    } else if (info.is_mock) {
      warning.hidden = false;
      warning.textContent =
        "Placeholder model: no checkpoint found, so the magic brush returns a band around your stroke. Train or drop a .pth into model/ to enable real predictions.";
    } else {
      warning.hidden = true;
    }
  }
}
