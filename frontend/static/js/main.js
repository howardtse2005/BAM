import { EditorCanvas } from "./canvas.js";
import { FileSystem } from "./file_system.js";
import { Settings } from "./settings.js";
import { bindShortcuts } from "./shortcuts.js";
import { UI } from "./ui.js";

const boot = window.BAM_BOOTSTRAP;

function boot0() {
  if (!boot) {
    console.error("[bam] missing BAM_BOOTSTRAP");
    return;
  }

  const stage = document.getElementById("stage");
  const wrap = document.getElementById("canvas-wrap");
  if (!stage || !wrap) return;

  const split = boot.split || "train";
  const fs = new FileSystem(boot.dataset, split);
  const canvas = new EditorCanvas({
    stage,
    wrap,
    dataset: boot.dataset,
    split,
    image: boot.image,
    cropSize: boot.cropSize,
    cropMin: boot.cropMin,
    cropRatio: boot.cropRatio,
    stride: boot.cropStride || 128,
    maxCrops: boot.maxCrops,
    lineWidth: boot.lineWidth,
    fillClose: boot.fillClose,
    magicWidth: boot.magicWidth,
    promptRatio: boot.magicPromptRatio,
    promptMin: boot.magicPromptMin,
    promptMax: boot.magicPromptMax,
  });
  const settings = new Settings(canvas);
  const ui = new UI({ canvas, fs, settings });
  ui.settings = settings;

  bindShortcuts({ canvas, ui });

  (async () => {
    try {
      const health = await fs.health();
      ui.updateModel(health.model);
    } catch (error) {
      ui.onError(error);
    }

    try {
      const config = await fs.config();
      canvas.shortSide = config.config.short_side;
    } catch {
      /* keep the default */
    }

    if (boot.image) {
      await ui.loadImage(boot.image);
    }
    // No image named in the URL: the editor stays empty and the stage asks for
    // an upload. Auto-opening the first file in the folder would hand the user
    // somebody else's image — the point of the site is their own.

    requestAnimationFrame(() => canvas.fit());
  })();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot0);
} else {
  boot0();
}
