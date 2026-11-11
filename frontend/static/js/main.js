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

  const fs = new FileSystem(boot.dataset);
  const canvas = new EditorCanvas({
    stage,
    wrap,
    dataset: boot.dataset,
    image: boot.image,
    cropSize: boot.cropSize,
    stride: boot.stride,
    maxCrops: boot.maxCrops,
  });
  const settings = new Settings(canvas);
  const ui = new UI({ canvas, fs, settings, bootstrap: boot });
  ui.settings = settings;

  bindShortcuts({ canvas, ui });

  (async () => {
    try {
      const health = await fs.health();
      ui.updateModel(health.model);
    } catch (error) {
      ui.onError(error);
    }

    if (boot.image) {
      await ui.loadImage(boot.image);
    } else {
      // Opening a dataset without naming an image should land on the first one,
      // not on an empty canvas plus a modal.
      try {
        const images = await fs.images();
        if (images.length) {
          await ui.loadImage(images[0]);
        } else {
          await ui.openPicker();
        }
      } catch {
        await ui.openPicker();
      }
    }

    requestAnimationFrame(() => canvas.fit());
  })();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot0);
} else {
  boot0();
}
