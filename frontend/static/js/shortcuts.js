const TYPING_TAGS = new Set(["INPUT", "SELECT", "TEXTAREA", "BUTTON"]);

function isTyping(event) {
  const target = event.target;
  if (!target) return false;
  if (target.isContentEditable) return true;
  return TYPING_TAGS.has(target.tagName);
}

export function bindShortcuts({ canvas, ui }) {
  window.addEventListener("keydown", (event) => {
    if (event.key === " " && !isTyping(event)) {
      canvas.spaceDown = true;
      event.preventDefault();
      return;
    }
    if (isTyping(event)) return;

    const meta = event.ctrlKey || event.metaKey;
    const key = event.key.toLowerCase();

    if (meta && key === "z") {
      event.preventDefault();
      if (event.shiftKey) canvas.redo();
      else canvas.undo();
      return;
    }
    if (meta && key === "y") {
      event.preventDefault();
      canvas.redo();
      return;
    }
    if (meta && key === "s") {
      event.preventDefault();
      ui.save();
      return;
    }
    if (meta) return;

    switch (key) {
      case "b":
        ui.setTool("brush");
        break;
      case "e":
        ui.setTool("eraser");
        break;
      case "m":
        ui.setTool("magic");
        break;
      case "f":
        ui.setTool("fill");
        break;
      case "h":
        ui.setTool("pan");
        break;
      case "[":
        ui.settings.el.brush.value = Math.max(1, Number(ui.settings.el.brush.value) - 2);
        ui.settings.el.brush.dispatchEvent(new Event("input"));
        break;
      case "]":
        ui.settings.el.brush.value = Math.min(80, Number(ui.settings.el.brush.value) + 2);
        ui.settings.el.brush.dispatchEvent(new Event("input"));
        break;
      case "+":
      case "=":
        canvas.zoomIn();
        break;
      case "-":
        canvas.zoomOut();
        break;
      case "0":
        canvas.fit();
        break;
      case "1":
        canvas.actualSize();
        break;
      case "tab":
        event.preventDefault();
        if (event.shiftKey) {
          const toggle = ui.settings.el.showEdge;
          toggle.checked = !toggle.checked;
          toggle.dispatchEvent(new Event("change"));
        } else {
          const toggle = ui.settings.el.showMask;
          toggle.checked = !toggle.checked;
          toggle.dispatchEvent(new Event("change"));
        }
        break;
      case "?":
        ui.hint("B brush · E eraser · M magic · F fill holes · Space pan · [ ] size · Ctrl+Z undo · Tab mask");
        break;
      default:
        break;
    }
  });

  window.addEventListener("keyup", (event) => {
    if (event.key === " ") canvas.spaceDown = false;
  });

  window.addEventListener("blur", () => {
    canvas.spaceDown = false;
  });
}
