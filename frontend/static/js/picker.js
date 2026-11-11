import { initTheme } from "./theme.js";

initTheme(document.getElementById("theme-toggle"));

const badge = document.getElementById("model-badge");
const count = document.getElementById("dataset-count");
const cards = document.querySelectorAll(".dataset-card").length;

if (count) {
  count.textContent = `${cards} dataset${cards === 1 ? "" : "s"}`;
}

async function refreshHealth() {
  if (!badge) return;
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    const model = data.model || {};
    if (!model.available) {
      badge.dataset.state = "error";
      badge.textContent = "no model";
      badge.title = model.error || "no checkpoint loaded";
    } else if (model.is_mock) {
      badge.dataset.state = "warn";
      badge.textContent = "mock model";
      badge.title = "No checkpoint found; predictions are a placeholder.";
    } else {
      badge.dataset.state = "ok";
      badge.textContent = `${model.model} · ${model.device}`;
      badge.title = `data tier ${model.data_tier || "?"}`;
    }
  } catch (error) {
    badge.dataset.state = "error";
    badge.textContent = "server offline";
    badge.title = String(error);
  }
}

refreshHealth();
