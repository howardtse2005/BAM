const STORAGE_KEY = "bam.theme";

export function currentTheme() {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem(STORAGE_KEY, theme);
}

export function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(next);
  return next;
}

export function initTheme(button, onChange) {
  applyTheme(currentTheme());
  if (!button) return;
  button.addEventListener("click", () => {
    const theme = toggleTheme();
    onChange?.(theme);
  });
}
