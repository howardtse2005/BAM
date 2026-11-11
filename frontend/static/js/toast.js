const HOST_ID = "toast-host";

export function toast(message, type = "info", timeout = 3400) {
  const host = document.getElementById(HOST_ID);
  if (!host) {
    console.warn("[bam]", message);
    return;
  }
  const el = document.createElement("div");
  el.className = "toast";
  el.dataset.type = type;
  el.textContent = message;
  host.appendChild(el);

  const remove = () => {
    el.style.opacity = "0";
    el.style.transform = "translateY(4px)";
    setTimeout(() => el.remove(), 180);
  };
  setTimeout(remove, timeout);
}

export function errorMessage(error) {
  if (!error) return "unknown error";
  if (typeof error === "string") return error;
  return error.message || String(error);
}
