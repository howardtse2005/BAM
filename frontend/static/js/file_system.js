async function parse(response) {
  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  if (!response.ok || (data && data.status === "error")) {
    const message = (data && data.message) || `HTTP ${response.status}`;
    throw new Error(message);
  }
  return data;
}

export class FileSystem {
  constructor(dataset, split = "train") {
    this.dataset = dataset;
    this.split = split;
  }

  setSplit(split) {
    this.split = split;
  }

  async health() {
    return parse(await fetch("/api/health"));
  }

  async config() {
    return parse(await fetch("/api/config"));
  }

  async images() {
    const data = await parse(
      await fetch(
        `/api/datasets/${encodeURIComponent(this.dataset)}/images?split=${encodeURIComponent(
          this.split
        )}&limit=1000`
      )
    );
    return data.images || [];
  }

  async meta(name) {
    return parse(
      await fetch(
        `/api/datasets/${encodeURIComponent(this.dataset)}/${encodeURIComponent(
          this.split
        )}/${encodeURIComponent(name)}/meta`
      )
    );
  }

  async mask(name) {
    const response = await fetch(
      `/api/mask/${encodeURIComponent(this.dataset)}/${encodeURIComponent(
        this.split
      )}/${encodeURIComponent(name)}`
    );
    if (response.status === 404) return null;
    const data = await parse(response);
    return data.mask_base64 || null;
  }

  async saveMask(name, maskBase64) {
    return parse(
      await fetch("/api/mask/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset: this.dataset,
          split: this.split,
          image: name,
          mask_base64: maskBase64,
        }),
      })
    );
  }

  async upload(name, dataUrl) {
    return parse(
      await fetch("/api/image/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset: this.dataset,
          split: this.split,
          name,
          image_base64: dataUrl,
        }),
      })
    );
  }

  async segment(payload) {
    return parse(
      await fetch("/api/segment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
    );
  }
}

export function imageUrl(dataset, split, name) {
  return `/media/${encodeURIComponent(dataset)}/${encodeURIComponent(split)}/im/${encodeURIComponent(
    name
  )}`;
}

export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("image failed to load"));
    image.src = src;
  });
}
