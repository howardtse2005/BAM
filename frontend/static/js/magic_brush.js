/**
 * Magic brush transport: crop collection and the two server calls.
 *
 * The brush stroke is the only prompt. Crops are emitted by arc length, not by
 * pointer events, so a fast flick and a slow drag produce the same uniform
 * coverage along the line.
 *
 * Crop size is not fixed. It is a fraction of the canvas short side, clamped to
 * [cropMin, cropMax] and snapped to a multiple of 32, so a small image is not
 * handed a crop that contains the whole object.
 */
export class CropCollector {
  constructor({ cropSize = 256, stride = null, maxCrops = 64 } = {}) {
    this.maxCrops = maxCrops;
    this.cropSize = cropSize;
    this.stride = stride || Math.max(16, Math.round(cropSize / 2));
    this.reset();
  }

  configure({ cropSize, stride = null } = {}) {
    if (cropSize) this.cropSize = cropSize;
    this.stride = stride || Math.max(16, Math.round(this.cropSize / 2));
  }

  reset(cropSize = null) {
    if (cropSize) this.configure({ cropSize });
    this.crops = [];
    this.last = null;
    this.full = false;
    this.length = 0;
  }

  add(x, y) {
    if (this.full) return false;

    if (this.last) {
      const step = Math.hypot(x - this.last.x, y - this.last.y);
      this.length += step;
      if (step < this.stride) return false;
    }

    if (this.crops.length >= this.maxCrops) {
      this.full = true;
      return false;
    }

    const half = this.cropSize / 2;
    this.crops.push({
      x0: Math.round(x - half),
      y0: Math.round(y - half),
      size: this.cropSize,
    });
    this.last = { x, y };
    return true;
  }

  get count() {
    return this.crops.length;
  }
}

async function postJSON(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok || !data || data.status !== "success") {
    const message = (data && data.message) || `request failed (HTTP ${response.status})`;
    throw new Error(message);
  }
  return data;
}

/** Rough stroke in, predicted clean line out. Nothing is committed yet. */
export function sendSegment(payload) {
  return postJSON("/api/segment", payload);
}

/** Accepted line in, filled object mask out. */
export function sendFill(payload) {
  return postJSON("/api/fill", payload);
}
