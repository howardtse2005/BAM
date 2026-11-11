/**
 * Crop collection along a stroke.
 *
 * Crops are emitted by arc length, not by pointer events, so a fast flick and a
 * slow drag produce the same uniform coverage. This is the fix for the reference
 * implementation's `crop_every_len * crops.length` counter, which drifts when a
 * stroke is retried.
 */
export class CropCollector {
  constructor({ cropSize = 256, stride = 128, maxCrops = 64 } = {}) {
    this.cropSize = cropSize;
    this.stride = stride;
    this.maxCrops = maxCrops;
    this.reset();
  }

  reset() {
    this.crops = [];
    this.last = null;
    this.full = false;
    this.length = 0;
  }

  add(x, y) {
    if (this.full) return false;

    if (this.last) {
      this.length += Math.hypot(x - this.last.x, y - this.last.y);
      if (Math.hypot(x - this.last.x, y - this.last.y) < this.stride) {
        return false;
      }
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

export async function sendSegment(payload) {
  const response = await fetch("/api/segment", {
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
    const message = (data && data.message) || `segmentation failed (HTTP ${response.status})`;
    throw new Error(message);
  }
  return data;
}
