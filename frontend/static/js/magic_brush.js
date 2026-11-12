/**
 * Magic brush transport: the two server calls.
 *
 * The stroke is the whole request. The server derives the single square that
 * contains it, so there is no crop bookkeeping on this side: a fast flick and a
 * slow drag both end in exactly one window and exactly one forward pass.
 */

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
