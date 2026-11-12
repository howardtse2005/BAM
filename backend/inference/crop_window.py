"""The single square the magic brush runs the model on.

The stroke is the prompt, so the window is derived from it: the smallest square
that contains the whole stroke -- plus the band drawn around it and a little
context -- is the entire view the model gets. One square means one forward pass
and nothing to merge, so there is no seam where two neighbouring predictions
disagreed and no crop count to tune.

Two sizes are kept apart:

``size``
    The square in canvas pixels: what the window covers on the user's image. It
    is snapped up to a multiple of 32 because the UNet has four pooling stages,
    and it is floored at ``crop_min`` so even a dot is seen with context around
    it rather than as a picture of its own thickness.

``model_size``
    The side the network is actually run at. It equals ``size`` unless the
    window is larger than ``crop_size``, the biggest crop the model was trained
    on; then the window is downscaled to fit. It is never upscaled: a short
    stroke blown up would turn a 12 px prompt into a 30 px blob outside the
    training distribution, and the model would be asked to unlearn its scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Every side handed to the UNet has to be a multiple of this.
GRID = 32


def snap_up(value: int, step: int = GRID) -> int:
    step = max(1, int(step))
    return ((int(value) + step - 1) // step) * step


@dataclass(frozen=True)
class CropWindow:
    """A square canvas window and the size it is run at."""

    x0: int
    y0: int
    size: int
    model_size: int

    @property
    def scale(self) -> float:
        """Model pixels per canvas pixel; 1.0 unless the window was shrunk."""
        return self.model_size / float(self.size)

    @property
    def downscaled(self) -> bool:
        return self.model_size != self.size


def stroke_bounds(stroke: np.ndarray) -> tuple[int, int, int, int] | None:
    """Inclusive bounding box of the painted pixels, or None for an empty stroke."""
    raw = np.asarray(stroke)
    on = raw if raw.dtype == np.bool_ else raw > 127
    ys, xs = np.nonzero(on)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def stroke_window(
    stroke: np.ndarray,
    *,
    pad: int = 0,
    min_side: int = 64,
    max_side: int = 256,
) -> CropWindow:
    """The minimum square around ``stroke``, sized for the network.

    ``pad`` grows the square on every side. It exists for two reasons: the
    visible brush band is wider than the prompt stroke stored in the mask, so
    the band would otherwise stick out of the window; and the model needs image
    context around the boundary instead of the boundary sitting on the crop
    edge, which is where it has the least information.

    Raises ``ValueError`` for an empty stroke: there is nothing to crop around,
    and silently returning a window would run the model on an empty prompt.
    """
    bounds = stroke_bounds(stroke)
    if bounds is None:
        raise ValueError("stroke is empty")

    x0, y0, x1, y1 = bounds
    pad = max(0, int(pad))

    cap = max(GRID, (int(max_side) // GRID) * GRID)
    floor = max(GRID, min(cap, snap_up(int(min_side))))

    extent = max(x1 - x0 + 1, y1 - y0 + 1) + 2 * pad
    size = max(snap_up(extent), floor)

    # Centre on the stroke's bounding box, not on its pixels' centroid: a lopsided
    # stroke (a long arc with a hook) would otherwise pull the window away from
    # its own far end.
    cx = (x0 + x1 + 1) / 2.0
    cy = (y0 + y1 + 1) / 2.0
    left = int(round(cx - size / 2.0))
    top = int(round(cy - size / 2.0))

    return CropWindow(
        x0=left,
        y0=top,
        size=size,
        model_size=min(size, cap),
    )


__all__ = ["CropWindow", "GRID", "snap_up", "stroke_bounds", "stroke_window"]
