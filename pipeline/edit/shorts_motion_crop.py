#!/usr/bin/env python3
"""Build the shared frame-evaluated portrait crop for motion Shorts."""

from __future__ import annotations

from typing import Any


class CropSpecError(ValueError):
    """A motion crop would be ambiguous or unsafe."""


def _bounded(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CropSpecError(f"{name} must be numeric")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise CropSpecError(f"{name} must be between 0 and 1")
    return value


def build_clip_filter(duration: float, shot: dict[str, Any], width: int,
                      height: int) -> str:
    """Return an FFmpeg crop/scale filter with an optional smooth x pan.

    ``cx0`` and ``cx1`` are normalized source-frame centers. The crop filter's
    x expression is evaluated every frame, so a motion source now receives the
    same smoothstep horizontal reframe already used for stills. Zoom remains
    deliberately static: the historical renderer never implemented dynamic
    clip zoom, and this helper does not pretend otherwise.
    """
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
        raise CropSpecError("duration must be positive")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise CropSpecError("output dimensions must be positive integers")
    cx0 = _bounded("cx0", shot.get("cx0", 0.5))
    cx1 = _bounded("cx1", shot.get("cx1", cx0))
    cy = _bounded("cy", shot.get("cy", 0.5))
    z0 = _bounded("z0", shot.get("z0", 1.0))
    if z0 <= 0:
        raise CropSpecError("z0 must be greater than zero")

    if abs(cx1 - cx0) < 1e-12:
        center_x = f"{cx0:.9f}"
    else:
        u = f"clip(t/{float(duration):.9f},0,1)"
        eased = f"(({u})*({u})*(3-2*({u})))"
        center_x = f"({cx0:.9f}+({cx1 - cx0:.9f})*{eased})"

    layout = shot.get('layout', 'portrait')
    if layout not in ('portrait', 'wide-panel'):
        raise CropSpecError('unknown motion layout')
    aspect = 9 / 16
    if layout == 'wide-panel':
        aspect = shot.get('panelAspect', 4 / 3)
        if isinstance(aspect, bool) or not isinstance(aspect, (int, float)) or not 1 <= aspect <= 16 / 9:
            raise CropSpecError('panelAspect must be between 1 and 16/9')
    crop_w = f"min(ih*{z0:.9f}*{aspect:.9f},iw)"
    crop_h = f"min(ih*{z0:.9f},iw/{aspect:.9f})"
    x = f"clip(({center_x})*iw-{crop_w}/2,0,iw-{crop_w})"
    y = f"clip({cy:.9f}*ih-{crop_h}/2,0,ih-{crop_h})"
    panel_height = 2 * round(width / aspect / 2) if layout == 'wide-panel' else height
    result = (
        f"crop='{crop_w}':'{crop_h}':'{x}':'{y}',"
        f"scale={width}:{panel_height}:flags=lanczos,setsar=1"
    )
    if layout == 'wide-panel':
        # A deliberately framed moving image, not a frozen/blurred duplicate.
        # Leaves the house question above and phrase captions below the picture.
        top_fraction = _bounded('panelTop', shot.get('panelTop', .3125))
        top = 2 * round(height * top_fraction / 2)
        if top + panel_height > height:
            raise CropSpecError('wide panel does not fit the target canvas')
        result += f',pad={width}:{height}:0:{top}:color=0x090c12'
    return result
