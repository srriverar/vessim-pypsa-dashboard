"""Daily shapes, ported from the ED/UC Simulator's ``ED.PROFILES``.

Values are per-unit multipliers of a load's peak (or a generator's rating) and
are taken verbatim from ``ed-uc-simulator/ed-core.js`` in
github.com/adjiebrotot/tool (GPL-3.0).  ``d`` is the weekday factor, Monday
first, used to give a shape a week.
"""

from __future__ import annotations

PROFILES: dict[str, dict] = {
    "sys": dict(
        label="System day (peaks at 18:00)", kind="load",
        h=[0.62, 0.58, 0.56, 0.55, 0.56, 0.60, 0.68, 0.77, 0.83, 0.86, 0.88, 0.89,
           0.89, 0.88, 0.87, 0.87, 0.89, 0.94, 1.00, 0.99, 0.95, 0.87, 0.77, 0.68],
        d=[0.97, 1, 1, 1, 0.97, 0.88, 0.84],
    ),
    "res": dict(
        label="Residential", kind="load",
        h=[0.45, 0.40, 0.38, 0.37, 0.38, 0.45, 0.62, 0.78, 0.72, 0.62, 0.58, 0.56,
           0.55, 0.54, 0.55, 0.60, 0.70, 0.86, 1.00, 0.98, 0.90, 0.78, 0.64, 0.52],
        d=[0.98, 0.97, 0.97, 0.98, 1, 1.03, 1.02],
    ),
    "com": dict(
        label="Commercial", kind="load",
        h=[0.38, 0.36, 0.35, 0.35, 0.36, 0.40, 0.52, 0.72, 0.88, 0.96, 1.00, 1.00,
           0.98, 0.99, 0.98, 0.95, 0.88, 0.74, 0.60, 0.52, 0.47, 0.44, 0.41, 0.39],
        d=[0.98, 1, 1, 1, 0.96, 0.85, 0.75],
    ),
    "ind": dict(
        label="Industrial (flat)", kind="load",
        h=[0.78, 0.77, 0.76, 0.76, 0.77, 0.80, 0.88, 0.96, 1.00, 1.00, 0.99, 0.97,
           0.95, 0.98, 1.00, 0.99, 0.96, 0.90, 0.86, 0.84, 0.82, 0.81, 0.80, 0.79],
        d=[0.98, 1, 1, 1, 0.98, 0.88, 0.82],
    ),
    "pv": dict(
        label="Solar (spring/autumn)", kind="gen",
        h=[0, 0, 0, 0, 0, 0, 0.03, 0.15, 0.35, 0.55, 0.70, 0.79,
           0.82, 0.79, 0.70, 0.55, 0.35, 0.15, 0.03, 0, 0, 0, 0, 0],
    ),
    "pveq": dict(
        label="Solar (equator, Bogotá-like)", kind="gen",
        h=[0, 0, 0, 0, 0, 0, 0.02, 0.14, 0.34, 0.54, 0.70, 0.80,
           0.83, 0.78, 0.66, 0.49, 0.29, 0.10, 0.01, 0, 0, 0, 0, 0],
    ),
    "wnd": dict(
        label="Wind", kind="gen",
        h=[0.62, 0.66, 0.70, 0.72, 0.70, 0.66, 0.60, 0.52, 0.44, 0.38, 0.33, 0.30,
           0.29, 0.31, 0.35, 0.40, 0.46, 0.52, 0.56, 0.58, 0.60, 0.61, 0.60, 0.61],
    ),
}

PROFILE_DEFAULT = "sys"


def shape(name: str, hours: int = 24) -> list[float]:
    """Return ``hours`` multipliers of a named daily shape."""
    prof = PROFILES.get(name)
    if prof is None:
        raise KeyError(f"Unknown profile '{name}'. Available: {', '.join(PROFILES)}")
    daily = prof["h"]
    out = [daily[h % 24] for h in range(hours)]
    if prof.get("d"):
        day_factors = prof["d"]
        out = [value * day_factors[(h // 24) % 7] for h, value in enumerate(out)]
    return out
