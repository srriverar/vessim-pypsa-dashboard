"""Computing workloads: the reason Vessim exists.

Vessim is a co-simulation testbed for *carbon-aware computing*. Its actors are
not generic loads: they are containers, batch jobs, inference services, GPU
clusters and the storage behind them. This module gives every case of the
dashboard an explicit computing layer, so that the same five examples that teach
dispatch also teach what is being dispatched for.

A workload is described by :class:`ComputingSpec` — a kind, an IT power, a PUE,
and the share that can move in time — and turned into a power trace on the
simulation grid by :func:`trace_for`. The trace is what a ``vessim.Consumer``
receives, so PyPSA sees the sum and Vessim sees the program.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

#: Human labels and the one-line justification used in the tables.
KIND_INFO: dict[str, tuple[str, str]] = {
    "edge": ("Edge device", "always on, a few watts, latency-bound: it cannot move"),
    "inference": ("Inference service", "follows its users: flat with a daytime peak, not deferrable"),
    "training": ("Model training", "batch, interruptible and deadline-free: the deferrable part"),
    "batch": ("Batch / ETL job", "runs in a nightly window between two deadlines"),
    "storage": ("Storage & backup", "NAS and object storage: mostly flat, a nightly backup bump"),
    "network": ("Network fabric", "switches and optics, proportional to the traffic"),
    "cooling": ("Cooling overhead", "the PUE term: it follows the IT load, it is not a workload"),
}


@dataclass(frozen=True)
class ComputingSpec:
    """One computing workload, in the units the plant cares about.

    ``kw`` is the **IT** power; the site sees ``kw * pue``.  ``flexible`` is the
    share of the IT power that may be moved in time (``1.0`` = fully
    deferrable).  ``detail`` is what the dashboard prints next to the number.
    """

    name: str
    kind: str
    kw: float
    pue: float = 1.1
    flexible: float = 0.0
    detail: str = ""

    @property
    def site_kw(self) -> float:
        return self.kw * self.pue

    @property
    def label(self) -> str:
        return KIND_INFO.get(self.kind, (self.kind.title(), ""))[0]


def water_fill(dirty: np.ndarray, cap: float = 2.0) -> np.ndarray:
    """Move a unit mean of work out of the dirty steps, without losing any.

    Returns weights with ``mean == 1``: zero on dirty steps while the clean ones
    can absorb the work at up to ``cap`` times the mean, and the remainder left
    on the dirty steps when they cannot.  Energy is conserved by construction —
    the property a load profile in an LP can never verify.
    """
    dirty = np.asarray(dirty, dtype=bool)
    n = len(dirty)
    clean = ~dirty
    n_clean = int(clean.sum())
    weights = np.zeros(n, dtype=float)
    if n_clean == 0:
        return np.ones(n, dtype=float)
    if n / n_clean <= cap:
        weights[clean] = n / n_clean
    else:
        weights[clean] = cap
        dirty_mask = dirty
        n_dirty = n - n_clean
        if n_dirty:
            weights[dirty_mask] = (n - cap * n_clean) / n_dirty
        else:                                    # pragma: no cover - defensive
            weights[:] = 1.0
    return weights


def trace_for(spec: ComputingSpec, steps: int, step_s: int,
              carbon: Callable[[np.ndarray], np.ndarray] | None = None,
              threshold: float = 0.35) -> np.ndarray:
    """Power this workload draws from the site, in kW, on the simulation grid.

    Every shape is expressed as a multiple of the IT power, then multiplied by
    the PUE and by the flexible share, so the daily energy stays the same no
    matter how the work is placed.
    """
    hours = (np.arange(steps) * step_s / 3600.0) % 24.0
    kind = spec.kind
    if kind == "inference":
        shape = 0.88 + 0.12 * np.sin(2 * np.pi * (hours - 10.0) / 24.0)
    elif kind == "edge":
        shape = 0.95 + 0.05 * np.sin(2 * np.pi * (hours - 6.0) / 24.0)
    elif kind == "batch":
        inside = (hours >= 22.0) | (hours < 6.0)
        shape = np.where(inside, 24.0 / 8.0, 0.0)          # 8 h window, same energy
    elif kind == "storage":
        shape = 1.0 + 0.25 * np.exp(-(((hours - 3.0) / 1.5) ** 2))
        shape = shape / shape.mean()
    elif kind == "network":
        shape = 0.9 + 0.10 * np.sin(2 * np.pi * (hours - 14.0) / 24.0)
    elif kind == "training":
        shape = np.ones(steps)
    else:                                                   # flat by default
        shape = np.ones(steps)

    flexible = float(np.clip(spec.flexible, 0.0, 1.0))
    if kind == "training" and flexible > 0.0 and carbon is not None:
        dirty = carbon(hours) > threshold
        weights = water_fill(dirty, cap=2.0)
        # the flexible share follows the clean hours, the rest is flat
        shape = (1.0 - flexible) * np.ones(steps) + flexible * weights
    return spec.kw * spec.pue * shape


def build(specs: tuple[ComputingSpec, ...], steps: int, step_s: int,
          carbon: Callable[[np.ndarray], np.ndarray] | None = None,
          threshold: float = 0.35) -> dict[str, np.ndarray]:
    """All the computing traces of a case, keyed by actor name (kW)."""
    return {spec.name: trace_for(spec, steps, step_s, carbon, threshold) for spec in specs}


def summary(specs: tuple[ComputingSpec, ...], traces: dict[str, np.ndarray],
            dt_hours: float, site_kwh: float) -> list[dict]:
    """Rows for the table: what the workload is, how much it draws, how much can move."""
    rows = []
    for spec in specs:
        trace = traces.get(spec.name)
        kwh = float(trace.sum() * dt_hours) if trace is not None else 0.0
        it_kwh = kwh / max(spec.pue, 1e-9)
        rows.append({
            "Workload": spec.name,
            "What it is": spec.label,
            "IT (kW)": round(spec.kw, 4),
            "PUE": spec.pue,
            "Site (kW)": round(spec.site_kw, 4),
            "Flexible": f"{spec.flexible:.0%}",
            "kWh/day": round(kwh, 3),
            "of site": f"{(kwh / site_kwh):.1%}" if site_kwh > 1e-9 else "—",
            "Deferrable kWh": round(it_kwh * spec.flexible * spec.pue, 3),
            "Detail": spec.detail,
        })
    return rows
