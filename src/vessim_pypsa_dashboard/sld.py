"""Single-line diagram renderer for the dashboard.

The two AdjieBrotot simulators draw the network as a single-line diagram: thick
bus bars, a symbol per device (circle for a generator, panel for PV, rotor for
wind, hatched square for the external grid, arrow for a load, plates for a
shunt, rectangle for a battery), two overlapping circles for a transformer,
converter boxes for an HVDC link, orthogonal routes, flow arrows along each
branch and the results printed next to every element.

This module re-implements that vocabulary in Python and returns plain SVG, so
every tab can show the *same* network the way those tools show it.  No
JavaScript is copied: geometry, symbols and labels are written from scratch and
every number comes from this repository's own solvers.  The visual conventions
are credited to the original tools (GPL-3.0, see ``NOTICE``).

Public API
----------
``layout(net, positions=None)``      place buses and route branches
``flows_from_pf(result)``            results of a Newton-Raphson run
``flows_from_dispatch(...)``         results of an ED/UC run (one hour)
``flows_from_site(...)``             results of a PyPSA site OPF (one hour)
``draw(net, flows, style=...)``      the SVG markup

All coordinates are in SVG pixels.  ``draw`` returns a self-contained
``<svg>`` element with no external fonts, scripts or images.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping

# --------------------------------------------------------------------------- #
# Visual tokens
# --------------------------------------------------------------------------- #

SANS = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

INK = "#dbe7f7"
MUTED = "#7f92b4"
PANEL = "#0d1629"
ACCENT = "#38bdf8"
GOOD = "#34d399"
WARN = "#fbbf24"
BAD = "#f87171"
COOL = "#1f3355"

BAR_W = 5.0
DEV_R = 11.0
STUB = 22.0
LANE_OFS = 4.4
GLYPH_GAP = 44.0
NODE_R = 5.0


def esc(text: object) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _hex(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def mix(a: str, b: str, t: float) -> str:
    """Blend two hex colours (t=0 -> a, t=1 -> b)."""
    t = max(0.0, min(1.0, t))
    x, y = _hex(a), _hex(b)
    return "rgb(%d,%d,%d)" % tuple(round(x[i] + (y[i] - x[i]) * t) for i in range(3))


def r1(value: float) -> float:
    return round(float(value), 1)


# --------------------------------------------------------------------------- #
# Number formatting (a value past a limit never prints as the limit)
# --------------------------------------------------------------------------- #

def fmt_power(value: float, unit: str = "MW") -> str:
    if value is None or not math.isfinite(value):
        return "–"
    if abs(value) < 5e-4:
        value = 0.0
    a = abs(value)
    digits = 1 if a >= 100 else 2 if a >= 1 else 3 if a >= 0.01 else 2 if a == 0 else 4
    return f"{value:,.{digits}f} {unit}".replace("-", "−")


def fmt_pu(value: float, lo: float = 0.95, hi: float = 1.05) -> str:
    if value is None or not math.isfinite(value):
        return "–"
    for digits in range(3, 7):
        text = f"{value:.{digits}f}"
        if not (value < lo - 1e-9 and float(text) >= lo) and not (value > hi + 1e-9 and float(text) <= hi):
            return text
    return f"{value:.3f}"


def fmt_pct(value: float, limit: float = 100.0) -> str:
    if value is None or not math.isfinite(value):
        return "–"
    for digits in range(0, 4):
        text = f"{value:.{digits}f}"
        if not (value > limit + 1e-9 and float(text) <= limit):
            return text
    return f"{value:.1f}"


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

@dataclass
class BusGeom:
    name: str
    cx: float
    cy: float
    half: float
    vertical: bool = False

    @property
    def x1(self) -> float:
        return self.cx if self.vertical else self.cx - self.half

    @property
    def x2(self) -> float:
        return self.cx if self.vertical else self.cx + self.half

    @property
    def y1(self) -> float:
        return self.cy - self.half if self.vertical else self.cy

    @property
    def y2(self) -> float:
        return self.cy + self.half if self.vertical else self.cy

    def point(self, offset: float) -> tuple[float, float]:
        """Point on the bar at ``offset`` from its centre."""
        if self.vertical:
            return self.cx, self.cy + offset
        return self.cx + offset, self.cy


@dataclass
class BranchGeom:
    name: str
    kind: str            # Line | Transformer | Link
    bus0: str
    bus1: str
    pts: list[tuple[float, float]]

    @property
    def mid(self) -> tuple[float, float]:
        return midpoint(self.pts)

    def length(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.pts, self.pts[1:]))


@dataclass
class DeviceGeom:
    name: str
    kind: str            # generator | grid | solar | wind | load | shunt | storage
    bus: str
    x: float
    y: float             # centre of the symbol
    ux: float            # unit vector from the bar to the symbol
    uy: float
    sx: float            # where the stub leaves the bar
    sy: float
    scale: float = 1.0   # symbol size multiplier (p_nom)


@dataclass
class Layout:
    bus: dict[str, BusGeom]
    branch: dict[str, BranchGeom]
    device: dict[str, DeviceGeom]
    width: float
    height: float


def midpoint(pts: list[tuple[float, float]]) -> tuple[float, float]:
    total = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    if total <= 1e-9:
        return pts[0]
    target, walked = total / 2.0, 0.0
    for a, b in zip(pts, pts[1:]):
        seg = math.dist(a, b)
        if walked + seg >= target and seg > 1e-9:
            t = (target - walked) / seg
            return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        walked += seg
    return pts[-1]


def _direction(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    norm = math.hypot(dx, dy) or 1.0
    return dx / norm, dy / norm


# --------------------------------------------------------------------------- #
# Automatic placement
# --------------------------------------------------------------------------- #

def auto_positions(net, dx: float = 250.0, dy: float = 132.0) -> dict[str, tuple[float, float]]:
    """Breadth-first tree from the slack bus: one column per electrical level.

    Deterministic, so the same network always draws the same way; edited
    networks (new buses, deleted lines) still land somewhere sensible.
    """
    buses = list(net.buses.index)
    if not buses:
        return {}

    edges: dict[str, list[str]] = {b: [] for b in buses}
    for frame in (net.lines, net.transformers, net.links):
        for _, row in frame.iterrows():
            a, b = str(row["bus0"]), str(row["bus1"])
            if a in edges and b in edges and a != b:
                edges[a].append(b)
                edges[b].append(a)

    def is_slack(name: str) -> bool:
        return str(net.buses.at[name, "type"]).lower() == "slack"

    roots = [b for b in buses if is_slack(b)] or [buses[0]]
    level: dict[str, int] = {}
    order: list[str] = []
    queue = list(roots)
    for root in roots:
        level[root] = 0
    while queue:
        current = queue.pop(0)
        order.append(current)
        for neighbour in edges[current]:
            if neighbour not in level:
                level[neighbour] = level[current] + 1
                queue.append(neighbour)
    for bus in buses:                      # disconnected islands keep their own tree
        if bus in level:
            continue
        level[bus] = 0
        queue = [bus]
        while queue:
            current = queue.pop(0)
            order.append(current)
            for neighbour in edges[current]:
                if neighbour not in level or level[neighbour] > level[current] + 1:
                    level[neighbour] = level[current] + 1
                    queue.append(neighbour)

    columns: dict[int, list[str]] = {}
    for bus in buses:
        columns.setdefault(level.get(bus, 0), []).append(bus)
    position: dict[str, tuple[float, float]] = {}
    tallest = max(len(v) for v in columns.values())
    for col, members in sorted(columns.items()):
        offset = (tallest - len(members)) / 2.0
        for row, bus in enumerate(members):
            position[bus] = (110.0 + col * dx, 78.0 + (row + offset) * dy)
    return position


# --------------------------------------------------------------------------- #
# Hand layouts for the bundled presets (grid units -> px)
# --------------------------------------------------------------------------- #

#: (x, y) of each bus bar centre, in px.  Presets not listed here are laid out
#: automatically from the slack bus.
POSITIONS: dict[str, dict[str, tuple[float, float]]] = {
    # WSCC 9-bus: two generator rows around a central 230 kV ring.
    "wscc9": {
        "Bus 5": (170, 110), "Bus 4": (170, 300), "Bus 1": (170, 585),
        "Bus 6": (430, 470), "Bus 7": (430, 210), "Bus 2": (430, 42),
        "Bus 9": (740, 110), "Bus 3": (740, 300), "Bus 8": (740, 470),
    },
    "3bus": {"Bus 1": (170, 210), "Bus 2": (520, 100), "Bus 3": (520, 330)},
    "feeder": {"MV 20 kV": (150, 210), "LV board": (400, 210),
               "Pillar 1": (640, 100), "Pillar 2": (640, 210), "Pillar 3": (640, 320)},
    "hvdc": {"Area A": (140, 210), "Converter A": (500, 210),
             "Converter B": (800, 210), "Area B": (1140, 210)},
    "ecopetrol": {"POI 110 kV": (150, 120), "Solar 110 kV": (560, 120),
                  "Plant 110 kV": (560, 360), "Data center 110 kV": (900, 240)},
    # ED/UC presets
    "pjm5": {"A": (200, 130), "B": (520, 90), "C": (820, 90),
             "D": (520, 330), "E": (820, 330)},
    "uc3": {"System": (420, 230)},
    "solar": {"Town": (420, 230)},
    "areas": {"North": (300, 150), "South": (300, 350)},
    # 4-bus site model of the integrated tab
    "site": {"POI": (170, 120), "PV": (170, 350), "Site": (540, 235), "TG": (900, 120)},
}


def _device_side(kind: str) -> int:
    """+1 = above the bar, -1 = below."""
    return -1 if kind in ("load", "shunt") else 1


def _device_kind(net, name: str, row, table: str) -> str:
    if table == "loads":
        return "load"
    if table == "shunts":
        return "shunt"
    if table == "storages":
        return "storage"
    carrier = str(row.get("carrier", "") or "").lower()
    if carrier == "solar":
        return "solar"
    if carrier == "wind":
        return "wind"
    if carrier == "grid":
        return "grid"
    return "generator"


def layout(net, positions: Mapping[str, tuple[float, float]] | None = None,
           spacing: float = 46.0) -> Layout:
    """Place the buses, route the branches and hang the devices on the bars."""
    buses = list(net.buses.index)
    if not buses:
        return Layout({}, {}, {}, 320.0, 160.0)
    position = dict(positions or auto_positions(net))

    # --- ports: how many things attach to each bar (sets its length) --------
    branch_rows = []
    for table in ("lines", "transformers", "links"):
        frame = getattr(net, table)
        for name, row in frame.iterrows():
            branch_rows.append((str(name), table[:-1].capitalize(), str(row["bus0"]), str(row["bus1"])))

    device_rows = []
    for table in ("generators", "loads", "shunts", "storages"):
        frame = getattr(net, table)
        for name, row in frame.iterrows():
            bus = row.get("bus")
            if bus is None or str(bus) not in net.buses.index:
                continue
            device_rows.append((str(name), table, row, str(bus)))

    ends: dict[str, int] = {b: 0 for b in buses}
    for _, _, b0, b1 in branch_rows:
        ends[b0] = ends.get(b0, 0) + 1
        ends[b1] = ends.get(b1, 0) + 1
    upper: dict[str, int] = {b: 0 for b in buses}
    lower: dict[str, int] = {b: 0 for b in buses}
    for name, table, row, bus in device_rows:
        kind = _device_kind(net, name, row, table)
        (upper if _device_side(kind) > 0 else lower)[bus] += 1

    geom: dict[str, BusGeom] = {}
    for bus in buses:
        need = spacing * max(ends[bus], upper[bus] + lower[bus]) + 30.0
        geom[bus] = BusGeom(bus, *position.get(bus, (200.0, 200.0)), max(56.0, need / 2.0))

    # --- route branches: orthogonal Z, leaving the bar on the far side ------
    # Branches that share a bar get their own elbow, otherwise two routes would
    # run on top of each other and read as one line.
    elbow_use: dict[tuple[str, bool], int] = {}
    branch: dict[str, BranchGeom] = {}
    for name, kind, b0, b1 in branch_rows:
        if b0 not in geom or b1 not in geom:
            continue
        g0, g1 = geom[b0], geom[b1]
        if b0 == b1:
            pts = [(g0.x2, g0.y1 + 8), (g0.x2 + 46, g0.y1 + 8),
                   (g0.x2 + 46, g0.y1 - 8), (g0.x2, g0.y1 - 8)]
        else:
            rightward = g1.cx >= g0.cx
            start = (g0.x2, g0.cy) if rightward else (g0.x1, g0.cy)
            end = (g1.x1, g1.cy) if rightward else (g1.x2, g1.cy)
            if abs(g1.cy - g0.cy) < 0.5:
                pts = [start, end]
            else:
                key = (b0, rightward)
                index = elbow_use.get(key, 0)
                elbow_use[key] = index + 1
                # alternate the elbow to either side of the midpoint
                shift = 0.0 if index == 0 else (26.0 * ((index + 1) // 2)) * (1 if index % 2 else -1)
                midx = (start[0] + end[0]) / 2.0 + shift
                pts = [start, (midx, start[1]), (midx, end[1]), end]
        branch[name] = BranchGeom(name, kind, b0, b1, pts)

    # --- distribute attach offsets along every bar -------------------------
    offsets: dict[tuple[str, str, str], float] = {}   # (table, name, end) -> offset
    slots: dict[str, list[tuple[float, str, str, str]]] = {b: [] for b in buses}
    for name, kind, b0, b1 in branch_rows:
        if b0 not in geom or b1 not in geom:
            continue
        if b0 == b1:
            continue
        for end, (me, other) in enumerate(((b0, b1), (b1, b0))):
            want = (geom[other].cx - geom[me].cx) * 0.35
            slots[me].append((want, "branch", name, str(end)))
    for name, table, row, bus in device_rows:
        kind = _device_kind(net, name, row, table)
        side = _device_side(kind)
        want = 0.0
        # devices sit on their own half of the bar, ordered left to right
        slots[bus].append((want + side * 0.01, table, name, ""))

    for bus, items in slots.items():
        items = [it for it in items if it[1] != "branch"] + [it for it in items if it[1] == "branch"]
        g = geom[bus]
        limit = max(g.half - 16.0, 0.0)
        if not items:
            continue
        # devices first, packed around the centre by side; branches on the ends
        devices = [it for it in items if it[1] != "branch"]
        branches = [it for it in items if it[1] == "branch"]
        branches.sort(key=lambda it: it[0])
        used: list[float] = []
        step = min(spacing * 1.25, (2 * limit) / max(len(devices), 1)) if devices else spacing
        start = -step * (len(devices) - 1) / 2.0
        for i, (_, table, name, _end) in enumerate(devices):
            offset = max(-limit, min(limit, start + i * step))
            offsets[(table, name, "")] = offset
            used.append(offset)
        branch_slots = list(_branch_offsets(len(branches), limit))
        taken = list(used)
        for index, (_, table, name, end) in enumerate(branches):
            slot = branch_slots[index]
            # nudge the branch port until it clears every device port on the bar
            for _ in range(40):
                if all(abs(slot - other) >= 18.0 for other in taken):
                    break
                slot += 4.0
                if slot > limit:
                    slot = max(-limit, branch_slots[index] - 4.0 * (index + 1))
            offsets[(table, name, end)] = max(-limit, min(limit, slot))
            taken.append(slot)

    # --- devices ------------------------------------------------------------
    device: dict[str, DeviceGeom] = {}
    for name, table, row, bus in device_rows:
        kind = _device_kind(net, name, row, table)
        g = geom[bus]
        offset = offsets.get((table, name, ""), 0.0)
        sx, sy = g.point(offset)
        side = _device_side(kind)
        ux, uy = (0.0, -float(side)) if not g.vertical else (-float(side), 0.0)
        distance = STUB + DEV_R
        p_nom = float(row.get("p_nom", 0) or 0)
        scale = 1.0
        device[name] = DeviceGeom(name, kind, bus, sx + ux * distance, sy + uy * distance,
                                  ux, uy, sx, sy, scale)

    # --- extent -------------------------------------------------------------
    xs = [g.x1 for g in geom.values()] + [g.x2 for g in geom.values()]
    ys = [g.y1 for g in geom.values()] + [g.y2 for g in geom.values()]
    for b in branch.values():
        xs += [p[0] for p in b.pts]
        ys += [p[1] for p in b.pts]
    for d in device.values():
        xs += [d.x - 26, d.x + 26]
        ys += [d.y - 26, d.y + 26]
    pad = 62.0
    width = (max(xs) - min(xs)) + 2 * pad if xs else 320.0
    height = (max(ys) - min(ys)) + 2 * pad if ys else 160.0
    # shift everything so the drawing starts near the origin
    ox, oy = (min(xs) - pad, min(ys) - pad) if xs else (0.0, 0.0)
    for g in geom.values():
        g.cx -= ox
        g.cy -= oy
    for b in branch.values():
        b.pts = [(p[0] - ox, p[1] - oy) for p in b.pts]
    for d in device.values():
        d.x -= ox
        d.y -= oy
        d.sx -= ox
        d.sy -= oy
    return Layout(geom, branch, device, width, height)


def _branch_offsets(count: int, limit: float) -> Iterable[float]:
    """Evenly spaced offsets from -limit to +limit (``count`` of them)."""
    if count <= 0:
        return []
    if count == 1:
        return [0.0]
    return [-limit + 2 * limit * i / (count - 1) for i in range(count)]


# --------------------------------------------------------------------------- #
# Results -> a uniform "flows" structure
# --------------------------------------------------------------------------- #

@dataclass
class Flows:
    """Numbers to print on the diagram, keyed by element name."""

    bus: dict[str, dict] = field(default_factory=dict)
    branch: dict[str, dict] = field(default_factory=dict)
    device: dict[str, dict] = field(default_factory=dict)
    lam: float | None = None
    label: str = ""


def flows_from_pf(result) -> Flows:
    """From :class:`grid.PfResult` (or its cached dict twin) — NR or DC."""
    buses = result["buses"] if isinstance(result, dict) else result.buses
    branches = result["branches"] if isinstance(result, dict) else result.branches
    flows = Flows()
    for name, row in buses.iterrows():
        flows.bus[str(name)] = {
            "v": float(row.get("v_pu", float("nan"))),
            "angle": float(row.get("angle_deg", float("nan"))),
            "p": float(row.get("p_inj", float("nan"))),
            "q": float(row.get("q_inj", float("nan"))),
        }
    for name, row in branches.iterrows():
        # grid branch tables carry a RangeIndex and the real name in a column
        flows.branch[str(row.get("name", name))] = {
            "p0": float(row.get("p0", 0.0)), "q0": float(row.get("q0", 0.0)),
            "p1": float(row.get("p1", 0.0)), "q1": float(row.get("q1", 0.0)),
            "loading": float(row.get("loading_pct", float("nan"))),
            "loss": float(row.get("p_loss", 0.0)),
        }
    return flows


def flows_from_dispatch(net, p_by_unit: Mapping[str, float], *,
                        costs: Mapping[str, float] | None = None,
                        status: Mapping[str, bool] | None = None,
                        soc: Mapping[str, float] | None = None,
                        lam: float | None = None,
                        load_scale: float = 1.0,
                        label: str = "") -> Flows:
    """From an ED/UC hour: one active power per unit, loads scaled by the user."""
    flows = Flows(lam=lam, label=label)
    load_total = 0.0
    for name, row in net.loads.iterrows():
        p = float(row.get("p_set", 0.0)) * load_scale
        load_total += p
        flows.device[str(name)] = {"p": p, "q": float(row.get("q_set", 0.0))}
    for name, row in net.generators.iterrows():
        key = str(name)
        flows.device[key] = {
            "p": float(p_by_unit.get(key, 0.0)),
            "q": 0.0,
            "cost": float(costs[key]) if costs and key in costs else float(row.get("marginal_cost", 0.0)),
            "on": bool(status[key]) if status and key in status else True,
            "pmax": float(row.get("p_nom", 0.0) or 0.0),
        }
    for name in net.storages.index:
        key = str(name)
        flows.device[key] = {"p": float(p_by_unit.get(key, 0.0)),
                             "soc": float(soc[key]) if soc and key in soc else None}
    # injection per bus, for the bar labels
    for bus in net.buses.index:
        flows.bus[str(bus)] = {"p": 0.0, "q": 0.0, "v": float("nan"), "angle": float("nan")}
    for name, row in net.generators.iterrows():
        key = str(name)
        if key in flows.device and str(row.get("bus")) in flows.bus:
            flows.bus[str(row["bus"])]["p"] += flows.device[key]["p"]
    for name, row in net.loads.iterrows():
        key = str(name)
        if str(row.get("bus")) in flows.bus:
            flows.bus[str(row["bus"])]["p"] -= flows.device[key]["p"]
    return flows


def flows_from_site(dispatch: Mapping[str, float], prices: Mapping[str, float] | None = None,
                    *,
                    battery: float = 0.0, soc: float | None = None,
                    lam: float | None = None, label: str = "") -> Flows:
    """From one row of a PyPSA site dispatch (the four-bus model)."""
    mapping = {"Grid import": "grid", "Solar": "solar", "Gas turbine": "gas", "Battery": "battery"}
    flows = Flows(lam=lam, label=label)
    solar = float(dispatch.get("Solar", 0.0))
    grid = float(dispatch.get("Grid import", 0.0))
    gas = float(dispatch.get("Gas turbine", 0.0))
    load = float(dispatch.get("Load", 0.0))
    flows.device["__grid__"] = {"p": grid}
    flows.device["__solar__"] = {"p": solar}
    flows.device["__gas__"] = {"p": gas}
    flows.device["__battery__"] = {"p": battery, "soc": soc}
    flows.device["__load__"] = {"p": load}
    if prices:
        flows.bus["POI"] = {"price": float(prices.get("POI", float("nan")))}
        flows.bus["Site"] = {"price": float(prices.get("Site", float("nan")))}
        flows.bus["PV"] = {"price": float(prices.get("PV", float("nan")))}
        flows.bus["TG"] = {"price": float(prices.get("TG", float("nan")))}
    return flows


def microgrid_network(*, with_solar: bool = True, with_battery: bool = True,
                      with_gas: bool = False, with_export: bool = False):
    """The two-bus stand-in that mirrors a Vessim ``Microgrid``.

    ``POI`` holds the grid connection, ``Site`` holds the load, the battery and
    the solar plant; the captive gas turbine hangs off ``TG`` when the case has
    one — the same bus name the site network uses.  Only names and topology matter here — every number printed on the
    diagram comes from the Vessim ``Recorder``.
    """
    import pandas as pd

    from .grid import PowerNetwork

    names = ["POI", "Site"] + (["TG"] if with_gas else [])
    net = PowerNetwork()
    net.buses = pd.DataFrame(
        {"v_nom": [110.0] * len(names), "v_set": [1.0] * len(names),
         "type": ["slack"] + ["pq"] * (len(names) - 1), "carrier": ["AC"] * len(names)},
        index=names,
    )
    lines = pd.DataFrame(
        {"bus0": ["POI"], "bus1": ["Site"], "type": [""], "length": [1.0],
         "num_parallel": [1], "r": [0.0], "x": [0.02], "b": [0.0], "g": [0.0],
         "s_nom": [100.0]},
        index=["POI–Site"],
    )
    if with_gas:
        lines.loc["Turbine feeder"] = {"bus0": "TG", "bus1": "Site", "type": "", "length": 1.0,
                                   "num_parallel": 1, "r": 0.0, "x": 0.02, "b": 0.0, "g": 0.0,
                                   "s_nom": 100.0}
    net.lines = lines

    def one_port(kind: str, carrier: str, bus: str, p_nom: float, **extra):
        base = {"bus": bus, "control": "PQ", "p_set": 0.0, "q_set": 0.0, "p_nom": p_nom,
                "carrier": carrier, "marginal_cost": 0.0, "marginal_cost_quadratic": 0.0,
                "committable": False, "p_min_pu": 0.0, "init": 0, "start_up_cost": 0.0,
                "shut_down_cost": 0.0, "min_up_time": 0, "min_down_time": 0,
                "ramp_limit_up": 1.0, "ramp_limit_down": 1.0, "co2_emissions": 0.0}
        base.update(extra)
        return base

    generators = {"Grid connection": one_port("gen", "grid", "POI", 100.0)}
    if with_solar:
        generators["Solar"] = one_port("gen", "solar", "Site", 10.0)
    if with_gas:
        generators["Gas turbine"] = one_port("gen", "gas", "TG", 10.0)
    net.generators = pd.DataFrame(generators).T

    net.loads = pd.DataFrame({"bus": ["Site"], "p_set": [1.0], "q_set": [0.0]}, index=["Load"])

    storages = {}
    if with_battery:
        storages["Battery"] = {"bus": "Site", "p_set": 0.0, "q_set": 0.0, "p_nom": 1.0,
                               "max_hours": 1.0, "efficiency_store": 1.0,
                               "efficiency_dispatch": 1.0, "soc0": 0.5, "marginal_cost": 0.0}
    net.storages = pd.DataFrame(storages).T if storages else net.storages
    return net


def site_network(site_bus_names: tuple[str, ...] = ("POI", "Site", "PV", "TG"),
                 only_lines: tuple[str, ...] | None = None):
    """A tiny stand-in network with the topology of ``build_site_network``.

    Only the diagram needs it: names, buses and the three lines.  Keeping it
    here avoids importing PyPSA into the drawing layer.
    """
    from .grid import PowerNetwork

    net = PowerNetwork()
    import pandas as pd

    buses = pd.DataFrame(
        {"v_nom": [110.0, 110.0, 110.0, 110.0], "v_set": [1.0] * 4,
         "type": ["slack", "pq", "pv", "pv"], "carrier": ["AC"] * 4},
        index=list(site_bus_names),
    )
    net.buses = buses
    lines = pd.DataFrame(
        {"bus0": ["POI", "Site", "Site"], "bus1": ["Site", "PV", "TG"],
         "type": ["", "", ""], "length": [1.0] * 3, "num_parallel": [1] * 3,
         "r": [0.0] * 3, "x": [0.02, 0.05, 0.05], "b": [0.0] * 3, "g": [0.0] * 3,
         "s_nom": [12.0, 2.4, 1.2]},
        index=["POI–Site", "Site–PV collector", "Turbine feeder"],
    )
    if only_lines:
        lines = lines.loc[[name for name in only_lines if name in lines.index]]
    net.lines = lines
    return net


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

@dataclass
class Style:
    show_p: bool = True
    show_q: bool = False
    show_v: bool = True
    show_angle: bool = False
    show_loading: bool = True
    show_loss: bool = False
    show_cost: bool = False
    show_flows: bool = True
    colour_by: str = "voltage"      # voltage | loading | none
    v_min: float = 0.95
    v_max: float = 1.05
    load_max: float = 100.0
    unit: str = "MW"
    title: str = ""
    subtitle: str = ""
    animate: bool = True            # SMIL arrows travelling along each feeder
    anim_speed: float = 54.0        # px/s at full flow


def _v_colour(value: float, style: Style) -> str:
    if value is None or not math.isfinite(value):
        return INK
    if value < style.v_min - 1e-9 or value > style.v_max + 1e-9:
        return BAD
    spread = max(1.0 - style.v_min, style.v_max - 1.0, 1e-6)
    dev = abs(value - 1.0) / spread
    return mix(INK, WARN, max(0.0, min(1.0, (dev - 0.55) / 0.45)))


def _l_colour(loading: float, style: Style) -> str:
    if loading is None or not math.isfinite(loading):
        return INK
    if loading > style.load_max + 1e-9:
        return BAD
    return mix(INK, WARN, max(0.0, min(1.0, (loading / max(style.load_max, 1e-9) - 0.55) / 0.45)))


def _text(x: float, y: float, value: object, *, anchor: str = "start", size: float = 10.5,
          fill: str = INK, mono: bool = False, bold: bool = False, halo: bool = True) -> str:
    """One label.

    The halo is drawn as a separate stroked copy *under* the filled one
    (``paint-order`` is not supported by every SVG renderer, and an unsupported
    ``paint-order`` would paint the stroke over the glyphs).
    """
    family = MONO if mono else SANS
    weight = "700" if bold else ("500" if mono else "400")
    anchor_attr = f' text-anchor="{anchor}"' if anchor != "start" else ""
    common = (f'x="{r1(x)}" y="{r1(y)}" font-family="{family}" font-size="{size}"'
              f' font-weight="{weight}"{anchor_attr}')
    under = (f'<text {common} fill="none" stroke="{PANEL}" stroke-width="3.2" '
             f'stroke-linejoin="round">{esc(value)}</text>') if halo else ""
    return under + f'<text {common} fill="{fill}">{esc(value)}</text>'


def _label_box(x: float, y: float, value: object, size: float, mono: bool,
               anchor: str, bold: bool = True) -> tuple[float, float, float, float]:
    """Approximate rectangle a label will cover (used to avoid overlaps).

    The factors are measured, not guessed: an under-estimate would let a label
    run past the fitted viewBox or sit on top of a bar.
    """
    text = str(value)
    if mono:
        width = len(text) * size * 0.645
    else:
        wide = sum(1 for ch in text if ch.isupper() or ch in "mwMW@%#")
        width = (len(text) * (0.55 if not bold else 0.585) + wide * 0.075) * size
    left = x if anchor == "start" else (x - width / 2 if anchor == "middle" else x - width)
    return left - 1.5, y - size * 0.92, width + 3.0, size * 1.25


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _group(tip: str, *fragments: str) -> str:
    """One element plus its native hover tooltip."""
    return (f"<g><title>{esc(tip)}</title>" + "".join(f for f in fragments if f) + "</g>")


def _path(pts: list[tuple[float, float]]) -> str:
    return "".join(("M" if i == 0 else "L") + f"{r1(p[0])} {r1(p[1])}" for i, p in enumerate(pts))


def _cut_path(pts: list[tuple[float, float]], s0: float, s1: float) -> tuple[list[tuple[float, float]], float]:
    """Sub-path between arc lengths ``s0`` and ``s1`` and its length."""
    if s1 <= s0:
        return [], 0.0
    a = _point_at(pts, s0)
    b = _point_at(pts, s1)
    out = [(a[0], a[1])]
    walked = 0.0
    for p, q in zip(pts, pts[1:]):
        walked += math.dist(p, q)
        if s0 < walked < s1:
            out.append(q)
    out.append((b[0], b[1]))
    length = sum(math.dist(p, q) for p, q in zip(out, out[1:]))
    return out, length


def _on_bars(x: float, y: float, bars) -> bool:
    """True when a point lands on a bus bar (marks there would be hidden)."""
    for rx, ry, rw, rh in bars or ():
        if rx <= x <= rx + rw and ry <= y <= ry + rh:
            return True
    return False


def _along(route, target):
    """Point, unit direction and normal at arc length ``target`` of a polyline."""
    walked = 0.0
    for a, b in zip(route, route[1:]):
        seg = math.dist(a, b)
        if seg <= 1e-9:
            continue
        if walked + seg >= target:
            t = (target - walked) / seg
            x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            ux, uy = (b[0] - a[0]) / seg, (b[1] - a[1]) / seg
            return x, y, ux, uy
        walked += seg
    x, y = route[-1]
    return x, y, 1.0, 0.0


def _flow_animation(pts: list[tuple[float, float]], power: float, colour: str,
                    max_power: float, style: Style) -> str:
    """Arrows that travel along the feeder, the way the reference tools do it.

    Same technique as ``ed-draw.js`` in the AdjieBrotot tool: one ``<path>``
    glyph per arrow, moved by SMIL ``animateMotion`` along the branch polyline
    with ``rotate="auto"``, repeated forever.  The direction follows the sign of
    the active power and the speed follows its magnitude, so a heavily loaded
    line visibly runs faster than a nearly idle one.
    """
    if not style.animate or not math.isfinite(power):
        return ""
    total = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    if total < 42 or abs(power) < 1e-6:
        return ""
    route = pts if power >= 0 else list(reversed(pts))
    share = min(1.0, abs(power) / max(max_power, 1e-9))
    speed = style.anim_speed * (0.35 + 0.65 * share)          # px / s
    duration = max(total / max(speed, 1e-6), 0.35)
    count = max(1, min(4, int(total // 86)))
    path = _path(route)
    out = []
    for k in range(count):
        begin = -duration * k / count
        out.append(
            f'<path class="vflow" d="M6.4 0L-3.2 -3.5L-3.2 3.5Z" fill="{colour}" opacity="0.92">'
            f'<animateMotion path="{path}" rotate="auto" dur="{duration:.2f}s" '
            f'begin="{begin:.2f}s" repeatCount="indefinite"/></path>'
        )
    return "".join(out)


def _arrows(pts: list[tuple[float, float]], power: float, reactive: float,
            colour_p: str, max_power: float, style: Style,
            bars=None) -> str:
    """Flow arrows along a route: filled arrow = P, open chevron = Q."""
    if not style.show_flows or not math.isfinite(power):
        return ""
    total = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    if total < 26 or abs(power) < max(1e-6, 0.004 * max_power):
        return ""
    route = pts if power >= 0 else list(reversed(pts))
    count = max(1, min(4, int(total // 95) + 1))
    marks = []
    for i in range(count):
        target = (i + 0.5) * total / count
        spot = None
        for k in (0, 0.06, -0.06, 0.12, -0.12):          # nudge off a bar if needed
            x, y, ux, uy = _along(route, min(total - 1.0, max(1.0, target + k * total)))
            if not _on_bars(x, y, bars):
                spot = (x, y, ux, uy)
                break
        if spot is None:
            continue
        x, y, ux, uy = spot
        nx, ny = -uy, ux
        angle = math.degrees(math.atan2(uy, ux))
        px, py = x + nx * LANE_OFS, y + ny * LANE_OFS
        marks.append(
            f'<g transform="translate({r1(px)},{r1(py)}) rotate({r1(angle)}) scale(1.55)">'
            f'<path d="M4.2 0L-1.1 -3.3L-1.1 -1L-4.4 -1L-4.4 1L-1.1 1L-1.1 3.3Z" fill="{colour_p}"/></g>'
        )
        if style.show_q and math.isfinite(reactive) and abs(reactive) > 1e-6:
            qx, qy = x - nx * LANE_OFS, y - ny * LANE_OFS
            direction = 1 if reactive >= 0 else -1
            marks.append(
                f'<g transform="translate({r1(qx)},{r1(qy)}) rotate({r1(angle if direction > 0 else angle + 180)}) scale(1.4)">'
                f'<path d="M-1.8 -3L2 0L-1.8 3" fill="none" stroke="{WARN}" '
                f'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></g>'
            )
    marks.append(_flow_animation(route, power, colour_p, max_power, style))
    return "".join(marks)


def _symbol(kind: str, x: float, y: float, ux: float, uy: float, colour: str,
            dim: bool = False) -> str:
    """One device symbol, centred on (x, y) with the stub along (ux, uy)."""
    dash = ' stroke-dasharray="3 3"' if dim else ""
    stroke = f'fill="none" stroke="{colour}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"'
    fill_bg = f'fill="{PANEL}" stroke="{colour}" stroke-width="1.8"'
    vx, vy = -uy, ux                      # across the symbol

    def pt(u: float, v: float) -> str:
        return f"{r1(x + ux * u + vx * v)} {r1(y + uy * u + vy * v)}"

    if kind in ("generator", "wind"):
        if kind == "wind":
            blades = ""
            for deg in (-90, 30, 150):
                a = math.radians(deg)
                c, s = math.cos(a), math.sin(a)
                tip = (x + 8.6 * c, y + 8.6 * s)
                root = (x + 1.9 * c, y + 1.9 * s)
                w = (-s * 1.9, c * 1.9)
                midp = (x + 4.1 * c, y + 4.1 * s)
                blades += (f"M{r1(root[0])} {r1(root[1])}L{r1(midp[0] + w[0])} {r1(midp[1] + w[1])}"
                           f"L{r1(tip[0])} {r1(tip[1])}L{r1(midp[0] - w[0] * .4)} {r1(midp[1] - w[1] * .4)}Z")
            return (f'<circle cx="{r1(x)}" cy="{r1(y)}" r="11" {fill_bg}{dash}/>'
                    f'<path d="{blades}" fill="{colour}" stroke="{colour}" stroke-width="0.8"/>'
                    f'<circle cx="{r1(x)}" cy="{r1(y)}" r="1.7" fill="{PANEL}" stroke="{colour}" stroke-width="1.2"/>')
        return (f'<circle cx="{r1(x)}" cy="{r1(y)}" r="11" {fill_bg}{dash}/>'
                f'<path d="M{r1(x - 6.4)} {r1(y)}q3.2 -6.4 6.4 0t6.4 0" {stroke}{dash}/>')
    if kind == "grid":
        return (f'<rect x="{r1(x - 11)}" y="{r1(y - 11)}" width="22" height="22" rx="1.5" {fill_bg}{dash}/>'
                f'<path d="M{r1(x - 11)} {r1(y + 3)}l8 -8M{r1(x - 11)} {r1(y + 11)}l22 -22'
                f'M{r1(x - 3)} {r1(y + 11)}l14 -14M{r1(x + 5)} {r1(y + 11)}l6 -6" fill="none" '
                f'stroke="{colour}" stroke-width="1.1"{dash}/>')
    if kind == "solar":
        return (f'<rect x="{r1(x - 11)}" y="{r1(y - 11)}" width="22" height="22" rx="1.5" {fill_bg}{dash}/>'
                f'<path d="M{r1(x - 11)} {r1(y - 3.7)}h22M{r1(x - 11)} {r1(y + 3.7)}h22'
                f'M{r1(x)} {r1(y - 11)}v22" fill="none" stroke="{colour}" stroke-width="1.1"{dash}/>')
    # These three glyphs hang *below/above* the bar: they are centred on (x, y)
    # like every other symbol, so the stub always meets the drawing.
    if kind == "load":
        return (f'<path d="M{pt(-9, 0)}L{pt(-4, 0)}" {stroke}{dash}/>'
                f'<path d="M{pt(9, 0)}L{pt(-4, -7.2)}L{pt(-4, 7.2)}Z" fill="{colour}" '
                f'stroke="{colour}" stroke-width="1.2" stroke-linejoin="round"{dash}/>')
    if kind == "shunt":
        return (f'<path d="M{pt(-9, -9)}L{pt(-9, 9)}M{pt(-4, -9)}L{pt(-4, 9)}M{pt(-4, 0)}L{pt(2, 0)}'
                f'M{pt(2, -7)}L{pt(2, 7)}M{pt(5, -4.5)}L{pt(5, 4.5)}M{pt(8, -2)}L{pt(8, 2)}" {stroke}{dash}/>')
    if kind == "storage":
        return (f'<path d="M{pt(-9, -8)}L{pt(7, -8)}L{pt(7, 8)}L{pt(-9, 8)}Z" fill="{PANEL}" '
                f'stroke="{colour}" stroke-width="1.8" stroke-linejoin="round"{dash}/>'
                f'<path d="M{pt(-4, -5)}L{pt(-4, 5)}M{pt(1, -3)}L{pt(1, 3)}" fill="none" '
                f'stroke="{colour}" stroke-width="2.4"{dash}/>')
    return ""


def _rect_of_segment(a: tuple[float, float], b: tuple[float, float], pad: float = 3.0):
    x0, x1 = sorted((a[0], b[0]))
    y0, y1 = sorted((a[1], b[1]))
    return x0 - pad, y0 - pad, (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad


def _label_rect(x: float, y: float, lines: list[str], size: float, mono: bool,
                anchor: str, bold: bool = True) -> tuple[float, float, float, float]:
    """Bounding box of a multi-line label whose first baseline is (x, y)."""
    width = max(_label_box(x, y, line, size, mono, anchor, bold if i == 0 else False)[2]
                for i, line in enumerate(lines))
    first = _label_box(x, y, lines[0], size, mono, anchor)
    height = (len(lines) - 1) * 12.5 + size * 1.3
    left = first[0] if anchor != "end" else first[0]
    if anchor == "middle":
        left = x - width / 2
    elif anchor == "end":
        left = x - width
    return left, y - size * 0.92, width, height


def _overlap(a, b) -> float:
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


@dataclass
class _Request:
    """One block of text waiting for a free spot."""

    lines: list[str]
    candidates: list[tuple[float, float]]
    anchors: list[str]
    fills: list[str]
    size: float
    mono: bool
    priority: int
    bold: bool = False
    ref: tuple[float, float] | None = None


def _place(requests: list[_Request], blockers, placed: list,
           bars: list | None = None) -> list[str]:
    """Greedy placement: the candidate covering the least ink wins.

    Same idea as the original tool: every block offers a few positions and the
    one that covers the fewest labels, symbols and lines is chosen.  Covering a
    bus bar is a *hard* penalty, so a branch label never sits on a bar, and a
    block with nowhere clean to go is dropped rather than printed on top of
    something else.
    """
    bars = bars or []
    out: list[str] = []
    for request in sorted(requests, key=lambda r: (r.priority, len(r.lines))):
        best = None
        for index, (x, y) in enumerate(request.candidates):
            anchor = request.anchors[index]
            rect = _label_rect(x, y, request.lines, request.size, request.mono, anchor)
            on_bar = sum(_overlap(rect, bar) for bar in bars)
            score = index * 8.0
            if request.ref is not None:
                # a label belongs to its own line: drifting away reads as orphaned
                score += 0.55 * math.hypot(x - request.ref[0], y - request.ref[1])
            score += 3.0 * sum(_overlap(rect, other) for other in blockers if other not in bars)
            score += 4.0 * sum(_overlap(rect, other) for other in placed)
            score += 40.0 * on_bar
            if best is None or score < best[0]:
                best = (score, x, y, anchor, rect)
        assert best is not None
        score, x, y, anchor, rect = best
        if score > 2400 and request.priority > 0:      # nowhere clean: stay quiet
            continue
        placed.append(rect)
        for i, line in enumerate(request.lines):
            out.append(_text(x, y + 12.5 * i, line, anchor=anchor,
                             size=request.size if i == 0 else request.size - 0.8,
                             bold=request.bold if i == 0 else False,
                             mono=request.mono if i else False,
                             fill=request.fills[i]))
    return out


def draw(net, flows: Flows | None = None, *, style: Style | None = None,
         positions: Mapping[str, tuple[float, float]] | None = None,
         layout_obj: Layout | None = None) -> str:
    """Render the network as an SVG single-line diagram."""
    style = style or Style()
    lay = layout_obj or layout(net, positions)
    flows = flows or Flows()

    bar_rects = [(b.x1 - 8, b.y1 - 8, (b.x2 - b.x1) + 16, (b.y2 - b.y1) + 16)
                 for b in lay.bus.values()]

    max_p = 1e-9
    max_q = 1e-9
    for payload in list(flows.branch.values()) + list(flows.device.values()):
        max_p = max(max_p, abs(float(payload.get("p0", payload.get("p", 0.0)) or 0.0)),
                    abs(float(payload.get("p1", 0.0) or 0.0)))
        max_q = max(max_q, abs(float(payload.get("q0", payload.get("q", 0.0)) or 0.0)))

    body: list[str] = []
    requests: list[_Request] = []
    blockers: list[tuple[float, float, float, float]] = []
    barriers: list[tuple[float, float, float, float]] = []

    # --- branches ----------------------------------------------------------
    for name, g in lay.branch.items():
        payload = flows.branch.get(name, {})
        loading = payload.get("loading")
        colour = _l_colour(loading, style) if style.colour_by in ("loading", "both") else INK
        weight = 2.0
        if style.colour_by in ("loading", "both") and loading is not None and math.isfinite(loading):
            weight = 1.6 + 2.6 * max(0.0, min(1.0, loading / max(style.load_max, 1.0)))
        frag = []
        if g.kind == "Link":
            total = g.length()
            cut = max(34.0, min(total * 0.28, total / 2.0 - 10.0))
            a = _point_at(g.pts, cut)
            b = _point_at(g.pts, total - cut)
            frag.append(f'<path d="{_path(g.pts)}" fill="none" stroke="{colour}" stroke-width="2" '
                        f'stroke-dasharray="10 3 2 3" stroke-linejoin="round"/>')
            frag.append(_converter(a[0], a[1], a[2], a[3], colour))
            frag.append(_converter(b[0], b[1], -b[2], -b[3], colour))
            blockers.append((a[0] - 12, a[1] - 12, 24, 24))
            blockers.append((b[0] - 12, b[1] - 12, 24, 24))
        else:
            frag.append(f'<path d="{_path(g.pts)}" fill="none" stroke="{colour}" '
                        f'stroke-width="{r1(weight)}" stroke-linejoin="round" stroke-linecap="round"/>')
        for a, b in zip(g.pts, g.pts[1:]):
            blockers.append(_rect_of_segment(a, b, 5.5))

        if g.kind == "Transformer":
            mx, my = g.mid
            ux, uy = _direction(g.pts[-2], g.pts[-1]) if len(g.pts) > 1 else (1.0, 0.0)
            discs = ""
            for k in (-1, 1):
                discs += (f'<circle cx="{r1(mx + k * 5.5 * ux)}" cy="{r1(my + k * 5.5 * uy)}" r="8.5" fill="{PANEL}"/>')
            for k in (-1, 1):
                discs += (f'<circle cx="{r1(mx + k * 5.5 * ux)}" cy="{r1(my + k * 5.5 * uy)}" r="8.5" '
                          f'fill="none" stroke="{colour}" stroke-width="1.8"/>')
            frag.append(discs)
            blockers.append((mx - 17, my - 17, 34, 34))

        p0 = float(payload.get("p0", 0.0) or 0.0)
        q0 = float(payload.get("q0", 0.0) or 0.0)
        tip = (f"{name} ({g.kind}) · {g.bus0} → {g.bus1}\n"
               f"P {p0:,.2f} MW → {-p0:,.2f} MW\n"
               f"Q {q0:,.2f} MVAr"
               + (f"\nloading {loading:,.1f} %" if loading is not None else "")
               + (f"\nloss {float(payload.get('loss', 0.0)):,.3f} MW" if payload.get("loss") else ""))
        frag.append(_arrows(g.pts, p0, q0, colour if colour != INK else ACCENT, max_p, style, bar_rects))
        body.append(_group(tip, *frag))

        # --- text
        lines: list[str] = [name]
        fills: list[str] = [INK]
        if abs(p0) > 1e-9 or abs(float(payload.get("p1", 0.0) or 0.0)) > 1e-9:
            send = p0 if abs(p0) > 1e-9 else float(payload.get("p1", 0.0))
            lines.append(fmt_power(abs(send), style.unit))
        if style.show_q and abs(q0) > 1e-6:
            lines.append(fmt_power(q0, "MVAr"))
        if style.show_loading and loading is not None and math.isfinite(loading):
            flag = " ▲" if loading > style.load_max + 1e-9 else ""
            lines.append(f"{fmt_pct(loading, style.load_max)}%{flag}")
        if style.show_loss and payload.get("loss"):
            lines.append(f"loss {fmt_power(float(payload['loss']), style.unit)}")
        fills += [BAD if extra.endswith("▲") else MUTED for extra in lines[1:]]

        candidates, anchors = [], []
        if g.kind == "Transformer":
            # the transformer symbol sits mid-route; keep its label beside the coils
            mx, my = g.mid
            candidates += [(mx + 20, my - 10), (mx - 20, my - 10),
                           (mx + 20, my + 14), (mx - 20, my + 14)]
            anchors += ["start", "end", "start", "end"]
        for a, b in zip(g.pts, g.pts[1:]):
            ux, uy = _direction(a, b)
            nx, ny = -uy, ux
            for frac in (0.5, 0.3, 0.7):
                px, py = a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac
                for side in (1, -1):
                    for reach in (18.0, 32.0, 48.0):
                        candidates.append((px + nx * side * reach, py + ny * side * reach + 3.5))
                        anchors.append("middle")
        requests.append(_Request(lines, candidates, anchors, fills, 10.0, False, 2, True, g.mid))

    # --- devices -----------------------------------------------------------
    for name, d in lay.device.items():
        payload = flows.device.get(name, {})
        bus_flow = flows.bus.get(d.bus, {})
        dim = payload.get("on") is False
        base = INK if style.colour_by == "none" else _v_colour(bus_flow.get("v", float("nan")), style)
        colour = mix(base, PANEL, 0.55) if dim else base
        frag = []
        if not dim:
            frag.append(f'<path d="M{r1(d.sx)} {r1(d.sy)}L{r1(d.x - d.ux * 11)} {r1(d.y - d.uy * 11)}" '
                        f'fill="none" stroke="{colour}" stroke-width="1.8" stroke-linecap="round"/>')
        frag.append(_symbol(d.kind, d.x, d.y, d.ux, d.uy, colour, dim))
        p = float(payload.get("p", 0.0) or 0.0)
        tip = (f"{name} ({d.kind}) at {d.bus}\nP {p:,.3f} {style.unit}"
               + (f" of {float(payload['pmax']):,.1f}" if payload.get("pmax") else "")
               + (f"\ncost {float(payload['cost']):,.2f} $/MWh" if payload.get("cost") is not None else "")
               + (f"\nSoC {float(payload['soc']):.1%}" if payload.get("soc") is not None else "")
               + ("\nOFFLINE" if payload.get("on") is False else ""))
        blockers.append((d.x - 13, d.y - 13, 26, 26))
        blockers.append(_rect_of_segment((d.sx, d.sy), (d.x, d.y), 3.0))

        p = float(payload.get("p", 0.0) or 0.0)
        if abs(p) > 1e-9 and not dim:
            # +p consumes for a load (arrow away from the bar), +p injects for a
            # generator or a storage unit (arrow towards the bar)
            into_bus = 1.0 if d.kind in ("load", "shunt") else -1.0
            frag.append(_arrows([(d.sx, d.sy), (d.x, d.y)], p * into_bus, 0.0, ACCENT, max_p, style, bar_rects))
        body.append(_group(tip, *frag))

        lines = [name]
        fills = [BAD if dim else INK]
        if style.show_p and "p" in payload:
            lines.append(fmt_power(p, style.unit))
        if style.show_q and payload.get("q"):
            lines.append(fmt_power(float(payload["q"]), "MVAr"))
        if style.show_cost and payload.get("cost") is not None:
            lines.append(f"{float(payload['cost']):,.1f} $/MWh")
        if payload.get("pmax"):
            lines.append(f"of {float(payload['pmax']):,.0f} {style.unit}")
        if payload.get("soc") is not None:
            lines.append(f"SoC {float(payload['soc']):.0%}")
        fills += [MUTED] * (len(lines) - 1)

        far = {"load": 17.0, "shunt": 17.0, "storage": 15.0}.get(d.kind, 14.0)
        # beyond the far end of the symbol, along the stub
        bx, by = d.x + d.ux * (far + 16), d.y + d.uy * (far + 16)
        candidates = [
            (bx, by + (4 if d.uy > 0 else -4)),
            (d.x + 18, d.y + 4), (d.x - 18, d.y + 4),
            (d.x + 18, d.y + 28), (d.x - 18, d.y - 24),
            (d.x, d.y - 24),
        ]
        anchors = ["middle", "start", "end", "start", "end", "middle"]
        if d.kind == "shunt":
            # the plates hang under the bar; the far side of the bar is often free
            candidates = [(d.x + 26, d.y - d.uy * 36), (d.x, d.y - d.uy * 51)] + candidates
            anchors = ["start", "middle"] + anchors
        requests.append(_Request(lines, candidates, anchors, fills, 10.0, False, 1, bold=True))

    # --- bus bars ----------------------------------------------------------
    for name, g in lay.bus.items():
        payload = flows.bus.get(name, {})
        v = payload.get("v", float("nan"))
        colour = _v_colour(v, style) if (style.colour_by in ("voltage", "both")) else INK
        details = [f"{name}: {net.buses.at[name, 'v_nom']:g} kV, {net.buses.at[name, 'type']}"]
        if v is not None and math.isfinite(float(v)):
            details.append(f"V {float(v):.4f} pu")
        if payload.get("price") is not None and math.isfinite(float(payload["price"])):
            details.append(f"LMP {float(payload['price']):,.2f} $/MWh")
        if payload.get("p") is not None and math.isfinite(float(payload["p"])):
            details.append(f"net injection {float(payload['p']):,.2f} {style.unit}")
        body.append(_group("\n".join(details),
                           f'<path d="M{r1(g.x1)} {r1(g.y1)}L{r1(g.x2)} {r1(g.y2)}" '
                           f'stroke="{colour}" stroke-width="{BAR_W}" stroke-linecap="round"/>'))
        blockers.append((g.x1 - 6, g.y1 - 6, (g.x2 - g.x1) + 12, (g.y2 - g.y1) + 12))
        barriers.append((g.x1 - 6, g.y1 - 6, (g.x2 - g.x1) + 12, (g.y2 - g.y1) + 12))

        lines = [name]
        fills = [INK]
        if style.show_v and v is not None and math.isfinite(float(v)):
            flag = " ▲" if float(v) > style.v_max + 1e-9 else (" ▼" if float(v) < style.v_min - 1e-9 else "")
            lines.append(fmt_pu(float(v), style.v_min, style.v_max) + " pu" + flag)
        if payload.get("price") is not None and math.isfinite(float(payload["price"])):
            lines.append(f"{float(payload['price']):,.1f} $/MWh")
        if style.show_p and payload.get("p") is not None and math.isfinite(float(payload["p"])) \
                and abs(float(payload["p"])) > 1e-9:
            lines.append(f"∑P {fmt_power(float(payload['p']), style.unit)}")
        if style.show_q and payload.get("q") is not None and math.isfinite(float(payload["q"])) \
                and abs(float(payload["q"])) > 1e-6:
            lines.append(f"∑Q {fmt_power(float(payload['q']), 'MVAr')}")
        fills += [BAD if extra.endswith(("▲", "▼")) else MUTED for extra in lines[1:]]

        top = g.y1 - 8 - 12.5 * (len(lines) - 1)
        bottom = g.y2 + 17
        if g.vertical:
            candidates = [(g.x2 + 14, g.y1 + 6), (g.x1 - 14, g.y1 + 6),
                          (g.x2 + 14, g.y2 + 14), (g.x1 - 14, g.y2 + 14)]
            anchors = ["start", "end", "start", "end"]
        else:
            # A bar label never sits on its own bar: below, above, then beside
            # one of its ends (so it always reads as belonging to the bar).
            candidates = [(g.x1, bottom), (g.x1, top),
                          (g.x2 + 14, g.cy + 4), (g.x1 - 14, g.cy + 4)]
            anchors = ["start", "start", "start", "end"]
        requests.append(_Request(lines, candidates, anchors, fills, 10.6, False, 0, bold=True))

    placed: list[tuple[float, float, float, float]] = []
    labels = _place(requests, blockers, placed, barriers)

    # The viewBox is fitted to what was actually drawn, labels included, so a
    # long label on the rightmost bar is never clipped.
    xs = [0.0, lay.width] + [r[0] for r in placed] + [r[0] + r[2] for r in placed]
    ys = [0.0, lay.height] + [r[1] for r in placed] + [r[1] + r[3] for r in placed]
    x0, y0 = min(xs) - 22.0, min(ys) - 22.0
    x1, y1 = max(xs) + 22.0, max(ys) + 22.0

    # --- frame -------------------------------------------------------------
    width, height = x1 - x0, y1 - y0
    title = ""
    if style.title:
        title = (_text(x0 + 14, y0 + 22, style.title, size=12.5, bold=True)
                 + (_text(x0 + 14, y0 + 38, style.subtitle, size=10.2, fill=MUTED) if style.subtitle else ""))
    lam = ""
    if flows.lam is not None:
        lam = _text(x1 - 14, y0 + 22, f"λ = {flows.lam:,.2f} $/MWh", anchor="end", size=11.5,
                    mono=True, fill=WARN)
    # The travelling arrows are pure decoration: anyone who asked the operating
    # system for less motion still gets the static arrows underneath.
    motion = ('<style>.vflow{ }@media (prefers-reduced-motion:reduce){.vflow{display:none}}</style>'
              if style.animate else '')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{r1(x0)} {r1(y0)} '
            f'{r1(width)} {r1(height)}" width="100%" role="img" style="display:block;max-width:100%">'
            f'{motion}{title}{lam}{"".join(body)}{"".join(labels)}</svg>')


def _point_at(pts: list[tuple[float, float]], distance: float) -> tuple[float, float, float, float]:
    walked = 0.0
    for a, b in zip(pts, pts[1:]):
        seg = math.dist(a, b)
        if walked + seg >= distance and seg > 1e-9:
            t = (distance - walked) / seg
            ux, uy = (b[0] - a[0]) / seg, (b[1] - a[1]) / seg
            return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, ux, uy
        walked += seg
    a, b = pts[-2], pts[-1]
    ux, uy = _direction(a, b)
    return b[0], b[1], ux, uy


def _converter(x: float, y: float, ix: float, iy: float, colour: str) -> str:
    s = 8.0
    angle = math.degrees(math.atan2(iy, ix))
    return (f'<g transform="translate({r1(x)},{r1(y)}) rotate({r1(angle)})">'
            f'<rect x="-8" y="-8" width="16" height="16" rx="2" fill="{PANEL}" stroke="{colour}" stroke-width="1.8"/>'
            f'<path d="M-8 8L8 -8" stroke="{colour}" stroke-width="1.4"/>'
            f'<path d="M-5 -3q1.5 -2.4 3 0t3 0" fill="none" stroke="{colour}" stroke-width="1.2"/>'
            f'<path d="M1 3h5.5M1 5.5h5.5" stroke="{colour}" stroke-width="1.2"/></g>')
