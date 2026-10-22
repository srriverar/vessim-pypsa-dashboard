"""Electric-network model and the educational AC load flow.

This module is the Python port of the *Load Flow Simulator* by Adjie Brotot
(https://tool.adjiebrotots.com/loadflow-simulator/, source in
github.com/adjiebrotot/tool).  The original tool keeps its model as a list of
PyPSA elements (`{c: class, id, ...PyPSA attributes in PyPSA units}`) and hands
it to PSA.js, a JavaScript port of PyPSA, for solving.  Here the same model is
kept as one table per PyPSA class, and the Newton-Raphson solver is written out
in NumPy so that every step can be inspected in the dashboard.

Conventions (identical to PyPSA's own power flow, so that results line up):

* bus voltages are per-unit on ``v_nom`` (kV);
* line impedances are in ohm and are converted with ``z_pu = z / v_nom**2``
  (PyPSA works on a 1 MVA system base);
* transformer impedances are p.u. on the transformer's own ``s_nom`` and are
  converted with ``z_pu = z / s_nom``;
* a transformer is stamped as an ideal tap/phase-shift transformer with the
  series admittance split between the two sides, and a T-model (``model='t'``)
  with a magnetising branch is converted to its equivalent PI first;
* links are not part of the AC Y-bus: they inject ``p_set`` at ``bus0`` and
  ``-p_set * efficiency`` at ``bus1``, exactly like PyPSA's power flow.

Sign convention: generator injections are positive, load withdrawals are
negative.  ``p_set`` is in MW, ``q_set`` in MVAr, impedances in ohm (lines) or
p.u. (transformers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Literal

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Table schemas.  One row per element, one table per PyPSA class, so the
# dashboard can show editable "sheets" like the Adjie Brotot tool does.
# --------------------------------------------------------------------------- #

BUS_COLS = {
    "v_nom": 110.0,          # kV
    "v_set": 1.0,            # p.u. voltage setpoint (PV / Slack buses)
    "type": "PQ",            # PQ | PV | Slack
    "carrier": "AC",
}

LINE_COLS = {
    "bus0": "",
    "bus1": "",
    "type": "",              # standard type name (optional)
    "length": 0.0,           # km
    "num_parallel": 1,
    "r": 0.0,                # ohm, total
    "x": 0.0,                # ohm, total
    "b": 0.0,                # S, total shunt susceptance
    "g": 0.0,
    "s_nom": 0.0,            # MVA rating (0 = no limit)
}

TRAFO_COLS = {
    "bus0": "",
    "bus1": "",
    "type": "",
    "s_nom": 100.0,          # MVA
    "r": 0.0,                # p.u. on own s_nom
    "x": 0.1,                # p.u. on own s_nom
    "g": 0.0,
    "b": 0.0,
    "tap_ratio": 1.0,
    "tap_side": 0,
    "phase_shift": 0.0,      # degrees
    "model": "t",            # t | pi
    "num_parallel": 1,
    "tap_position": 0,
    "tap_neutral": 0,
    "tap_step": 0.0,         # % per tap
    "length": 0.0,
}

GEN_COLS = {
    "bus": "",
    "control": "PQ",         # PQ | PV | Slack (kept for round-tripping)
    "p_set": 0.0,            # MW
    "q_set": 0.0,            # MVAr
    "p_nom": 0.0,            # MW
    "carrier": "gas",
    "marginal_cost": 0.0,    # USD/MWh
    "marginal_cost_quadratic": 0.0,  # USD/MWh^2 (PyPSA convention)
    "committable": False,
    "p_min_pu": 0.0,
    "init": 1,               # running before the first hour (1) or not (0)
    "start_up_cost": 0.0,
    "shut_down_cost": 0.0,
    "min_up_time": 0,
    "min_down_time": 0,
    "ramp_limit_up": 1.0,
    "ramp_limit_down": 1.0,
    "co2_emissions": 0.0,    # tCO2/MWh
}

LOAD_COLS = {
    "bus": "",
    "p_set": 0.0,            # MW
    "q_set": 0.0,            # MVAr
}

SHUNT_COLS = {
    "bus": "",
    "g": 0.0,                # S at v_nom
    "b": 0.0,                # S at v_nom (capacitive positive, as PyPSA)
}

STORAGE_COLS = {
    "bus": "",
    "p_set": 0.0,
    "q_set": 0.0,
    "p_nom": 0.0,
    "max_hours": 4.0,
    "efficiency_store": 0.95,
    "efficiency_dispatch": 0.95,
    "soc0": 0.5,
    "marginal_cost": 0.0,
}

LINK_COLS = {
    "bus0": "",
    "bus1": "",
    "p_set": 0.0,
    "p_nom": 0.0,
    "efficiency": 1.0,
}

TABLE_SPECS = {
    "buses": BUS_COLS,
    "lines": LINE_COLS,
    "transformers": TRAFO_COLS,
    "generators": GEN_COLS,
    "loads": LOAD_COLS,
    "shunts": SHUNT_COLS,
    "storages": STORAGE_COLS,
    "links": LINK_COLS,
}

TABLE_LABELS = {
    "buses": "🟦 Buses",
    "lines": "➖ Lines",
    "transformers": "🔁 Transformers",
    "generators": "⚡ Generators",
    "loads": "🏭 Loads",
    "shunts": "🧲 Shunts",
    "storages": "🔋 Storage",
    "links": "🔗 Links",
}



def norm_bus_type(value) -> str:
    """Normalise a bus type string ('PV', 'Pv', 'pv') to 'PQ' | 'PV' | 'Slack'."""
    text = str(value).strip().lower()
    if text.startswith("slack"):
        return "Slack"
    if text.startswith("pv") or text == "p":
        return "PV"
    return "PQ"


def empty_table(kind: str) -> pd.DataFrame:
    """Return an empty DataFrame with the canonical schema of ``kind``."""
    spec = TABLE_SPECS[kind]
    df = pd.DataFrame({col: pd.Series(dtype=_dtype(val)) for col, val in spec.items()})
    df.index.name = "name"
    return df


def _dtype(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int64"
    if isinstance(value, float):
        return "float64"
    return "object"


def _coerce_table(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Coerce a user-edited table back to the canonical schema."""
    spec = TABLE_SPECS[kind]
    out = pd.DataFrame(index=pd.Index(df.index.astype(str), name="name"))
    for col, default in spec.items():
        if col in df.columns:
            series = df[col]
        else:
            series = pd.Series([default] * len(df), index=df.index)
        if isinstance(default, bool):
            out[col] = series.astype("boolean").fillna(False).astype(bool) if len(series) else series.astype(bool)
        elif isinstance(default, int):
            out[col] = pd.to_numeric(series, errors="coerce").fillna(default).astype(int)
        elif isinstance(default, float):
            out[col] = pd.to_numeric(series, errors="coerce").fillna(default).astype(float)
        else:
            out[col] = series.astype(str).fillna(str(default)).replace("nan", str(default))
    return out


# --------------------------------------------------------------------------- #
# Standard types.  Read from the PyPSA installation when available (the same
# tables PSA.js bundles), with a small fallback so the four Quick Starts work
# even without PyPSA.
# --------------------------------------------------------------------------- #

FALLBACK_LINE_TYPES = {
    # name: (r_per_length ohm/km, x_per_length ohm/km, c_per_length nF/km, i_nom kA)
    "243-AL1/39-ST1A 110.0": (0.1188, 0.390, 9.0, 0.645),
    "NAYY 4x150 SE": (0.2080, 0.080, 261.0, 0.270),
    "Al/St 240/40 2-bundle 220.0": (0.0600, 0.301, 12.5, 1.290),
}

FALLBACK_TRANSFORMER_TYPES = {
    # name: dict of the PyPSA standard type columns
    "0.63 MVA 20/0.4 kV": dict(
        f_nom=50.0, s_nom=0.63, v_nom_0=20.0, v_nom_1=0.4, vsc=6.0, vscr=1.206,
        pfe=1.65, i0=0.2619, phase_shift=150.0, tap_side=0, tap_neutral=0,
        tap_min=-2, tap_max=2, tap_step=2.5,
    ),
}


def _load_standard_types() -> tuple[Dict[str, dict], Dict[str, dict]]:
    try:  # pragma: no cover - depends on the installed PyPSA
        import os

        import pypsa

        base = os.path.join(os.path.dirname(pypsa.__file__), "data", "standard_types")
        lines = pd.read_csv(os.path.join(base, "line_types.csv"))
        trafos = pd.read_csv(os.path.join(base, "transformer_types.csv"))
        line_types = {
            row["name"]: dict(
                f_nom=float(row["f_nom"]),
                r_per_length=float(row["r_per_length"]),
                x_per_length=float(row["x_per_length"]),
                c_per_length=float(row["c_per_length"]),
                i_nom=float(row["i_nom"]),
            )
            for _, row in lines.iterrows()
        }
        trafo_types = {
            row["name"]: {k: row[k] for k in trafos.columns}
            for _, row in trafos.iterrows()
        }
        return line_types, trafo_types
    except Exception:  # pragma: no cover - fallback path
        return (
            {
                k: dict(f_nom=50.0, r_per_length=v[0], x_per_length=v[1],
                        c_per_length=v[2], i_nom=v[3])
                for k, v in FALLBACK_LINE_TYPES.items()
            },
            {k: dict(v) for k, v in FALLBACK_TRANSFORMER_TYPES.items()},
        )


LINE_TYPES, TRANSFORMER_TYPES = _load_standard_types()


# --------------------------------------------------------------------------- #
# Network container
# --------------------------------------------------------------------------- #

@dataclass
class PowerNetwork:
    """A PyPSA-shaped network kept as one DataFrame per component class."""

    name: str = "network"
    buses: pd.DataFrame = field(default_factory=lambda: empty_table("buses"))
    lines: pd.DataFrame = field(default_factory=lambda: empty_table("lines"))
    transformers: pd.DataFrame = field(default_factory=lambda: empty_table("transformers"))
    generators: pd.DataFrame = field(default_factory=lambda: empty_table("generators"))
    loads: pd.DataFrame = field(default_factory=lambda: empty_table("loads"))
    shunts: pd.DataFrame = field(default_factory=lambda: empty_table("shunts"))
    storages: pd.DataFrame = field(default_factory=lambda: empty_table("storages"))
    links: pd.DataFrame = field(default_factory=lambda: empty_table("links"))
    source: str = ""  # provenance, e.g. "Adjie Brotot preset: WSCC 9-bus"

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_rows(cls, name: str, rows: Dict[str, Iterable[dict]], source: str = "") -> "PowerNetwork":
        net = cls(name=name, source=source)
        for kind, records in rows.items():
            records = list(records)
            if not records:
                continue
            df = pd.DataFrame(records)
            name_col = None
            for candidate in ("name", "id"):
                if candidate in df.columns:
                    name_col = candidate
                    break
            if name_col:
                df = df.set_index(name_col)
                if name_col == "id":
                    df.index = [f"{r.get('c', '')} {i}".strip() for i, r in zip(df.index, records)]
            setattr(net, kind, _coerce_table(df, kind))
        return net

    def copy(self) -> "PowerNetwork":
        return PowerNetwork(
            name=self.name,
            buses=self.buses.copy(),
            lines=self.lines.copy(),
            transformers=self.transformers.copy(),
            generators=self.generators.copy(),
            loads=self.loads.copy(),
            shunts=self.shunts.copy(),
            storages=self.storages.copy(),
            links=self.links.copy(),
            source=self.source,
        )

    # -- validation -------------------------------------------------------- #
    def known_buses(self) -> set[str]:
        return set(self.buses.index)

    def validate(self) -> list[str]:
        """Return a list of human-readable problems (empty means valid)."""
        problems: list[str] = []
        known = self.known_buses()
        if not known:
            return ["The network has no buses."]
        for kind, cols in (("lines", ("bus0", "bus1")), ("transformers", ("bus0", "bus1")),
                           ("links", ("bus0", "bus1"))):
            table = getattr(self, kind)
            for name, row in table.iterrows():
                for col in cols:
                    if row[col] not in known:
                        problems.append(f"{kind[:-1].title()} '{name}': bus '{row[col]}' does not exist.")
                if row["bus0"] == row["bus1"]:
                    problems.append(f"{kind[:-1].title()} '{name}' connects a bus to itself.")
        for kind in ("generators", "loads", "shunts", "storages"):
            table = getattr(self, kind)
            for name, row in table.iterrows():
                if row["bus"] not in known:
                    problems.append(f"{kind[:-1].title()} '{name}': bus '{row['bus']}' does not exist.")
        types = self.buses["type"].map(norm_bus_type)
        if not (types == "Slack").any():
            problems.append(
                "No slack bus: mark at least one bus as Slack (or add a generator with control='Slack')."
            )
        return problems

    # -- topology ---------------------------------------------------------- #
    def islands(self) -> list[list[str]]:
        """Connected components of the AC graph (lines and transformers)."""
        adjacency: dict[str, set[str]] = {b: set() for b in self.buses.index}
        for table in (self.lines, self.transformers):
            for _, row in table.iterrows():
                a, b = row["bus0"], row["bus1"]
                if a in adjacency and b in adjacency:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
        seen: set[str] = set()
        out: list[list[str]] = []
        for bus in self.buses.index:
            if bus in seen:
                continue
            stack, comp = [bus], []
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                comp.append(cur)
                stack.extend(adjacency[cur] - seen)
            out.append(comp)
        return out

    # -- per-unit conversion ---------------------------------------------- #
    def resolve_types(self) -> "PowerNetwork":
        """Return a copy with line/transformer parameters filled from standard types."""
        net = self.copy()
        lines = net.lines.copy()
        for name, row in lines.iterrows():
            tname = str(row.get("type", "") or "")
            if not tname:
                continue
            t = LINE_TYPES.get(tname)
            if t is None:
                raise ValueError(f"Unknown line type '{tname}' on line '{name}'.")
            n = max(float(row.get("num_parallel", 1) or 1), 1e-9)
            length = float(row.get("length", 0.0) or 0.0)
            lines.at[name, "r"] = t["r_per_length"] * length / n
            lines.at[name, "x"] = t["x_per_length"] * length / n
            lines.at[name, "b"] = 2 * np.pi * 1e-9 * t["f_nom"] * t["c_per_length"] * length * n
        net.lines = lines

        trafos = net.transformers.copy()
        for name, row in trafos.iterrows():
            tname = str(row.get("type", "") or "")
            if not tname:
                continue
            t = TRANSFORMER_TYPES.get(tname)
            if t is None:
                raise ValueError(f"Unknown transformer type '{tname}' on transformer '{name}'.")
            n = max(float(row.get("num_parallel", 1) or 1), 1e-9)
            s_nom = float(t["s_nom"])
            r = float(t["vscr"]) / 100.0
            x = float(np.sqrt(max((float(t["vsc"]) / 100.0) ** 2 - r**2, 0.0)))
            g = float(t["pfe"]) / (1000.0 * s_nom)
            b = -float(np.sqrt(max((float(t["i0"]) / 100.0) ** 2 - g**2, 0.0)))
            tap_pos = float(row.get("tap_position", 0) or 0)
            neutral = float(t.get("tap_neutral", 0) or 0)
            step = float(t.get("tap_step", 0) or 0)
            tap = 1.0 + (tap_pos - neutral) * step / 100.0
            trafos.at[name, "s_nom"] = s_nom
            trafos.at[name, "r"] = r / n
            trafos.at[name, "x"] = x / n
            trafos.at[name, "g"] = g * n
            trafos.at[name, "b"] = b * n
            trafos.at[name, "tap_ratio"] = tap
            trafos.at[name, "tap_side"] = int(t.get("tap_side", 0) or 0)
            trafos.at[name, "phase_shift"] = float(t.get("phase_shift", 0.0) or 0.0)
            trafos.at[name, "model"] = "t"
        net.transformers = trafos
        return net

    def branch_frame(self) -> pd.DataFrame:
        """Uniform branch table in system per-unit (1 MVA base)."""
        rows = []
        for name, r in self.lines.iterrows():
            v_nom = float(self.buses.at[r["bus0"], "v_nom"])
            b_pu = float(r.get("b", 0.0)) * v_nom**2
            g_pu = float(r.get("g", 0.0)) * v_nom**2
            rows.append(dict(
                name=name, kind="Line", bus0=r["bus0"], bus1=r["bus1"],
                r_pu=float(r["r"]) / v_nom**2, x_pu=float(r["x"]) / v_nom**2,
                g_pu=g_pu, b_pu=b_pu, tau_hv=1.0, tau_lv=1.0, phase_shift=0.0,
                s_nom=float(r.get("s_nom", 0.0) or 0.0),
            ))
        for name, r in self.transformers.iterrows():
            s_nom = max(float(r.get("s_nom", 1.0) or 1.0), 1e-9)
            r_pu = float(r["r"]) / s_nom
            x_pu = float(r["x"]) / s_nom
            g_pu = float(r.get("g", 0.0)) * s_nom
            b_pu = float(r.get("b", 0.0)) * s_nom
            if str(r.get("model", "t")) == "t" and (abs(g_pu) > 0 or abs(b_pu) > 0):
                r_pu, x_pu, g_pu, b_pu = _t_to_pi(r_pu, x_pu, g_pu, b_pu)
            tau = float(r.get("tap_ratio", 1.0) or 1.0) or 1.0
            side = int(r.get("tap_side", 0) or 0)
            rows.append(dict(
                name=name, kind="Transformer", bus0=r["bus0"], bus1=r["bus1"],
                r_pu=r_pu, x_pu=x_pu, g_pu=g_pu, b_pu=b_pu,
                tau_hv=tau if side == 0 else 1.0,
                tau_lv=tau if side == 1 else 1.0,
                phase_shift=float(r.get("phase_shift", 0.0) or 0.0),
                s_nom=s_nom,
            ))
        return pd.DataFrame(rows)


def _t_to_pi(r_pu: float, x_pu: float, g_pu: float, b_pu: float):
    """Convert a T-model transformer to its equivalent PI (PyPSA's wye-delta)."""
    z_series = complex(r_pu, x_pu)
    y_shunt = complex(g_pu, b_pu)
    if y_shunt == 0:
        return r_pu, x_pu, g_pu, b_pu
    z1, z2, z3 = z_series / 2, z_series / 2, 1 / y_shunt
    total = z1 * z2 + z2 * z3 + z3 * z1
    za = total / z2
    zc = total / z3
    return zc.real, zc.imag, (2 / za).real, (2 / za).imag


# --------------------------------------------------------------------------- #
# Y-bus and Newton-Raphson
# --------------------------------------------------------------------------- #

@dataclass
class PfResult:
    buses: pd.DataFrame
    branches: pd.DataFrame
    ybus: np.ndarray
    history: pd.DataFrame
    converged: bool
    iterations: int
    max_mismatch: float
    bus_order: list[str]
    summary: dict = field(default_factory=dict)


def build_ybus(net: PowerNetwork) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Assemble the complex bus admittance matrix and return branch data."""
    net = net.resolve_types()
    buses = list(net.buses.index)
    index = {b: i for i, b in enumerate(buses)}
    n = len(buses)
    Y = np.zeros((n, n), dtype=complex)
    branches = net.branch_frame()

    for _, br in branches.iterrows():
        i, j = index[br["bus0"]], index[br["bus1"]]
        y_se = 1 / complex(br["r_pu"], br["x_pu"])
        y_sh = complex(br["g_pu"], br["b_pu"])
        tau_hv, tau_lv = br["tau_hv"], br["tau_lv"]
        shift = np.exp(1j * np.deg2rad(br["phase_shift"]))
        Y[i, i] += (y_se + 0.5 * y_sh) / tau_hv**2
        Y[j, j] += (y_se + 0.5 * y_sh) / tau_lv**2
        Y[i, j] += -y_se / (tau_hv * tau_lv * np.conj(shift))
        Y[j, i] += -y_se / (tau_hv * tau_lv * shift)

    for _, sh in net.shunts.iterrows():
        i = index[sh["bus"]]
        v_nom = float(net.buses.at[sh["bus"], "v_nom"])
        Y[i, i] += complex(float(sh["g"]), float(sh["b"])) * v_nom**2

    return Y, branches, buses


def _injections(net: PowerNetwork) -> tuple[pd.Series, pd.Series]:
    """Specified P and Q injection per bus in MW / MVAr (links included)."""
    p = pd.Series(0.0, index=net.buses.index)
    q = pd.Series(0.0, index=net.buses.index)
    for _, row in net.generators.iterrows():
        p[row["bus"]] += float(row["p_set"])
        q[row["bus"]] += float(row["q_set"])
    for _, row in net.storages.iterrows():
        p[row["bus"]] += float(row["p_set"])
        q[row["bus"]] += float(row["q_set"])
    for _, row in net.links.iterrows():
        # p_set is measured at bus0 and flowing into the link (PyPSA's convention)
        p[row["bus0"]] -= float(row["p_set"])
        p[row["bus1"]] += float(row["p_set"]) * float(row["efficiency"])
    for _, row in net.loads.iterrows():
        p[row["bus"]] -= float(row["p_set"])
        q[row["bus"]] -= float(row["q_set"])
    return p, q


def run_pf(
    net: PowerNetwork,
    tol: float = 1e-8,
    max_iter: int = 30,
    start: Literal["flat", "dc"] = "flat",
) -> PfResult:
    """Polar Newton-Raphson AC power flow with an analytic Jacobian."""
    problems = net.validate()
    if problems:
        raise ValueError("\n".join(problems))

    Y, branches, buses = build_ybus(net)
    n = len(buses)
    index = {b: i for i, b in enumerate(buses)}
    types = net.buses["type"].map(norm_bus_type).to_numpy()
    slack = np.array([t == "Slack" for t in types])
    pv = np.array([t == "PV" for t in types])
    pq = np.array([t == "PQ" for t in types])

    for comp in net.islands():
        if not any(slack[index[b]] for b in comp):
            raise ValueError(
                "Island without a slack bus: " + ", ".join(comp)
                + ". Mark one bus of every island as Slack."
            )

    p_spec, q_spec = _injections(net)
    p_spec = p_spec.to_numpy()
    q_spec = q_spec.to_numpy()

    V = np.array([float(net.buses.at[b, "v_set"]) if types[i] in ("PV", "Slack") else 1.0
                  for i, b in enumerate(buses)])
    theta = np.zeros(n)

    if start == "dc":
        theta = _dc_seed(net, buses)

    G, B = Y.real, Y.imag
    non_slack = ~slack
    history = []
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        Vc = V * np.exp(1j * theta)
        S = Vc * np.conj(Y @ Vc)
        p_calc, q_calc = S.real, S.imag
        mismatch = np.concatenate([
            p_spec[non_slack] - p_calc[non_slack],
            q_spec[pq] - q_calc[pq],
        ])
        norm = float(np.max(np.abs(mismatch))) if len(mismatch) else 0.0
        history.append({"iteration": it - 1, "max_mismatch": norm})
        if norm < tol:
            converged = True
            break
        J = _jacobian(G, B, V, theta, p_calc, q_calc, non_slack, pq)
        try:
            dx = np.linalg.solve(J, mismatch)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - pathological networks
            raise ValueError(f"Singular Jacobian: {exc}") from exc
        n_theta = int(non_slack.sum())
        theta[non_slack] += dx[:n_theta]
        V[pq] += dx[n_theta:]

    Vc = V * np.exp(1j * theta)
    S = Vc * np.conj(Y @ Vc)

    bus_table = net.buses.copy()
    bus_table["v_pu"] = V
    bus_table["angle_deg"] = np.degrees(theta)
    bus_table["v_kv"] = V * net.buses["v_nom"].to_numpy()
    bus_table["p_inj"] = S.real
    bus_table["q_inj"] = S.imag

    # Bus-level generation: the difference between what flows into the bus and
    # what was specified there (the slack picks up losses, PV buses pick up Q).
    gen_p_spec = pd.Series(0.0, index=buses)
    gen_q_spec = pd.Series(0.0, index=buses)
    for _, row in net.generators.iterrows():
        gen_p_spec[row["bus"]] += float(row["p_set"])
        gen_q_spec[row["bus"]] += float(row["q_set"])

    bus_table = net.buses.copy()
    bus_table["v_pu"] = V
    bus_table["angle_deg"] = np.degrees(theta)
    bus_table["v_kv"] = V * net.buses["v_nom"].to_numpy()
    bus_table["p_inj"] = S.real
    bus_table["q_inj"] = S.imag

    load_p = pd.Series(0.0, index=buses)
    load_q = pd.Series(0.0, index=buses)
    for _, row in net.loads.iterrows():
        load_p[row["bus"]] += float(row["p_set"])
        load_q[row["bus"]] += float(row["q_set"])

    p_gen = S.real - (p_spec - gen_p_spec.to_numpy())
    q_gen = S.imag - (q_spec - gen_q_spec.to_numpy())
    bus_table["p_gen"] = p_gen
    bus_table["q_gen"] = q_gen
    bus_table["p_load"] = load_p.to_numpy()
    bus_table["q_load"] = load_q.to_numpy()

    branch_table = _branch_flows(net, branches, Vc, buses, index)
    summary = {
        "losses_mw": float(branch_table["p_loss"].sum()),
        "losses_mvar": float(branch_table["q_loss"].sum()),
        "generation_mw": float(bus_table["p_gen"].sum()),
        "load_mw": float(bus_table["p_load"].sum()),
        "max_loading_pct": float(branch_table["loading_pct"].max()) if len(branch_table) else 0.0,
        "min_v_pu": float(V.min()),
        "max_v_pu": float(V.max()),
        "iterations": it,
        "converged": converged,
        "max_mismatch": float(history[-1]["max_mismatch"]),
    }
    summary["losses_pct"] = (
        100 * summary["losses_mw"] / summary["generation_mw"] if summary["generation_mw"] else 0.0
    )
    return PfResult(
        buses=bus_table,
        branches=branch_table,
        ybus=Y,
        history=pd.DataFrame(history),
        converged=converged,
        iterations=it,
        max_mismatch=summary["max_mismatch"],
        bus_order=buses,
        summary=summary,
    )


def _dc_seed(net: PowerNetwork, buses: list[str]) -> np.ndarray:
    """Angles from a DC power flow, used as a Newton-Raphson starting point."""
    try:
        dc = run_dc_pf(net)
        return np.deg2rad(dc.buses["angle_deg"].reindex(buses).to_numpy())
    except Exception:
        return np.zeros(len(buses))


def _jacobian(G, B, V, theta, p_calc, q_calc, non_slack, pq):
    """Analytic polar Jacobian of the power-flow equations."""
    n = len(V)
    idx_ns = np.where(non_slack)[0]
    idx_pq = np.where(pq)[0]
    dth = theta[:, None] - theta[None, :]
    cos_d, sin_d = np.cos(dth), np.sin(dth)
    VV = np.outer(V, V)
    Vrow = np.broadcast_to(V[:, None], (n, n))

    dP_dth = VV * (G * sin_d - B * cos_d)
    dP_dV = Vrow * (G * cos_d + B * sin_d)
    dQ_dth = -VV * (G * cos_d + B * sin_d)
    dQ_dV = Vrow * (G * sin_d - B * cos_d)

    k = np.arange(n)
    dP_dth[k, k] = -q_calc - B.diagonal() * V**2
    dP_dV[k, k] = p_calc / V + G.diagonal() * V
    dQ_dth[k, k] = p_calc - G.diagonal() * V**2
    dQ_dV[k, k] = q_calc / V - B.diagonal() * V

    return np.block([
        [dP_dth[np.ix_(idx_ns, idx_ns)], dP_dV[np.ix_(idx_ns, idx_pq)]],
        [dQ_dth[np.ix_(idx_pq, idx_ns)], dQ_dV[np.ix_(idx_pq, idx_pq)]],
    ])


def _branch_flows(net: PowerNetwork, branches: pd.DataFrame, Vc: np.ndarray,
                  buses: list[str], index: dict[str, int]) -> pd.DataFrame:
    rows = []
    for _, br in branches.iterrows():
        i, j = index[br["bus0"]], index[br["bus1"]]
        y_se = 1 / complex(br["r_pu"], br["x_pu"])
        y_sh = complex(br["g_pu"], br["b_pu"])
        tau_hv, tau_lv = br["tau_hv"], br["tau_lv"]
        shift = np.exp(1j * np.deg2rad(br["phase_shift"]))
        Y00 = (y_se + 0.5 * y_sh) / tau_hv**2
        Y01 = -y_se / (tau_hv * tau_lv * np.conj(shift))
        Y10 = -y_se / (tau_hv * tau_lv * shift)
        Y11 = (y_se + 0.5 * y_sh) / tau_lv**2
        Vi, Vj = Vc[i], Vc[j]
        I0 = Y00 * Vi + Y01 * Vj
        I1 = Y10 * Vi + Y11 * Vj
        S0 = Vi * np.conj(I0)
        S1 = Vj * np.conj(I1)
        s_nom = float(br["s_nom"])
        s_max = max(abs(S0), abs(S1))
        rows.append(dict(
            name=br["name"], kind=br["kind"], bus0=br["bus0"], bus1=br["bus1"],
            p0=S0.real, q0=S0.imag, p1=S1.real, q1=S1.imag,
            p_loss=S0.real + S1.real, q_loss=S0.imag + S1.imag,
            loading_pct=(100 * s_max / s_nom) if s_nom > 0 else np.nan,
            s_nom=s_nom,
        ))
    # links: fixed injections, no Y-bus coupling
    for name, row in net.links.iterrows():
        p0 = float(row["p_set"])
        p1 = -p0 * float(row["efficiency"])
        rows.append(dict(
            name=name, kind="Link", bus0=row["bus0"], bus1=row["bus1"],
            p0=p0, q0=0.0, p1=p1, q1=0.0, p_loss=p0 + p1, q_loss=0.0,
            loading_pct=(100 * abs(p0) / float(row["p_nom"])) if float(row["p_nom"]) > 0 else np.nan,
            s_nom=float(row["p_nom"]),
        ))
    return pd.DataFrame(rows)


def run_dc_pf(net: PowerNetwork) -> PfResult:
    """Linear DC power flow on the same model (lossless, flat voltages)."""
    problems = net.validate()
    if problems:
        raise ValueError("\n".join(problems))
    net = net.resolve_types()
    buses = list(net.buses.index)
    index = {b: i for i, b in enumerate(buses)}
    n = len(buses)
    Bbus = np.zeros((n, n))
    branches = net.branch_frame()
    for _, br in branches.iterrows():
        i, j = index[br["bus0"]], index[br["bus1"]]
        b = 1 / br["x_pu"]
        Bbus[i, i] += b
        Bbus[j, j] += b
        Bbus[i, j] -= b
        Bbus[j, i] -= b
    types = net.buses["type"].map(norm_bus_type).to_numpy()
    slack = types == "Slack"
    p_spec, _ = _injections(net)
    # phase shifters inject an extra term: p0 = (theta_i - theta_j - phi) / x
    p_eff = p_spec.to_numpy().astype(float).copy()
    for _, br in branches.iterrows():
        phi = np.deg2rad(br["phase_shift"])
        if phi:
            i, j = index[br["bus0"]], index[br["bus1"]]
            p_eff[i] += phi / br["x_pu"]
            p_eff[j] -= phi / br["x_pu"]
    theta = np.zeros(n)
    idx = ~slack
    if idx.any():
        theta[idx] = np.linalg.solve(Bbus[np.ix_(idx, idx)], p_eff[idx])
    flows = []
    for _, br in branches.iterrows():
        i, j = index[br["bus0"]], index[br["bus1"]]
        p0 = (theta[i] - theta[j] - np.deg2rad(br["phase_shift"])) / br["x_pu"]
        s_nom = float(br["s_nom"])
        flows.append(dict(
            name=br["name"], kind=br["kind"], bus0=br["bus0"], bus1=br["bus1"],
            p0=p0, q0=0.0, p1=-p0, q1=0.0, p_loss=0.0, q_loss=0.0,
            loading_pct=(100 * abs(p0) / s_nom) if s_nom > 0 else np.nan, s_nom=s_nom,
        ))
    for name, row in net.links.iterrows():
        p0 = float(row["p_set"])
        flows.append(dict(name=name, kind="Link", bus0=row["bus0"], bus1=row["bus1"],
                          p0=p0, q0=0.0, p1=-p0 * float(row["efficiency"]), q1=0.0,
                          p_loss=p0 * (1 - float(row["efficiency"])), q_loss=0.0,
                          loading_pct=(100 * abs(p0) / float(row["p_nom"])) if float(row["p_nom"]) > 0 else np.nan,
                          s_nom=float(row["p_nom"])))
    bus_table = net.buses.copy()
    bus_table["v_pu"] = 1.0
    bus_table["angle_deg"] = np.degrees(theta)
    bus_table["v_kv"] = net.buses["v_nom"].to_numpy()
    bus_table["p_inj"] = p_spec.to_numpy()
    bus_table["q_inj"] = 0.0
    branch_table = pd.DataFrame(flows)
    return PfResult(
        buses=bus_table, branches=branch_table, ybus=Bbus,
        history=pd.DataFrame([{"iteration": 1, "max_mismatch": 0.0}]),
        converged=True, iterations=1, max_mismatch=0.0, bus_order=buses,
        summary={"losses_mw": 0.0, "losses_pct": 0.0, "min_v_pu": 1.0, "max_v_pu": 1.0,
                 "max_loading_pct": float(branch_table["loading_pct"].max()) if len(branch_table) else 0.0,
                 "iterations": 1, "converged": True, "max_mismatch": 0.0,
                 "generation_mw": float(branch_table["p_loss"].sum() + net.loads["p_set"].sum())},
    )


# --------------------------------------------------------------------------- #
# Bridge to PyPSA
# --------------------------------------------------------------------------- #

def to_pypsa(net: PowerNetwork, snapshots=("now",)):
    """Build a :class:`pypsa.Network` that mirrors this model exactly."""
    import pypsa

    net = net.resolve_types()
    n = pypsa.Network()
    n.set_snapshots(list(snapshots))
    n.sn_mva = 1.0
    for carrier, co2 in (("AC", 0.0), ("gas", 0.35), ("solar", 0.0), ("wind", 0.0), ("grid", 0.4)):
        try:
            n.add("Carrier", carrier, co2_emissions=co2)
        except Exception:
            pass
    for name, bus in net.buses.iterrows():
        n.add("Bus", name, v_nom=float(bus["v_nom"]), carrier="AC")
    for name, row in net.lines.iterrows():
        n.add("Line", name, bus0=row["bus0"], bus1=row["bus1"], r=float(row["r"]),
              x=float(row["x"]), b=float(row["b"]), g=float(row["g"]),
              s_nom=float(row["s_nom"]), length=float(row.get("length", 0.0) or 0.0),
              num_parallel=int(row.get("num_parallel", 1) or 1))
    for name, row in net.transformers.iterrows():
        n.add("Transformer", name, bus0=row["bus0"], bus1=row["bus1"], s_nom=float(row["s_nom"]),
              r=float(row["r"]), x=float(row["x"]), g=float(row["g"]), b=float(row["b"]),
              tap_ratio=float(row["tap_ratio"]), tap_side=int(row["tap_side"]),
              phase_shift=float(row["phase_shift"]), model=str(row["model"]))
    for name, row in net.generators.iterrows():
        bus_type = norm_bus_type(net.buses.at[row["bus"], "type"])
        control = "Slack" if bus_type == "Slack" else ("PV" if bus_type == "PV" else str(row["control"]))
        n.add("Generator", name, bus=row["bus"], control=control,
              p_set=float(row["p_set"]), q_set=float(row["q_set"]), p_nom=float(row["p_nom"]),
              carrier=str(row["carrier"]), marginal_cost=float(row["marginal_cost"]))
    for name, bus in net.buses.iterrows():
        if norm_bus_type(bus["type"]) in ("PV", "Slack"):
            n.buses.at[name, "v_mag_pu_set"] = float(bus["v_set"])
    for name, row in net.loads.iterrows():
        n.add("Load", name, bus=row["bus"], p_set=float(row["p_set"]), q_set=float(row["q_set"]))
    for name, row in net.shunts.iterrows():
        n.add("ShuntImpedance", name, bus=row["bus"], g=float(row["g"]), b=float(row["b"]))
    for name, row in net.storages.iterrows():
        n.add("StorageUnit", name, bus=row["bus"], p_set=float(row["p_set"]), q_set=float(row["q_set"]),
              p_nom=float(row["p_nom"]))
    for name, row in net.links.iterrows():
        n.add("Link", name, bus0=row["bus0"], bus1=row["bus1"], p_set=float(row["p_set"]),
              p_nom=float(row["p_nom"]), efficiency=float(row["efficiency"]))
    return n
