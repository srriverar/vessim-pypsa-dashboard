"""The computing layer of a case: what the microgrid is dispatching *for*.

Vessim is a testbed for carbon-aware **computing** systems. Every case in this
dashboard therefore carries explicit computing workloads — an edge node, an
inference service, a training cluster, storage, and the PUE overhead that
follows them — built by :mod:`vessim_pypsa_dashboard.computing` and simulated as
``vessim.Consumer`` actors. This panel shows them: what each one is, how much it
draws, how much of it can move, and what share of the site's electricity it owns.

PyPSA never sees any of this. It sees ``loads_t.p_set``: the sum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..computing import summary
from . import theme

COLORS = ["#38bdf8", "#a78bfa", "#fbbf24", "#34d399", "#f472b6", "#fb923c"]


def render(run: dict) -> None:
    spec = run["spec"]
    specs = getattr(spec, "computing", ())
    if not specs:
        theme.note("This case has no computing layer configured.")
        return
    df: pd.DataFrame = run["df"]
    dt = float(run["dt"])
    traces = {}
    for item in specs:
        column = f"load_{item.name}_mw"
        traces[item.name] = (df[column].to_numpy(dtype=float) * 1000.0
                             if column in df.columns else np.zeros(len(df)))

    site_kwh = float(df["load_mw"].sum() * dt) * 1000.0
    compute_kwh = sum(float(t.sum() * dt) for t in traces.values())
    it_kwh = sum(float(t.sum() * dt) / max(s.pue, 1e-9)
                 for s, t in ((s, traces[s.name]) for s in specs))
    deferrable_kwh = sum(float(t.sum() * dt) * s.flexible / max(s.pue, 1e-9)
                         for s, t in ((s, traces[s.name]) for s in specs))
    overhead_kwh = compute_kwh - it_kwh

    theme.section("The computing layer",
                  "the workloads Vessim is actually dispatching — and what PyPSA only sees as a sum",
                  "COMPUTING")
    theme.kpis([
        ("Computing share", f"{100 * compute_kwh / max(site_kwh, 1e-9):.1f}", "%",
         f"{compute_kwh:,.2f} of {site_kwh:,.2f} kWh on the site", "accent"),
        ("Workloads", f"{len(specs)}", "",
         " · ".join(sorted({item.kind for item in specs})), ""),
        ("Deferrable", f"{deferrable_kwh:,.2f}", "kWh",
         "IT energy the schedule may move in time", "good" if deferrable_kwh > 1e-9 else ""),
        ("PUE overhead", f"{overhead_kwh:,.2f}", "kWh",
         f"{100 * overhead_kwh / max(compute_kwh, 1e-9):.1f} % cooling and losses", ""),
    ])

    st.dataframe(pd.DataFrame(summary(specs, traces, dt, site_kwh)), hide_index=True,
                 **theme.wide(), height=min(90 + 36 * len(specs), 260))

    figure = go.Figure()
    for index, item in enumerate(specs):
        figure.add_trace(go.Scatter(
            x=df.index, y=traces[item.name], name=item.name, mode="lines",
            stackgroup="compute", line=dict(width=0.5, color=COLORS[index % len(COLORS)]),
            fillcolor=COLORS[index % len(COLORS)], opacity=0.75,
            hovertemplate="%{y:.3f} kW"))
    figure.add_trace(go.Scatter(x=df.index, y=df["load_mw"] * 1000.0, name="Site load (all actors)",
                                mode="lines", line=dict(color="#f8fafc", width=2),
                                hovertemplate="%{y:.1f} kW"))
    theme.show(theme.figure(figure, height=300, y_title="kW", x_title=""))
    figure.update_yaxes(ticksuffix=" kW")

    flexible_names = [item.name for item in specs if item.flexible > 0]
    if flexible_names:
        theme.note(
            "The workloads in colour are the ones the schedule can move: "
            f"<b>{', '.join(flexible_names)}</b>. Their traces are built so that the daily energy "
            "is identical however the work is placed — the property that lets the optimiser and the "
            "controller be compared at all. PyPSA receives <code>p_set</code> = the white line, with "
            "no way to know which part of it could have run at another hour."
        )
    else:
        theme.note(
            "None of these workloads can be moved: that is a design choice of the case, and the "
            "reason its dispatch gap is what it is. Compare with case 03 or 05."
        )
