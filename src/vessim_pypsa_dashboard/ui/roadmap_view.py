"""Case 06 — the roadmap case, page by page.

Four things Vessim's roadmap says are coming, reduced to what runs here, each in
its own section and each labelled with what it is *not*:

* the dataset (WattTime MOER, bundled with Vessim) instead of a sine wave;
* the SAM-style photovoltaic model (irradiance → cell temperature → power);
* the Monte Carlo (N full Vessim runs, P10/P50/P90);
* the battery degradation (cycle + calendar fade, priced per day).

The page ends with the honest part: what it would take to have these *inside*
Vessim rather than in the app around it — which is the conversation this case is
meant to start.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import roadmap
from ..vessim_runner import CASES
from . import theme

COLOURS = theme.COLORS


def _dataset_section(spec) -> None:
    theme.section("1 · A dataset, not a formula",
                  "the carbon signal is the WattTime trace that ships with Vessim", "DATASET")
    info = roadmap.dataset_summary()
    hours = np.arange(spec.steps) * spec.step_s / 3600.0
    series = roadmap.carbon_dataset(spec.steps, spec.step_s)
    synthetic = 0.42 + 0.06 * np.sin(2 * np.pi * (hours - 6) / 24.0)

    theme.kpis([
        ("Source", "watttime_example.csv", "", "bundled with Vessim (MIT)", ""),
        ("Rows", f"{info['rows']:,}", "", f"{info['days']} days at 5-minute resolution", ""),
        ("Mean", f"{info['mean']:.3f}", "kg/kWh", "read as g CO₂/kWh, as Vessim's example does", ""),
        ("Range", f"{info['min']:.3f}–{info['max']:.3f}", "kg/kWh", "one dataset day used here", ""),
        ("What the case used before", "a sine", "", "smooth, monotone, and far too easy", "warn"),
    ])
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=hours, y=series * 1000, name="WattTime dataset",
                                line=dict(color=COLOURS["red"], width=2.4)))
    figure.add_trace(go.Scatter(x=hours, y=synthetic * 1000, name="the sine the other cases use",
                                line=dict(color=COLOURS["muted"], width=1.6, dash="dot")))
    theme.figure(figure, height=300, x_title="hour of the simulated day",
                 y_title="g CO₂/kWh")
    theme.show(figure, key="rm_dataset")
    theme.note(
        "The dataset has steps the sine cannot: short dirty spikes and clean plateaus inside "
        "the same hour. A policy that only knows the daily shape cannot react to them, which is "
        "exactly what makes a dataset worth shipping. The file runs "
        f"{info['start'][:10]} → {info['end'][:10]}. <b>What this is not:</b> a live connector — "
        "the file ships in <code>src/vessim_pypsa_dashboard/data/</code>, and the roadmap item is "
        "to integrate such sources inside Vessim."
    )


def _pv_section(spec) -> None:
    theme.section("2 · SAM-style photovoltaic, from irradiance",
                  "PVWatts inside SAM: NOCT cell temperature, temperature-corrected efficiency",
                  "SAM · PVWATTS")
    info = roadmap.dataset_summary()
    left, middle, right = st.columns(3)
    with left:
        cloud = st.slider("Cloud severity", 0.0, 1.0, 0.6, 0.05, key="rm_cloud",
                          help="0 = perfectly clear sky, 1 = the cloud index used at full strength")
    with middle:
        ambient = st.slider("Ambient offset (°C)", -5.0, 10.0, 0.0, 0.5, key="rm_temp")
    with right:
        latitude = st.slider("Latitude (°N)", -30.0, 55.0, 4.6, 0.4, key="rm_lat",
                             help="Bogotá is 4.6° N: almost no season, which is why the profile "
                                  "repeats every day")

    model = roadmap.PVModel(kwp=float(spec.solar_kwp))
    hours = np.arange(spec.steps) * spec.step_s / 3600.0
    rng = np.random.default_rng(3)
    cloud_index = roadmap.cloud_factor(rng, spec.steps)
    ghi = roadmap.solar_geometry(hours, latitude_deg=latitude) * (1.0 - cloud * (1.0 - cloud_index))
    temperature = 20.0 + 6.0 * np.sin(2 * np.pi * (hours - 9) / 24.0) + ambient
    power = model.power_kw(ghi, temperature)
    cell = model.cell_c(ghi, temperature)
    naive = model.kwp * ghi / 1000.0 * model.derate          # no temperature derating

    theme.kpis([
        ("Roof", f"{spec.solar_kwp:,.0f}", "kWp", f"{model.derate:.0%} derate, γ = {model.gamma}", ""),
        ("Peak irradiance", f"{ghi.max():,.0f}", "W/m²", "clear-sky envelope this day", ""),
        ("Cell temperature", f"{cell.max():,.0f}", "°C",
         f"NOCT {model.noct_c:.0f} °C, ambient {temperature.max():.0f} °C", "warn"),
        ("Energy", f"{power.sum() * spec.step_s / 3600:,.0f}", "kWh", "after the temperature term", "good"),
        ("Lost to heat", f"{(naive - power).sum() * spec.step_s / 3600:,.1f}", "kWh",
         f"{100 * (naive - power).sum() / max(naive.sum(), 1e-9):.1f} % of a constant-efficiency model",
         "bad"),
    ])

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=hours, y=ghi, name="irradiance (W/m²)",
                                line=dict(color=COLOURS["amber"], width=2)))
    figure.add_trace(go.Scatter(x=hours, y=power, name="PV power (kW)",
                                line=dict(color=COLOURS["solar"], width=2.4)))
    figure.add_trace(go.Scatter(x=hours, y=naive, name="no temperature derating (kW)",
                                line=dict(color=COLOURS["muted"], width=1.4, dash="dot")))
    figure.add_trace(go.Scatter(x=hours, y=cell, name="cell temperature (°C)", yaxis="y2",
                                line=dict(color=COLOURS["red"], width=1.2, dash="dash")))
    figure.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))
    theme.figure(figure, height=330, x_title="hour", y_title="W/m² · kW")
    theme.show(figure, key="rm_pv")
    theme.note(
        "<b>What this is not:</b> SAM. It is the PVWatts sequence that SAM wraps — plane-of-array "
        "irradiance, cell temperature, temperature-corrected efficiency — with a clear-sky "
        "envelope instead of a weather file and no inverter model. What it <i>does</i> show is the "
        "part a constant-efficiency model gets wrong: a hot panel at noon produces less than the "
        "same irradiance at 09:00."
    )


def _monte_carlo_section(spec, run: dict) -> None:
    theme.section("3 · Monte Carlo over full simulations",
                  "one draw per run: weather, load, price and carbon", "PROBABILISTIC")
    left, middle, right = st.columns(3)
    with left:
        n = st.slider("Scenarios", 4, 32, 12, 4, key="rm_n")
    with middle:
        seed = st.slider("Seed", 1, 40, 5, 1, key="rm_seed")
    with right:
        sigma = st.slider("Load spread σ", 0.01, 0.12, 0.05, 0.01, key="rm_sigma")

    with st.spinner(f"Running {n} complete Vessim simulations…"):
        result = roadmap.monte_carlo(n=int(n), seed=int(seed), carbon_price=float(spec.carbon_price),
                                     load_sigma=float(sigma))
    summary = result["summary"].copy()
    st.dataframe(summary.round(2), hide_index=True, **theme.wide(), height=290)

    frame = result["frame"]
    left, right = st.columns(2)
    with left:
        figure = go.Figure(go.Histogram(x=frame["bill USD"], nbinsx=max(6, int(n / 2)),
                                        marker_color=COLOURS["sky"], opacity=.85))
        figure.add_vline(x=float(run["baseline"]["total_cost_usd"]),
                         line=dict(color=COLOURS["amber"], dash="dash", width=2),
                         annotation_text="the deterministic day")
        theme.figure(figure, height=280, x_title="bill for the day (USD)", y_title="scenarios")
        theme.show(figure, key="rm_hist")
    with right:
        figure = go.Figure(go.Scatter(
            x=frame["bill USD"], y=frame["CO₂ kg"], mode="markers",
            marker=dict(size=9, color=frame["PV kWh"], colorscale="YlOrRd", showscale=True,
                        colorbar=dict(title="PV kWh")),
            hovertemplate="scenario %{pointNumber}<br>%{x:,.0f} USD · %{y:,.0f} kg<extra></extra>"))
        theme.figure(figure, height=280, x_title="bill USD", y_title="CO₂ kg")
        theme.show(figure, key="rm_scatter")

    spread = summary.set_index("metric").loc["bill USD", "spread"]
    p50 = summary.set_index("metric").loc["bill USD", "P50"]
    theme.note(
        f"The bill moves <b>{spread:,.0f} USD</b> between P10 and P90 on the same plant and the "
        f"same rules — <b>{100 * spread / max(p50, 1e-9):.0f} %</b> of the median. None of that "
        "range is visible in a single deterministic run, and none of it is a modelling error: it "
        "is the weather. <b>What this is not:</b> inside Vessim. Each sample is a complete "
        "`env.run`, so a 1000-sample study costs 1000 simulations; the roadmap item is a "
        "distribution-aware interface that would let one run carry the sampling."
    )


def _degradation_section(spec, run: dict) -> None:
    theme.section("4 · The battery ages, and the rule decides how fast",
                  "cycle fade from equivalent full cycles + calendar fade, priced per day",
                  "BLAST-LITE")
    wear = roadmap.degradation(run["df"]["soc_mwh"].to_numpy(), float(run["dt"]),
                               float(spec.battery_kwh) / 1000.0)
    discharged = max(wear["discharged_mwh"], 1e-12)
    implied = wear["wear_usd"] / discharged
    theme.kpis([
        ("Equivalent full cycles", f"{wear['efc']:.2f}", "/day", "discharged ÷ capacity", ""),
        ("Throughput", f"{wear['throughput_mwh']:.2f}", "MWh", "moved in both directions", ""),
        ("Fade today", f"{100 * (wear['cycle_fade'] + wear['calendar_fade']):.3f}", "%",
         f"cycle {100 * wear['cycle_fade']:.3f} % + calendar {100 * wear['calendar_fade']:.3f} %", ""),
        ("Wear cost", f"{wear['wear_usd']:.2f}", "USD/day",
         f"{implied:,.1f} USD per MWh discharged", "warn"),
        ("The rule assumes", "20.0", "USD/MWh", "MeritOrderPolicy's degradation cost", "bad"),
    ])

    table = roadmap.degradation_by_policy(spec.key)
    st.dataframe(table.drop(columns=["key"]).round(2), hide_index=True, **theme.wide(), height=270)

    cycles = np.linspace(1000, 20000, 120)
    figure = go.Figure()
    for name, life, colour in (("NMC (3 000)", 3000, COLOURS["red"]),
                               ("LFP (6 000)", 6000, COLOURS["green"]),
                               ("LTO (20 000)", 20000, COLOURS["sky"])):
        cost = 0.20 * wear["efc"] / life * float(spec.battery_kwh) * 250.0
        figure.add_hline(y=cost, line=dict(color=colour, dash="dash", width=1.6),
                         annotation_text=name, annotation_position="right")
    figure.add_hline(y=20.0 * discharged, line=dict(color=COLOURS["amber"], width=3),
                     annotation_text="what the policy charged for the same duty",
                     annotation_position="left")
    theme.figure(figure, height=260, x_title="", y_title="wear cost of this day (USD)")
    theme.show(figure, key="rm_chem")
    theme.note(
        f"Three chemistries, the same duty cycle: the day costs the NMC cell "
        f"{0.20 * wear['efc'] / 3000 * float(spec.battery_kwh) * 250:.2f} USD and the LTO cell "
        f"{0.20 * wear['efc'] / 20000 * float(spec.battery_kwh) * 250:.2f} USD. The flat "
        f"20 USD/MWh inside the policy charges <b>{20.0 * discharged:.2f} USD</b> for the same "
        f"trace — {20.0 * discharged / max(wear['wear_usd'], 1e-9):.1f}× the physics. A rule that "
        "over-prices wear leaves the battery idle; this section is why BLAST-Lite is on the "
        "roadmap. <b>What this is not:</b> BLAST-Lite — no depth-of-discharge dependence, no "
        f"temperature, no rainflow counting. The case's cell is {float(spec.battery_kwh):,.0f} kWh."
    )


def _inside_vessim(spec) -> None:
    theme.section("5 · What it would take to have this inside Vessim",
                  "the four roadmap items, as interfaces rather than features", "FOR THE MAINTAINERS")
    st.markdown(
        """
        | Roadmap item | What this case does today | What an interface inside Vessim would need |
        |---|---|---|
        | Probabilistic modelling | draws in `roadmap.scenario`, one `env.run` per draw | a sampler handed to `Environment.run`, so one call walks the distribution and the `Recorder` keeps the sample index |
        | Datasets | reads a bundled CSV and passes arrays through `signals=` | a `Signal` source that resolves a dataset by name and resamples to `step_size` before the run starts |
        | SAM | `pvwatts(ghi, ambient)` in a few lines | a `Producer` whose power is a subsystem call, with SAM's weather file as the input |
        | BLAST-Lite | capacity fade after the fact, priced per day | a `Storage` whose capacity degrades *during* the run, so the policy sees the fade it causes |
        """,
    )
    st.code(
        "# the shape that would make the four items composable\n"
        "env = vs.Environment(sim_start='2026-01-01', step_size=1800, sampler=MonteCarlo(n=200))\n"
        "env.add_microgrid(\n"
        "    actors=[vs.Consumer('campus', campus),\n"
        "            vs.Producer('solar', vs.SAM('weather.epw', kwp=180)),\n"
        "            vs.Consumer('dc', computing)],\n"
        "    dispatchables=[vs.DegradingBattery('battery', 300_000, model='blast-lite'),\n"
        "                   vs.GridConnection(max_import=260_000)],\n"
        "    policy=MeritOrderPolicy(carbon_price_usd_per_t=60),\n"
        "    grid_signals={'carbon': vs.dataset('watttime', 'Caiso_North'), 'price': tariff},\n"
        ")\n",
        language="python",
    )
    theme.note(
        "Every one of those four lines exists in this app as a function the reader can call "
        "today — <code>roadmap.monte_carlo</code>, <code>roadmap.carbon_dataset</code>, "
        "<code>roadmap.PVModel</code>, <code>roadmap.degradation</code> — and none of them is "
        "inside Vessim, which is precisely the point of the case: it measures today's workaround "
        "so that tomorrow's interface has a reference to beat.",
    )


def render(case_key: str, run: dict, spec, carbon: float) -> None:
    theme.hero(
        "🎲", "The roadmap case",
        f"CASE {spec.number} · {spec.title.upper()} · MONTE CARLO",
        "Vessim's roadmap lists four things: probabilistic modelling, datasets, SAM as a "
        "subsystem, and BLAST-Lite for battery degradation. This case is those four, reduced to "
        "what fits in a page and labelled item by item — not as a claim that Vessim does them, "
        "but as a measurement of what it takes.",
        [("Scenarios", "N full runs"), ("Dataset", "WattTime"), ("PV", "PVWatts"),
         ("Storage", "fade priced")],
    )
    st.markdown(
        '<div class="small" style="margin:.2rem 0 1rem 0">'
        "The case itself is a 120 kW campus with 180 kWp of PV, a 300 kWh battery and a 260 kW "
        "connection, with the same computing workloads as the other cases. Every number below is "
        "computed now, from <code>src/vessim_pypsa_dashboard/roadmap.py</code>.</div>",
        unsafe_allow_html=True)

    _dataset_section(spec)
    _pv_section(spec)
    _monte_carlo_section(spec, run)
    _degradation_section(spec, run)
    _inside_vessim(spec)
