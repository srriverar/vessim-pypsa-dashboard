"""Papers that use Vessim — their test systems, run inside the app.

Five publications, five test systems, five reproductions.  Each tab answers the
same three questions:

* **what the paper asks** — the hypothesis, in one sentence;
* **what the test system is** — the hardware, the signal, the controller;
* **what happens here** — the same system, shrunk to a page, with the paper's
  number next to the reproduced one and an honest account of the gap.

The numbers on the right-hand side of every table are computed at run time from
:mod:`vessim_pypsa_dashboard.papers`; nothing is transcribed from a figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import papers
from . import data, theme

COLOURS = theme.COLORS


def _hours(n: int, dt: float) -> np.ndarray:
    return np.arange(n) * dt


def _table(rows: list[tuple[str, str, str, str]]) -> None:
    frame = pd.DataFrame(rows, columns=["Quantity", "Paper", "This page", "Why they differ"])
    st.dataframe(frame, hide_index=True, **theme.wide())


def _paper_header(title: str, reference: str, question: str, system: str, verdict: str) -> None:
    theme.section(title, reference, "paper")
    st.markdown(
        f'<div class="small" style="margin:.2rem 0 .8rem 0">'
        f"<b>Question.</b> {question}<br>"
        f"<b>Test system.</b> {system}<br>"
        f"<b>Result of the paper.</b> {verdict}</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# 1 · Vessim (SIGENERGY EIR, 2024) — the reference microgrid
# --------------------------------------------------------------------------- #

def _study_1() -> None:
    _paper_header(
        "The reference microgrid",
        "Wiesner et al., <i>Vessim: A Testbed for Carbon-Aware Applications and "
        "Systems</i>, ACM SIGENERGY EIR 4(5), 2024 — doi 10.1145/3727200.3727210",
        "can an application that knows the grid's carbon intensity shift work into the "
        "clean hours and cut emissions <i>without</i> buying anything new?",
        "a three-bus microgrid — a consumer, a photovoltaic producer, a battery and a "
        "grid connection — simulated in 60 s steps against a real carbon-intensity "
        "signal; the application is a controller that reads the signal and moves a "
        "flexible share of its workload.",
        "yes, and the size of the win is set by the signal's spread, not by the "
        "controller; the testbed exists so that this win can be measured before "
        "anybody rewires a data centre.",
    )
    run = data.vessim_case("carbon_aware", 60.0, 1.0, 1.0)
    frame = run["df"]
    dt = float(run.get("dt", 0.25))
    hours = _hours(len(frame), dt)
    load = frame["load_mw"].to_numpy() * 1000.0
    carbon = frame["carbon_kg_mwh"].to_numpy() / 1000.0

    carbon_price = st.slider("Carbon price (USD/t)", 0, 200, 60, 10, key="pp1_price")
    flexible = st.slider("Flexible share of the load", 0.0, 0.8, 0.35, 0.05, key="pp1_flex")
    result = papers.shifting_value(load, carbon, float(carbon_price), float(flexible))

    theme.kpis([
        ("Energy", f"{result['energy_kwh']:,.1f}", "kWh", "identical before and after", ""),
        ("Emissions before", f"{result['carbon_before_kg']:,.1f}", "kg", "", "bad"),
        ("Emissions after", f"{result['carbon_after_kg']:,.1f}", "kg",
         f"{papers_pct(result['avoided_kg'] / max(result['carbon_before_kg'], 1e-9))} saved", "good"),
        ("Avoided", f"{result['avoided_kg']:,.1f}", "kg",
         f"≈ {result['avoided_usd']:,.2f} USD at {carbon_price} USD/t", "good"),
        ("Energy that could not move", f"{result['left_unmoved_kwh']:,.1f}", "kWh",
         "the cleanest steps filled up", "warn"),
    ])

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=hours, y=result["profile_before"], name="load, as it came",
                                line=dict(color=COLOURS["muted"], width=2)))
    figure.add_trace(go.Scatter(x=hours, y=result["profile_after"], name="load, after shifting",
                                line=dict(color=COLOURS["green"], width=2.4),
                                fill="tonexty", fillcolor="rgba(52,211,153,.12)"))
    figure.add_trace(go.Scatter(x=hours, y=carbon * 1000.0, name="carbon intensity (g/kWh)",
                                yaxis="y2", line=dict(color=COLOURS["red"], width=1.4, dash="dot")))
    figure.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                     title="g CO₂/kWh"))
    theme.figure(figure, height=360, x_title="hour of the simulated day", y_title="kW")
    theme.show(figure, key="pp1_profile")

    _table([
        ("Load moved", "a controlled share of a synthetic application",
         f"{flexible*100:.0f}% of this day's load",
         "the page exposes the share that the paper's controller decides internally"),
        ("Signal", "real marginal carbon intensity, 60 s resolution",
         f"case 03's trace, {carbon.min()*1000:.0f}–{carbon.max()*1000:.0f} g/kWh, "
         f"Δt = {dt*60:.0f} min",
         "the dashboard runs the same signal at the case's own step so it lines up with "
         "load flow and dispatch"),
        ("Emissions", "lower after shifting, same energy",
         f"{result['carbon_before_kg']:,.1f} → {result['carbon_after_kg']:,.1f} kg",
         "this page packs the flexible energy into the cleanest steps (at most 2× the "
         "original there), which is the paper's behaviour with a simpler scheduler"),
    ])
    theme.note(
        "This is <b>case 03 · carbon-aware workload</b> in this app. Open it and the same "
        "shift appears inside Vessim's own controller, with the policy log that produced it.",
    )


def papers_pct(value: float) -> str:
    return f"{100.0 * value:,.1f} %"


# --------------------------------------------------------------------------- #
# 2 · Software-in-the-loop (SPE, 2023)
# --------------------------------------------------------------------------- #

def _study_2() -> None:
    _paper_header(
        "Software in the loop",
        "Wiesner et al., <i>Software-in-the-Loop Simulation for Developing and Testing "
        "Carbon-Aware Applications</i>, SPE 53(12), 2023 — doi 10.1002/spe.3275",
        "can a real application process be run <i>inside</i> the simulation, so that what "
        "is measured is the software's own energy and not a model of it?",
        "the application runs for real; a small agent samples its CPU share every second, "
        "the CPU share is mapped to watts through a linear power model fitted on the "
        "machine at hand (idle power + slope × CPU), and the simulation clock waits for "
        "the process instead of the other way round.",
        "the loop is faithful where it matters — the energy trace follows the "
        "application's own ramp — and the price is speed: the wall clock is the "
        "simulation clock.",
    )
    model = papers.PowerModel()
    seconds = np.arange(1.0, 601.0)
    cpu = np.clip(np.linspace(0.05, 0.98, len(seconds)) + 0.03 * np.sin(seconds / 17.0), 0.02, 1.0)
    result = papers.sil_energy(seconds, cpu, model)

    theme.kpis([
        ("Model", f"{model.idle_w:.0f} + {model.slope_w:.0f}·CPU", "W",
         f"R² = {model.r2:.2f} on the fitting session", ""),
        ("Mean power", f"{result['mean_w']:,.1f}", "W", f"at {result['mean_cpu']*100:.0f}% CPU", ""),
        ("Peak power", f"{result['peak_w']:,.1f}", "W", "", "warn"),
        ("Energy of the run", f"{result['energy_wh']:,.2f}", "Wh", "10 simulated minutes", "good"),
    ])

    left, right = st.columns([1, 1.35])
    with left:
        grid_cpu = np.linspace(0, 1, 60)
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=grid_cpu * 100, y=model.idle_w + model.slope_w * grid_cpu,
                                    name="fitted model", line=dict(color=COLOURS["sky"], width=2.4)))
        figure.add_trace(go.Scatter(x=cpu * 100, y=result["power_w"], name="the run",
                                    mode="markers", marker=dict(size=4,
                                                                color=COLOURS["amber"], opacity=.75)))
        theme.figure(figure, height=300, x_title="CPU utilisation (%)", y_title="package power (W)")
        theme.show(figure, key="pp2_fit")
    with right:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=seconds, y=result["power_w"], name="power (W)",
                                    line=dict(color=COLOURS["vessim"], width=2)))
        figure.add_trace(go.Scatter(x=seconds, y=cpu * 100, name="CPU (%)", yaxis="y2",
                                    line=dict(color=COLOURS["muted"], width=1.2, dash="dot")))
        figure.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                         range=[0, 105], title="%"))
        theme.figure(figure, height=300, x_title="second inside the loop", y_title="W")
        theme.show(figure, key="pp2_run")

    _table([
        ("What is modelled", "nothing about the application: it runs",
         "a CPU trace replayed through the same kind of linear model",
         "the page cannot launch a process per visitor in a shared deployment; "
         "<b>case 04 · hardware in the loop</b> does launch one, locally"),
        ("Sampling", "sub-second, from the operating system",
         f"1 s, {len(seconds)} samples", "enough to resolve a 10-minute ramp"),
        ("Model form", "P = P_idle + k·CPU", f"P = {model.idle_w:.0f} + {model.slope_w:.0f}·CPU W",
         "same form, coefficients from this workspace's machine"),
        ("Energy", "integrated from the sampled power",
         f"{result['energy_wh']:,.2f} Wh over {len(seconds)/60:.0f} min", "—"),
    ])
    theme.note(
        "The paper's real contribution is the <b>contract</b>: the simulator hands the "
        "application a clock and a price, the application hands back its power. "
        "<b>🔌 Hardware in the loop</b> in this app writes that contract out line by line.",
    )


# --------------------------------------------------------------------------- #
# 3 · Battery models (LOCO, 2024)
# --------------------------------------------------------------------------- #

def _study_3() -> None:
    _paper_header(
        "Choosing the right battery model",
        "Kilian, Wiesner, Kao, <i>Choosing the right battery model for data-center "
        "simulations</i>, LOCO 2024 — arXiv 2506.17739",
        "does the storage model change the answer, or is a lossless battery good enough?",
        "one day of a solar-plus-storage site, replayed through four models: lossless, "
        "linear (one efficiency), KiBaM (two tanks, power-limited discharge) and Peukert "
        "(capacity falls with the discharge rate).",
        "the model matters exactly when the battery is asked to work hard; when it is "
        "never binding, all four agree and the linear model is the right choice — which is "
        "the paper's recommendation for short experiments.",
    )
    run = data.vessim_case("solar_battery", 60.0, 1.0, 1.0)
    frame = run["df"]
    dt = float(run.get("dt", 0.25))
    hours = _hours(len(frame), dt)
    load = frame["load_mw"].to_numpy() * 1000.0
    solar = frame["solar_mw"].to_numpy() * 1000.0

    left, right = st.columns([1, 1])
    with left:
        capacity = st.slider("Battery capacity (kWh)", 2.0, 24.0, 12.0, 1.0, key="pp3_cap")
    with right:
        power = st.slider("Inverter rating (kW)", 1.0, 12.0, 6.0, 0.5, key="pp3_pow")

    results = {name: papers.battery_day(load, solar, float(capacity), float(power), model=name)
               for name in ("ideal", "linear", "kibam", "peukert")}
    labels = {"ideal": "lossless", "linear": "linear (η = 0.95)",
              "kibam": "KiBaM", "peukert": "Peukert"}

    theme.kpis([(labels[name], f"{res['unmet_kwh']:,.2f}", "kWh unmet",
                 f"SOC end {res['soc_end_kwh']:,.2f} kWh", "bad" if res["unmet_kwh"] > 1 else "good")
                for name, res in results.items()])

    figure = go.Figure()
    for name, colour in (("ideal", COLOURS["muted"]), ("linear", COLOURS["sky"]),
                         ("kibam", COLOURS["battery"]), ("peukert", COLOURS["amber"])):
        res = results[name]
        soc = res["soc_kwh"][1:]
        figure.add_trace(go.Scatter(x=hours, y=soc, name=labels[name],
                                    line=dict(color=colour, width=2)))
    theme.figure(figure, height=330, x_title="hour", y_title="state of charge (kWh)")
    theme.show(figure, key="pp3_soc")

    hard = power / max(capacity, 1e-9)
    theme.note(
        f"At {capacity:.0f} kWh and {power:.1f} kW this battery runs at {hard:.2f} C. "
        "KiBaM separates from the others once the available tank empties — its power sags "
        "before its energy does. Peukert only bites above 1 C, so on a gentle day it prints "
        "the same numbers as the linear model; <b>that is the paper's point</b>, not a bug, "
        "and the table says so instead of inventing a difference."
    )

    _table([
        ("Models compared", "lossless, linear, KiBaM (and PyBaMM DFN/SPM)",
         "lossless, linear, KiBaM, Peukert", "PyBaMM is a dependency the app does not ship"),
        ("Recommendation", "linear for short runs, physics-based when storage binds",
         "the spread appears at ≥ 0.5 C", "same conclusion, visible on this case"),
        ("Unmet demand", "depends on the model once the battery binds",
         " · ".join(f"{labels[n]} {results[n]['unmet_kwh']:.2f} kWh" for n in results), "—"),
        ("Test system", "a data-centre site with on-site generation",
         "case 02 · solar + battery · one day", "the dashboard's own trace"),
    ])


# --------------------------------------------------------------------------- #
# 4 · LLM inference (PECS @ Euro-PAR, 2025)
# --------------------------------------------------------------------------- #

def _study_4() -> None:
    _paper_header(
        "Energy and carbon of LLM inference",
        "Özcan, Wiesner, Weiß, Kao, <i>Quantifying the Energy Consumption and Carbon "
        "Emissions of LLM Inference via Simulations</i>, PECS@Euro-PAR 2025 — arXiv 2507.11417",
        "how much energy and carbon does a serving fleet emit, and how much of it can a "
        "carbon-aware scheduler take back?",
        "Vidur produces the per-request timing and the model-flops saturation; the GPU "
        "power follows P = P_idle + (P_max − P_idle)·(mfu/mfu_sat)^γ; Vessim places the "
        "fleet on a grid with a real carbon signal and lets the scheduler move the "
        "deferrable requests.",
        "energy is dominated by the fleet size and the utilisation, carbon by the hour; "
        "shifting the deferrable share moves emissions by a double-digit percentage "
        "without touching the hardware.",
    )
    run = data.vessim_case("carbon_aware", 60.0, 1.0, 1.0)
    frame = run["df"]
    dt = float(run.get("dt", 0.25))
    hours = _hours(len(frame), dt)
    carbon = frame["carbon_kg_mwh"].to_numpy() / 1000.0

    left, right, third = st.columns(3)
    with left:
        gpu_name = st.selectbox("Accelerator", list(papers.GPUS), key="pp4_gpu")
    with right:
        fleet = st.slider("GPUs in the fleet", 8, 128, 64, 8, key="pp4_fleet")
    with third:
        shiftable = st.slider("Deferrable share of requests", 0.0, 0.9, 0.45, 0.05, key="pp4_shift")

    gpu = papers.GPUS[gpu_name]
    requests = 400.0 + 3600.0 * np.exp(-((np.arange(len(frame)) - len(frame) * 0.55) ** 2) / 40.0)
    result = papers.inference_day(requests, carbon, gpu, gpus=int(fleet),
                                  pue=1.2, shiftable=float(shiftable))

    theme.kpis([
        ("Fleet", f"{fleet}", "GPUs", f"{gpu.name} · {gpu.idle_w:.0f}–{gpu.max_w:.0f} W", ""),
        ("Site energy", f"{result['energy_kwh']:,.0f}", "kWh", "over the simulated day", ""),
        ("Emissions", f"{result['emissions_kg']:,.1f}", "kg",
         f"{result['wh_per_request']:.3f} Wh/request", "warn"),
        ("Carbon-aware offset", papers_pct(result["offset"]), "",
         f"after moving {shiftable*100:.0f}% of requests", "good"),
        ("Peak GPU power", f"{result['peak_gpu_w']/1000:,.2f}", "kW", "", ""),
    ])

    figure = go.Figure()
    figure.add_trace(go.Bar(x=hours, y=result["site_kw"], name="site load, as it came",
                            marker_color=COLOURS["muted"], opacity=.55))
    figure.add_trace(go.Bar(x=hours, y=result["shifted_kw"], name="after carbon-aware shifting",
                            marker_color=COLOURS["green"], opacity=.8))
    figure.add_trace(go.Scatter(x=hours, y=carbon * 1000.0, name="g CO₂/kWh", yaxis="y2",
                                line=dict(color=COLOURS["red"], width=1.4, dash="dot")))
    figure.update_layout(barmode="overlay",
                         yaxis2=dict(overlaying="y", side="right", showgrid=False))
    theme.figure(figure, height=360, x_title="hour", y_title="kW")
    theme.show(figure, key="pp4_load")

    _table([
        ("Timing model", "Vidur, per-request", "a Gaussian request window over 96 steps",
         "no trace of a real serving workload ships with this app"),
        ("Power model", "P = P_idle + (P_max − P_idle)(mfu/mfu_sat)^γ",
         f"same form, γ = {gpu.gamma}, A100/H100/A40 coefficients from the paper", "—"),
        ("Offset", "up to 69.2 % of emissions, workload-dependent",
         f"{result['offset']*100:.1f} % here", "the dashboard's carbon trace and fleet size, "
         "and the shift only moves requests into cleaner <i>steps</i>"),
        ("Test system", "a serving fleet on one grid",
         f"{fleet} × {gpu.name}", "shrunk by ~an order of magnitude to fit a page"),
    ])
    theme.note(
        "The same argument is what <b>case 05 · Ecopetrol</b> shows at industrial scale: "
        "the training and inference loads are the flexible part of a plant's demand.",
    )


# --------------------------------------------------------------------------- #
# 5 · Pretraining during curtailment windows (2026)
# --------------------------------------------------------------------------- #

def _study_5() -> None:
    _paper_header(
        "Pretraining in curtailment windows",
        "Wiesner et al., <i>Distributed LLM Pretraining During Renewable Curtailment "
        "Windows</i>, technical report 2026 — arXiv 2602.22760",
        "if a cluster trains only when renewable energy would otherwise be curtailed, and "
        "the three clusters sit in different time zones, how much carbon does that save?",
        "three geo-distributed training clusters (a 561 M-parameter model, 12.8 B tokens, "
        "WattTime signals); each site pauses, trains locally or averages periodically so "
        "that the global step lands in the cleanest available window.",
        "the fleet runs at 5–12 % of the emissions of a single site training around the "
        "clock — most of the win comes from <i>when</i>, not from <i>what</i>.",
    )
    run = data.vessim_case("ecopetrol", 60.0, 1.0, 1.0)
    frame = run["df"]
    dt = float(run.get("dt", 0.25))
    hours = _hours(len(frame), dt)
    base = frame["carbon_kg_mwh"].to_numpy() / 1000.0
    n = len(base)
    carbons = [np.roll(base, k * n // 3) for k in range(3)]
    threshold = st.slider("Curtailment threshold (kg CO₂/kWh)", 0.05, 0.45, 0.25, 0.01,
                          key="pp5_thr")
    result = papers.pretraining_day(carbons, float(threshold))

    theme.kpis([
        ("Window", f"{result['hours_available']:,.1f}", "h",
         "at least one site is curtailed", ""),
        ("Energy", f"{result['energy_policy_kwh']:,.0f}", "kWh",
         f"vs {result['energy_baseline_kwh']:,.0f} kWh round the clock",
         "good" if result["energy_policy_kwh"] <= result["energy_baseline_kwh"] else "warn"),
        ("Emissions", f"{result['emissions_policy_kg']:,.0f}", "kg",
         f"vs {result['emissions_baseline_kg']:,.0f} kg", "good"),
        ("Ratio", papers_pct(result["ratio"]), "of the baseline",
         "emissions, not energy", "good"),
        ("Intensity", f"{result['g_per_kwh_policy']:,.0f}", "g/kWh",
         f"baseline {result['g_per_kwh_baseline']:,.0f}", ""),
    ])

    figure = go.Figure()
    for index, carbon in enumerate(carbons):
        figure.add_trace(go.Scatter(x=hours, y=carbon * 1000, name=f"site {index+1}",
                                    line=dict(width=1.5)))
    figure.add_hline(y=float(threshold) * 1000, line=dict(color=COLOURS["red"], dash="dash"),
                     annotation_text="curtailment threshold")
    theme.figure(figure, height=320, x_title="hour (local, phase-shifted)", y_title="g CO₂/kWh")
    theme.show(figure, key="pp5_sites")

    windows = pd.DataFrame({
        "site": [f"site {i+1}" for i in range(3) for _ in np.flatnonzero(result["windows"][i])],
        "start": [hours[j] for i in range(3) for j in np.flatnonzero(result["windows"][i])],
        "length": [dt for i in range(3) for _ in np.flatnonzero(result["windows"][i])],
    })
    if not windows.empty:
        figure = go.Figure()
        for index, row in windows.iterrows():
            figure.add_trace(go.Bar(x=[row["length"]], y=[row["site"]], base=[row["start"]],
                                    orientation="h", marker_color=COLOURS["green"],
                                    showlegend=False, hovertemplate=f"h {row['start']:.2f}"))
        theme.figure(figure, height=220, x_title="hour of the day", y_title="")
        theme.show(figure, key="pp5_gantt")
    else:
        theme.note("No step is below the threshold: the fleet would idle all day.", "bad")

    sweep = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    ratios = [papers.pretraining_day(carbons, value)["ratio"] * 100 for value in sweep]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=sweep, y=ratios, mode="lines+markers",
                                line=dict(color=COLOURS["vessim"], width=2)))
    figure.add_vline(x=float(threshold), line=dict(color=COLOURS["amber"], dash="dash"))
    theme.figure(figure, height=280, x_title="threshold (kg CO₂/kWh)", y_title="% of baseline emissions")
    theme.show(figure, key="pp5_sweep")
    theme.note(
        "The curve is not monotone in the way intuition suggests: a looser threshold buys "
        "more hours but at dirtier intensities, and the two effects cross. " 
        "The paper's 5–12 % needs <b>three clusters whose clean hours interleave</b> — this "
        "page only rolls the same trace by a third of a day, so it reproduces the mechanism, "
        "not the paper's number.",
    )

    _table([
        ("Clusters", "3, in three grids", "3, the same trace phase-shifted",
         "one real signal is all this app ships; the interleaving is simulated"),
        ("Model", "561 M parameters, 12.8 B tokens", "a 100 kW IT load",
         "the page shows the energy bookkeeping, not the training"),
        ("Result", "5–12 % of single-site emissions",
         f"{result['ratio']*100:.1f} % at {threshold:.2f} kg/kWh",
         "different baseline and only one signal"),
    ])


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

def render() -> None:
    theme.hero(
        "📄",
        "Papers that use Vessim",
        "THEIR TEST SYSTEMS, RUNNING HERE",
        "Five publications built on Vessim. For each one: the question, the test system as "
        "the authors describe it, and that same system recomputed in front of you, with the "
        "paper's number beside the reproduced one and the reason for every gap.",
        [("Papers", "5"), ("Test systems", "5"), ("Numbers", "recomputed"),
         ("Source", "papers.py")],
    )
    st.markdown(
        '<div class="small" style="margin:.2rem 0 1rem 0">'
        "Every panel below runs live: move a slider and the figure is recomputed from "
        "the same functions the rest of the app uses "
        "(<code>src/vessim_pypsa_dashboard/papers.py</code>). Where a paper reports a "
        "figure we cannot reproduce at this scale, the table says so instead of hiding it.</div>",
        unsafe_allow_html=True,
    )

    tab_map, tab_1, tab_2, tab_3, tab_4, tab_5 = st.tabs([
        "🗺️ Map", "🛰️ Reference microgrid", "🧪 Software in the loop",
        "🔋 Battery models", "🤖 LLM inference", "🌍 Curtailment pretraining",
    ])
    with tab_map:
        theme.section("From paper to case", "which test system lives where in this app", "map")
        st.dataframe(pd.DataFrame([
            ("SIGENERGY EIR 2024", "three-bus microgrid, carbon-aware controller",
             "03 · carbon-aware workload", "the same controller, with its policy log"),
            ("SPE 2023", "a real process inside the simulation",
             "04 · hardware in the loop", "the contract, written out and executed"),
            ("LOCO 2024", "solar + battery site, four storage models",
             "02 · solar + battery", "the model comparison on the case's own day"),
            ("PECS 2025", "GPU fleet on a carbon signal",
             "05 · Ecopetrol (training + inference)", "industrial-scale flexible load"),
            ("arXiv 2602.22760", "three distributed training clusters",
             "03 / 05 · signals", "phase-shifted copies of a real signal"),
        ], columns=["Paper", "Test system", "Case in this app", "What the case adds"]),
            hide_index=True, **theme.wide())
        theme.note(
            "The adjacent question — why the <b>LoadFlow</b> and <b>ED-UC</b> sheets use "
            "these cases and not somebody else's IEEE buses — is answered in every case "
            "tab, at the top of both sheets.",
        )
    with tab_1:
        _study_1()
    with tab_2:
        _study_2()
    with tab_3:
        _study_3()
    with tab_4:
        _study_4()
    with tab_5:
        _study_5()
