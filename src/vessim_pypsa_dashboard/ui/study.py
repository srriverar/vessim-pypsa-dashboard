"""The Vessim viewer, ported: Results · Time series · Run details.

`python -m vessim view results/study` serves a small React app with three pages
and two kinds of entity — a **configuration** (one parameter set) and a **run**
(one execution of it). The options of that app are reproduced here against real
studies generated in this dashboard:

* **Results** — configurations or runs; search; sort column and direction;
  column picker; statistic per configuration (median / mean / min / max);
  three plot views (by configuration, scatter with a Pareto front, empirical
  distribution); axis pickers; Pareto objectives (minimise / maximise); the
  Pareto front as a selection.
* **Time series** — overlay selected runs *or* configurations (aggregated with
  the median or the mean, with a min–max or IQR band); time alignment (elapsed
  or wall clock); signal picker.
* **Run details** — the parameters, the metrics and the recorded dispatch of one
  run, with a jump into the overlay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..study import METRICS, aggregate, ecdf, pareto_front
from . import data, theme

STUDIES = {
    "optuna-search": {
        "label": "Optuna search · dispatch policy",
        "what": "A multi-objective Optuna study (cost × CO₂) over the merit-order policy, "
                "two noise realisations per trial.",
    },
    "dispatch-comparison": {
        "label": "Dispatch comparison · demand realisations",
        "what": "Seven policies, three demand levels: the same rule on different days.",
    },
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _metric_label(key: str) -> str:
    info = METRICS.get(key, {})
    unit = info.get("unit")
    return f"{info.get('label', key)}" + (f" ({unit})" if unit else "")


def _runs_frame(study: dict) -> pd.DataFrame:
    rows = []
    for run in study["runs"]:
        row = {"run": run["key"], "configuration": run["configuration_name"]}
        row.update({p: run["parameters"].get(p) for p in study["parameters"]})
        row.update({m: run.get(m) for m in study["metrics"]})
        rows.append(row)
    return pd.DataFrame(rows)


def _config_frame(study: dict, stat: str, columns: list[str]) -> pd.DataFrame:
    rows = []
    for config in study["configurations"]:
        row = {"configuration": config["name"], "runs": len(config["runs"])}
        row.update({p: config["parameters"].get(p) for p in study["parameters"]})
        for metric in columns:
            row[metric] = aggregate(study["runs"], config["key"], metric, stat)
        rows.append(row)
    return pd.DataFrame(rows)


def _pareto_records(rows: pd.DataFrame, x: str, y: str) -> list[dict]:
    """The rows of a table as points the Pareto routine can compare."""
    return [{"key": label, x: row[x], y: row[y]}
            for label, (_, row) in zip(rows.index, rows.iterrows())]


def _pareto_chart(study: dict, rows: pd.DataFrame, x: str, y: str,
                  x_sense: str, y_sense: str, show_front: bool) -> tuple[go.Figure, list[str]]:
    records = _pareto_records(rows, x, y)
    keys = pareto_front(records, x, y, x_sense, y_sense) if show_front else []
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=rows[x], y=rows[y], mode="markers", name="configurations",
        text=rows.index, marker=dict(size=9, color="#38bdf8", opacity=0.75,
                                     line=dict(width=0.5, color="#0b1220")),
        hovertemplate="%{text}<br>x=%{x:.3g}<br>y=%{y:.3g}<extra></extra>"))
    if keys:
        front = rows.loc[rows.index.intersection(keys)]
        figure.add_trace(go.Scatter(
            x=front[x], y=front[y], mode="markers", name="Pareto front",
            marker=dict(size=14, color="#fbbf24", symbol="diamond",
                        line=dict(width=1, color="#0b1220")),
            hovertemplate="%{text}<br>Pareto<extra></extra>"))
    theme.figure(figure, height=360, x_title=_metric_label(x), y_title=_metric_label(y))
    return figure, keys


def _ecdf_chart(study: dict, rows: pd.DataFrame, metric: str) -> go.Figure:
    figure = go.Figure()
    groups = rows.groupby("configuration")[metric]
    if groups.ngroups <= 8:
        for name, series in groups:
            x, y = ecdf(series.tolist())
            if len(x):
                figure.add_trace(go.Scatter(x=x, y=y, mode="lines", name=str(name)))
    else:
        x, y = ecdf(rows[metric].tolist())
        figure.add_trace(go.Scatter(x=x, y=y, mode="lines", name="all runs"))
    theme.figure(figure, height=340, x_title=_metric_label(metric), y_title="fraction of runs")
    figure.update_yaxes(range=[0, 1.02])
    return figure


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #

def _results(study: dict, case_key: str) -> None:
    frame = _runs_frame(study)
    theme.section("Results", "the same tables and plots as the Vessim viewer", "VIEWER")

    top = st.columns([1.1, 1, 1, 1])
    mode = top[0].radio("Rows", ["configurations", "runs"], horizontal=True,
                        key=f"study_mode_{case_key}")
    query = top[1].text_input("Search", "", placeholder="name, parameter…",
                             key=f"study_query_{case_key}")
    stat = top[2].selectbox("Statistic", ["median", "mean", "min", "max"], index=0,
                            key=f"study_stat_{case_key}",
                            disabled=mode == "runs")
    plot_type = top[3].selectbox("Plot view",
                                 ["By configuration", "Scatter plot", "Distribution"],
                                 key=f"study_plottype_{case_key}")

    if query:
        mask = frame.astype(str).apply(
            lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        frame = frame[mask]

    if mode == "runs":
        columns = st.multiselect("Columns", study["parameters"] + study["metrics"],
                                 default=study["parameters"][:2] + ["bill_usd", "co2_kg"],
                                 format_func=_metric_label, key=f"study_cols_runs_{case_key}")
        view = frame[["run", "configuration"] + [c for c in columns if c in frame.columns]].copy()
        sort = st.selectbox("Sort by", ["run", "configuration"] + columns, key=f"study_sort_runs_{case_key}")
        descending = st.toggle("Descending", value=False, key=f"study_desc_runs_{case_key}")
        view = view.sort_values(sort, ascending=not descending, na_position="last")
        st.dataframe(view.round(3), hide_index=True, **theme.wide(), height=330)
        st.caption(f"{len(view)} runs · each row is one Vessim execution.")
        frame_for_plot = frame
    else:
        metrics = st.multiselect("Metrics", study["metrics"],
                                 default=["bill_usd", "co2_kg", "gas_mwh"],
                                 format_func=_metric_label, key=f"study_cols_cfg_{case_key}")
        table = _config_frame(study, stat, metrics)
        show_params = st.multiselect("Parameters", study["parameters"],
                                     default=study["parameters"][:3],
                                     key=f"study_params_cfg_{case_key}")
        view = table[["configuration", "runs"] + show_params + metrics]
        sort = st.selectbox("Sort by", ["configuration"] + metrics + show_params,
                            key=f"study_sort_cfg_{case_key}")
        descending = st.toggle("Descending", value=False, key=f"study_desc_cfg_{case_key}")
        selected_first = st.toggle("Selected first", value=False,
                                   key=f"study_first_cfg_{case_key}")
        view = view.sort_values(sort, ascending=not descending, na_position="last")
        st.dataframe(view.round(3), hide_index=True, **theme.wide(), height=330)
        st.caption(f"{len(table)} configurations × {len(study['runs']) // max(len(table), 1)} runs · "
                   f"statistic: {stat}.")
        frame_for_plot = table

    if plot_type == "Scatter plot":
        left, right, third, fourth = st.columns([1, 1, 1, 1])
        x = left.selectbox("Horizontal axis", study["metrics"], index=0,
                           format_func=_metric_label, key=f"study_x_{case_key}")
        y = right.selectbox("Vertical axis", study["metrics"], index=1,
                            format_func=_metric_label, key=f"study_y_{case_key}")
        show_front = third.toggle("Pareto front", value=True, key=f"study_pareto_{case_key}")
        x_sense = fourth.selectbox("Horizontal objective", ["min", "max"], key=f"study_xs_{case_key}")
        y_sense = st.selectbox("Vertical objective", ["min", "max"], index=0,
                               key=f"study_ys_{case_key}")
        chart_rows = frame_for_plot.reset_index(drop=True)
        chart_rows.index = chart_rows["configuration"] if "configuration" in chart_rows else chart_rows.index
        figure, front = _pareto_chart(study, chart_rows, x, y, x_sense, y_sense, show_front)
        theme.show(figure)
        if front:
            st.caption(f"{len(front)} non-dominated {'configuration' if mode == 'configurations' else 'run'}"
                       "(s): " + ", ".join(str(f) for f in front[:12]))
    elif plot_type == "Distribution":
        metric = st.selectbox("Metric", study["metrics"], index=0, format_func=_metric_label,
                              key=f"study_ecdf_{case_key}")
        theme.show(_ecdf_chart(study, frame_for_plot if mode == "configurations" else frame,
                               metric))
        st.caption("Empirical distribution of the selected metric across runs; bands are not "
                   "confidence intervals.")


def _time_series(study: dict, case_key: str) -> None:
    theme.section("Time series", "overlay the recordings of the study", "VIEWER")
    controls = st.columns([1.2, 1, 1, 1])
    mode = controls[0].radio("Overlay", ["configurations", "runs"], horizontal=True,
                             key=f"study_ts_mode_{case_key}")
    alignment = controls[1].selectbox("Time alignment",
                                      ["Elapsed from start (h)", "Recorded time"],
                                      key=f"study_ts_align_{case_key}")
    stat = controls[2].selectbox("Aggregate", ["median", "mean"], key=f"study_ts_stat_{case_key}",
                                 disabled=mode == "runs")
    spread = controls[3].selectbox("Band", ["min–max", "IQR", "none"],
                                   key=f"study_ts_spread_{case_key}", disabled=mode == "runs")

    if mode == "configurations":
        options = {c["key"]: c["name"] for c in study["configurations"]}
        chosen = st.multiselect("Configurations", list(options), format_func=options.get,
                                default=list(options)[:2], key=f"study_ts_cfg_{case_key}")
    else:
        options = {r["key"]: f"{r['key']} · {r['configuration_name']}" for r in study["runs"]}
        chosen = st.multiselect("Runs", list(options), format_func=options.get,
                                default=list(options)[:2], key=f"study_ts_run_{case_key}")

    signals = st.multiselect(
        "Signals", ["load_mw", "solar_mw", "grid_mw", "battery_mw", "gas_mw", "soc_mwh",
                    "carbon_kg_mwh", "price_usd_mwh"],
        default=["load_mw", "grid_mw", "battery_mw"], key=f"study_ts_sig_{case_key}")

    if not chosen or not signals:
        st.info("Choose at least one configuration or run and one signal to plot recorded "
                "trajectories.")
        return

    band_signal = signals[0]
    if mode == "configurations" and spread != "none":
        band_signal = st.selectbox("Signal that gets the band", signals, index=0,
                                   key=f"study_ts_band_{case_key}")

    traces = study["traces"]
    series: dict[str, pd.DataFrame | pd.Series] = {}
    if mode == "runs":
        series = {key: traces[key] for key in chosen if key in traces}
    else:
        for config_key in chosen:
            config = next(c for c in study["configurations"] if c["key"] == config_key)
            frames = [traces[r] for r in config["runs"] if r in traces]
            if frames:
                series[config["name"]] = pd.concat(frames, keys=range(len(frames)))

    def _x(index: pd.DatetimeIndex) -> np.ndarray:
        if alignment.startswith("Elapsed"):
            return (index - index[0]).total_seconds() / 3600.0
        return index

    figure = go.Figure()
    for label, frame in series.items():
        if mode == "runs":
            x = _x(frame.index)
            for signal in signals:
                figure.add_trace(go.Scatter(
                    x=x, y=frame[signal], mode="lines",
                    name=f"{label} · {signal}" if len(signals) > 1 else label))
            continue
        # configurations: aggregate the runs of the configuration
        index = frame[["load_mw"]].groupby(level=1).head(1).index.get_level_values(1)
        x = _x(index)
        grouped = frame.groupby(level=1)
        for signal in signals:
            values = grouped[signal]
            middle = values.median() if stat == "median" else values.mean()
            figure.add_trace(go.Scatter(x=x, y=middle.to_numpy(float), mode="lines",
                                        name=f"{label} · {signal}" if len(signals) > 1 else label))
            if signal != band_signal:
                continue
            if spread == "min–max":
                low, high = values.min(), values.max()
            else:                                    # interquartile range
                low, high = values.quantile(0.25), values.quantile(0.75)
            figure.add_trace(go.Scatter(
                x=list(x) + list(x)[::-1], y=list(high) + list(low)[::-1],
                fill="toself", mode="lines", line=dict(width=0), opacity=0.18,
                name=f"{label} · {spread}", showlegend=False, hoverinfo="skip"))

    theme.figure(figure, height=400, x_title="h from start" if alignment.startswith("Elapsed")
                 else "recorded time", y_title="mixed units")
    theme.show(figure)
    st.caption("Signals come from Vessim's `Recorder`; MW, MWh, kg/MWh and USD/MWh share the "
               "axis, so read the legend. " + (
                   f"Lines are the {stat} of each configuration's runs; the shaded band is the "
                   f"{spread} of **{band_signal}** — the observed spread of the realisations, not "
                   "a confidence interval." if mode == "configurations" else
                   "Each line is one run."))

    table = pd.DataFrame({label: (frame.mean(numeric_only=True) if mode == "runs"
                                  else frame.groupby(level=1).median().mean())
                          for label, frame in series.items()})
    with st.expander("⬇️ Take the overlay away as CSV", expanded=False):
        st.dataframe(table.round(3), **theme.wide(), height=240)
        st.download_button("Download the per-signal means (CSV)", table.to_csv().encode(),
                           file_name=f"{study['study_id']}_{case_key}_overlay.csv",
                           mime="text/csv", **theme.wide(),
                           key=f"study_ts_csv_{case_key}")


def _run_details(study: dict, case_key: str) -> None:
    theme.section("Run details", "one execution, its parameters and its recording", "VIEWER")
    options = {r["key"]: f"{r['key']} · {r['configuration_name']}" for r in study["runs"]}
    key = st.selectbox("Run", list(options), format_func=options.get, index=0,
                       key=f"study_run_{case_key}")
    run = next(r for r in study["runs"] if r["key"] == key)
    left, right = st.columns(2)
    with left:
        st.markdown("**Parameters**")
        st.dataframe(pd.DataFrame({"parameter": list(run["parameters"]),
                                   "value": [str(v) for v in run["parameters"].values()]}),
                     hide_index=True, **theme.wide(), height=250)
    with right:
        st.markdown("**Metrics**")
        st.dataframe(pd.DataFrame({
            "metric": [_metric_label(m) for m in study["metrics"]],
            "value": [round(float(run[m]), 3) for m in study["metrics"]],
        }), hide_index=True, **theme.wide(), height=250)

    frame = study["traces"].get(key)
    if frame is not None:
        figure = go.Figure()
        for signal, colour in (("load_mw", "#f8fafc"), ("solar_mw", "#fbbf24"),
                               ("grid_mw", "#38bdf8"), ("battery_mw", "#a78bfa")):
            figure.add_trace(go.Scatter(x=frame.index, y=frame[signal], name=signal,
                                        line=dict(color=colour, width=1.8)))
        theme.show(theme.figure(figure, height=320, y_title="MW"))
        st.caption(f"Recording of {key}: {len(frame)} steps.")


def render() -> None:
    theme.hero(
        "📊", "Studies, the Vessim viewer way", "RESULTS · TIME SERIES · RUN DETAILS",
        "The official viewer opens a directory of recorded runs and lets you compare "
        "configurations and realisations: tables, distributions, scatter plots with a Pareto "
        "front, overlays and run details. This page generates real studies — an Optuna search over "
        "the dispatch policy and a policy comparison across demand realisations — and gives you "
        "the same three pages and the same options, over live simulations.",
        [("Studies", "2"), ("Pages", "3"), ("Solver", "Optuna TPE"), ("Source", "Vessim viewer")],
    )

    theme.section("The study", "the unit the viewer indexes everything by", "STUDY")
    columns = st.columns([1.4, 1])
    study_id = columns[0].selectbox("Study", list(STUDIES), format_func=lambda k: STUDIES[k]["label"],
                                    key="study_id")
    case_options = {"ecopetrol": "05 · Ecopetrol industrial",
                    "carbon_aware": "03 · Carbon-aware workload",
                    "solar_battery": "02 · Solar + battery"}
    case_key = columns[1].selectbox("Case", list(case_options), format_func=case_options.get,
                                    key="study_case_key")
    st.caption(STUDIES[study_id]["what"])

    study = data.study_case(study_id, case_key)
    best = study.get("best")
    kpis = [
        ("Configurations", f"{len(study['configurations'])}", "",
         f"{len(study['runs'])} runs", ""),
        ("Parameters", f"{len(study['parameters'])}", "",
         ", ".join(study["parameters"][:3]) + ("…" if len(study["parameters"]) > 3 else ""), ""),
        ("Metrics", f"{len(study['metrics'])}", "", "cost, CO₂, gas, grid, peak…", ""),
    ]
    if best:
        kpis.append(("Best trial", f"#{best['trial']}", "",
                     f"{best['bill_usd']:,.2f} USD · {best['co2_kg']:,.1f} kg", "good"))
    theme.kpis(kpis)

    tab_results, tab_time, tab_run = st.tabs(["📋 Results", "📈 Time series", "🔍 Run details"])
    with tab_results:
        _results(study, case_key)
    with tab_time:
        _time_series(study, case_key)
    with tab_run:
        _run_details(study, case_key)

    theme.note(
        "Everything here is a real simulation: each run is a <code>vessim.Environment</code> with "
        "actors, dispatchables, a policy and a <code>Recorder</code>. The Optuna study asks the "
        "sampler for a policy configuration, runs it twice with different signal noise, and keeps "
        "the pair (cost, CO₂) — which is why the scatter plot has a Pareto front instead of a "
        "winner. That is the same structure the official viewer reads from "
        "<code>results/study</code>, reproduced live."
    )
