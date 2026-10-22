"""Vessim ↔ PyPSA — interactive dashboard.

Entry point for Streamlit Community Cloud (``Main file: app.py``) and for local
runs (``streamlit run app.py``).  The importable library lives under ``src/``;
this file only wires it to the page.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import importlib.metadata as metadata

import streamlit as st

st.set_page_config(
    page_title="Vessim ↔ PyPSA · microgrid lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from vessim_pypsa_dashboard.ui import (  # noqa: E402
    blindspots,
    data,
    decision,
    explain,
    hil,
    maths,
    method,
    optimal,
    overview,
    policy_guide,
    policy_lab,
    roadmap_view,
    study,
    theme,
    view_educ,
    view_integrated,
    view_loadflow,
    view_papers,
    view_vessim,
)
from vessim_pypsa_dashboard.vessim_runner import CASES, CASE_ORDER  # noqa: E402

theme.inject_css()

NAV_OVERVIEW = "🏠 Overview"
NAV_METHOD = "🧭 Method"
NAV_DECISION = "🧩 When to use what"
NAV_EXPLAIN = "🕶️ Blind spots & policies"
NAV_STUDY = "📊 Studies · viewer"
NAV_PAPERS = "📄 Papers · test systems"
NAV_HIL = "🔌 Hardware in the loop"
CASE_LABELS = {key: f"{CASES[key].number} · {CASES[key].title}" for key in CASE_ORDER}
NAV_LABELS = {NAV_OVERVIEW: NAV_OVERVIEW, NAV_METHOD: NAV_METHOD,
              NAV_DECISION: NAV_DECISION, NAV_EXPLAIN: NAV_EXPLAIN,
              NAV_HIL: NAV_HIL, NAV_STUDY: NAV_STUDY,
              NAV_PAPERS: NAV_PAPERS, **CASE_LABELS}

#: The sidebar is two levels: a section, then a page inside it.  Fourteen flat
#: entries were the reason the app felt like a maze; four groups of two to five
#: is a table of contents.
SECTIONS: dict[str, list[str]] = {
    "🏠 Start here": [NAV_OVERVIEW, NAV_METHOD],
    "🔬 The six cases": list(CASE_ORDER),
    "🧭 Guides": [NAV_DECISION, NAV_EXPLAIN, NAV_HIL],
    "📊 Studies & papers": [NAV_STUDY, NAV_PAPERS],
}

def _section_of(page: str) -> str:
    for section, pages in SECTIONS.items():
        if page in pages:
            return section
    return next(iter(SECTIONS))

# a button on the landing page can move the navigation before the radios render
if "pending_nav" in st.session_state:
    st.session_state["nav"] = st.session_state.pop("pending_nav")
st.session_state.setdefault("nav", NAV_OVERVIEW)
if st.session_state.get("nav") not in NAV_LABELS:
    st.session_state["nav"] = NAV_OVERVIEW
# the section widget owns its own state, so it has to be moved when the page is
# changed from code (the "Open case" buttons) rather than by the reader
_expected_section = _section_of(st.session_state["nav"])
if st.session_state.get("section") != _expected_section:
    st.session_state["section"] = _expected_section

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:.55rem; margin-bottom:.2rem">
          <div style="font-size:1.65rem">⚡</div>
          <div>
            <div style="font-weight:800; font-size:1.02rem; color:#eaf4ff; letter-spacing:-.01em">
              Vessim ↔ PyPSA</div>
            <div style="font-size:.72rem; color:#8ea3c7; letter-spacing:.06em">
              MICROGRID LABORATORY</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    section = st.radio("Section", list(SECTIONS), key="section",
                       label_visibility="collapsed")
    pages = SECTIONS[section]
    if st.session_state.get("nav") not in pages:      # the section was just changed
        st.session_state["nav"] = pages[0]
    nav = st.radio("Page", pages, format_func=NAV_LABELS.get, key="nav",
                   label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)

    if nav in CASE_ORDER:
        case_nav = nav
        spec = CASES[case_nav]
        st.markdown('<div class="label" style="font-size:.68rem; letter-spacing:.1em; '
                    'text-transform:uppercase; color:#8ea3c7">Shared signals</div>',
                    unsafe_allow_html=True)
        carbon = st.slider("Carbon price (USD/t)", 0, 300, int(spec.carbon_price), 10,
                           key=f"sb_carbon_{case_nav}",
                           help="Applied to Vessim's merit-order policy and to PyPSA's marginal cost")
        solar_scale = st.slider("Solar scale", 0.0, 1.8, 1.0, 0.05, key=f"sb_solar_{case_nav}",
                                help="Multiplies the photovoltaic production trace")
        load_scale = st.slider("Demand scale", 0.5, 1.5, 1.0, 0.05, key=f"sb_load_{case_nav}")
        if st.button("↺ Case defaults", **theme.wide(), key=f"sb_reset_{case_nav}"):
            for key in ("sb_carbon", "sb_solar", "sb_load"):
                st.session_state.pop(f"{key}_{case_nav}", None)
            st.rerun()
    else:
        carbon, solar_scale, load_scale = float(CASES[CASE_ORDER[0]].carbon_price), 1.0, 1.0

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="small">Every case opens four views: <b>Vessim only</b> — with the run, the '
        'policies and the optimum as sheets — load flow, dispatch/commitment, and the PyPSA OPF '
        'fed by Vessim; case <b>06</b> adds a fifth, <b>🎲 Roadmap</b>, for the four items on '
        'Vessim\'s published roadmap. Guides: <b>🧩 When to use what</b>, <b>🕶️ Blind spots & policies</b> and '
        '<b>🔌 Hardware in the loop</b>. The maths sits inside each tab, in a 🧮 expander next to '
        'the number it explains.</div>',
        unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.session_state.setdefault(theme.DENSITY_KEY, "Standard")
    st.radio("Density", theme.DENSITY_LEVELS, key=theme.DENSITY_KEY, horizontal=True,
             help="Focus folds long explanations behind a “▸ Read the rest” link; Full shows every "
                  "paragraph at once. No text is ever removed, only folded.")
    st.markdown(
        '<div class="small" style="margin-top:-.35rem">Nothing is deleted at any level: '
        '<b>Focus</b> keeps the headline and folds the reasoning, <b>Full</b> unfolds everything.</div>',
        unsafe_allow_html=True)

    with st.expander("⚙️ Versions", expanded=False):
        rows = []
        for package in ("vessim", "pypsa", "streamlit", "pandas", "plotly", "highspy"):
            try:
                rows.append(f"{package} {metadata.version(package)}")
            except Exception:
                rows.append(f"{package} —")
        st.code("\n".join(rows), language="text")


# --------------------------------------------------------------------------- #
# Safe rendering
# --------------------------------------------------------------------------- #

def _safe(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # keep one broken view from taking down the page
        theme.note(f"<b>This view failed:</b> {type(exc).__name__}: {exc}", "bad")
        with st.expander("Technical detail", expanded=False):
            st.exception(exc)


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #

if nav == NAV_OVERVIEW:
    overview.render()
elif nav == NAV_METHOD:
    method.render()
elif nav == NAV_DECISION:
    decision.render()
elif nav == NAV_EXPLAIN:
    _safe(explain.render)
elif nav == NAV_STUDY:
    study.render()
elif nav == NAV_HIL:
    hil.render()
elif nav == NAV_PAPERS:
    _safe(view_papers.render)
else:
    spec = CASES[nav]
    theme.hero(
        spec.number,
        spec.title,
        spec.subtitle.upper(),
        spec.description,
        [("Origen", spec.source), ("Pasos", f"{spec.steps}"),
         ("Δt", f"{spec.step_s/60:.0f} min"), ("Horizonte", f"{spec.horizon_hours:.0f} h"),
         ("CO₂", f"{carbon:.0f} USD/t")],
    )

    with st.spinner("Running Vessim…" if spec.key != "sil" else "Running the SiL case (launches a real process)…"):
        run = data.vessim_case(nav, float(carbon), float(solar_scale), float(load_scale))

    labels = [
        "🔮 Vessim only",
        "⚡ LoadFlow · AdjieBrotot",
        "🏭 ED-UC · AdjieBrotot",
        "🔗 Vessim + PyPSA",
    ]
    if spec.key == "montecarlo":          # the roadmap case brings its own fifth tab
        labels.append("🎲 Roadmap · MC · SAM · fade")
    tabs = st.tabs(labels)
    tab_vessim, tab_lf, tab_educ, tab_pypsa = tabs[:4]

    with tab_vessim:
        sheet_run, sheet_policy, sheet_optimal = st.tabs([
            "▶️ The run",
            "🎛️ Policies",
            "🏁 Follow the optimum",
        ])
        with sheet_run:
            _safe(view_vessim.render, nav, run, spec, float(carbon))
        with sheet_policy:
            _safe(policy_lab.current_policy_block, run, spec, nav, float(carbon))
            _safe(policy_lab.scenario_block, nav, run, spec, float(carbon),
                  float(solar_scale), float(load_scale))
        with sheet_optimal:
            _safe(optimal.render, nav, run, spec, float(carbon))
    with tab_lf:
        _safe(view_loadflow.render, nav, run, spec, float(carbon))
    with tab_educ:
        _safe(view_educ.render, nav, run, spec, float(carbon))
    with tab_pypsa:
        _safe(view_integrated.render, nav, run, spec, float(carbon))
    if spec.key == "montecarlo":
        with tabs[4]:
            _safe(roadmap_view.render, nav, run, spec, float(carbon))


    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="small" style="display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap">'
        '<span><b>Vessim</b> brings the simulation and the signals · <b>PyPSA</b> brings the optimum '
        'with a network · the two <b>AdjieBrotot</b> tools bring the manual method, the editable '
        'sheets and the diagram vocabulary.</span>'
        '<span>MIT · teaching and research use</span></div>',
        unsafe_allow_html=True)
