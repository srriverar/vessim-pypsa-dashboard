"""Visual system of the dashboard: CSS, KPI cards, Plotly theme.

Everything here is presentation only.  The design goal is a "control room"
look: dense, readable, dark, with a single accent colour per concept
(Vessim = violet, LoadFlow = cyan, ED/UC = amber, PyPSA = green).
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import plotly.graph_objects as go
import streamlit as st
from streamlit import __version__ as _st_version

# --------------------------------------------------------------------------- #
# Colour language (shared by every view)
# --------------------------------------------------------------------------- #

COLORS = {
    "vessim": "#a78bfa",
    "loadflow": "#38bdf8",
    "educ": "#fbbf24",
    "pypsa": "#34d399",
    "grid": "#f472b6",
    "solar": "#facc15",
    "gas": "#fb923c",
    "battery": "#22d3ee",
    "load": "#e2e8f0",
    "export": "#818cf8",
    "curtail": "#94a3b8",
    "red": "#f87171",
    "green": "#34d399",
    "amber": "#fbbf24",
    "sky": "#38bdf8",
    "muted": "#93a4c4",
}

CARRIER_COLORS = {
    "coal": "#64748b",
    "gas": "#fb923c",
    "diesel": "#f97316",
    "solar": "#facc15",
    "wind": "#7dd3fc",
    "hydro": "#38bdf8",
    "battery": "#22d3ee",
    "grid": "#f472b6",
    "nuclear": "#c084fc",
    "": "#94a3b8",
}

CSS = """
<style>
/* ---------------------------------------------------------------- layout */
.block-container {padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1460px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(56,189,248,.14);}
[data-testid="stSidebar"] .block-container {padding-top: 1.2rem;}

/* ------------------------------------------------------------------ hero */
.hero {
  position: relative;
  background:
    radial-gradient(900px 260px at 8% -40%, rgba(56,189,248,.22), transparent 60%),
    radial-gradient(700px 240px at 100% 0%, rgba(167,139,250,.16), transparent 62%),
    linear-gradient(135deg, #0f1b30 0%, #0b1424 70%);
  border: 1px solid #22314f;
  border-radius: 18px;
  padding: 1.05rem 1.25rem 1.1rem;
  margin-bottom: 1rem;
}
.hero .eyebrow {font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; color:#7dd3fc; font-weight:600;}
.hero h1 {font-size:1.72rem; margin:.2rem 0 .15rem; color:#eef6ff; letter-spacing:-.01em;}
.hero p {color:#a9bcdb; margin:0; font-size:.92rem; line-height:1.55; max-width: 82ch;}
.hero .num {
  position:absolute; top:1rem; right:1.15rem; font-size:.8rem; font-weight:700;
  color:#7dd3fc; background:rgba(56,189,248,.10); border:1px solid rgba(56,189,248,.35);
  padding:.18rem .55rem; border-radius:999px; letter-spacing:.08em;
}

/* ----------------------------------------------------------------- chips */
.chips {display:flex; flex-wrap:wrap; gap:.35rem; margin-top:.7rem;}
.chip {
  font-size:.72rem; color:#b9cbe8; background:#13203a; border:1px solid #24365a;
  padding:.18rem .55rem; border-radius:999px;
}
.chip b {color:#e8f1ff; font-weight:600;}

/* ------------------------------------------------------------------ KPIs */
.kpi-grid {display:grid; grid-template-columns:repeat(auto-fit, minmax(168px, 1fr)); gap:.7rem; margin:.35rem 0 1.4rem;}
.kpi {
  background:linear-gradient(160deg,#12203a 0%, #0d1629 100%);
  border:1px solid #22314f; border-radius:14px; padding:.75rem .9rem .8rem;
}
.kpi .label {font-size:.68rem; letter-spacing:.09em; text-transform:uppercase; color:#8ea3c7; font-weight:600;}
.kpi .value {font-size:1.42rem; font-weight:700; color:#eef6ff; line-height:1.18; margin-top:.12rem;}
.kpi .unit {font-size:.78rem; font-weight:500; color:#9db1d1; margin-left:.18rem;}
.kpi .sub {font-size:.72rem; color:#7f92b4; margin-top:.05rem;}
.kpi.good {border-color:rgba(52,211,153,.45);} .kpi.good .value {color:#6ee7b7;}
.kpi.bad  {border-color:rgba(248,113,113,.45);} .kpi.bad .value {color:#fca5a5;}
.kpi.accent{border-color:rgba(56,189,248,.45);} .kpi.accent .value {color:#7dd3fc;}
.kpi.warn {border-color:rgba(251,191,36,.45);} .kpi.warn .value {color:#fcd34d;}

/* -------------------------------------------------------------- pipeline */
.pipe {display:flex; align-items:stretch; gap:.35rem; margin:.2rem 0 1.1rem; flex-wrap:wrap;}
.pipe-step {
  flex:1 1 180px; background:linear-gradient(160deg,#111d33 0%, #0c1526 100%);
  border:1px solid #22314f; border-radius:12px; padding:.55rem .7rem;
}
.pipe-step .k {font-size:.67rem; letter-spacing:.12em; text-transform:uppercase; color:#7dd3fc; font-weight:700;}
.pipe-step .d {font-size:.78rem; color:#a9bcdb; line-height:1.45; margin-top:.15rem;}
.pipe-arrow {align-self:center; color:#3b5478; font-size:1.05rem; font-weight:700;}

/* --------------------------------------------------------------- section */
.section {display:flex; align-items:baseline; gap:.6rem; margin:1.7rem 0 .7rem; flex-wrap:wrap;}
.section:first-child {margin-top:.4rem;}
.section .tag {font-size:.68rem; letter-spacing:.14em; text-transform:uppercase; color:#7dd3fc; font-weight:700;}
.section .title {font-size:1.08rem; font-weight:700; color:#e9f2ff;}
.section .sub {font-size:.82rem; color:#8ea3c7; max-width:78ch;}

/* ----------------------------------------------------------------- notes */
.note {
  border-left:3px solid #38bdf8; background:rgba(56,189,248,.07);
  padding:.7rem 1rem; border-radius:0 10px 10px 0; color:#b9cbe8; font-size:.85rem;
  line-height:1.6; margin:.4rem 0 .9rem; max-width:100ch;
}
.note.warn {border-left-color:#fbbf24; background:rgba(251,191,36,.07);}
.note.good {border-left-color:#34d399; background:rgba(52,211,153,.07);}
.note.bad  {border-left-color:#f87171; background:rgba(248,113,113,.07);}
.note b {color:#e8f1ff;}

/* ----------------------------------------------------------------- misc */
.mono {font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.83rem;}
.small {font-size:.78rem; color:#8ea3c7; line-height:1.5;}
.stCaption, [data-testid="stCaptionContainer"] {color:#8ea3c7; line-height:1.5;}
.pill {display:inline-block; font-size:.7rem; padding:.1rem .5rem; border-radius:999px;
       border:1px solid #2b3f66; color:#b9cbe8; background:#101d36;}
.pill.ok {border-color:rgba(52,211,153,.5); color:#6ee7b7;}
.pill.bad{border-color:rgba(248,113,113,.5); color:#fca5a5;}
hr {border-color:#1c2b47;}

/* ----------------------------------------------------------------- tabs */
.stTabs [data-baseweb="tab-list"] {gap:.25rem; border-bottom:1px solid #1e2d4b;}
.stTabs [data-baseweb="tab"] {
  height:2.55rem; padding:0 .9rem; border-radius:10px 10px 0 0;
  color:#9db1d1; font-size:.9rem; font-weight:600;
}
.stTabs [aria-selected="true"] {color:#eaf4ff; background:rgba(56,189,248,.10);}

/* buttons */
.stButton>button, .stDownloadButton>button {
  border-radius:10px; border:1px solid #2a3f66; font-weight:600;
}
div[data-testid="stMetricValue"] {font-size:1.35rem;}

/* dataframe / editor */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {border-radius:12px; overflow:hidden;}

/* hide streamlit chrome a bit */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
</style>
"""


def wide() -> dict:
    """``width='stretch'`` on new Streamlit, ``use_container_width`` on old ones."""
    try:
        major, minor = (int(part) for part in _st_version.split(".")[:2])
    except Exception:  # pragma: no cover
        return {"use_container_width": True}
    return {"width": "stretch"} if (major, minor) >= (1, 49) else {"use_container_width": True}


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Density: how much prose is unfolded at once
# --------------------------------------------------------------------------- #

DENSITY_KEY = "density"
DENSITY_LEVELS = ("Focus", "Standard", "Full")

#: characters of prose a note may occupy before the rest is folded away
DENSITY_BUDGET = {"Focus": 280, "Standard": 460, "Full": 10 ** 9}
#: how many sentences to keep when folding, at least
DENSITY_KEEP = {"Focus": 1, "Standard": 2, "Full": 99}

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z<“(])")


def density() -> str:
    """The reader's chosen density.  ``Focus`` folds, it never removes."""
    level = st.session_state.get(DENSITY_KEY, "Standard")
    return level if level in DENSITY_LEVELS else "Standard"


def _split_prose(text: str) -> tuple[str, str] | None:
    """Cut a run of HTML prose after a sentence, without breaking a tag."""
    budget = DENSITY_BUDGET[density()]
    if len(text) <= budget:
        return None
    cuts = [match.start() for match in _SENTENCE.finditer(text)]
    usable = [cut for cut in cuts if 80 <= cut <= budget]
    if not usable:                      # one long sentence: cut on a tag boundary
        usable = [pos for pos in (text.rfind("</b>.", 0, budget), text.rfind(". ", 0, budget))
                  if pos > 80]
    if not usable:
        return None
    cut = max(usable)
    return text[:cut + 1], text[cut + 1:].strip()


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

def hero(number: str, title: str, subtitle: str, description: str,
         chips: Sequence[tuple[str, str]] = ()) -> None:
    chip_html = "".join(
        f'<span class="chip">{label} <b>{value}</b></span>' for label, value in chips
    )
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">{subtitle}</div>
          <h1>{title}</h1>
          <p>{description}</p>
          <div class="chips">{chip_html}</div>
          <div class="num">{number}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, sub: str = "", tag: str = "") -> None:
    st.markdown(
        f"""<div class="section">
              <span class="tag">{tag}</span>
              <span class="title">{title}</span>
              <span class="sub">{sub}</span>
            </div>""",
        unsafe_allow_html=True,
    )


def kpis(items: Iterable[tuple]) -> None:
    """Render KPI cards.

    Each item is ``(label, value, unit, sub, tone)`` with the last two optional.
    """
    cards = []
    for item in items:
        label, value = item[0], item[1]
        unit = item[2] if len(item) > 2 else ""
        sub = item[3] if len(item) > 3 else ""
        tone = item[4] if len(item) > 4 else ""
        unit_html = f'<span class="unit">{unit}</span>' if unit else ""
        sub_html = f'<div class="sub">{sub}</div>' if sub else ""
        cards.append(
            f'<div class="kpi {tone}"><div class="label">{label}</div>'
            f'<div class="value">{value}{unit_html}</div>{sub_html}</div>'
        )
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def pipeline(steps: Sequence[tuple[str, str]]) -> None:
    """One row that says: what comes in → what this sheet does → what leaves.

    Every sheet of a case starts with it, because the pipeline is the question
    a newcomer actually asks ("where does this number come from?") and the
    answer is different in each of the four tabs.
    """
    cells = []
    for index, (label, detail) in enumerate(steps):
        cells.append(f'<div class="pipe-step"><div class="k">{label}</div>'
                     f'<div class="d">{detail}</div></div>')
        if index < len(steps) - 1:
            cells.append('<div class="pipe-arrow">→</div>')
    st.markdown(f'<div class="pipe">{"".join(cells)}</div>', unsafe_allow_html=True)


def note(text: str, tone: str = "") -> None:
    """A highlighted explanation.

    At the reader's density level, a long note keeps its first sentences and
    folds the rest into one click — the text is never shortened, only folded,
    and ``Full`` shows everything at once.
    """
    parts = _split_prose(text)
    if parts is None:
        st.markdown(f'<div class="note {tone}">{text}</div>', unsafe_allow_html=True)
        return
    lead, rest = parts
    st.markdown(f'<div class="note {tone}">{lead}</div>', unsafe_allow_html=True)
    with st.expander("▸ Read the rest", expanded=False):
        st.markdown(f'<div class="note {tone}" style="margin:.1rem 0 .2rem">{rest}</div>',
                    unsafe_allow_html=True)


def spacer(height: int = 8) -> None:
    st.markdown(f'<div style="height:{height}px"></div>', unsafe_allow_html=True)


def pill(text: str, tone: str = "") -> str:
    return f'<span class="pill {tone}">{text}</span>'


# --------------------------------------------------------------------------- #
# Plotly helpers
# --------------------------------------------------------------------------- #

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color="#c9d8ef", size=12),
        colorway=["#38bdf8", "#a78bfa", "#fbbf24", "#34d399", "#f472b6",
                  "#fb923c", "#22d3ee", "#94a3b8"],
        xaxis=dict(gridcolor="rgba(148,163,184,.14)", zerolinecolor="rgba(148,163,184,.25)",
                   linecolor="rgba(148,163,184,.25)", ticks="outside", tickcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="rgba(148,163,184,.14)", zerolinecolor="rgba(148,163,184,.25)",
                   linecolor="rgba(148,163,184,.25)", ticks="outside", tickcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=10, r=10, t=44, b=10),
        hoverlabel=dict(bgcolor="#0f1b30", bordercolor="#2a3f66",
                        font=dict(color="#e6eefc", size=12)),
        hovermode="x unified",
    )
)


def figure(fig: go.Figure, height: int = 340, title: str = "", x_title: str = "",
           y_title: str = "", legend: bool = True, unified: bool = True,
           margin_top: int = 46) -> go.Figure:
    fig.update_layout(template=PLOTLY_TEMPLATE, height=height, title_text=title,
                      margin=dict(l=10, r=10, t=margin_top, b=10))
    if x_title:
        fig.update_xaxes(title_text=x_title)
    if y_title:
        fig.update_yaxes(title_text=y_title)
    if not legend:
        fig.update_layout(showlegend=False)
    fig.update_layout(hovermode="x unified" if unified else "closest")
    return fig


def show(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, config={"displaylogo": False}, key=key, **wide())


def bar_palette(names: Sequence[str]) -> list[str]:
    return [CARRIER_COLORS.get(str(n).lower(), COLORS["sky"]) for n in names]
