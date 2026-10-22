"""Streamlit wrapper around :mod:`vessim_pypsa_dashboard.sld`.

The single-line diagram is plain SVG, so it is inlined with ``st.markdown``:
it scales to the column width, needs no JavaScript and survives the sandboxed
preview that blocks external scripts.
"""

from __future__ import annotations

import streamlit as st

from .. import sld

PANEL_CSS = """
<style>
.sld-panel {
  background: linear-gradient(170deg, #0d1729 0%, #0a1322 100%);
  border: 1px solid #1f3355; border-radius: 14px;
  padding: .55rem .7rem .35rem; margin: .15rem 0 .55rem;
  overflow: hidden;
}
.sld-panel .sld-head {
  display: flex; justify-content: space-between; align-items: baseline;
  gap: .8rem; padding: 0 .15rem .3rem;
}
.sld-panel .sld-title {font-size: .78rem; letter-spacing: .08em; text-transform: uppercase;
  color: #7f92b4; font-weight: 700;}
.sld-panel .sld-hint {font-size: .74rem; color: #64789b;}
.sld-legend {display:flex; flex-wrap:wrap; gap:.75rem; font-size:.72rem; color:#8ea3c7;
  padding:.15rem .2rem .1rem;}
.sld-legend span {display:inline-flex; align-items:center; gap:.32rem;}
.sld-legend i {display:inline-block; width:14px; height:3px; border-radius:2px; background:#dbe7f7;}
.sld-legend i.arrow {width:0; height:0; background:none; border-left:7px solid #38bdf8;
  border-top:4px solid transparent; border-bottom:4px solid transparent;}
.sld-legend i.open {width:8px; height:8px; background:none; border-left:2px solid #fbbf24;
  border-top:2px solid transparent; border-bottom:2px solid transparent;}
.sld-legend i.good {background:#34d399;} .sld-legend i.warn {background:#fbbf24;}
.sld-legend i.bad {background:#f87171;}
</style>
"""


def render(svg: str, *, caption: str = "", hint: str = "hover any element for detail",
           legend: bool = True) -> None:
    """Show one diagram inside the standard panel."""
    if not svg:
        st.markdown('<div class="small">No network to draw.</div>', unsafe_allow_html=True)
        return
    parts = [PANEL_CSS, '<div class="sld-panel">']
    if caption or hint:
        parts.append('<div class="sld-head">'
                     f'<span class="sld-title">{caption}</span>'
                     f'<span class="sld-hint">{hint}</span></div>')
    parts.append(svg)
    parts.append("</div>")
    if legend:
        parts.append(
            '<div class="sld-legend">'
            '<span><i></i>bus bar</span>'
            '<span><i class="arrow"></i>active power, direction of flow</span>'
            '<span><i class="open"></i>reactive power</span>'
            '<span><i class="good"></i>within limits</span>'
            '<span><i class="warn"></i>near a limit</span>'
            '<span><i class="bad"></i>past a limit</span>'
            '</div>'
        )
    st.markdown("".join(parts), unsafe_allow_html=True)


def draw(net, flows=None, *, positions=None, title: str = "", subtitle: str = "",
         show_q: bool = False, show_loading: bool = True, show_cost: bool = False,
         show_loss: bool = False, colour_by: str = "voltage",
         v_min: float = 0.95, v_max: float = 1.05, unit: str = "MW") -> str:
    """Convenience wrapper: network + flows -> SVG string."""
    style = sld.Style(
        show_q=show_q, show_loading=show_loading, show_cost=show_cost, show_loss=show_loss,
        colour_by=colour_by, v_min=v_min, v_max=v_max, unit=unit,
        title=title, subtitle=subtitle,
    )
    return sld.draw(net, flows, style=style, positions=positions)
