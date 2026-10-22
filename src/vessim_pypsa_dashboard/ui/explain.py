"""One page for the two halves of the same argument: the blind spots and the policy.

They used to be two pages, and the split was an artefact of how they were
written, not of what they say.  The blind spots are *what the LP cannot
represent*; a dispatch policy is *what you write instead*.  Reading them apart
left the reader with the diagnosis and no treatment.  Here they are three tabs of
one page, with one case selector shared by all of them:

* 🕶️ **What the LP cannot see** — the boundary map and the three live probes;
* 🛠️ **How a policy is written** — the contract, the plug-in point, the eight
  shipped rules, an editor that compiles and executes your own class;
* 🎚️ **Tune one and measure it** — six numbers, a live run, and the price of a
  constraint no price can express.
"""

from __future__ import annotations

import streamlit as st

from . import blindspots, policy_guide, theme


def render() -> None:
    theme.hero(
        "🕶️", "Blind spots & policies",
        "WHAT THE LP CANNOT SEE · AND WHAT YOU WRITE INSTEAD",
        "An LP optimises a model: linear constraints, a horizon it already knows, a scalar "
        "efficiency. Vessim executes software against a plant. The first tab measures what falls "
        "outside the model on a real run; the second shows the one API that fills the gap — a "
        "subclass of <code>DispatchPolicy</code>; the third tunes a rule and prices the constraint "
        "you imposed.",
        [("Blind spots", "6"), ("Probes", "3 live"), ("Policies", "8 + yours")],
    )
    case = blindspots.case_controls()
    tab_blind, tab_write, tab_tune = st.tabs([
        "🕶️ What the LP cannot see",
        "🛠️ How a policy is written",
        "🎚️ Tune one and measure it",
    ])
    with tab_blind:
        blindspots.body(case)
        blindspots.closing_note()
    with tab_write:
        policy_guide.body(case[1])
    with tab_tune:
        blindspots.playground(case)
