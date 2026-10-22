"""The ⚡ load-flow network and the 🔗 four-node OPF are the same system.

The dashboard's claim is that a reader can carry a number from one tab to the
other.  That is only true if the two networks are related by an explicit map and
if the numbers reconcile on the same trace — which is what these tests check,
case by case, instead of trusting the drawings.
"""

from __future__ import annotations

import numpy as np
import pytest

from vessim_pypsa_dashboard import grid, vessim_system
from vessim_pypsa_dashboard.pypsa_models import SiteSpec, build_site_network
from vessim_pypsa_dashboard.ui import data
from vessim_pypsa_dashboard.vessim_runner import CASES, CASE_ORDER


@pytest.fixture(scope="module")
def cases():
    out = {}
    for key in CASE_ORDER:
        spec = CASES[key]
        run = data.vessim_case.__wrapped__(key, float(spec.carbon_price), 1.0, 1.0)
        out[key] = (spec, run, vessim_system.system_for(spec))
    return out


@pytest.mark.parametrize("key", CASE_ORDER)
def test_every_ac_bus_appears_exactly_once(cases, key):
    spec, run, system = cases[key]
    mapping = vessim_system.correspondence(system)
    buses = list(vessim_system.site_network(run, spec, 0).buses.index)
    assert sorted(mapping["AC bus"].tolist()) == sorted(buses)
    assert not mapping["AC bus"].duplicated().any()


@pytest.mark.parametrize("key", CASE_ORDER)
def test_mapped_nodes_exist_in_the_opf(cases, key):
    spec, run, system = cases[key]
    mapping = vessim_system.correspondence(system)
    params = data.site_spec_from(run["df"], spec, float(spec.carbon_price),
                                 spec.grid_import_kw / 1000.0, spec.battery_kw / 1000.0,
                                 spec.battery_kwh / max(spec.battery_kw, 1e-9),
                                 20.0, 20.0, 20.0).__dict__
    net = build_site_network(run["df"], SiteSpec(**params), float(run["dt"]))
    assert set(mapping["PyPSA node"]) <= set(net.buses.index)
    # the names are shared, not merely similar: the load flow calls its PV bus "PV"
    assert "PV" in net.buses.index and net.generators.at["Solar", "bus"] == "PV"


@pytest.mark.parametrize("key", CASE_ORDER)
def test_the_two_models_differ_by_the_losses_only(cases, key):
    """AC slack − the trace's import = the network's losses, at every step checked."""
    spec, run, system = cases[key]
    df = run["df"]
    steps = np.linspace(0, len(df) - 1, min(6, len(df))).astype(int)
    for step in steps:
        result = grid.run_pf(vessim_system.site_network(run, spec, int(step)), start="dc")
        assert result.converged
        slack = float(result.buses.loc["POI", "p_gen"])
        trace = float(df["grid_mw"].iloc[int(step)])
        losses = float(result.summary["losses_mw"])
        assert abs(slack - (trace + losses)) < 1e-6, (key, step, slack, trace, losses)
        # the load the OPF is given is exactly the load the meters draw
        assert abs(result.summary["load_mw"] - float(df["load_mw"].iloc[int(step)])) < 1e-9


@pytest.mark.parametrize("key", CASE_ORDER)
def test_the_loss_gap_is_small_and_non_negative(cases, key):
    spec, run, system = cases[key]
    gap = vessim_system.loss_gap(run, spec, samples=6)
    assert gap["mean_loss_mw"] >= 0.0
    assert 0.0 <= gap["share_pct"] <= 10.0
    assert gap["energy_mwh"] >= 0.0 and gap["usd"] >= 0.0


def test_the_diagram_uses_the_shared_names():
    """The OPF diagram draws the buses the OPF actually has."""
    spec = CASES["ecopetrol"]
    run = data.vessim_case.__wrapped__("ecopetrol", 120.0, 1.0, 1.0)
    params = data.site_spec_from(run["df"], spec, 120.0, 12.0, 10.0, 4.0, 12.0, 25.0, 20.0).__dict__
    opt = data.site_case(run["df"], params, run["dt"])
    from vessim_pypsa_dashboard.ui.view_integrated import _case_svg

    svg, balance = _case_svg(opt, run, spec, 30)
    for name in ("POI", "Site", "PV", "TG"):
        assert name in svg
    # every bus of the case's network is drawn: this is the ⚡ network, not a
    # four-node contraction of it
    for name in opt["buses"]:
        assert str(name) in svg
    assert abs(balance["residual"]) < 2e-2, balance
