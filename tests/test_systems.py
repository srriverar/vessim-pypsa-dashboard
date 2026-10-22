"""Every case must open its *own* system, not a generic envelope.

These tests are the contract behind the two AdjieBrotot sheets: the network a
case shows is the system that case installed, the load flow balances against the
simulation's own import, and the dispatch the sheet solves is the fleet the
policy actually ran.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from vessim_pypsa_dashboard import grid, sld
from vessim_pypsa_dashboard.ui import data
from vessim_pypsa_dashboard.vessim_runner import CASES, CASE_ORDER
from vessim_pypsa_dashboard.vessim_system import (
    SYSTEMS,
    ed_at,
    fleet_table,
    layout_for,
    site_network,
    snapshot,
    system_for,
    uc_day,
)


@pytest.fixture(scope="module")
def runs():
    return {key: data.vessim_case(key, float(CASES[key].carbon_price), 1.0, 1.0)
            for key in CASE_ORDER}


# --------------------------------------------------------------------------- #
# The systems are declared, per case
# --------------------------------------------------------------------------- #

def test_every_case_declares_its_own_system():
    assert set(SYSTEMS) == set(CASE_ORDER)
    for key, system in SYSTEMS.items():
        assert system.key == key
        assert system.number == CASES[key].number
        covered = {actor for group in system.groups for actor in group.actors}
        assert covered, key
        assert system.grid[1] > 0 and system.solar[1] > 0


def test_systems_are_not_the_same_envelope():
    """The five cases must differ in equipment, not only in numbers."""
    shapes = {key: (tuple(name for _, name, _ in system.equipment()), system.mv_kv)
              for key, system in SYSTEMS.items()}
    assert len(set(shapes.values())) == len(SYSTEMS), shapes


def test_ecopetrol_is_the_industrial_system():
    system = system_for(CASES["ecopetrol"])
    assert system.mv_kv == 110.0 and system.site_kv == 13.8
    assert system.gas is not None and system.gas[1] == 20.0
    assert system.battery[1] == 40.0 and system.battery[2] == 10.0
    names = {group.name: group for group in system.groups}
    assert "Data centre" in names and "Plant panel" in names
    assert names["Data centre"].kv == 0.4
    assert layout_for(CASES["ecopetrol"]).tx_mva >= 40.0


def test_small_cases_stay_at_low_voltage():
    for key in ("basic", "solar_battery", "carbon_aware", "sil"):
        system = system_for(CASES[key])
        assert system.mv_kv == 20.0 and system.site_kv == 0.4
        assert system.gas is None
        assert all(group.kv == 0.4 for group in system.groups)


# --------------------------------------------------------------------------- #
# The network is the case's, at the case's powers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("key", CASE_ORDER)
def test_network_carries_one_meter_per_actor(key, runs):
    run = runs[key]
    net = site_network(run, CASES[key], 0)
    actors = set(snapshot(run, CASES[key], 0)["actors"])
    assert actors <= set(net.loads.index), (actors, set(net.loads.index))
    assert all(name in net.buses.index for name in net.loads["bus"])
    assert all(name in net.buses.index for name in net.lines["bus1"])
    assert all(name in net.buses.index for name in net.transformers["bus1"])
    assert net.validate() == []


@pytest.mark.parametrize("key", CASE_ORDER)
def test_load_flow_converges_over_the_whole_day(key, runs):
    run = runs[key]
    steps = range(0, len(run["df"]), max(1, len(run["df"]) // 12))
    for step in steps:
        result = grid.run_pf(site_network(run, CASES[key], step), start="dc")
        assert result.converged, (key, step, result.iterations)
        assert result.buses["v_pu"].min() > 0.80, (key, step, result.buses["v_pu"].min())


@pytest.mark.parametrize("key", CASE_ORDER)
def test_slack_reproduces_the_import_up_to_the_losses(key, runs):
    run = runs[key]
    spec = CASES[key]
    for step in (0, len(run["df"]) // 2, len(run["df"]) - 1):
        net = site_network(run, spec, step)
        result = grid.run_pf(net, start="dc")
        snap = snapshot(run, spec, step)
        slack = float(result.buses["p_gen"].reindex(["POI"]).iloc[0]) * 1000.0
        losses_kw = float(result.summary["losses_mw"]) * 1000.0
        assert abs(slack - (snap["grid_mw"] * 1000.0 + losses_kw)) < 1e-3, (key, step)


def test_cable_impedance_is_per_km_times_length(runs):
    """Length is physics, not a label: r = r_per_km × km, by voltage class."""
    for key in CASE_ORDER:
        net = site_network(runs[key], CASES[key], 0)
        for name, row in net.lines.iterrows():
            kv = float(net.buses.at[row["bus0"], "v_nom"])
            r_km, x_km = (0.32, 0.08) if kv <= 1.0 else (0.16, 0.24)
            assert row["r"] == pytest.approx(r_km * row["length"]), (key, name)
            assert row["x"] == pytest.approx(x_km * row["length"]), (key, name)
    # and the two voltage classes really are different cables
    eco = site_network(runs["ecopetrol"], CASES["ecopetrol"], 0)
    assert eco.lines.loc["Site–PV collector", "r"] / eco.lines.loc["Site–PV collector", "length"] \
        == pytest.approx(0.16)
    basic = site_network(runs["basic"], CASES["basic"], 0)
    tail = basic.lines.loc["Site–Consumer panel feeder"]
    assert tail["r"] / tail["length"] == pytest.approx(0.32)


@pytest.mark.parametrize("key", CASE_ORDER)
def test_the_diagram_draws_every_bus(key, runs):
    net = site_network(runs[key], CASES[key], 0)
    svg = sld.draw(net)
    assert svg.startswith("<svg") or svg.lstrip().startswith("<svg")
    for bus in net.buses.index:
        assert str(bus) in svg, bus


# --------------------------------------------------------------------------- #
# The dispatch is the case's fleet, and it is the one the policy ran
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("key", CASE_ORDER)
def test_fleet_table_marks_the_decisions(key, runs):
    spec = CASES[key]
    table = fleet_table(runs[key], spec, 0, float(spec.carbon_price))
    decisions = set(table.loc[table["In the λ decision"] == "yes", "Equipment"])
    assert decisions == {system_for(spec).grid[0]} | (
        {system_for(spec).gas[0]} if system_for(spec).gas else set())


@pytest.mark.parametrize("key", CASE_ORDER)
def test_manual_dispatch_matches_vessim_on_the_same_fleet(key, runs):
    """At the peak step the two methods agree, because they see the same units."""
    run, spec = runs[key], CASES[key]
    carbon = float(spec.carbon_price)
    step = int(np.argmax(run["df"]["load_mw"].to_numpy()))
    ed = ed_at(run, spec, step, carbon)
    manual = dict(zip(ed["dispatch"]["unit"], ed["dispatch"]["mw"]))
    vessim = dict(zip(ed["vessim"]["unit"], ed["vessim"]["mw"]))
    for unit, value in vessim.items():
        assert abs(manual.get(unit, 0.0) - value) < 0.02 * max(1.0, abs(value)), (key, unit)


@pytest.mark.parametrize("key", CASE_ORDER)
def test_commitment_serves_the_day(key, runs):
    run, spec = runs[key], CASES[key]
    uc = uc_day(run, spec, float(spec.carbon_price))
    assert uc["unserved_mwh"] < 1e-6
    assert uc["frame"]["grid_mw"].max() <= spec.grid_import_kw / 1000.0 + 1e-6
    assert np.isfinite(uc["cost_usd"]) and uc["cost_usd"] > 0.0
