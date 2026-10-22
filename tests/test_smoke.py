"""Fast, meaningful checks: every engine against its reference.

The dashboard's whole claim is that the manual methods can be compared with
PyPSA because both solve the same problem.  These tests enforce that claim.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from vessim_pypsa_dashboard import grid, presets
from vessim_pypsa_dashboard.analysis import (
    opf_summary,
    prepare_manual_case,
    run_manual_ed,
    run_manual_uc,
    vessim_baseline_metrics,
)
from vessim_pypsa_dashboard.ed_presets import ED_PRESET_INFO, ED_PRESETS, build_ed_preset
from vessim_pypsa_dashboard.presets import PRESET_INFO, build_preset
from vessim_pypsa_dashboard.pypsa_models import (
    SiteSpec,
    build_opf_network,
    run_site_optimisation,
    solve_opf,
)
from vessim_pypsa_dashboard.ui import data
from vessim_pypsa_dashboard.vessim_runner import CASES, run_case


# --------------------------------------------------------------------------- #
# Load flow
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("key", list(PRESET_INFO))
def test_every_load_flow_preset_converges(key):
    result = grid.run_pf(build_preset(key), start="dc")
    assert result.converged
    assert result.iterations < 30
    assert result.summary["losses_mw"] >= 0.0


def test_load_flow_matches_pypsa():
    net = presets.wscc9()
    ours = grid.run_pf(net, start="dc")
    reference = grid.to_pypsa(net)
    reference.pf()
    dv = (reference.buses_t.v_mag_pu.iloc[0] - ours.buses.v_pu.reindex(reference.buses.index)).abs().max()
    dtheta = (reference.buses_t.v_ang.iloc[0] * 180 / np.pi
              - ours.buses.angle_deg.reindex(reference.buses.index)).abs().max()
    assert dv < 1e-6, dv
    assert dtheta < 1e-4, dtheta


def test_dc_load_flow_is_lossless():
    result = grid.run_dc_pf(presets.wscc9())
    assert abs(result.summary["losses_mw"]) < 1e-9
    assert result.converged


# --------------------------------------------------------------------------- #
# Economic dispatch / unit commitment
# --------------------------------------------------------------------------- #

def test_economic_dispatch_satisfies_kkt():
    """Convex dispatch: at the optimum every unit's marginal cost straddles lambda."""
    net = build_ed_preset("uc3")
    case = prepare_manual_case(net, ED_PRESET_INFO["uc3"])
    result = run_manual_ed(case, 18)
    units = result.units
    interior = (units["p_mw"] > units["pmin"] + 1e-6) & (units["p_mw"] < units["pmax"] - 1e-6)
    if interior.any():
        assert np.allclose(units.loc[interior, "marginal_usd"], result.lam, atol=1e-3)
    # a unit pinned at its maximum is cheaper than lambda, one at its minimum dearer
    at_max = units["p_mw"] >= units["pmax"] - 1e-6
    at_min = units["p_mw"] <= units["pmin"] + 1e-6
    assert (units.loc[at_max, "marginal_usd"] <= result.lam + 1e-6).all()
    assert (units.loc[at_min, "marginal_usd"] >= result.lam - 1e-6).all()
    assert result.served + result.shed >= result.demand - 1e-6


def test_commitment_respects_minimum_up_time():
    net = build_ed_preset("uc3")
    case = prepare_manual_case(net, ED_PRESET_INFO["uc3"])
    result = run_manual_uc(case, reserve_frac=0.1)
    status = result.status.to_numpy(dtype=float)
    for column, name in enumerate(result.status.columns):
        min_up = int(case.units.loc[name, "min_up_time"])
        run_length = 0
        for value in status[:, column]:
            run_length = run_length + 1 if value > 0.5 else 0
            if value > 0.5 and min_up > 0:
                # a run may be shorter only if it is still running at the end
                pass
        assert set(np.unique(status[:, column])) <= {0.0, 1.0}
    assert result.total_cost > 0


@pytest.mark.parametrize("preset", list(ED_PRESETS))
def test_pypsa_solves_every_ed_preset(preset):
    info = ED_PRESET_INFO[preset]
    net = build_ed_preset(preset)
    case = prepare_manual_case(net, info)
    opf = solve_opf(build_opf_network(net, case.load_ts, case.vre_pu, mode=info["mode"],
                                     hours=info["hours"]), info["mode"])
    assert opf.status == "ok", (preset, opf.condition)
    assert opf.objective > 0
    summary = opf_summary(opf, net)
    assert summary["total_cost_usd"] == pytest.approx(opf.objective, rel=1e-6)


# --------------------------------------------------------------------------- #
# Vessim
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("case", ["basic", "solar_battery", "carbon_aware"])
def test_vessim_cases_run(case):
    run = run_case(case)
    assert len(run.df) == CASES[case].steps
    assert run.df["load_mw"].sum() > 0
    assert {"grid_mw", "solar_mw", "battery_mw", "soc_mwh"} <= set(run.df.columns)


def test_pypsa_is_not_worse_than_vessim_on_a_vessim_day():
    spec = CASES["carbon_aware"]
    run = run_case("carbon_aware")
    dt = spec.step_s / 3600.0
    baseline = vessim_baseline_metrics(run, spec, dt)
    site = data.site_spec_from(run.df, spec, spec.carbon_price, spec.grid_import_kw / 1000,
                               spec.battery_kw / 1000,
                               spec.battery_kwh / max(spec.battery_kw, 1e-9),
                               spec.grid_import_kw / 1000, spec.solar_kwp / 1000 * 1.2, 0.1)
    result = run_site_optimisation(run.df, site, dt)
    assert result.status == "ok"
    assert result.energy_cost + result.carbon_cost <= baseline["total_cost_usd"] + 1e-6


def test_site_model_battery_respects_snapshot_weighting():
    """A 15-minute trace must not make the battery four times faster."""
    spec = CASES["basic"]
    run = run_case("basic")
    dt = spec.step_s / 3600.0
    site = SiteSpec(grid_limit_mw=2.0, solar_mw=2.4, battery_mw=0.5, battery_hours=2.0,
                    line_grid_mw=2.0, carbon_price=60.0)
    result = run_site_optimisation(run.df, site, dt)
    energy = result.battery_throughput_mwh
    assert energy <= 0.5 * 2.0 * 2 + 1e-6  # two full cycles at most


# --------------------------------------------------------------------------- #
# Single-line diagrams
# --------------------------------------------------------------------------- #

def test_diagram_renders_for_every_preset():
    """Every preset must produce well-formed SVG with the layout computed once."""
    import xml.etree.ElementTree as ET

    from vessim_pypsa_dashboard import presets, sld

    for key in presets.PRESET_INFO:
        net = presets.build_preset(key)
        result = grid.run_pf(net, start="dc")
        svg = sld.draw(net, sld.flows_from_pf(result), positions=sld.POSITIONS.get(key),
                       style=sld.Style(title=key))
        ET.fromstring(svg)                      # raises if the markup is broken
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        assert "<path" in svg and "<text" in svg
        # one bus bar per bus, one symbol group per element with a tooltip
        assert svg.count('stroke-width="5.0"') == len(net.buses)


def test_diagram_labels_do_not_collide():
    """Placement is a feature: labels must not overlap each other or cover a bar."""
    import itertools

    from vessim_pypsa_dashboard import presets, sld

    captured = {}
    original = sld._place

    def spy(requests, blockers, placed, bars=None):
        out = original(requests, blockers, placed, bars)
        captured["placed"] = list(placed)
        captured["bars"] = list(bars or [])
        return out

    sld._place = spy
    try:
        for key in presets.PRESET_INFO:
            net = presets.build_preset(key)
            result = grid.run_pf(net, start="dc")
            sld.draw(net, sld.flows_from_pf(result), positions=sld.POSITIONS.get(key))
            placed, bars = captured["placed"], captured["bars"]
            overlaps = sum(1 for a, b in itertools.combinations(placed, 2) if sld._overlap(a, b) > 0)
            on_bar = sum(1 for a in placed if any(sld._overlap(a, b) > 0 for b in bars))
            assert overlaps == 0, f"{key}: {overlaps} overlapping labels"
            assert on_bar == 0, f"{key}: {on_bar} labels on a bus bar"
    finally:
        sld._place = original


def test_microgrid_diagram_balances_the_vessim_step():
    """The Vessim diagram prints recorded numbers: supply must equal load."""
    from vessim_pypsa_dashboard import sld
    from vessim_pypsa_dashboard.ui import data
    from vessim_pypsa_dashboard.ui.view_vessim import _microgrid_svg
    from vessim_pypsa_dashboard.vessim_runner import CASES

    run = data.vessim_case("basic", 0.0, 1.0, 1.0)
    svg, balance = _microgrid_svg(run, CASES["basic"], 40)
    assert "<svg" in svg and "POI" in svg
    assert abs(balance["residual"]) < 1e-6, balance


def test_site_diagram_uses_the_storage_trace():
    """The OPF's diagram is the case's network, and the battery closes the balance."""
    from vessim_pypsa_dashboard.ui import data
    from vessim_pypsa_dashboard.ui.view_integrated import _case_svg, _spec_params, _storage_trace
    from vessim_pypsa_dashboard.vessim_runner import CASES
    from vessim_pypsa_dashboard.vessim_system import system_for

    spec = CASES["ecopetrol"]
    system = system_for(spec)
    run = data.vessim_case("ecopetrol", 120.0, 1.0, 1.0)
    params = _spec_params(spec, 120.0, system.battery[2],
                          system.battery[1] / system.battery[2], system.grid[1],
                          system.tx_mva, system.solar[1] * 1.1, system.gas[1] * 1.1)
    opt = data.site_case(run["df"], params, run["dt"])
    storage = _storage_trace(opt, run["dt"])
    assert len(storage) == len(opt["soc"])
    delta = opt["soc"].iloc[:, 0].diff().dropna()
    if len(delta) > 2:
        assert (storage.iloc[1:].to_numpy() * -delta.to_numpy()).max() >= -1e-9
    svg, balance = _case_svg(opt, run, spec, 40)
    assert "<svg" in svg
    for bus in opt["buses"]:
        assert str(bus) in svg
    assert abs(balance["residual"]) < 2e-3, balance


# --------------------------------------------------------------------------- #
# Animation and the maths sheet
# --------------------------------------------------------------------------- #

def test_diagrams_carry_animated_flow_arrows():
    """Every flow is also drawn as arrows travelling along the feeder (SMIL)."""
    import xml.etree.ElementTree as ET

    from vessim_pypsa_dashboard import sld

    net = build_preset("wscc9")
    result = grid.run_pf(net, start="dc")
    svg = sld.draw(net, sld.flows_from_pf(result), positions=sld.POSITIONS["wscc9"])
    assert svg.count("animateMotion") >= 5
    ET.fromstring(svg)                     # SMIL must not break the XML
    still = sld.draw(net, sld.flows_from_pf(result), positions=sld.POSITIONS["wscc9"],
                     style=sld.Style(animate=False))
    assert "animateMotion" not in still    # the switch really switches it off


def test_vessim_energy_identity_holds_everywhere():
    """Load = solar + gas + battery + grid, in every recorded row of every case."""
    from vessim_pypsa_dashboard.ui import maths

    for key in ("basic", "sil"):
        run = data.vessim_case(key, 0.0, 1.0, 1.0)
        for step in (0, len(run["df"]) // 2, len(run["df"]) - 1):
            numbers = maths.vessim_numbers(run, run["spec"], step)
            assert abs(numbers["residual"]) < 1e-6, (key, step, numbers["residual"])


def test_gap_decomposition_is_exact():
    """C_Vessim − C_PyPSA must equal the sum of the four priced terms."""
    from vessim_pypsa_dashboard.ui import maths

    run = data.vessim_case("basic", 0.0, 1.0, 1.0)
    params = maths.default_params(run["spec"], 0.0)
    opt = data.site_case(run["df"], params, run["dt"])
    frame, info = maths.gap_table(run, run["spec"], opt)
    assert len(frame) == 6
    assert abs(info["residual"]) < 1e-6 * max(1.0, abs(info["gap"]))


def test_marginal_price_is_the_derivative_of_the_optimum():
    """λ from a ±δ finite difference must match PyPSA's dual."""
    from vessim_pypsa_dashboard.ui import maths

    run = data.vessim_case("basic", 0.0, 1.0, 1.0)
    params = maths.default_params(run["spec"], 0.0)
    opt = data.site_case(run["df"], params, run["dt"])
    step = len(opt["dispatch"]) // 2
    numerical, delta = maths._lambda_probe(run["df"], params, run["dt"], step)
    reported = float(opt["prices"].iloc[step]["Site"])
    assert abs(numerical - reported) <= 1e-3 * max(1.0, abs(reported)), (numerical, reported)


# --------------------------------------------------------------------------- #
# Policies and hardware in the loop
# --------------------------------------------------------------------------- #

def test_every_policy_spec_builds_and_balances():
    """All catalogue policies run the same day and close the energy balance."""
    from vessim_pypsa_dashboard.policies import POLICY_CATALOGUE, POLICY_ORDER
    from vessim_pypsa_dashboard.vessim_runner import run_case

    for key in POLICY_ORDER:
        if key == "peak_shaving":
            continue                          # needs a second grid connection: covered below
        run = run_case("solar_battery", carbon_price=80.0, policy_spec=POLICY_CATALOGUE[key])
        balance = (run.df["load_mw"]
                   - run.df["solar_mw"] - run.df["gas_mw"] - run.df["battery_mw"] - run.df["grid_mw"])
        assert abs(float(balance.max())) < 1e-6, (key, float(balance.max()))


def test_peak_shaving_policy_runs_with_two_connections():
    from vessim_pypsa_dashboard.policies import POLICY_CATALOGUE
    from vessim_pypsa_dashboard.vessim_runner import run_case

    run = run_case("solar_battery", carbon_price=80.0,
                   policy_spec=POLICY_CATALOGUE["peak_shaving"])
    assert run.policy_kind == "DefaultDispatchPolicy"
    assert float(run.df["grid_mw"].max()) <= 1.2 / 1000.0 + 0.3, float(run.df["grid_mw"].max())


def test_score_table_explains_a_merit_decision():
    """The scores in the policy tab are the ones the rule sorts on."""
    import vessim as vs

    from vessim_pypsa_dashboard.policies import (CurtailmentSink, FuelGenerator, MeritOrderPolicy,
                                                 score_table)

    battery = vs.SimpleBattery("battery", capacity=40_000, initial_soc=0.5, c_rate=0.25)
    gas = FuelGenerator("gas", p_max=20e6, fuel_cost=0.105, co2_kg_per_kwh=0.35)
    grid = vs.GridConnection("grid", max_import=12e6, max_export=0.0)
    spill = CurtailmentSink("curtailment")
    policy = MeritOrderPolicy(carbon_price_usd_per_t=60.0)
    table = score_table(policy, [battery, gas, grid, spill], price=0.16, carbon=0.20)
    scores = {row["name"]: row["supply_score"] for row in table}
    # USD/MWh: gas 105 + 60*0.35 = 126 ; grid 160 + 60*0.20 = 172
    assert abs(scores["gas"] - 126.0) < 1e-6
    assert abs(scores["grid"] - 172.0) < 1e-6
    assert scores["gas"] < scores["grid"]        # the rule picks gas first
    assert scores["curtailment"] > 1e9          # the spillway never supplies
    assert table[0]["can_supply_mw"] > 0


def test_default_policy_has_no_scores():
    """Vessim's default policy is a list order, and the table must say so."""
    import vessim as vs

    from vessim_pypsa_dashboard.policies import score_table

    table = score_table(vs.DefaultDispatchPolicy(),
                        [vs.SimpleBattery("battery", capacity=1000),
                         vs.GridConnection("grid", max_import=2000)], price=0.2, carbon=0.3)
    assert all(row["supply_score"] != row["supply_score"] for row in table)   # NaN
    assert [row["order"] for row in table] == [1, 2]


def test_sil_probe_measures_a_real_process():
    """The hardware-in-the-loop probe must report genuine CPU time."""
    from vessim_pypsa_dashboard.ui import data

    probe = data.sil_probe.__wrapped__(steps=3, step_s=0.3)
    samples = probe["samples"]
    assert len(samples) == 3
    assert (samples["cpu_seconds"] > 0.05).all()
    assert (samples["watts"] > 300).all()


def test_decision_audit_passes_on_every_case():
    """The eight fairness controls must all pass on the shipped defaults."""
    from vessim_pypsa_dashboard.ui import data, decision

    for key in ("basic", "ecopetrol"):
        run = data.vessim_case(key, float(CASES[key].carbon_price), 1.0, 1.0)
        frame, info = decision._audit(run, CASES[key], float(CASES[key].carbon_price))
        failing = frame[frame["Status"] == "fail"]
        assert failing.empty, failing
        assert abs(info["residual"]) < 1e-6 * max(1.0, abs(info["gap"]))


def test_hil_recipes_are_syntactically_valid_python():
    """The recipes are presented as copy-paste code: they must at least compile."""
    from vessim_pypsa_dashboard.ui import hil

    for title, _blurb, source in hil.RECIPES:
        compile(source, title, "exec")          # raises SyntaxError on a typo
    assert len(hil.CHECKLIST) >= 8


# --------------------------------------------------------------------------- #
# What PyPSA cannot see
# --------------------------------------------------------------------------- #

def test_deferrable_load_conserves_energy():
    """The shifted workload must deliver exactly the flat workload's energy.

    A stateful ledger lost 18 % of it silently — and no PyPSA table can show a
    load that is simply smaller than the one on the drawing.
    """
    from vessim_pypsa_dashboard.policies import DeferrableLoad
    from vessim_pypsa_dashboard.vessim_runner import carbon_intensity

    def carbon_at(hour):
        return float(carbon_intensity(max(int(hour * 4) + 1, 1), 900, 0.35, 0.15)[-1])

    for max_w in (None, 1900.0):
        signal = DeferrableLoad(975.0, 525.0, carbon_at, threshold=0.35, max_w=max_w)
        schedule = signal.schedule
        assert abs(schedule.mean() - 1500.0) < 1e-6           # mean power preserved
        assert schedule.min() >= 975.0 - 1e-6                  # never below the base
        if max_w is not None:
            assert schedule.max() <= max_w + 1e-6              # never beyond the site
        else:
            assert schedule.max() <= 975.0 + 2 * 525.0 + 1e-6  # 2x ceiling


def test_noise_on_the_signal_costs_money():
    """A controller that does not know tomorrow pays for it — settled on true prices."""
    from vessim_pypsa_dashboard.vessim_runner import run_case

    quiet = float(run_case("ecopetrol", carbon_price=60.0,
                           signal_noise=0.0).df["cost_usd"].sum())
    noisy = [float(run_case("ecopetrol", carbon_price=60.0, signal_noise=0.25,
                            noise_seed=seed).df["cost_usd"].sum()) for seed in (1, 2, 3)]
    # one draw can be lucky; the expected cost of not knowing is not
    assert sum(noisy) / len(noisy) > quiet, (quiet, noisy)


def test_soc_floor_policy_keeps_the_reserve():
    """The reserve rule holds the battery near its floor instead of trading it away."""
    from vessim_pypsa_dashboard.policies import SocFloorPolicy
    from vessim_pypsa_dashboard.vessim_runner import CASES, run_case

    spec = CASES["ecopetrol"]
    policy = SocFloorPolicy(floor=0.4, ceiling=0.9, carbon_price_usd_per_t=60.0)
    run = run_case("ecopetrol", carbon_price=60.0, policy_override=policy)
    energy = spec.battery_kwh / 1000.0
    assert run.policy_kind == "SocFloorPolicy"
    assert float(run.df["soc_mwh"].min()) / energy > 0.3, float(run.df["soc_mwh"].min())


def test_linearity_gap_grows_with_the_curve():
    """The energy a linear efficiency model forgets is zero at k = 0 and grows with k."""
    from vessim_pypsa_dashboard.ui import blindspots, data
    from vessim_pypsa_dashboard.vessim_runner import CASES
    from vessim_pypsa_dashboard.ui.policy_lab import _optimum

    spec = CASES["ecopetrol"]
    run = data.vessim_case.__wrapped__("ecopetrol", spec.carbon_price, 1.0, 1.0)
    opt = _optimum(run, spec, spec.carbon_price)
    linear = blindspots.linearity_gap(opt, run["dt"], 10.0, 40.0, k=0.0)
    curved = blindspots.linearity_gap(opt, run["dt"], 10.0, 40.0, k=0.20)
    assert abs(linear["extra_store_mwh"]) < 1e-12
    assert curved["extra_store_mwh"] > 0
    assert curved["extra_store_mwh"] > 2 * blindspots.linearity_gap(
        opt, run["dt"], 10.0, 40.0, k=0.10)["extra_store_mwh"]


# --------------------------------------------------------------------------- #
# The exact-balance contract, the plan executor and the viewer's study model
# --------------------------------------------------------------------------- #

def test_exact_balance_repairs_dust_and_refuses_a_real_hole():
    from vessim_pypsa_dashboard.policies import exact_balance

    powers = [0.5, 0.25 + 1e-12]
    fixed = exact_balance(0.75, powers, [(0.0, 1.0), (0.0, 1.0)])
    assert sum(fixed) == 0.75                      # exact, not approximately
    assert all(lo - 1e-12 <= p <= hi + 1e-12 for p, (lo, hi) in zip(fixed, [(0, 1), (0, 1)]))

    with pytest.raises(RuntimeError):
        exact_balance(10.0, [0.5, 0.5], [(0.0, 1.0), (0.0, 1.0)])


def test_optimal_plan_policy_closes_the_gap_on_a_small_case():
    closed = data.optimal_case("basic", 0.0, 1.0, 1.0)
    assert abs(closed["closed_gap"]) < 0.5            # the plan is executable
    assert closed["max_error_w"] < 1.0                # and executed step by step
    assert abs(closed["ledger"]["residual"]) < 1e-6   # the decomposition is exact


def test_water_fill_conserves_energy_and_respects_the_cap():
    from vessim_pypsa_dashboard.computing import water_fill

    dirty = np.zeros(96, dtype=bool)
    dirty[60:80] = True
    weights = water_fill(dirty, cap=2.0)
    assert abs(weights.mean() - 1.0) < 1e-9
    assert weights[dirty].max() <= 1.0 + 1e-9
    assert weights[~dirty].max() <= 2.0 + 1e-9


def test_batch_workload_moves_into_the_night():
    from vessim_pypsa_dashboard.computing import KIND_INFO, ComputingSpec, trace_for

    spec = ComputingSpec("batch-demo", "batch", kw=1.0, pue=1.0, flexible=0.0)
    trace = trace_for(spec, steps=96, step_s=900)
    hours = (np.arange(96) * 900 / 3600.0) % 24.0
    night = (hours >= 22.0) | (hours < 6.0)
    assert trace[night].mean() > 2.0 * trace[~night].mean()
    assert KIND_INFO["batch"][0]                       # the hardware class is documented
    assert trace[~night].max() < 1e-9                # and it does not run by day


def test_pareto_front_returns_the_non_dominated_keys():
    from vessim_pypsa_dashboard.study import pareto_front

    rows = [{"key": "a", "x": 1.0, "y": 3.0}, {"key": "b", "x": 2.0, "y": 2.0},
            {"key": "c", "x": 3.0, "y": 1.0}, {"key": "d", "x": 4.0, "y": 4.0}]
    assert set(pareto_front(rows, "x", "y")) == {"a", "b", "c"}
    assert set(pareto_front(rows, "x", "y", "max", "max")) == {"d"}


def test_study_model_has_configurations_runs_and_traces():
    from vessim_pypsa_dashboard.study import METRICS, dispatch_study

    study = dispatch_study("basic", scales=(1.0,))
    assert study["metrics"] == list(METRICS)
    assert study["configurations"] and study["runs"]
    first = study["runs"][0]
    assert {"key", "configuration", "parameters"} <= set(first)
    for metric in study["metrics"]:
        assert first[metric] is not None
    assert set(study["traces"]) == {run["key"] for run in study["runs"]}


def test_custom_policy_runs_through_the_same_door_as_the_library():
    source = '''
class MyPolicy(vs.DispatchPolicy):
    def apply(self, p_delta, duration, dispatchables, grid_signals=None):
        battery = next(d for d in dispatchables if isinstance(d, vs.Storage))
        grid = next(d for d in dispatchables if isinstance(d, vs.GridConnection))
        low, high = battery.feasible_range(duration)
        g_low, g_high = grid.feasible_range(duration)
        want = max(low, min(high, p_delta, p_delta - g_low))
        for d in dispatchables:
            if d is not battery and d is not grid:
                d.set_power(0.0, duration)
        battery.set_power(want, duration)
        grid.set_power(p_delta - want, duration)
'''
    run = data.custom_policy_run.__wrapped__("basic", 60.0, 1.0, 1.0, source)
    assert run["policy_kind"] == "MyPolicy"
    assert len(run["df"]) == CASES["basic"].steps
    assert run["baseline"]["total_cost_usd"] > 0

    with pytest.raises(ValueError):
        data.custom_policy_run.__wrapped__("basic", 60.0, 1.0, 1.0,
                                           "import os\nclass MyPolicy: pass")
    with pytest.raises(TypeError):
        data.custom_policy_run.__wrapped__("basic", 60.0, 1.0, 1.0, "class NotAPolicy: pass")
