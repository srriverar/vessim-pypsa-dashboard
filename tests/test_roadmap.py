"""Tests for case 06 — the four items on Vessim's published roadmap.

The point of these tests is not coverage for its own sake: each one pins a
property the page *claims*, so that a claim cannot quietly become false.
"""

from __future__ import annotations

import numpy as np
import pytest

from vessim_pypsa_dashboard import roadmap
from vessim_pypsa_dashboard.vessim_runner import CASES, CASE_ORDER


# --------------------------------------------------------------------------- #
# 1 · The photovoltaic model
# --------------------------------------------------------------------------- #

def test_pv_is_monotone_in_irradiance_and_never_negative() -> None:
    model = roadmap.PVModel(kwp=180.0)
    ghi = np.linspace(0.0, 1000.0, 60)
    ambient = np.full_like(ghi, 25.0)
    power = model.power_kw(ghi, ambient)
    assert power.min() >= 0.0
    assert power[0] == 0.0
    assert np.all(np.diff(power) > 0.0)          # more sun, more power
    assert power.max() <= model.kwp             # and never above the roof


def test_heat_costs_more_than_it_gives() -> None:
    """Same irradiance at 20 °C and at 45 °C: the hot cell must produce less."""
    model = roadmap.PVModel(kwp=180.0)
    cool = model.power_kw(np.array([900.0]), np.array([20.0]))[0]
    hot = model.power_kw(np.array([900.0]), np.array([45.0]))[0]
    assert hot < cool
    cell_cool = model.cell_c(np.array([900.0]), np.array([20.0]))[0]
    cell_hot = model.cell_c(np.array([900.0]), np.array([45.0]))[0]
    expected = cool * (1.0 + model.gamma * (cell_hot - 25.0)) / (1.0 + model.gamma * (cell_cool - 25.0))
    assert hot == pytest.approx(expected, rel=1e-9)


def test_solar_geometry_has_a_day() -> None:
    hours = np.arange(48) * 0.5
    ghi = roadmap.solar_geometry(hours)
    assert ghi.min() == 0.0                      # night exists in Bogotá too
    assert ghi.max() == pytest.approx(1000.0, rel=0.02)
    noon = ghi[hours % 24 == 12.0]
    assert noon.max() == pytest.approx(ghi.max())   # highest sun at solar noon
    assert np.all(ghi >= 0.0)


def test_cloud_index_stays_inside_its_range() -> None:
    rng = np.random.default_rng(11)
    cloud = roadmap.cloud_factor(rng, 48)
    assert cloud.shape == (48,)
    assert cloud.min() >= 0.25 - 1e-12 and cloud.max() <= 1.0 + 1e-12


# --------------------------------------------------------------------------- #
# 2 · The dataset
# --------------------------------------------------------------------------- #

def test_the_bundled_dataset_is_what_the_page_says() -> None:
    info = roadmap.dataset_summary()
    assert info["rows"] > 8000
    assert info["days"] == pytest.approx(30, abs=2)
    assert 0.2 < info["mean"] < 0.5              # kg CO₂/kWh
    assert info["min"] < info["mean"] < info["max"]
    assert info["start"][:4] == "2023"


def test_carbon_dataset_is_resampled_to_the_case() -> None:
    spec = CASES["montecarlo"]
    trace = roadmap.carbon_dataset(spec.steps, spec.step_s)
    assert trace.shape == (spec.steps,)
    info = roadmap.dataset_summary()
    assert trace.min() >= info["min"] - 1e-9
    assert trace.max() <= info["max"] + 1e-9
    # the point of the dataset: it is not a smooth sine
    assert np.abs(np.diff(trace)).max() > 1e-3


# --------------------------------------------------------------------------- #
# 3 · Degradation
# --------------------------------------------------------------------------- #

def test_degradation_is_monotone_in_usage() -> None:
    capacity = 0.3                                  # MWh
    quiet = np.full(49, capacity / 2)
    busy = np.concatenate([np.linspace(0.05, 0.25, 25), np.linspace(0.25, 0.05, 24)])
    a = roadmap.degradation(quiet, 0.5, capacity)
    b = roadmap.degradation(busy, 0.5, capacity)
    assert a["efc"] == pytest.approx(0.0, abs=1e-9)
    assert b["efc"] > 0.5
    assert b["wear_usd"] > a["wear_usd"]
    assert b["soh_end"] < a["soh_end"]
    # calendar fade alone, when the battery never moves
    assert a["soh_end"] == pytest.approx(1.0 - a["calendar_fade"], rel=1e-9)


def test_wear_price_is_the_physics_not_the_policy() -> None:
    """A 6 000-cycle cell at 250 USD/kWh costs ~8.3 USD per MWh moved."""
    trace = np.concatenate([np.linspace(0.3, 0.0, 25), np.linspace(0.0, 0.3, 24)])
    wear = roadmap.degradation(trace, 0.5, 0.3)
    implied = wear["wear_usd"] / wear["discharged_mwh"]
    assert implied == pytest.approx(0.2 / 6000 * 250 * 1000, rel=0.02)
    assert implied < 20.0                        # what MeritOrderPolicy charges


# --------------------------------------------------------------------------- #
# 4 · The Monte Carlo
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def small_run():
    return roadmap.monte_carlo(n=4, seed=3)


def test_monte_carlo_runs_the_case_itself(small_run) -> None:
    assert small_run["n"] == 4
    frame = small_run["frame"]
    assert len(frame) == 4
    assert (frame["bill USD"] > 0).all()
    assert (frame["CO₂ kg"] > 0).all()
    assert frame["PV kWh"].max() < 24 * CASES["montecarlo"].solar_kwp


def test_percentiles_are_ordered(small_run) -> None:
    summary = small_run["summary"].set_index("metric")
    for metric in ("bill USD", "CO₂ kg", "PV kWh"):
        row = summary.loc[metric]
        assert row["P10"] <= row["P50"] <= row["P90"]
        assert row["spread"] == pytest.approx(row["P90"] - row["P10"])


def test_the_same_seed_gives_the_same_study(small_run) -> None:
    again = roadmap.monte_carlo(n=4, seed=3)
    assert np.allclose(small_run["frame"]["bill USD"], again["frame"]["bill USD"])
    other = roadmap.monte_carlo(n=4, seed=4)
    assert not np.allclose(small_run["frame"]["bill USD"], other["frame"]["bill USD"])


def test_samples_are_not_copies(small_run) -> None:
    """Every draw must differ in weather, load, price and carbon alike."""
    draws = small_run["signals"]
    for field in ("solar_kw", "carbon", "price", "load_factor"):
        first, second = draws[0][field], draws[1][field]
        assert not np.allclose(first, second), field


# --------------------------------------------------------------------------- #
# 5 · The case is registered everywhere it has to be
# --------------------------------------------------------------------------- #

def test_case_06_is_part_of_the_app() -> None:
    assert "montecarlo" in CASE_ORDER
    spec = CASES["montecarlo"]
    assert spec.number == "06"
    assert (spec.solar_kwp, spec.battery_kwh, spec.grid_import_kw) == (180, 300, 260)
    from vessim_pypsa_dashboard import vessim_system
    assert "montecarlo" in vessim_system.SYSTEMS


def test_case_06_has_a_network_that_converges() -> None:
    from vessim_pypsa_dashboard import grid, vessim_system
    from vessim_pypsa_dashboard.ui import data

    spec = CASES["montecarlo"]
    run = data.vessim_case("montecarlo", float(spec.carbon_price), 1.0, 1.0)
    net = vessim_system.site_network(run, spec, spec.horizon_hours)
    result = grid.run_pf(net, start="dc")
    assert result.converged
    assert result.summary["losses_mw"] >= 0.0
    system = vessim_system.SYSTEMS["montecarlo"]
    assert {"training", "inference", "storage"} <= {a for g in system.groups for a in g.actors}
