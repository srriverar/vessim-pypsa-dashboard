"""The five paper reproductions: identities, ranges and the page that shows them."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from vessim_pypsa_dashboard import papers


def _day(n: int = 96, seed: int = 3):
    rng = np.random.default_rng(seed)
    hours = np.arange(n) / (n / 24.0)
    load = 8.0 + 3.0 * np.sin(2 * np.pi * (hours - 9) / 24.0) + rng.normal(0, 0.2, n)
    solar = np.clip(7.0 * np.sin(np.pi * (hours - 6) / 12.0), 0.0, None)
    carbon = 0.35 + 0.2 * np.sin(2 * np.pi * (hours - 4) / 24.0)
    return np.clip(load, 0.5, None), solar, np.clip(carbon, 0.03, None)


# --------------------------------------------------------------------------- #
# 1 · carbon-aware shifting
# --------------------------------------------------------------------------- #

def test_shifting_conserves_energy_and_never_adds_carbon():
    load, _, carbon = _day()
    result = papers.shifting_value(load, carbon, 60.0, flexible=0.35)
    before = float(result["profile_before"].sum())
    after = float(result["profile_after"].sum())
    assert abs(before - after) < 1e-6 * max(before, 1.0)
    assert result["carbon_after_kg"] <= result["carbon_before_kg"] + 1e-9
    assert result["avoided_kg"] >= 0.0
    assert result["avoided_usd"] == pytest.approx(result["avoided_kg"] / 1000.0 * 60.0, rel=1e-9)


def test_shifting_moves_only_the_flexible_share():
    load, _, carbon = _day()
    for share in (0.0, 0.35, 0.8):
        result = papers.shifting_value(load, carbon, 0.0, flexible=share)
        assert result["share_moved"] <= 1.0
        if share == 0.0:
            assert result["avoided_kg"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 2 · software in the loop
# --------------------------------------------------------------------------- #

def test_power_model_stays_between_idle_and_full():
    model = papers.PowerModel()
    seconds = np.arange(1.0, 301.0)
    cpu = np.linspace(0.0, 1.0, len(seconds))
    result = papers.sil_energy(seconds, cpu, model)
    assert result["power_w"].min() >= model.idle_w - 1e-9
    assert result["power_w"].max() <= model.idle_w + model.slope_w + 1e-9
    assert result["energy_wh"] == pytest.approx(result["mean_w"] * len(seconds) / 3600.0, rel=1e-9)
    assert result["mean_cpu"] == pytest.approx(float(cpu.mean()))


# --------------------------------------------------------------------------- #
# 3 · battery models
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("model", ["ideal", "linear", "kibam", "peukert"])
def test_battery_models_balance_and_respect_capacity(model):
    load, solar, _ = _day()
    capacity, power = 12.0, 6.0
    result = papers.battery_day(load, solar, capacity, power, model=model)
    net = load - solar
    assert np.allclose(result["served_kw"] + result["unmet_kw"], np.clip(net, 0, None), atol=1e-9)
    assert np.allclose(result["grid_kw"], net - result["served_kw"], atol=1e-9)
    assert result["soc_kwh"].min() >= -1e-9
    assert result["soc_kwh"].max() <= capacity + 1e-9
    assert result["unmet_kwh"] >= -1e-9


def test_lossless_battery_never_charges_more_than_the_rating():
    load, solar, _ = _day()
    result = papers.battery_day(load, solar, 12.0, 6.0, model="ideal")
    assert np.abs(result["served_kw"]).max() <= 6.0 + 1e-9


# --------------------------------------------------------------------------- #
# 4 · inference
# --------------------------------------------------------------------------- #

def test_inference_offset_never_exceeds_one_and_lowers_emissions():
    _, _, carbon = _day()
    requests = 400.0 + 3000.0 * np.exp(-((np.arange(len(carbon)) - 50) ** 2) / 40.0)
    for share in (0.0, 0.45, 0.9):
        result = papers.inference_day(requests, carbon, papers.GPUS["A100 SXM4"], shiftable=share)
        assert 0.0 <= result["offset"] <= 1.0
        assert result["emissions_after_kg"] <= result["emissions_kg"] + 1e-9
        assert result["energy_kwh"] > 0.0


def test_inference_power_rises_with_the_fleet():
    _, _, carbon = _day()
    requests = np.full(len(carbon), 1500.0)
    small = papers.inference_day(requests, carbon, papers.GPUS["H100 SXM5"], gpus=32)
    large = papers.inference_day(requests, carbon, papers.GPUS["H100 SXM5"], gpus=64)
    assert large["energy_kwh"] > small["energy_kwh"]


# --------------------------------------------------------------------------- #
# 5 · curtailment windows
# --------------------------------------------------------------------------- #

def test_pretraining_windows_follow_the_threshold():
    _, _, carbon = _day()
    carbons = [np.roll(carbon, k) for k in (0, 32, 64)]
    tight = papers.pretraining_day(carbons, 0.10)
    loose = papers.pretraining_day(carbons, 0.40)
    assert tight["hours_available"] <= loose["hours_available"]
    assert tight["energy_policy_kwh"] <= loose["energy_policy_kwh"] + 1e-9
    for window, trace in zip(loose["windows"], carbons):
        assert np.array_equal(window, trace <= 0.40)
    assert 0.0 <= loose["ratio"] <= 1.5
