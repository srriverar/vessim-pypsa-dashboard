"""Quick-Start networks for the ED/UC view.

Transcribed from ``ED.PRESETS`` in ``ed-uc-simulator/ed-core.js``
(github.com/adjiebrotot/tool, GPL-3.0), plus one new preset for the Ecopetrol
industrial case.  The profiles refer to :mod:`vessim_pypsa_dashboard.profiles`,
which carries the same daily shapes as the original tool.

The original tool treats ``s_nom = 0`` on a line as *no limit* (MATPOWER's
convention); PyPSA would read it as *zero capacity*, so
:func:`vessim_pypsa_dashboard.pypsa_models.build_opf_network` maps it to a large
rating.
"""

from __future__ import annotations

from .grid import PowerNetwork

# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def pjm5() -> PowerNetwork:
    """PJM 5-bus system (Li & Bo, 2010), the tool's default Quick Start."""
    Z = 230 * 230 / 100
    rows = {
        "buses": [
            dict(name="A", v_nom=230.0, type="Slack"),
            dict(name="B", v_nom=230.0, type="PQ"),
            dict(name="C", v_nom=230.0, type="PQ"),
            dict(name="D", v_nom=230.0, type="PQ"),
            dict(name="E", v_nom=230.0, type="PQ"),
        ],
        "lines": [
            dict(name="A-B", bus0="A", bus1="B", r=0.00281 * Z, x=0.0281 * Z, s_nom=400.0),
            dict(name="A-D", bus0="A", bus1="D", r=0.00304 * Z, x=0.0304 * Z, s_nom=0.0),
            dict(name="A-E", bus0="A", bus1="E", r=0.00064 * Z, x=0.0064 * Z, s_nom=0.0),
            dict(name="B-C", bus0="B", bus1="C", r=0.00108 * Z, x=0.0108 * Z, s_nom=0.0),
            dict(name="C-D", bus0="C", bus1="D", r=0.00297 * Z, x=0.0297 * Z, s_nom=0.0),
            dict(name="D-E", bus0="D", bus1="E", r=0.00297 * Z, x=0.0297 * Z, s_nom=240.0),
        ],
        "generators": [
            dict(name="Alta", bus="A", p_nom=40, marginal_cost=14, carrier="gas"),
            dict(name="Park City", bus="A", p_nom=170, marginal_cost=15, carrier="gas"),
            dict(name="Solitude", bus="C", p_nom=520, marginal_cost=30, carrier="coal"),
            dict(name="Sundance", bus="D", p_nom=200, marginal_cost=40, carrier="gas"),
            dict(name="Brighton", bus="E", p_nom=600, marginal_cost=10, carrier="coal"),
        ],
        "loads": [
            dict(name="Load B", bus="B", p_set=300.0),
            dict(name="Load C", bus="C", p_set=300.0),
            dict(name="Load D", bus="D", p_set=400.0),
        ],
    }
    return PowerNetwork.from_rows("PJM 5-bus", rows, source="Adjie Brotot ED preset: pjm5")


def uc3() -> PowerNetwork:
    """Three thermal units on one bus: baseload, combined cycle, peaker."""
    rows = {
        "buses": [dict(name="System", v_nom=220.0, type="Slack")],
        "generators": [
            dict(name="Coal", bus="System", p_nom=380, marginal_cost=28, committable=True,
                 p_min_pu=0.45, start_up_cost=18000, min_up_time=8, min_down_time=8,
                 ramp_limit_up=0.25, ramp_limit_down=0.25, carrier="coal"),
            dict(name="Combined cycle", bus="System", p_nom=240, marginal_cost=52,
                 committable=True, p_min_pu=0.40, start_up_cost=3000, min_up_time=4,
                 min_down_time=3, ramp_limit_up=0.5, ramp_limit_down=0.5, carrier="gas"),
            dict(name="Peaker", bus="System", p_nom=120, marginal_cost=115, committable=True,
                 p_min_pu=0.20, start_up_cost=900, min_up_time=1, min_down_time=1,
                 init=0, carrier="gas"),
        ],
        "loads": [dict(name="Demand", bus="System", p_set=640.0)],
    }
    return PowerNetwork.from_rows("Three-unit commitment", rows, source="Adjie Brotot ED preset: uc3")


def solar_battery() -> PowerNetwork:
    """Solar farm, gas turbine, peaker, battery and two demand shapes."""
    rows = {
        "buses": [dict(name="Town", v_nom=132.0, type="Slack")],
        "generators": [
            dict(name="Solar farm", bus="Town", p_nom=420, marginal_cost=0, carrier="solar"),
            dict(name="Gas turbine", bus="Town", p_nom=260, marginal_cost=62, committable=True,
                 p_min_pu=0.35, start_up_cost=4000, min_up_time=4, min_down_time=2,
                 ramp_limit_up=0.6, ramp_limit_down=0.6, carrier="gas"),
            dict(name="Peaker", bus="Town", p_nom=150, marginal_cost=140, committable=True,
                 p_min_pu=0.20, start_up_cost=600, min_up_time=1, min_down_time=1,
                 init=0, carrier="gas"),
        ],
        "storages": [
            dict(name="Battery", bus="Town", p_nom=100.0, max_hours=4.0, efficiency_store=0.88,
                 efficiency_dispatch=0.88, soc0=0.5, marginal_cost=0.0),
        ],
        "loads": [
            dict(name="Homes", bus="Town", p_set=300.0),
            dict(name="Shops and offices", bus="Town", p_set=180.0),
        ],
    }
    return PowerNetwork.from_rows("Solar and a battery", rows, source="Adjie Brotot ED preset: solar")


def areas() -> PowerNetwork:
    """Windy north and gas-fired south joined by a 250 MW HVDC link."""
    rows = {
        "buses": [
            dict(name="North", v_nom=275.0, type="Slack"),
            dict(name="South", v_nom=275.0, type="PQ"),
        ],
        "links": [dict(name="HVDC link", bus0="North", bus1="South", p_nom=250.0, efficiency=0.97)],
        "generators": [
            dict(name="Wind farms", bus="North", p_nom=600, marginal_cost=0, carrier="wind"),
            dict(name="Hydro", bus="North", p_nom=150, marginal_cost=18, ramp_limit_up=0.5,
                 ramp_limit_down=0.5, carrier="hydro"),
            dict(name="Combined cycle", bus="South", p_nom=350, marginal_cost=55, committable=True,
                 p_min_pu=0.40, start_up_cost=7000, min_up_time=6, min_down_time=4,
                 ramp_limit_up=0.4, ramp_limit_down=0.4, carrier="gas"),
            dict(name="Peakers", bus="South", p_nom=250, marginal_cost=120, committable=True,
                 p_min_pu=0.20, start_up_cost=1200, min_up_time=1, min_down_time=1,
                 init=0, carrier="gas"),
        ],
        "loads": [
            dict(name="North load", bus="North", p_set=220.0),
            dict(name="South load", bus="South", p_set=620.0),
        ],
    }
    return PowerNetwork.from_rows("Two areas, wind and an HVDC link", rows,
                                  source="Adjie Brotot ED preset: areas")


def ecopetrol() -> PowerNetwork:
    """Industrial case: a plant and a data centre behind a constrained corridor."""
    rows = {
        "buses": [
            dict(name="POI", v_nom=110.0, type="Slack", carrier="grid"),
            dict(name="Plant", v_nom=110.0, type="PQ"),
            dict(name="Data center", v_nom=110.0, type="PQ"),
        ],
        "lines": [
            dict(name="POI–Plant", bus0="POI", bus1="Plant", r=2.6, x=8.6, s_nom=120.0),
            dict(name="POI–Data center", bus0="POI", bus1="Data center", r=3.2, x=10.4, s_nom=30.0),
            dict(name="Plant–Data center", bus0="Plant", bus1="Data center", r=1.8, x=6.2, s_nom=30.0),
        ],
        "generators": [
            dict(name="National grid", bus="POI", p_nom=40, marginal_cost=95, carrier="grid"),
            dict(name="Gas turbine", bus="Plant", p_nom=20, marginal_cost=55, committable=True,
                 p_min_pu=0.30, start_up_cost=2500, min_up_time=3, min_down_time=2,
                 ramp_limit_up=0.5, ramp_limit_down=0.5, carrier="gas"),
            dict(name="Solar farm", bus="Plant", p_nom=25, marginal_cost=0, carrier="solar"),
        ],
        "storages": [
            dict(name="BESS", bus="Data center", p_nom=10.0, max_hours=4.0, efficiency_store=0.95,
                 efficiency_dispatch=0.95, soc0=0.5, marginal_cost=0.0),
        ],
        "loads": [
            dict(name="Industrial plant", bus="Plant", p_set=18.0),
            dict(name="Data center", bus="Data center", p_set=6.0),
        ],
    }
    return PowerNetwork.from_rows("Ecopetrol industrial (ED/UC)", rows,
                                  source="New preset for the Ecopetrol case study")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

ED_PRESETS = {
    "pjm5": pjm5,
    "uc3": uc3,
    "solar": solar_battery,
    "areas": areas,
    "ecopetrol": ecopetrol,
}

ED_PRESET_INFO = {
    "pjm5": dict(
        title="PJM 5-bus",
        blurb="Li & Bo's five-bus system, dispatched over a day. At the 18:00 peak it reproduces "
              "the published nodal prices; A-B and D-E are the only congested corridors.",
        mode="edm", hours=24, hour=18, reserve=0.05, voll=10_000.0,
        load_profiles={"Load B": "sys", "Load C": "sys", "Load D": "sys"},
        gen_profiles={},
    ),
    "uc3": dict(
        title="Three-unit commitment",
        blurb="A slow cheap baseload unit, a combined cycle that can cycle daily and a quick dear "
              "peaker: the classic case where the priority list and the MILP disagree.",
        mode="uc", hours=24, hour=18, reserve=0.10, voll=10_000.0,
        load_profiles={"Demand": "sys"},
        gen_profiles={},
    ),
    "solar": dict(
        title="Solar and a battery",
        blurb="A solar farm, a must-run gas turbine, a peaker and a four-hour battery serving "
              "homes and shops: renewables, storage and commitment in one screen.",
        mode="uc", hours=24, hour=12, reserve=0.05, voll=10_000.0,
        load_profiles={"Homes": "res", "Shops and offices": "com"},
        gen_profiles={"Solar farm": "pv"},
    ),
    "areas": dict(
        title="Two areas",
        blurb="A windy north and a gas-fired south joined by a 250 MW HVDC link over three days. "
              "Prices split when the link is full.",
        mode="uc", hours=72, hour=18, reserve=0.05, voll=10_000.0,
        load_profiles={"North load": "ind", "South load": "sys"},
        gen_profiles={"Wind farms": "wnd"},
    ),
    "ecopetrol": dict(
        title="Ecopetrol industrial",
        blurb="Plant, data centre, captive gas turbine, solar farm and BESS behind a weak 110 kV "
              "corridor: congestion and carbon cost meet unit commitment.",
        mode="uc", hours=24, hour=18, reserve=0.08, voll=20_000.0,
        load_profiles={"Industrial plant": "ind", "Data center": "sys"},
        gen_profiles={"Solar farm": "pveq"},
    ),
}

ED_PRESET_DEFAULT = "pjm5"


def build_ed_preset(key: str) -> PowerNetwork:
    if key not in ED_PRESETS:
        raise KeyError(f"Unknown ED preset '{key}'. Available: {', '.join(ED_PRESETS)}")
    return ED_PRESETS[key]()


def scaled_loads(net: PowerNetwork, profiles: dict[str, str], hours: int):
    """Return a DataFrame (hours x loads) of absolute MW demand."""
    from .profiles import shape

    columns = {}
    for name, row in net.loads.iterrows():
        prof = profiles.get(name, "sys")
        multipliers = shape(prof, hours)
        columns[name] = [float(row["p_set"]) * m for m in multipliers]
    return __import__("pandas").DataFrame(columns)
