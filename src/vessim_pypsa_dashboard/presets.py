"""Quick-Start networks, ported 1:1 from Adjie Brotot's Load Flow Simulator.

Every preset in this module is a transcription of ``LF.PRESETS`` in
``loadflow-simulator/lf-core.js`` (github.com/adjiebrotot/tool, GPL-3.0): the
same buses, the same impedances, the same set points, converted from the saved
form of that tool to :class:`~vessim_pypsa_dashboard.grid.PowerNetwork`.

The fifth preset, *Ecopetrol industrial*, is new: it is the electrical
skeleton used by the industrial case study of the dashboard.
"""

from __future__ import annotations

from .grid import PowerNetwork, LINE_TYPES

Z_WSCC = 230 * 230 / 100  # ohm base of the WSCC 9-bus system (100 MVA, 230 kV)


def wscc9() -> PowerNetwork:
    """WSCC 9-bus system (Anderson & Fouad), published load flow."""
    rows: dict[str, list[dict]] = {
        "buses": [
            dict(name="Bus 1", v_nom=16.5, v_set=1.04, type="Slack"),
            dict(name="Bus 2", v_nom=18.0, v_set=1.025, type="PV"),
            dict(name="Bus 3", v_nom=13.8, v_set=1.025, type="PV"),
            dict(name="Bus 4", v_nom=230.0, type="PQ"),
            dict(name="Bus 5", v_nom=230.0, type="PQ"),
            dict(name="Bus 6", v_nom=230.0, type="PQ"),
            dict(name="Bus 7", v_nom=230.0, type="PQ"),
            dict(name="Bus 8", v_nom=230.0, type="PQ"),
            dict(name="Bus 9", v_nom=230.0, type="PQ"),
        ],
        "transformers": [
            dict(name="T1", bus0="Bus 4", bus1="Bus 1", x=0.0576 * 247.5 / 100, s_nom=247.5),
            dict(name="T2", bus0="Bus 7", bus1="Bus 2", x=0.0625 * 192 / 100, s_nom=192.0),
            dict(name="T3", bus0="Bus 9", bus1="Bus 3", x=0.0586 * 128 / 100, s_nom=128.0),
        ],
        "lines": [
            dict(name="Line 4-5", bus0="Bus 4", bus1="Bus 5", r=0.010 * Z_WSCC, x=0.085 * Z_WSCC,
                 b=0.176 / Z_WSCC, s_nom=250),
            dict(name="Line 4-6", bus0="Bus 4", bus1="Bus 6", r=0.017 * Z_WSCC, x=0.092 * Z_WSCC,
                 b=0.158 / Z_WSCC, s_nom=250),
            dict(name="Line 5-7", bus0="Bus 5", bus1="Bus 7", r=0.032 * Z_WSCC, x=0.161 * Z_WSCC,
                 b=0.306 / Z_WSCC, s_nom=250),
            dict(name="Line 6-9", bus0="Bus 6", bus1="Bus 9", r=0.039 * Z_WSCC, x=0.170 * Z_WSCC,
                 b=0.358 / Z_WSCC, s_nom=250),
            dict(name="Line 7-8", bus0="Bus 7", bus1="Bus 8", r=0.0085 * Z_WSCC, x=0.072 * Z_WSCC,
                 b=0.149 / Z_WSCC, s_nom=250),
            dict(name="Line 8-9", bus0="Bus 8", bus1="Bus 9", r=0.0119 * Z_WSCC, x=0.1008 * Z_WSCC,
                 b=0.209 / Z_WSCC, s_nom=250),
        ],
        "generators": [
            dict(name="G1", bus="Bus 1", control="Slack", p_set=0.0, p_nom=247.5, carrier="gas"),
            dict(name="G2", bus="Bus 2", control="PV", p_set=163.0, p_nom=192.0, carrier="gas"),
            dict(name="G3", bus="Bus 3", control="PV", p_set=85.0, p_nom=128.0, carrier="gas"),
        ],
        "loads": [
            dict(name="Load A", bus="Bus 5", p_set=125.0, q_set=50.0),
            dict(name="Load B", bus="Bus 6", p_set=90.0, q_set=30.0),
            dict(name="Load C", bus="Bus 8", p_set=100.0, q_set=35.0),
        ],
    }
    return PowerNetwork.from_rows("WSCC 9-bus", rows, source="Adjie Brotot preset: wscc9")


def three_bus() -> PowerNetwork:
    """Three 110 kV buses in a ring: a grid connection, a power station, a town."""
    line_type = "243-AL1/39-ST1A 110.0"
    rows = {
        "buses": [
            dict(name="Bus 1", v_nom=110.0, type="Slack"),
            dict(name="Bus 2", v_nom=110.0, v_set=1.02, type="PV"),
            dict(name="Bus 3", v_nom=110.0, type="PQ"),
        ],
        "lines": [
            dict(name="Line 1-2", bus0="Bus 1", bus1="Bus 2", type=line_type, length=40.0,
                 s_nom=123.0, num_parallel=1),
            dict(name="Line 1-3", bus0="Bus 1", bus1="Bus 3", type=line_type, length=30.0,
                 s_nom=123.0, num_parallel=1),
            dict(name="Line 2-3", bus0="Bus 2", bus1="Bus 3", type=line_type, length=25.0,
                 s_nom=123.0, num_parallel=1),
        ],
        "generators": [
            dict(name="Grid", bus="Bus 1", control="Slack", grid=True, carrier="grid"),
            dict(name="Power station", bus="Bus 2", control="PV", p_set=60.0, p_nom=100.0, carrier="gas"),
        ],
        "loads": [dict(name="Town", bus="Bus 3", p_set=100.0, q_set=30.0)],
    }
    return PowerNetwork.from_rows("3-bus meshed 110 kV", rows, source="Adjie Brotot preset: 3bus")


def feeder() -> PowerNetwork:
    """0.63 MVA 20/0.4 kV substation feeding three street pillars."""
    cable = "NAYY 4x150 SE"
    rows = {
        "buses": [
            dict(name="MV 20 kV", v_nom=20.0, type="Slack"),
            dict(name="LV board", v_nom=0.4, type="PQ"),
            dict(name="Pillar 1", v_nom=0.4, type="PQ"),
            dict(name="Pillar 2", v_nom=0.4, type="PQ"),
            dict(name="Pillar 3", v_nom=0.4, type="PQ"),
        ],
        "transformers": [
            dict(name="Substation", bus0="MV 20 kV", bus1="LV board", type="0.63 MVA 20/0.4 kV"),
        ],
        "lines": [
            dict(name="Cable 1", bus0="LV board", bus1="Pillar 1", type=cable, length=0.15, s_nom=0.187),
            dict(name="Cable 2", bus0="Pillar 1", bus1="Pillar 2", type=cable, length=0.12, s_nom=0.187),
            dict(name="Cable 3", bus0="Pillar 2", bus1="Pillar 3", type=cable, length=0.12, s_nom=0.187),
        ],
        "generators": [
            dict(name="Grid", bus="MV 20 kV", control="Slack", grid=True, carrier="grid"),
            dict(name="Rooftop solar", bus="Pillar 3", control="PQ", p_set=0.04, p_nom=0.05, carrier="solar"),
        ],
        "loads": [
            dict(name="Homes 1", bus="Pillar 1", p_set=0.06, q_set=0.015),
            dict(name="Homes 2", bus="Pillar 2", p_set=0.06, q_set=0.015),
            dict(name="Homes 3", bus="Pillar 3", p_set=0.05, q_set=0.012),
        ],
        "shunts": [dict(name="Capacitor", bus="LV board", b=0.02 / (0.4 * 0.4))],
    }
    return PowerNetwork.from_rows("20/0.4 kV feeder", rows, source="Adjie Brotot preset: feeder")


def hvdc() -> PowerNetwork:
    """Two 220 kV areas, each with its own slack, joined by an HVDC link."""
    line_type = "Al/St 240/40 2-bundle 220.0"
    rows = {
        "buses": [
            dict(name="Area A", v_nom=220.0, type="Slack"),
            dict(name="Converter A", v_nom=220.0, type="PQ"),
            dict(name="Converter B", v_nom=220.0, type="PQ"),
            dict(name="Area B", v_nom=220.0, v_set=1.01, type="Slack"),
        ],
        "lines": [
            dict(name="Line A", bus0="Area A", bus1="Converter A", type=line_type, length=80.0,
                 num_parallel=2, s_nom=440.0),
            dict(name="Line B", bus0="Converter B", bus1="Area B", type=line_type, length=120.0,
                 num_parallel=2, s_nom=440.0),
        ],
        "links": [
            dict(name="HVDC link", bus0="Converter A", bus1="Converter B", p_set=150.0,
                 efficiency=0.97, p_nom=200.0),
        ],
        "generators": [
            dict(name="Grid A", bus="Area A", control="Slack", grid=True, carrier="grid"),
            dict(name="Grid B", bus="Area B", control="Slack", grid=True, carrier="grid"),
        ],
        "loads": [
            dict(name="City A", bus="Converter A", p_set=50.0, q_set=15.0),
            dict(name="City B", bus="Area B", p_set=260.0, q_set=60.0),
        ],
    }
    return PowerNetwork.from_rows("Two areas and an HVDC link", rows, source="Adjie Brotot preset: hvdc")


def ecopetrol() -> PowerNetwork:
    """Industrial 110 kV network for the Ecopetrol case study (new preset).

    A point of interconnection (slack), a solar farm, the process plant and the
    data centre.  The corridor into the data centre is deliberately weak, so
    line limits bind and locational prices separate once PyPSA optimises it.
    """
    line_type = "243-AL1/39-ST1A 110.0"
    rows = {
        "buses": [
            dict(name="POI 110 kV", v_nom=110.0, type="Slack", carrier="grid"),
            dict(name="Solar 110 kV", v_nom=110.0, type="PV", v_set=1.01),
            dict(name="Plant 110 kV", v_nom=110.0, type="PQ"),
            dict(name="Data center 110 kV", v_nom=110.0, type="PQ"),
        ],
        "lines": [
            dict(name="POI–Solar", bus0="POI 110 kV", bus1="Solar 110 kV", type=line_type,
                 length=18.0, s_nom=100.0),
            dict(name="POI–Plant", bus0="POI 110 kV", bus1="Plant 110 kV", type=line_type,
                 length=22.0, s_nom=120.0),
            dict(name="Solar–Plant", bus0="Solar 110 kV", bus1="Plant 110 kV", type=line_type,
                 length=15.0, s_nom=80.0),
            dict(name="Plant–Data center", bus0="Plant 110 kV", bus1="Data center 110 kV",
                 type=line_type, length=9.0, s_nom=45.0),
            dict(name="Solar–Data center", bus0="Solar 110 kV", bus1="Data center 110 kV",
                 type=line_type, length=14.0, s_nom=45.0),
        ],
        "generators": [
            dict(name="National grid", bus="POI 110 kV", control="Slack", p_nom=150.0,
                 marginal_cost=95.0, carrier="grid"),
            dict(name="Solar farm", bus="Solar 110 kV", control="PV", p_set=12.0, p_nom=25.0,
                 marginal_cost=0.0, carrier="solar"),
        ],
        "loads": [
            dict(name="Industrial plant", bus="Plant 110 kV", p_set=18.0, q_set=6.0),
            dict(name="Data center", bus="Data center 110 kV", p_set=6.0, q_set=1.5),
        ],
        "shunts": [dict(name="Capacitor bank", bus="Data center 110 kV", b=1.0 / (110.0**2))],
    }
    return PowerNetwork.from_rows("Ecopetrol industrial 110 kV", rows,
                                  source="New preset for the Ecopetrol case study")


PRESETS = {
    "wscc9": wscc9,
    "3bus": three_bus,
    "feeder": feeder,
    "hvdc": hvdc,
    "ecopetrol": ecopetrol,
}

PRESET_INFO = {
    "wscc9": dict(
        title="WSCC 9-bus",
        blurb="Anderson & Fouad's textbook system: three units, three loads, a 230 kV ring. "
              "The published voltages are reproduced to the third decimal.",
        limits=dict(v_min=0.95, v_max=1.05, loading=100.0),
    ),
    "3bus": dict(
        title="3-bus ring",
        blurb="Three 110 kV buses in a ring: a grid connection, a power station holding 1.02 p.u. "
              "and a town drawing 100 MW.",
        limits=dict(v_min=0.95, v_max=1.05, loading=100.0),
    ),
    "feeder": dict(
        title="20/0.4 kV feeder",
        blurb="A 0.63 MVA 20/0.4 kV substation feeding three street pillars over NAYY 4x150 cable, "
              "with rooftop solar at the far end.",
        limits=dict(v_min=0.95, v_max=1.05, loading=100.0),
    ),
    "hvdc": dict(
        title="HVDC link",
        blurb="Two 220 kV areas, each with its own slack, joined by a 150 MW HVDC link that loses 3%.",
        limits=dict(v_min=0.95, v_max=1.05, loading=100.0),
    ),
    "ecopetrol": dict(
        title="Ecopetrol industrial",
        blurb="The 110 kV network of the industrial case: POI, solar farm, process plant and data "
              "centre, with a weak corridor into the data centre.",
        limits=dict(v_min=0.95, v_max=1.05, loading=100.0),
    ),
}

PRESET_DEFAULT = "wscc9"


def build_preset(key: str) -> PowerNetwork:
    if key not in PRESETS:
        raise KeyError(f"Unknown preset '{key}'. Available: {', '.join(PRESETS)}")
    return PRESETS[key]()


def line_type_names() -> list[str]:
    return sorted(LINE_TYPES)
