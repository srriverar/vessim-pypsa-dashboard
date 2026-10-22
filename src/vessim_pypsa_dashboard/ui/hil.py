"""Hardware-in-the-loop: how to put a real device (or a real cluster) in the loop.

The page answers four questions in order:

1. **What are the three levels?**  Trace replay → software-in-the-loop (SiL) →
   hardware-in-the-loop (HIL): what is real, what is simulated, and what each one
   can therefore prove.
2. **What is the API contract?**  Vessim's ``Controller`` (``start``/``step``/
   ``finalize``) and ``MicrogridState``; the two adapter classes that make a real
   device look like a Vessim component (a ``Signal`` for measurement, a
   ``Dispatchable`` for actuation).
3. **Does it actually run?**  A live probe: a real process is launched, the
   kernel reports its CPU time with ``os.wait4``, and that time becomes watts.
   The numbers on screen are measured now, not quoted.
4. **What do I need for my own hardware?**  Ready-to-adapt recipes for Modbus
   TCP, OPC UA, MQTT/InfluxDB, Prometheus/Kepler and a Raspberry Pi, each with
   what it can and cannot prove, plus the safety and determinism checklist.
"""

from __future__ import annotations

import inspect

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import vessim as vs

from ..sil import SoftwareInTheLoopMonitor, WORKLOAD
from . import data, theme

LEVELS = [
    ("Trace replay", "nothing", "everything", "the numbers add up", "no feedback, no timing, no hardware"),
    ("Software-in-the-loop", "a real process", "the plant", "the controller reacts to a real workload",
     "the power model is an estimate (CPU time × watts)"),
    ("Hardware-in-the-loop", "a real device", "the grid or the site", "control loops, latency, safety",
     "scale: one device, not a fleet"),
]

RECIPES = [
    ("Measure a device: Modbus TCP meter",
     "`pymodbus` · most industrial meters and breakers expose holding registers",
     '''class ModbusMeter(vs.Signal):
    """Power of a real meter, in W, read at every step."""

    def __init__(self, host="192.168.1.50", unit=1, register=3000, scale=1.0):
        self.client = ModbusTcpClient(host, port=502)
        self.unit, self.register, self.scale = unit, register, scale
        self._w = 0.0

    def at(self, now=None):
        rr = self.client.read_holding_registers(self.register, count=2, slave=self.unit)
        raw = (rr.registers[0] << 16) | rr.registers[1]      # signed 32-bit
        self._w = raw / 1000.0 * self.scale                  # kW → W
        return self._w

# actor = vs.Consumer("hall", signal=ModbusMeter())'''),
    ("Actuate a device: a Dispatchable backed by hardware",
     "`pymodbus` / `asyncua` · the clean Vessim idiom: subclass and write",
     '''class ModbusBattery(vs.Dispatchable):
    """A battery that really charges: the setpoint goes out, the state comes back."""

    def __init__(self, name, capacity_wh, bus):
        super().__init__(name)
        self.capacity = capacity_wh
        self.bus = bus                                   # the live connection

    def feasible_range(self, duration):
        soc = self.state()["soc"]
        return (-min(self.capacity * soc / (duration / 3600), 1e9),
                min(self.capacity * (1 - soc) / (duration / 3600), 1e9))

    def set_power(self, power, duration=None):
        super().set_power(power, duration)
        self.bus.write_register(SETPOINT, int(power))     # ← the actuation

    def step(self, duration):
        self._measured = self.bus.read_register(POWER)     # ← the feedback

    def state(self):
        return {"soc": self.bus.read_register(SOC) / 100.0,
                "current_power": self._measured}'''),
    ("A real cluster: Prometheus + Kepler",
     "built in: `vs.PrometheusSignal` · Kubernetes power per pod",
     '''# Vessim ships the signal; nothing to write.
power = vs.PrometheusSignal(
    url="http://prometheus.monitoring:9090",
    query='sum(kepler_container_joules_total[1m]) / 60',   # W, averaged over a minute
    update_interval=15,
)
env.add_microgrid(actors=[vs.Consumer("gpu-pool", signal=power)])'''),
    ("A Raspberry Pi acting on the grid",
     "`gpiozero` · the cheapest HIL rig that still teaches the lesson",
     '''from gpiozero import LED, MCP3008

def at(self, now=None):                       # a Signal that reads a current clamp
    return MCP3008(channel=0).value * 230.0 * 4.2   # A × V × turns ratio

class RelayShed(vs.Dispatchable):             # a load the policy can curtail
    def set_power(self, power, duration=None):
        super().set_power(power, duration)
        (LED(17) if power > 0 else LED(27)).on() # one relay per shed step'''),
    ("Publish the whole simulation: the REST API",
     "built in: `vs.Api` · drives the microgrid from outside the process",
     '''env.add_controller(vs.Api(broker_port=8700, export_prometheus=True))
# GET  :8700/microgrids            current state of every microgrid
# POST :8700/microgrids/{name}/control   set a dispatchable from outside
# then scrape :8700 with Prometheus and plot the simulation next to the device'''),
]

CHECKLIST = [
    ("Units", "Vessim speaks W and seconds; meters speak kW, kWh and register words", "fail"),
    ("Sign", "Vessim is consumption-positive; most meters are the opposite", "fail"),
    ("Timeout", "a modbus read that hangs stalls the whole co-simulation", "fail"),
    ("Determinism", "record every input and output, or the run is not reproducible", "warn"),
    ("Safety layer", "the policy must never be the only thing standing between a setpoint and a wire", "fail"),
    ("Clock", "virtual time vs wall time: pacing is the experimenter's job (see the probe below)", "warn"),
    ("Fault injection", "pull the cable on purpose: does the policy fail safely?", "warn"),
    ("Permissions", "a dead-man switch on the actuator, not only in software", "fail"),
]


def _loop_svg() -> str:
    """A small block diagram of the loop — plain SVG, no dependencies."""
    boxes = [
        (24, "Signal\n(real device or trace)", "#38bdf8"),
        (232, "Dispatch policy\n(a rule, per step)", "#fbbf24"),
        (440, "Dispatchable\n(setpoint out, state in)", "#a78bfa"),
        (648, "Plant / grid\n(the physics)", "#34d399"),
    ]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 210" width="100%" '
             'style="display:block;max-width:100%" role="img">']
    for x, label, colour in boxes:
        parts.append(f'<rect x="{x}" y="52" width="168" height="74" rx="10" fill="#0d1629" '
                     f'stroke="{colour}" stroke-width="2"/>')
        for i, line in enumerate(label.split("\n")):
            parts.append(f'<text x="{x + 84}" y="{78 + 17 * i}" text-anchor="middle" '
                         f'font-family="Inter, system-ui" font-size="12" fill="#dbe7f7" '
                         f'font-weight="{700 if i == 0 else 400}">{line}</text>')
    for x in (192, 400, 608):
        parts.append(f'<path d="M{x} 89 L{x + 40} 89" stroke="#7f92b4" stroke-width="1.8"/>')
        parts.append(f'<path d="M{x + 40} 89l-7 -4v8z" fill="#7f92b4"/>')
    # feedback loop
    parts.append('<path d="M732 126 L732 176 L108 176 L108 126" fill="none" stroke="#7f92b4" '
                 'stroke-width="1.8" stroke-dasharray="6 4"/>')
    parts.append('<path d="M108 126l-4 7h8z" fill="#7f92b4"/>')
    parts.append('<text x="420" y="196" text-anchor="middle" font-family="Inter, system-ui" '
                 'font-size="11.5" fill="#8ea3c7">measured state back into the next decision '
                 '· the loop closes at the simulation clock</text>')
    parts.append("</svg>")
    return "".join(parts)


def render() -> None:
    theme.hero(
        "—", "Hardware in the loop",
        "MEASURE, ACT, AND CLOSE THE LOOP ON REAL EQUIPMENT",
        "Case 04 already runs a <b>real process</b>: a Python workload is launched for every "
        "simulation step, the kernel reports the CPU time it consumed, and the simulation is paced "
        "by it. This page shows that mechanism, proves it live, and lays out what changes when the "
        "process becomes a meter, a battery or a cluster.",
        [("Levels", "3"), ("Recipes", "5"), ("Live probe", "yes"), ("API", "vs.Controller")],
    )

    theme.section("Three levels of “real”", "what is real, and what it lets you claim", "CONCEPT")
    st.dataframe(pd.DataFrame(LEVELS, columns=["Level", "Real", "Simulated", "Can show", "Cannot show"]),
                 hide_index=True, **theme.wide())

    st.markdown(_loop_svg(), unsafe_allow_html=True)

    # ------------------------------------------------------------------ probe
    theme.section("Live proof: a real process, measured now", "no numbers are quoted", "PROBE")
    left, right = st.columns([1, 1.4])
    with left:
        steps = st.slider("Steps to measure", 3, 10, 6, key="hil_steps")
        step_s = st.slider("Simulated second per step", 0.2, 1.0, 0.4, 0.1, key="hil_step")
        st.caption("Each step launches a new Python process and blocks until it exits: the simulation "
                   "cannot run faster than the workload, which is what makes it software-in-the-loop.")
    with right:
        probe = data.sil_probe(int(steps), float(step_s))
        samples = probe["samples"]
        if len(samples):
            fig = go.Figure()
            fig.add_trace(go.Bar(x=samples["step"], y=samples["cpu_seconds"], name="CPU seconds",
                                 marker_color=theme.COLORS["loadflow"],
                                 hovertemplate="step %{x}: %{y:.3f} s CPU<extra></extra>"))
            fig.add_trace(go.Bar(x=samples["step"], y=samples["wall_seconds"], name="wall seconds",
                                 marker_color=theme.COLORS["vessim"],
                                 hovertemplate="step %{x}: %{y:.3f} s wall<extra></extra>"))
            theme.figure(fig, height=250, x_title="step", y_title="seconds")
            mean_cpu = float(samples["cpu_seconds"].mean())
            mean_wall = float(samples["wall_seconds"].mean())
            watts = float(samples["watts"].mean())
            theme.kpis([
                ("CPU per step", f"{mean_cpu:.3f}", "s", f"requested {step_s:.2f} s", ""),
                ("Wall per step", f"{mean_wall:.3f}", "s", "the simulation was paced by it", ""),
                ("Utilisation", f"{100 * mean_cpu / max(step_s, 1e-9):.0f}", "%", "of one core", ""),
                ("Power", f"{watts:,.0f}", "W", "PUE·(base + core·util)", "good"),
            ])
            st.dataframe(
                samples.rename(columns={"step": "Step", "cpu_seconds": "CPU (s)",
                                        "wall_seconds": "Wall (s)", "utilisation": "Utilisation",
                                        "watts": "Power (W)"})
                .style.format({"CPU (s)": "{:.3f}", "Wall (s)": "{:.3f}",
                               "Utilisation": "{:.2f}", "Power (W)": "{:.1f}"}),
                hide_index=True, **theme.wide(), height=210)
            theme.note(
                f"CPU ≈ wall because the workload saturates one core. The power model is explicit: "
                f"<code>P = PUE · (base_w + core_w · CPU_seconds / step_seconds)</code> = "
                f"1.2 · (300 + 12 · {mean_cpu / step_s:.2f}) = {watts:.0f} W. Change the workload "
                f"and these numbers change; they are not a table."
            )

    # ---------------------------------------------------------------- the API
    theme.section("The contract: one class, three methods", "what Vessim asks of a controller",
                  "API")
    left, right = st.columns([1, 1.15])
    with left:
        st.markdown(
            "<div class='note'>A <code>vs.Controller</code> is called at every step with the "
            "simulation time and a <code>MicrogridState</code>. Override <code>start</code> to open "
            "connections, <code>step</code> to measure/act, <code>finalize</code> to close them. "
            "That is the whole hardware interface.</div>", unsafe_allow_html=True)
        try:
            st.code(inspect.getsource(vs.Controller), language="python")
        except Exception:                    # pragma: no cover
            st.caption("Vessim's Controller source is unavailable in this installation.")
    with right:
        st.markdown("**What arrives in `step`, per microgrid:**")
        st.dataframe(pd.DataFrame([
            ("p_delta", "net actor + external power, W", "positive = surplus, negative = deficit"),
            ("p_grid", "net exchange across grid connections, W", "positive = export"),
            ("actor_states", "per actor observation", "signed power, W"),
            ("dispatch_states", "per dispatchable observation", "current_power, soc, charge_level"),
            ("policy_state", "diagnostics returned by the policy", "free-form"),
            ("grid_signals", "sampled exogenous signals", "price, carbon, anything you attach"),
        ], columns=["Field", "Meaning", "Sign / units"]), hide_index=True, **theme.wide())

    st.markdown("**The monitor this repository already runs (case 04):**")
    try:
        source = inspect.getsource(SoftwareInTheLoopMonitor)
        st.code(source, language="python", line_numbers=True)
    except Exception:                       # pragma: no cover
        st.code(WORKLOAD, language="python")

    # -------------------------------------------------------------- recipes
    theme.section("Recipes", "copy, adapt, and keep the units honest", "HARDWARE")
    for title, subtitle, code in RECIPES:
        with st.expander(f"{title} — {subtitle}", expanded=False):
            st.code(code, language="python")

    theme.section("Checklist before the first real run", "the eight things that break HIL rigs",
                  "SAFETY")
    st.dataframe(
        pd.DataFrame(CHECKLIST, columns=["Item", "Why it matters", "Severity"])
        .style.map(lambda value: ("background-color:#3a1414; color:#f87171" if value == "fail"
                                   else "background-color:#3a2a10; color:#fbbf24"),
                   subset=["Severity"]),
        hide_index=True, **theme.wide())
    theme.note(
        "<b>The policy must never be the last line of defence.</b> In a simulation a wrong setpoint "
        "costs a number; on a wire it costs a device. Put a dead-man timer, a current limit and a "
        "physical switch between the experiment and the plant — and test them by pulling the cable "
        "during a run."
    )

    theme.section("What a HIL run buys you scientifically", "and what it still cannot", "VALUE")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "- **Validity of the power model.** A measured trace replaces an assumption, and the "
            "error of the assumption becomes measurable.\n"
            "- **Latency and staleness.** The gap between the decision and its effect is real, and "
            "so is its cost.\n"
            "- **Faults.** Cables, timeouts and lost packets are part of the experiment.\n"
            "- **Credibility.** A reviewer accepts a measured trace where they would question a "
            "synthetic one."
        )
    with right:
        st.markdown(
            "- **It does not scale.** One device is not a fleet; the interesting economics live at "
            "fleet scale.\n"
            "- **It is not reproducible by default.** Record every input and output or nobody can "
            "repeat the run.\n"
            "- **It does not replace the optimum.** A measured trajectory still has no denominator; "
            "PyPSA still provides it.\n"
            "- **It does not remove the protocol.** Controls 1–8 apply exactly as before."
        )
