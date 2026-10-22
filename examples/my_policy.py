"""Write a dispatch policy in Vessim — a complete, runnable file.

    pip install vessim
    python examples/my_policy.py

A dispatch policy is a subclass of ``vessim.DispatchPolicy`` with one method:

    apply(p_delta, duration, dispatchables, grid_signals=None) -> None

``p_delta`` is what the microgrid still needs at this step, in watts, after the
actors have spoken: positive means the site has a surplus to absorb, negative
means it has a deficit to cover.  The policy's whole job is to *set* the
dispatchables so that the signed sum is exactly ``p_delta`` — Vessim computes
``p_delta - Σ power`` and raises if the remainder is not zero, not even by a
float ulp.  That is why the last dispatchable here receives the remainder
instead of a number of its own: ``a - b - (a - b)`` is exactly zero.

This file defines a rule that is not a price and not a merit order — it is a
*threshold* on the price, the kind of thing an operator writes on a whiteboard:
fill the battery when electricity is cheap, empty it when it is expensive,
otherwise stay out of the way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vessim as vs


class ThriftyPolicy(vs.DispatchPolicy):
    """Absorb your own surplus; spend it when the grid is expensive.

    The rule never imports to charge — the export tariff (35 % of the price) is
    below the import price, so buying to sell is a loss — and never discharges
    into the grid: it only serves the site's deficit.  Twelve lines of logic,
    and PyPSA cannot represent any of them as a constraint, because they are
    conditional on the *sign* of the balance at every step.
    """

    def __init__(self, expensive: float = 0.20, default_price: float = 0.18):
        super().__init__()
        self.expensive = float(expensive)      # USD/kWh above which the battery serves
        self.default_price = float(default_price)
        self.log: list[dict] = []              # optional: what the rule saw

    def apply(self, p_delta, duration, dispatchables, grid_signals=None) -> None:
        price = self.default_price
        if grid_signals and grid_signals.get("price") is not None:
            price = float(grid_signals["price"])

        battery = next(d for d in dispatchables if isinstance(d, vs.Storage))
        grid = next(d for d in dispatchables if isinstance(d, vs.GridConnection))
        low, high = battery.feasible_range(duration)
        g_low, g_high = grid.feasible_range(duration)

        # 1. what should the battery do? (positive = charge)
        if p_delta > 0:                       # surplus: absorb it, never more
            want = min(high, p_delta)
        elif price > self.expensive:          # expensive deficit: serve it, never more
            want = max(low, p_delta)
        else:
            want = 0.0

        # 2. the grid has to close the balance, so leave it room
        want = max(low, min(high, want, p_delta - g_low))

        # 3. write every dispatchable, so nothing keeps a stale setpoint
        for d in dispatchables:
            if d is not battery and d is not grid:
                d.set_power(0.0, duration)
        battery.set_power(want, duration)

        # 4. the grid takes exactly what is left: `a - b - (a - b)` is zero,
        #    which is the only residual Vessim accepts
        left = p_delta - want
        if not g_low <= left <= g_high:
            raise RuntimeError(f"the grid cannot absorb {left:,.1f} W at this step")
        grid.set_power(left, duration)

        self.log.append({"price": price, "battery_w": want, "grid_w": left})


def day(hours: int = 24, step_s: int = 900) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A synthetic day: a flat 700 W load, a winter sun, a price and a carbon shape."""
    t = np.arange(hours * 3600 // step_s) * step_s / 3600.0
    load = np.full_like(t, 700.0)
    solar = 1800.0 * np.maximum(np.sin(np.pi * (t % 24 - 8.0) / 10.0), 0.0) ** 1.5
    price = 0.12 + 0.10 * np.exp(-(((t % 24) - 19.0) / 3.0) ** 2)   # evening peak, USD/kWh
    return load, solar, price


def main() -> None:
    _, solar, price = day()
    step_s = 900

    def trace(values: np.ndarray, scale: float = 1.0) -> vs.Trace:
        index = np.arange(len(values)) * step_s
        return vs.Trace(pd.Series(values * scale, index=index))

    policy = ThriftyPolicy(expensive=0.20)

    env = vs.Environment(sim_start="2026-01-01", step_size=step_s, name="thrifty")
    env.add_microgrid(
        name="site",
        actors=[vs.Consumer("facility", signal=vs.StaticSignal(700.0)),
                vs.Producer("solar", signal=trace(solar, 1.0))],
        dispatchables=[vs.SimpleBattery("battery", capacity=6_000.0, initial_soc=0.4,
                                        c_rate=0.5),
                       vs.GridConnection("grid", max_import=3_000.0, max_export=3_000.0)],
        policy=policy,
        grid_signals={"price": trace(price, 1.0)},   # Vessim's Trace is in W/hour units
    )
    recorder = env.add_controller(vs.Recorder(run_id="thrifty"))
    env.run(until=24 * 3600)

    df = recorder.to_df()
    dt_h = step_s / 3600.0
    # Vessim's sign convention: for a dispatchable, positive = consumption of the
    # microgrid's surplus, except for the grid connection, whose recorder column
    # is positive when it *exports*.  The metrics are the authority:
    metrics = recorder.metrics["microgrids"]["site"]
    imported = metrics["grid_import"] / 1000.0        # kWh
    exported = metrics["grid_export"] / 1000.0        # kWh
    grid = pd.Series(-df["dispatch_states.grid.current_power"].to_numpy()) / 1000.0   # kW, positive = import
    battery = pd.Series(df["dispatch_states.battery.current_power"].to_numpy()) / 1000.0
    cost = float((grid.clip(lower=0) * price).sum() * dt_h) \
        - 0.35 * float((grid.clip(upper=0).abs() * price).sum() * dt_h)
    surplus = pd.Series(solar / 1000.0 - 0.7)                   # kW, solar minus load
    deficit = (-surplus).clip(lower=0.0)
    spilled = surplus.clip(lower=0.0)
    base_import = float((deficit * price).sum() * dt_h) - 0.35 * float((spilled * price).sum() * dt_h)

    print(f"steps recorded   {len(df)}")
    print(f"battery charged  {battery.clip(lower=0).sum() * dt_h:6.2f} kWh")
    print(f"battery emptied  {(-battery.clip(upper=0)).sum() * dt_h:6.2f} kWh")
    print(f"grid imported    {imported:6.2f} kWh   exported {exported:5.2f} kWh")
    print(f"energy bill      {cost:6.2f} USD")
    print(f"same day, no battery, no rule: {base_import:6.2f} USD")
    print(f"decisions logged {len(policy.log)}   peak import "
          f"{grid.max() / 1000.0:.2f} MW")


if __name__ == "__main__":
    main()
