"""Print the whole comparison without a browser.

    python examples/headless_report.py            # five Vessim cases
    python examples/headless_report.py --educ     # ED/UC presets too
    python examples/headless_report.py --csv      # also write reports/*.csv

Useful for a paper appendix, for CI artefacts or to check a change quickly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from vessim_pypsa_dashboard.analysis import (  # noqa: E402
    manual_summary,
    opf_summary,
    prepare_manual_case,
    run_manual_uc,
)
from vessim_pypsa_dashboard.ed_presets import ED_PRESET_INFO, build_ed_preset  # noqa: E402
from vessim_pypsa_dashboard.pypsa_models import (  # noqa: E402
    SiteSpec,
    build_opf_network,
    run_site_optimisation,
    solve_opf,
)
from vessim_pypsa_dashboard.vessim_runner import CASES, run_case  # noqa: E402
from vessim_pypsa_dashboard.analysis import vessim_baseline_metrics  # noqa: E402
from vessim_pypsa_dashboard.ui import maths  # noqa: E402


def cases_table() -> pd.DataFrame:
    rows = []
    for key, spec in CASES.items():
        run = run_case(key)
        dt = spec.step_s / 3600.0
        base = vessim_baseline_metrics(run, spec, dt)
        site = SiteSpec(
            grid_limit_mw=spec.grid_import_kw / 1000.0,
            solar_mw=max(spec.solar_kwp / 1000.0, 1e-3),
            gas_mw=spec.gas_kw / 1000.0,
            battery_mw=max(spec.battery_kw / 1000.0, 1e-3),
            battery_hours=max(spec.battery_kwh / max(spec.battery_kw, 1e-9), 1e-3),
            line_grid_mw=spec.grid_import_kw / 1000.0,
            grid_price=spec.price_base * 1000.0,
            grid_carbon=spec.grid_carbon,
            gas_price=spec.gas_cost_usd_kwh * 1000.0,
            gas_carbon=spec.gas_co2,
            carbon_price=spec.carbon_price,
        )
        opt = run_site_optimisation(run.df, site, dt)
        # the same six-term ledger the maths page shows, so the appendix and the
        # app can never disagree
        run_like = {"df": run.df, "dt": dt, "spec": spec, "baseline": base}
        opt_like = {
            "dispatch": opt.dispatch, "soc": opt.soc, "prices": opt.prices,
            "energy_cost": opt.energy_cost, "carbon_cost": opt.carbon_cost,
            "curtailment_mwh": opt.curtailment_mwh,
            "battery_throughput_mwh": opt.battery_throughput_mwh,
            "storage_discharge_mwh": float(
                opt.network.storage_units_t.p.clip(lower=0).to_numpy().sum() * dt)
            if len(opt.network.storage_units_t.p.columns) else 0.0,
            "storage_marginal_cost": float(site.storage_marginal_cost),
        }
        _, info = maths.gap_table(run_like, site, opt_like)
        rows.append({
            "case": key,
            "steps": spec.steps,
            "dt_min": spec.step_s / 60,
            "vessim_usd": info["vessim_bill"],
            "pypsa_usd": info["pypsa_cost"],
            "naive_gap": info["gap_bill"],
            "fair_gap": info["gap_fair"],
            "closed_gap": info["gap"],
            "ledger_residual": info["residual"],
            "saving_pct": (100 * info["gap_bill"] / info["vessim_bill"]
                           if abs(info["vessim_bill"]) > 1e-6 else float("nan")),
            "vessim_kg": base["emissions_t"] * 1000,
            "pypsa_kg": opt.emissions_t * 1000,
        })
    return pd.DataFrame(rows)


def educ_table() -> pd.DataFrame:
    rows = []
    for key, info in ED_PRESET_INFO.items():
        net = build_ed_preset(key)
        case = prepare_manual_case(net, info)
        uc = run_manual_uc(case, reserve_frac=info["reserve"], voll=info["voll"])
        opf = solve_opf(build_opf_network(net, case.load_ts, case.vre_pu, mode=info["mode"],
                                         hours=info["hours"],
                                         reserve_mw=info["reserve"] * float(case.net_load.max())),
                        info["mode"])
        manual = manual_summary(uc, net)
        optimal = opf_summary(opf, net)
        rows.append({
            "preset": key,
            "mode": info["mode"],
            "hours": info["hours"],
            "manual_usd": manual["total_cost_usd"],
            "pypsa_usd": optimal["total_cost_usd"],
            "gap_pct": 100 * (manual["total_cost_usd"] - optimal["total_cost_usd"])
                       / optimal["total_cost_usd"],
            "manual_tco2": manual["emissions_t"],
            "pypsa_tco2": optimal["emissions_t"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--educ", action="store_true", help="also run the ED/UC presets")
    parser.add_argument("--csv", action="store_true", help="write reports/*.csv")
    args = parser.parse_args()

    pd.set_option("display.width", 140)
    cases = cases_table()
    print("\nVessim vs PyPSA on the five official examples")
    print(cases.round(2).to_string(index=False))

    educ = educ_table() if args.educ else None
    if educ is not None:
        print("\nManual ED/UC vs PyPSA")
        print(educ.round(2).to_string(index=False))

    if args.csv:
        out = ROOT / "reports"
        out.mkdir(exist_ok=True)
        cases.to_csv(out / "cases.csv", index=False)
        if educ is not None:
            educ.to_csv(out / "educ.csv", index=False)
        print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
