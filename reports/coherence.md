# Coherence audit

**152 checks · 0 failed · score 100.0%**

Produced by `python tools/check_coherence.py`. Every row is a statement that must hold for the dashboard to be telling the truth about the simulation; the *worst* column is the largest deviation found, not an average, so a single bad step cannot hide.

## Versions

| package | version |
|---|---|
| vessim | 0.16.0 |
| pypsa | 1.2.4 |
| streamlit | 1.66.1 |
| pandas | 2.2.3 |
| numpy | 2.3.5 |
| optuna | 5.0.0 |
| highspy | 1.15.1 |

Mode: 12 sampled snapshots per case

## basic

| check | worst | tolerance | where | statement |
|---|---|---|---|---|
| ✅ balance | 2.17e-19 MW | ≤ 1e-09 MW | step 36 | solar + gas + grid + battery − load − curtail = 0 |
| ✅ soc | 5.42e-20 MWh | ≤ 1e-09 MWh | every step | soc[t+1] = soc[t] − battery[t]·Δt (the battery is lossless) |
| ✅ soc bounds | [0.0000, 0.0010] MWh | [0, 0.0010] MWh |  | the state of charge never leaves the tank |
| ✅ c-rate | 0.0005 MW | ≤ 0.0005 MW |  | c-rate 0.50 /h |
| ✅ grid limits | [-0.0007, 0.0008] MW | [-0.0010, 0.0020] MW |  | import and export within the connection |
| ✅ cost | 0.00e+00 USD | ≤ 1e-09 USD |  | the bill recomputed from the recorded powers |
| ✅ emissions | 0.00e+00 kg | ≤ 1e-09 kg |  | the CO₂ recomputed from the same powers |
| ✅ signals | carbon 0.00e+00 kg/MWh · price 0.00e+00 USD/MWh | ≤ 1e-3 |  | the columns are the modelled signals, step for step |
| ✅ actor sum | 0.00e+00 MW | ≤ 1e-9 MW |  | 3 metered actors |
| ✅ load flow | 9.00e-10 kW | ≤ 1e-03 kW | 12 snapshots | slack = grid + losses · max losses 0.00 kW |
| ✅ voltage | 1.0000 p.u. | [0.85, 1.1] p.u. | 12 snapshots | the case's system can carry its own powers |
| ✅ ed | 6.9044e-04 MW | ≤ 2 % of the setpoint | 12 snapshots | λ-iteration vs Vessim on the same fleet |
| ✅ uc | 9.02e-11 MWh | 0 MWh | whole day | 0 start(s) · 2.19 USD |
| ✅ uc grid | 0.0008 MW | ≤ 0.0020 MW | whole day | the commitment never imports more than the connection allows |
| ✅ lp | 1.11e-16 rel. | ≤ 1e-04 |  | objective 1.99 USD recomputed from its own dispatch |
| ✅ lp balance | 1.08e-19 MW | ≤ 1e-6 MW |  | the LP's dispatch serves the net load at every step |
| ✅ bus map | 8 AC buses → 1 row each | every bus mapped once |  | 8 buses of the case's own network |
| ✅ same buses | 8 OPF buses | = the 8 of the load flow | ok | built by the same call — vessim_system.site_network |
| ✅ same branches | 7 branches | = the 7 of the load flow | 6 lines · 1 transformers | same ratings, same names |
| ✅ serves the load | 0.00e+00 MWh unserved | < 1e-6 MWh | no shedding | the network carries every recorded MWh without an emergency generator |
| ✅ poi | 8.84e-13 MW | ≤ 1e-6 MW | 6 snapshots | AC slack = the trace's import + the losses of that snapshot |
| ✅ loss gap | 0.13 % of load | [0, 10] % | 0.00 MWh/day · 0.00 USD | the one term the lossless OPF does not charge for |
| ✅ diagram | 8 buses | every bus drawn |  |  |
| ✅ system | 3 actors · 1 boards | every actor wired |  | 01 · Basic simulation |

## solar_battery

| check | worst | tolerance | where | statement |
|---|---|---|---|---|
| ✅ balance | 4.34e-19 MW | ≤ 1e-09 MW | step 14 | solar + gas + grid + battery − load − curtail = 0 |
| ✅ soc | 1.08e-19 MWh | ≤ 1e-09 MWh | every step | soc[t+1] = soc[t] − battery[t]·Δt (the battery is lossless) |
| ✅ soc bounds | [0.0000, 0.0015] MWh | [0, 0.0030] MWh |  | the state of charge never leaves the tank |
| ✅ c-rate | 0.0013 MW | ≤ 0.0015 MW |  | c-rate 0.50 /h |
| ✅ grid limits | [-0.0000, 0.0013] MW | [-0.0020, 0.0025] MW |  | import and export within the connection |
| ✅ cost | 0.00e+00 USD | ≤ 1e-09 USD |  | the bill recomputed from the recorded powers |
| ✅ emissions | 1.39e-17 kg | ≤ 1e-09 kg |  | the CO₂ recomputed from the same powers |
| ✅ signals | carbon 0.00e+00 kg/MWh · price 0.00e+00 USD/MWh | ≤ 1e-3 |  | the columns are the modelled signals, step for step |
| ✅ actor sum | 0.00e+00 MW | ≤ 1e-9 MW |  | 3 metered actors |
| ✅ load flow | 9.29e-10 kW | ≤ 1e-03 kW | 12 snapshots | slack = grid + losses · max losses 0.00 kW |
| ✅ voltage | 1.0000 p.u. | [0.85, 1.1] p.u. | 12 snapshots | the case's system can carry its own powers |
| ✅ ed | 4.3368e-19 MW | ≤ 2 % of the setpoint | 12 snapshots | λ-iteration vs Vessim on the same fleet |
| ✅ uc | 5.10e-19 MWh | 0 MWh | whole day | 0 start(s) · 5.28 USD |
| ✅ uc grid | 0.0013 MW | ≤ 0.0025 MW | whole day | the commitment never imports more than the connection allows |
| ✅ lp | 2.03e-16 rel. | ≤ 1e-04 |  | objective 4.37 USD recomputed from its own dispatch |
| ✅ lp balance | 2.17e-19 MW | ≤ 1e-6 MW |  | the LP's dispatch serves the net load at every step |
| ✅ bus map | 8 AC buses → 1 row each | every bus mapped once |  | 8 buses of the case's own network |
| ✅ same buses | 8 OPF buses | = the 8 of the load flow | ok | built by the same call — vessim_system.site_network |
| ✅ same branches | 7 branches | = the 7 of the load flow | 6 lines · 1 transformers | same ratings, same names |
| ✅ serves the load | 0.00e+00 MWh unserved | < 1e-6 MWh | no shedding | the network carries every recorded MWh without an emergency generator |
| ✅ poi | 8.78e-13 MW | ≤ 1e-6 MW | 6 snapshots | AC slack = the trace's import + the losses of that snapshot |
| ✅ loss gap | 0.17 % of load | [0, 10] % | 0.00 MWh/day · 0.01 USD | the one term the lossless OPF does not charge for |
| ✅ diagram | 8 buses | every bus drawn |  |  |
| ✅ system | 3 actors · 1 boards | every actor wired |  | 02 · Solar + battery |

## carbon_aware

| check | worst | tolerance | where | statement |
|---|---|---|---|---|
| ✅ balance | 8.67e-19 MW | ≤ 1e-09 MW | step 21 | solar + gas + grid + battery − load − curtail = 0 |
| ✅ soc | 2.17e-19 MWh | ≤ 1e-09 MWh | every step | soc[t+1] = soc[t] − battery[t]·Δt (the battery is lossless) |
| ✅ soc bounds | [0.0004, 0.0020] MWh | [0, 0.0040] MWh |  | the state of charge never leaves the tank |
| ✅ c-rate | 0.0019 MW | ≤ 0.0020 MW |  | c-rate 0.50 /h |
| ✅ grid limits | [-0.0001, 0.0022] MW | [-0.0020, 0.0030] MW |  | import and export within the connection |
| ✅ cost | 0.00e+00 USD | ≤ 1e-09 USD |  | the bill recomputed from the recorded powers |
| ✅ emissions | 2.78e-17 kg | ≤ 1e-09 kg |  | the CO₂ recomputed from the same powers |
| ✅ signals | carbon 0.00e+00 kg/MWh · price 0.00e+00 USD/MWh | ≤ 1e-3 |  | the columns are the modelled signals, step for step |
| ✅ actor sum | 0.00e+00 MW | ≤ 1e-9 MW |  | 4 metered actors |
| ✅ load flow | 4.85e-10 kW | ≤ 1e-03 kW | 12 snapshots | slack = grid + losses · max losses 0.01 kW |
| ✅ voltage | 1.0000 p.u. | [0.85, 1.1] p.u. | 12 snapshots | the case's system can carry its own powers |
| ✅ ed | 9.4677e-05 MW | ≤ 2 % of the setpoint | 12 snapshots | λ-iteration vs Vessim on the same fleet |
| ✅ uc | 0.00e+00 MWh | 0 MWh | whole day | 0 start(s) · 6.98 USD |
| ✅ uc grid | 0.0022 MW | ≤ 0.0030 MW | whole day | the commitment never imports more than the connection allows |
| ✅ lp | 0.00e+00 rel. | ≤ 1e-04 |  | objective 6.11 USD recomputed from its own dispatch |
| ✅ lp balance | 4.34e-19 MW | ≤ 1e-6 MW |  | the LP's dispatch serves the net load at every step |
| ✅ bus map | 9 AC buses → 1 row each | every bus mapped once |  | 9 buses of the case's own network |
| ✅ same buses | 9 OPF buses | = the 9 of the load flow | ok | built by the same call — vessim_system.site_network |
| ✅ same branches | 8 branches | = the 8 of the load flow | 7 lines · 1 transformers | same ratings, same names |
| ✅ serves the load | 0.00e+00 MWh unserved | < 1e-6 MWh | no shedding | the network carries every recorded MWh without an emergency generator |
| ✅ poi | 5.14e-13 MW | ≤ 1e-6 MW | 6 snapshots | AC slack = the trace's import + the losses of that snapshot |
| ✅ loss gap | 0.18 % of load | [0, 10] % | 0.00 MWh/day · 0.02 USD | the one term the lossless OPF does not charge for |
| ✅ diagram | 9 buses | every bus drawn |  |  |
| ✅ system | 4 actors · 2 boards | every actor wired |  | 03 · Carbon-aware workload |

## sil

| check | worst | tolerance | where | statement |
|---|---|---|---|---|
| ✅ balance | 1.08e-19 MW | ≤ 1e-09 MW | step 1 | solar + gas + grid + battery − load − curtail = 0 |
| ✅ soc | 1.69e-21 MWh | ≤ 1e-09 MWh | every step | soc[t+1] = soc[t] − battery[t]·Δt (the battery is lossless) |
| ✅ soc bounds | [0.0000, 0.0000] MWh | [0, 0.0000] MWh |  | the state of charge never leaves the tank |
| ✅ c-rate | 0.0000 MW | ≤ 0.0000 MW |  | c-rate 1.00 /h |
| ✅ grid limits | [0.0004, 0.0005] MW | [-0.0005, 0.0010] MW |  | import and export within the connection |
| ✅ cost | 0.00e+00 USD | ≤ 1e-09 USD |  | the bill recomputed from the recorded powers |
| ✅ emissions | 0.00e+00 kg | ≤ 1e-09 kg |  | the CO₂ recomputed from the same powers |
| ✅ signals | carbon 0.00e+00 kg/MWh · price 0.00e+00 USD/MWh | ≤ 1e-3 |  | the columns are the modelled signals, step for step |
| ✅ actor sum | 0.00e+00 MW | ≤ 1e-9 MW |  | 3 metered actors |
| ✅ load flow | 4.77e-09 kW | ≤ 1e-03 kW | 8 snapshots | slack = grid + losses · max losses 0.00 kW |
| ✅ voltage | 1.0000 p.u. | [0.85, 1.1] p.u. | 8 snapshots | the case's system can carry its own powers |
| ✅ ed | 8.6600e-10 MW | ≤ 2 % of the setpoint | 8 snapshots | λ-iteration vs Vessim on the same fleet |
| ✅ uc | 0.00e+00 MWh | 0 MWh | whole day | 0 start(s) · 0.00 USD |
| ✅ uc grid | 0.0005 MW | ≤ 0.0010 MW | whole day | the commitment never imports more than the connection allows |
| ✅ lp | 0.00e+00 rel. | ≤ 1e-04 |  | objective 0.00 USD recomputed from its own dispatch |
| ✅ lp balance | 0.00e+00 MW | ≤ 1e-6 MW |  | the LP's dispatch serves the net load at every step |
| ✅ bus map | 8 AC buses → 1 row each | every bus mapped once |  | 8 buses of the case's own network |
| ✅ same buses | 8 OPF buses | = the 8 of the load flow | ok | built by the same call — vessim_system.site_network |
| ✅ same branches | 7 branches | = the 7 of the load flow | 6 lines · 1 transformers | same ratings, same names |
| ✅ serves the load | 0.00e+00 MWh unserved | < 1e-6 MWh | no shedding | the network carries every recorded MWh without an emergency generator |
| ✅ poi | 4.75e-12 MW | ≤ 1e-6 MW | 6 snapshots | AC slack = the trace's import + the losses of that snapshot |
| ✅ loss gap | 0.21 % of load | [0, 10] % | 0.00 MWh/day · 0.00 USD | the one term the lossless OPF does not charge for |
| ✅ diagram | 8 buses | every bus drawn |  |  |
| ✅ system | 3 actors · 1 boards | every actor wired |  | 04 · Software-in-the-loop |

## ecopetrol

| check | worst | tolerance | where | statement |
|---|---|---|---|---|
| ✅ balance | 7.11e-15 MW | ≤ 1e-09 MW | step 50 | solar + gas + grid + battery − load − curtail = 0 |
| ✅ soc | 0.00e+00 MWh | ≤ 1e-09 MWh | every step | soc[t+1] = soc[t] − battery[t]·Δt (the battery is lossless) |
| ✅ soc bounds | [11.6767, 20.0000] MWh | [0, 40.0000] MWh |  | the state of charge never leaves the tank |
| ✅ c-rate | 10.0000 MW | ≤ 10.0000 MW |  | c-rate 0.25 /h |
| ✅ grid limits | [-0.0000, 12.0000] MW | [-4.0000, 12.0000] MW |  | import and export within the connection |
| ✅ cost | 0.00e+00 USD | ≤ 1e-09 USD |  | the bill recomputed from the recorded powers |
| ✅ emissions | 0.00e+00 kg | ≤ 1e-09 kg |  | the CO₂ recomputed from the same powers |
| ✅ signals | carbon 0.00e+00 kg/MWh · price 0.00e+00 USD/MWh | ≤ 1e-3 |  | the columns are the modelled signals, step for step |
| ✅ actor sum | 0.00e+00 MW | ≤ 1e-9 MW |  | 4 metered actors |
| ✅ load flow | 2.67e-09 kW | ≤ 1e-03 kW | 12 snapshots | slack = grid + losses · max losses 928.15 kW |
| ✅ voltage | 0.9827 p.u. | [0.85, 1.1] p.u. | 12 snapshots | the case's system can carry its own powers |
| ✅ ed | 5.3291e-15 MW | ≤ 2 % of the setpoint | 12 snapshots | λ-iteration vs Vessim on the same fleet |
| ✅ uc | 2.14e-10 MWh | 0 MWh | whole day | 2 start(s) · 40,089.00 USD |
| ✅ uc grid | 12.0000 MW | ≤ 12.0000 MW | whole day | the commitment never imports more than the connection allows |
| ✅ lp | 0.00e+00 rel. | ≤ 1e-04 |  | objective 40,355.25 USD recomputed from its own dispatch |
| ✅ lp balance | 3.55e-15 MW | ≤ 1e-6 MW |  | the LP's dispatch serves the net load at every step |
| ✅ bus map | 10 AC buses → 1 row each | every bus mapped once |  | 10 buses of the case's own network |
| ✅ same buses | 10 OPF buses | = the 10 of the load flow | ok | built by the same call — vessim_system.site_network |
| ✅ same branches | 9 branches | = the 9 of the load flow | 5 lines · 4 transformers | same ratings, same names |
| ✅ serves the load | 0.00e+00 MWh unserved | < 1e-6 MWh | no shedding | the network carries every recorded MWh without an emergency generator |
| ✅ poi | 2.67e-12 MW | ≤ 1e-6 MW | 6 snapshots | AC slack = the trace's import + the losses of that snapshot |
| ✅ loss gap | 1.55 % of load | [0, 10] % | 7.89 MWh/day · 1,505.56 USD | the one term the lossless OPF does not charge for |
| ✅ diagram | 10 buses | every bus drawn |  |  |
| ✅ system | 4 actors · 2 boards | every actor wired |  | 05 · Ecopetrol industrial |

## montecarlo

| check | worst | tolerance | where | statement |
|---|---|---|---|---|
| ✅ balance | 5.55e-17 MW | ≤ 1e-09 MW | step 22 | solar + gas + grid + battery − load − curtail = 0 |
| ✅ soc | 6.94e-18 MWh | ≤ 1e-09 MWh | every step | soc[t+1] = soc[t] − battery[t]·Δt (the battery is lossless) |
| ✅ soc bounds | [0.0300, 0.1500] MWh | [0, 0.3000] MWh |  | the state of charge never leaves the tank |
| ✅ c-rate | 0.1291 MW | ≤ 0.1500 MW |  | c-rate 0.50 /h |
| ✅ grid limits | [-0.0000, 0.1760] MW | [-0.0800, 0.2600] MW |  | import and export within the connection |
| ✅ cost | 0.00e+00 USD | ≤ 1e-09 USD |  | the bill recomputed from the recorded powers |
| ✅ emissions | 4.44e-16 kg | ≤ 1e-09 kg |  | the CO₂ recomputed from the same powers |
| ✅ signals | carbon 0.00e+00 kg/MWh · price 0.00e+00 USD/MWh | ≤ 1e-3 |  | the columns are the modelled signals, step for step |
| ✅ actor sum | 0.00e+00 MW | ≤ 1e-9 MW |  | 4 metered actors |
| ✅ load flow | 3.98e-07 kW | ≤ 1e-03 kW | 12 snapshots | slack = grid + losses · max losses 16.45 kW |
| ✅ voltage | 0.9311 p.u. | [0.85, 1.1] p.u. | 12 snapshots | the case's system can carry its own powers |
| ✅ ed | 3.8164e-17 MW | ≤ 2 % of the setpoint | 12 snapshots | λ-iteration vs Vessim on the same fleet |
| ✅ uc | 1.05e-10 MWh | 0 MWh | whole day | 0 start(s) · 596.09 USD |
| ✅ uc grid | 0.1760 MW | ≤ 0.2600 MW | whole day | the commitment never imports more than the connection allows |
| ✅ lp | 0.00e+00 rel. | ≤ 1e-04 |  | objective 507.42 USD recomputed from its own dispatch |
| ✅ lp balance | 2.78e-17 MW | ≤ 1e-6 MW |  | the LP's dispatch serves the net load at every step |
| ✅ bus map | 9 AC buses → 1 row each | every bus mapped once |  | 9 buses of the case's own network |
| ✅ same buses | 9 OPF buses | = the 9 of the load flow | ok | built by the same call — vessim_system.site_network |
| ✅ same branches | 8 branches | = the 8 of the load flow | 7 lines · 1 transformers | same ratings, same names |
| ✅ serves the load | 0.00e+00 MWh unserved | < 1e-6 MWh | no shedding | the network carries every recorded MWh without an emergency generator |
| ✅ poi | 2.27e-10 MW | ≤ 1e-6 MW | 6 snapshots | AC slack = the trace's import + the losses of that snapshot |
| ✅ loss gap | 4.75 % of load | [0, 10] % | 0.18 MWh/day · 37.15 USD | the one term the lossless OPF does not charge for |
| ✅ diagram | 9 buses | every bus drawn |  |  |
| ✅ system | 4 actors · 2 boards | every actor wired |  | 06 · Probabilistic campus |

## papers

| check | worst | tolerance | where | statement |
|---|---|---|---|---|
| ✅ shifting energy | 0.00e+00 kWh | ≤ 1e-9 kWh | Vessim 2024 | the flexible share moves, the energy does not |
| ✅ sil energy | 0.00e+00 Wh | ≤ 1e-9 Wh | SPE 2023 | energy = mean power × time |
| ✅ battery ideal | 0.00e+00 kW | ≤ 1e-9 kW | LOCO 2024 | served + unmet = deficit, at every step |
| ✅ battery linear | 0.00e+00 kW | ≤ 1e-9 kW | LOCO 2024 | served + unmet = deficit, at every step |
| ✅ battery kibam | 0.00e+00 kW | ≤ 1e-9 kW | LOCO 2024 | served + unmet = deficit, at every step |
| ✅ battery peukert | 0.00e+00 kW | ≤ 1e-9 kW | LOCO 2024 | served + unmet = deficit, at every step |
| ✅ inference offset | 0.161 | [0, 1] | PECS 2025 | shifting can never create emissions |
| ✅ pretraining ratio | 0.467 | [0, 1.5] | arXiv 2602.22760 | the policy cannot emit more than 1.5× the baseline |

## Failures

None. Every invariant holds at every audited step.
