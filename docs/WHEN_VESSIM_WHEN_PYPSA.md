# When to use Vessim, when to use PyPSA, and when the answer is *both*

**A decision guide, and the scientific case for pairing the two**

> This is the document to read before writing a paper, a thesis chapter or a proposal on top of this
> repository. It answers three questions in order: *which tool answers which question*, *what must be
> held constant so the comparison means something*, and *why the resulting claims are defensible*.
>
> Every number quoted here is reproducible from the code: `PYTHONPATH=src python -m pytest tests -q`
> (27 tests) and `PYTHONPATH=src python examples/headless_report.py`.

---

## 0. The short answer

| Your question is about… | Use | Because |
|---|---|---|
| what a piece of **software** does against an energy system (a scheduler, an application, an RL agent, a real CPU workload) | **Vessim** | it simulates the closed loop, at a clock, with real components in the loop |
| the **best schedule** for a given equipment set (cost, emissions, curtailment), the **value of energy** at a bus, congestion, capacity | **PyPSA** | it solves the whole horizon at once and returns primal *and* dual information |
| how much a controller **loses** by being local, what a **price** would do to it, or how to build a controller that looks ahead | **both** | one produces the trajectory, the other produces the benchmark and the prices |
| whether a published heuristic is **any good** | **both** | without an optimum there is no denominator |

In one sentence each:

- **Vessim answers “what happens if I run this?”** It is a co-simulation testbed for carbon-aware
  applications and systems: actors, dispatchables, a dispatch policy, grid signals, a recorder, and
  software- or hardware-in-the-loop. It is *forward*, *closed-loop* and makes **no optimality claim**.
- **PyPSA answers “what is the best I could have done?”** It is an optimisation framework for power
  and energy systems: it minimises a cost over the whole horizon subject to network, storage and
  commitment constraints, and hands back the dual variables of that problem as **prices**.
- **Together they answer “how much is the difference worth, and where does it come from?”** The
  trajectory from Vessim is a *feasible point* of PyPSA's problem; the optimum is the best such point.
  The distance between them is a measurable quantity — a regret — and it can be decomposed into
  terms with prices on them (this repository does exactly that, to machine precision).

The three claims that make the pairing scientific rather than decorative:

1. **A bound.** If the Vessim trajectory is feasible for the PyPSA problem, then
   `C_PyPSA ≤ C_Vessim`. The gap is non-negative by construction and is *the value of the information
   PyPSA uses* (look-ahead, network, prices).
2. **A certificate.** The optimum comes with a dual solution; the dual is the marginal price, and it
   can be checked numerically (`∂C*/∂L_t`, finite differences, agreement to 10⁻⁶ in this repo).
3. **An exact decomposition.** Every cost in both models is linear in energy, so
   `C_V − C_P = Σ (Δenergy × price)`. The ledger adds up to the last cent, which means no cause can
   hide inside a residual.

---

## 1. What the two tools actually are, formally

### 1.1 Vessim is a forward simulation of a policy

Vessim steps a microgrid at a fixed clock. At step *t* it asks each actor for power, then calls a
dispatch policy to balance the microgrid:

```
x_{t+1} = f(x_t, u_t, w_t)          state: battery charge, counters
u_t     = π(x_t, s_t; θ)            policy: merit order, thresholds, a learned agent…
w_t                                 exogenous: load, solar, price, carbon intensity
y_t     = g(x_t, u_t, w_t)          what the Recorder writes down
```

There is no objective function anywhere in that loop. Whatever `π` does is *the* answer. The
guarantees Vessim gives are about **the simulation**, not about the decision: steps happen in order,
the balance is enforced, real components can be attached, and the recording is complete.

Consequences, which decide when it is the right tool:

- It accepts **any** policy, including ones that cannot be written as an optimisation problem:
  a scheduler in production, a Java service, a neural network, a rule with hysteresis, a process
  whose power draw you can only measure.
- It reproduces **timing**: step size, latency, control period, jitter, and the fact that a decision
  taken at *t* cannot use information from *t+1*.
- It supports **software- and hardware-in-the-loop**, which is the only honest way to study a system
  whose power trace you cannot model analytically.

### 1.2 PyPSA is an optimisation over a horizon

PyPSA solves, in the linear-programming case,

```
min_{g,p,c,d,e,f}  Σ_t Δt [ (π_t + τ e_t) g_t + (c_g + τ e_g) p_t + … ]
s.t.               nodal balance at every bus and snapshot
                   |f_ℓ,t| ≤ S_ℓ                    (lines)
                   e_{t+1} = e_t + Δt(η c_t − d_t/η) (storage)
                   p_min ≤ p_t ≤ p_max, …            (units)
```

and returns:

- the **primal** solution — what to do, in MW, per bus and per hour;
- the **dual** solution — the marginal value of each constraint, i.e. the nodal price λ and the
  congestion rent;
- a **certificate** — the solver's optimality (or infeasibility) status, with a bound on the gap.

Consequences:

- It can price things: λ is a derivative, not an average, and that is what a market or a controller
  needs in order to decide well.
- It sees the network: losses, congestion, voltage (in AC mode), and the fact that energy at one bus
  is not the same good as energy at another.
- It cannot represent a black-box controller. Anything non-convex must be approximated or turned
  into a MILP with a size and a runtime you have to live with.

### 1.3 The relationship that makes a comparison legitimate

Let *F* be the feasible set of PyPSA's problem for a given day — same equipment, same exogenous
signals, same horizon, same terminal condition. Let `u^V` be the trajectory Vessim produced.

**Claim.** If `u^V ∈ F`, then `C(u^P) ≤ C(u^V)`.

*Proof.* `u^P` is defined as the minimiser over *F*; `u^V` is a point of *F*; therefore its objective
cannot be smaller. ∎

Everything scientific about this project rests on that one line — and on being honest about the three
ways it can fail:

1. `u^V ∉ F`. Vessim can export to the grid; the site model here has no export sink. Vessim can
   curtail; the LP must be allowed to curtail too. If the heuristic has an option the LP lacks, the
   inequality has no force. **Fix:** reconcile the option sets explicitly (see §5).
2. **Different objectives.** Vessim's policy prices battery cycling at 20 USD/MWh; if the LP's
   storage is free, the LP is solving an easier problem and the “saving” is a modelling artefact.
   **Fix:** charge the same cycling cost in both.
3. **Different terminal conditions.** Over a finite horizon an LP with no terminal value *empties the
   battery* and books the proceeds as a saving. **Fix:** mark the final stock to market, or make both
   models end where they started.

This repository now does all three, and the ledger in §6 shows the size of each correction —
including the uncomfortable case where the sign of the “saving” flips.

---

## 2. Use Vessim alone when…

…the object of study is **the controller or the software**, and the energy system is the environment.

| Signal | Why PyPSA would be the wrong tool |
|---|---|
| You are testing a **real application** (case 04: a real CPU process drives the microgrid) | an LP cannot represent a process whose power draw exists only as measurement |
| The controller is a **black box**: RL agent, scheduler, production code | non-convex and non-analytic by nature; the LP cannot accept it |
| You care about **timing**: control period, latency, stale signals | an LP is atemporal; the horizon is a modelling choice, not a clock |
| You need **software/hardware-in-the-loop** | the loop must actually run in real time |
| You study **forecast error**: decide with a forecast, evaluate on the realisation | PyPSA is deterministic; a proper treatment needs scenario trees, which is a different (and much larger) study |
| You want **regression testing** of a controller, a digital twin, continuous testing | that is what a testbed is for |
| The “plant” is not a power system at all: a compute cluster, a cooling loop, a network link | out of scope for a power-system optimiser |

**Rule of thumb.** If you cannot write the controller as a convex objective with the variables you
care about, you need a simulator. If you *can*, you probably want the optimiser.

### What you must not claim when you use Vessim alone

- **Not** “this dispatch is efficient”, “this is the cheapest way”, “we saved X %”. There is no
  denominator. A heuristic can only be described, not evaluated, without a benchmark.
- **Not** “this is grid-compliant”. A single-bus microgrid model has no voltages, no loading, no
  congestion, no losses.
- **Not** “the price is …”. The signal Vessim consumes is an input, not a price formed by the system.

---

## 3. Use PyPSA alone when…

…the object of study is **the schedule, the price or the capacity**, and the controller is not part
of the question.

| Signal | Why Vessim would add nothing |
|---|---|
| Congestion, losses, nodal prices, voltage | only a network model produces them |
| Unit commitment with minimum up/down times, ramping, start-up costs | needs a MILP and a solver certificate |
| Storage arbitrage over days, cyclic SoC, seasonal storage | needs the whole horizon at once |
| **Prices**: LMP, congestion rent, value of the marginal MWh | prices *are* duals; a simulation has none |
| Capacity expansion, sizing, portfolio studies | an investment question, not a trajectory |
| Policy design: carbon tax, market rules, subsidies | you need to re-optimise counterfactuals |
| Publishing a **benchmark number** with a bound | an optimum is reproducible and certifiable; a heuristic is not |

**Rule of thumb.** If the sentence contains *optimal*, *value of*, *price*, *capacity*, *congestion*,
*curtailment cost* or *benchmark*, you need an optimiser.

### What you must not claim when you use PyPSA alone

- **Not** “this is what the operator would do”. An LP assumes a controller with perfect foresight,
  perfect obedience and no clock. Real dispatch is a control system.
- **Not** “this is implementable”. The schedule may require actions no real device can take at the
  required instant.
- **Not** “we validated the model against reality”. An LP is validated against *mathematics*
  (duality, feasibility), not against a running system.

---

## 4. Use both when…

…the claim is **comparative**, or **the controller needs a planner inside the loop**. There are three
architectures, and they answer different questions.

### 4.1 Open-loop benchmark — “what did the rules cost us?”

```
Vessim (full run)  ──► trace (load, solar, battery, grid, prices)
                          │
                          ├──► PyPSA on the same trace ──► optimum + λ
                          └──► ledger: C_V − C_P = Σ (Δenergy × price)
```

This is what the dashboard does. Vessim runs *first and alone*; PyPSA never influences it. Therefore:

- **It measures regret**, i.e. the value of the information PyPSA has and the policy does not
  (look-ahead, network, prices).
- **It does not measure what the controller would have done with that information.** No causal claim
  is possible, and none should be made.
- The comparison is only meaningful under the fairness conditions of §5.

*Use it for:* benchmarking a heuristic, teaching, a first paper, sanity checks, a proposal.

### 4.2 Rolling-horizon MPC — “what if the controller could look ahead?”

At every control step, solve a PyPSA problem over the next *H* hours from the *current* state, apply
only the first action, advance the simulation, repeat.

```
for t in horizon:
    plan  = PyPSA( window [t, t+H), x_t, forecast )
    apply(plan.action[0])            # only the first one
    x_{t+1} = f(x_t, action, w_t)    # Vessim advances the plant
```

This is where the two tools genuinely need each other:

- **Vessim** owns the clock, the state, the plant, the actors, and the possibility of real hardware.
- **PyPSA** owns the decision inside each window.
- The **regret** of the MPC against perfect foresight is now a *forecast-quality* quantity, not an
  information deficit — a completely different scientific question, and the one worth publishing.

Requirements before you claim anything from an MPC:

1. **Terminal condition inside the window.** Without one, the MPC drains the battery in the first
   window and behaves exactly like a myopic controller. Use a cyclic constraint, a terminal value, or
   a horizon long enough that the tail is negligible — and *say which*.
2. **Solve time < control period**, reported, not assumed.
3. **Forecast model stated.** Perfect-foresight MPC is an upper bound, not a realistic controller.
4. **Same accounting as the benchmark** (§5), otherwise the comparison is meaningless.

*Use it for:* the main contribution of a paper, controller design, anything claiming “look-ahead
helps”.

### 4.3 Advisory co-simulation — “what does a price do to the software?”

PyPSA runs offline (or on a slower clock) and publishes prices or limits; Vessim's actors consume
them as signals.

```
PyPSA (clearing, minutes ahead) ──► λ_t, congestion, limits ──► Vessim actors (seconds)
        ▲                                                            │
        └──────────────── load, storage state, bids ◄────────────────┘
```

This is the market-design experiment: does a nodal price make a batch scheduler shift work better than
a carbon signal does? Neither tool can answer that alone.

*Use it for:* mechanism design, the interface between computing and power systems, digital twins.

---

## 5. The comparison protocol: eight things that must be held constant

A benchmark is a controlled experiment. These are the controls, each with the failure mode it
prevents, the fix, and where it lives in the code.

| # | Confounder | What goes wrong if you ignore it | Fix in this repo |
|---|---|---|---|
| 1 | **Exogenous signals** | the two models optimise different days | PyPSA reads `price_usd_mwh` and `carbon_kg_mwh` from the Vessim trace, not from a separate assumption (`pypsa_models.build_site_network`) |
| 2 | **Snapshot weighting** | PyPSA defaults to 1 h per snapshot; on a 15-min trace the battery becomes four times too fast | `n.snapshot_weightings = Δt` — and a test pins it |
| 3 | **Cycling cost** | Vessim's policy prices discharge at 20 USD/MWh; a free LP battery cycles 6× more and “wins” | `SiteSpec.storage_marginal_cost` is passed to the `StorageUnit` |
| 4 | **Terminal condition** | an LP with no terminal value empties the tank and books it as a saving | the stock is marked to market at λ_T; the term appears in the ledger |
| 5 | **Export** | Vessim can sell to the grid; a site model with no export sink cannot | the export credit is a separate line in the ledger, never silently netted |
| 6 | **Losses** | the heuristic pays no losses; an OPF with resistance charges for them | the site OPF is deliberately lossless (`r = 0`); the load-flow tab shows the real wires |
| 7 | **Curtailment** | if the LP must consume every kWh of sun, it looks worse than the heuristic | `allow_curtailment` is explicit in both |
| 8 | **Carbon price, twice** | τ as a cost but not as a signal (or vice versa) makes one side blind | τ reaches both the policy and the marginal cost; the run's own spec carries the value |

Two more that belong to the *design*, not the code:

9. **Horizon length.** A 24 h trace with a 4 h battery is dominated by the terminal condition. If the
   claim is about steady-state behaviour, use a periodic extension (repeat the day, evaluate the
   middle one) or a rolled horizon.
10. **The heuristic must be a fair opponent.** A deliberately bad policy makes a beautiful number and
    an unpublishable result. Report which policy, which parameters, and whether it was tuned.

---

## 6. What the numbers say today

All five official cases, τ at each case's default, everything else at its default. “Bill” is what the
operator pays; “fair” adds the cycling cost to the heuristic's side; “closed” also values the stock
left at the horizon.

| Case | Vessim bill (USD) | PyPSA bill (USD) | Naive gap | Fair gap | Closed-book gap | Decomposition exact |
|---|---|---|---|---|---|---|
| 01 basic | 2.15 | 1.87 | 0.27 (12.7 %) | 0.28 | **0.28** | 2·10⁻¹⁶ |
| 02 solar + battery | 5.28 | 4.37 | 0.91 (17.3 %) | 0.94 | **0.94** | 6·10⁻¹⁶ |
| 03 carbon-aware | 6.98 | 6.11 | 0.87 (12.5 %) | 0.90 | **0.90** | 2·10⁻¹⁵ |
| 04 SiL | 0.00 | 0.00 | 0.00 | 0.00 | **0.01** | 1·10⁻²⁰ |
| 05 Ecopetrol | 40 085.04 | 39 003.76 | 1 081.27 (2.70 %) | 1 247.74 | **−223.52** | 2·10⁻¹² |

The Ecopetrol ledger, term by term (USD over the 24 h):

| Cause | Term | USD |
|---|---|---|
| Grid energy, valued at each hour's price | Σ Δt (I^V − I^P)·π_t | −357.44 |
| Gas energy, at the turbine's cost | Σ Δt (G^V − G^P)·c_g | +1 567.25 |
| Export credit the heuristic takes and the LP cannot | −Σ Δt X^V·φ·π_t | −0.00 |
| Carbon, at τ = 60 USD/t | τ·Σ Δt (ΔI·e_t + ΔG·e_g) | +271.47 |
| Battery discharge, same price on both sides | c_batt·(D^V − D^P) | −233.53 |
| Energy left in the tank | −λ_T·(SoC^V_T − SoC^P_T) | −1 471.26 |
| **Total** | | **−223.52** |

Three findings worth publishing, in decreasing order of comfort:

1. **The optimum substitutes gas for grid, not energy for energy.** It buys 3.3 MWh *more* grid and
   14.9 MWh *less* gas, and emits 4.5 t less CO₂ (99.58 → 95.05 t), because the captive turbine emits
   350 kg/MWh while the grid averages ~180 kg/MWh. A single-bus policy cannot see that trade at all.
2. **The naive saving is mostly accounting.** Of the 1 081.27 USD “saving”, 1 471.26 comes from
   ending with an empty battery and 233.53 from paying no cycling cost on the heuristic's side. Once
   both are corrected, the heuristic is — on this horizon — *cheaper*.
3. **That is not an argument for heuristics.** It is an argument that a 24 h horizon with a 40 MWh
   battery is dominated by the terminal condition. The right response is a rolled or periodic horizon
   and an MPC, not a shorter benchmark.

The small cases behave as theory says: positive gap, exact ledger, driven by the grid term (the
heuristic buys at the wrong hours).

---

## 7. Why this proposal is scientifically viable — ten arguments

1. **It is falsifiable.** Four predictions, each testable and each with a pass/fail criterion in CI:
   the ledger sums to the gap (10⁻⁶); λ equals the finite-difference derivative (10⁻³); every
   diagram's balance closes; PyPSA never loses to a feasible heuristic *when the protocol of §5 is
   applied* — and when it does, the ledger says why.
2. **The numerical core is validated against independent references.** AC load flow against
   `pypsa.Network.pf()`: ΔV = 7.6·10⁻¹² p.u. on the WSCC 9-bus. Economic dispatch against
   `n.optimize()`: 0 % on a copper plate. Unit commitment: the 3–57 % gap is *reported as a result*,
   not hidden. λ against a central finite difference: exact to 10⁻⁶.
3. **It is reproducible.** Pinned versions (`requirements.txt`), official MIT-licensed traces bundled,
   seeds and deterministic policies, a headless report that writes the tables, a CI workflow, MIT
   licence, and a `NOTICE` that states exactly what was re-implemented from the GPL-3.0 reference.
4. **It fills a hole that the two projects leave open on purpose.** Vessim is a testbed for
   *applications and systems*; it ships dispatch policies, not a network-constrained optimiser.
   PyPSA is an optimiser for *power systems*; it has no notion of a controller that runs at a clock.
   The gap between them is exactly “how good is my controller, and what would a price have done to
   it?” — a gap neither project claims.
5. **It turns a heuristic into a measurable object.** Before this, “carbon-aware scheduling saved
   14 %” has no denominator. With an optimum, the same sentence becomes “the heuristic captured 96 %
   of the achievable value on this day, and the missing 4 % is the look-ahead it cannot see”.
6. **The decomposition is a methodological contribution, not a figure.** Attributing regret to priced
   causes (grid timing, gas substitution, cycling, terminal stock, export, carbon) is exactly what a
   reviewer asks for and what a single “cost gap” column cannot provide.
7. **It is a teaching instrument with the equations attached.** The 🧮 page writes each result as an
   equation with the reader's own numbers substituted, then says what it means. That is reusable in a
   course, a workshop or a thesis appendix, and it is the cheapest possible defence against
   “I don't see where this number comes from”.
8. **It joins two standards instead of inventing a third.** Vessim is Mosaik-based co-simulation;
   PyPSA is an LP/MILP on HiGHS. Nothing exotic is required of either, so the addon is small,
   inspectable and mergeable — which matters for a proposal to maintainers.
9. **The validation is live, not claimed.** Every tab recomputes its own comparison from the data the
   reader edits. The artifact *is* the evidence.
10. **It is honest about its own limits** (see §8). A proposal that lists the ways it can be wrong is
    a proposal a reviewer can accept; one that does not is a proposal they must reject.

### Positioning in one paragraph

> Carbon-aware computing papers evaluate controllers on custom testbeds; power-system papers evaluate
> schedules on networks. Neither quantifies what a controller *gives up* by being local, and neither
> exposes the marginal price to the controller. This work joins the two open-source stacks, defines
> the fairness conditions under which the comparison is meaningful, and reports the regret of the
> official Vessim policies against a network-constrained optimum — decomposed into priced causes and
> verified numerically at every build.

---

## 8. Threats to validity, and how to answer them

| Threat | Effect on the claims | Mitigation (implemented / planned) |
|---|---|---|
| Copper plate vs network | the site model has no losses or congestion, so the network term is understated | implemented: the load-flow tab shows the full AC network; planned: an AC-OPF variant |
| Deterministic prices, no forecast error | the optimum is a *bound*, not an achievable number | stated as such everywhere; planned: scenario-based or rolling MPC |
| Finite horizon | can flip the sign of the gap (§6) | implemented: mark-to-market ledger; planned: periodic extension |
| Lossless OPF | understates the cost of moving energy | implemented: deliberate, documented, and shown in the load-flow tab |
| Linear costs | no minimum stable load, no start-up cost in the site model | implemented for ED/UC presets (MILP); planned for the site model |
| No battery degradation beyond a linear cost | overstates cycling | implemented: 20 USD/MWh parity; planned: rainflow/ageing model |
| One scenario per case | no distribution, no confidence interval | planned: Monte Carlo over price, carbon amplitude and load scale |
| The heuristic may be a strawman | inflates the gap | reported policy parameters; planned: tuned and published baselines |
| Carbon accounting | double counting risk when τ is both a cost and a signal | τ is applied once per MWh, on the same basis, and the ledger separates the term |
| Step size | 15 min vs 1 h changes storage behaviour | snapshot weighting enforced and tested |

---

## 9. The experimental programme

Each hypothesis is cheap and uses machinery that already exists.

| # | Hypothesis | Design | What would refute it |
|---|---|---|---|
| H1 | the gap grows with network tightness | sweep the POI–Site rating (100 % → 10 %), everything else fixed | gap flat or non-monotone in the rating |
| H2 | the gap grows with the carbon price | sweep τ ∈ {0, 50, 150, 300} USD/t | gap insensitive to τ |
| H3 | the gap is dominated by the terminal stock at short horizons | compare 24 h, periodic 3×24 h, and 7 d | terminal term explains < 10 % of the gap |
| H4 | an MPC closes most of the perfect-foresight gap | rolling horizon H ∈ {3, 6, 12} h, perfect and imperfect forecasts | MPC no better than the heuristic, or insensitive to H |
| H5 | a nodal price beats a carbon signal for a workload scheduler | advisory co-simulation: λ vs carbon intensity as the signal | no difference in cost or emissions |

Reporting standard for each: paired differences on identical traces, mean ± CI over N days,
effect size, and the ledger so the reader can see *which* term moved.

---

## 10. Reviewer FAQ

**“Isn't this just a dashboard?”**
The dashboard is the interface to four numerical claims: a validated load flow, a validated dispatch,
an optimality gap with a certificate, and an exact decomposition. The interface is what makes the
claims inspectable; it is not the contribution.

**“Why not just use PyPSA for everything?”**
Because PyPSA cannot take a real application, a black-box controller or a clock. Case 04 in this
repository is a real CPU process driving the microgrid; there is no LP for that.

**“Why not just use Vessim for everything?”**
Because then no number has a denominator. “The policy used 95.3 t of CO₂” is not a result; “it used
5.1 % more than the optimum, and 21 % of the gap is gas it should not have burned” is.

**“Is the comparison fair?”**
It is now — §5 lists the eight controls, and the repository fails CI if the ledger stops adding up.
Earlier versions of this dashboard were *not* fair: the LP was cycling the battery for free and
draining it at the end, which is where the original 4.15 % “saving” came from.

**“What is genuinely new?”**
(i) The regret decomposition into priced causes; (ii) the explicit fairness protocol for
simulation-vs-optimisation comparisons; (iii) exposing dual prices to a carbon-aware controller; (iv)
the open, reproducible artifact; (v) the pedagogical form: equations, numbers and diagrams on one
screen.

**“What is the biggest risk?”**
That the horizon artefacts dominate the effect. That is why H3 exists, and why the architecture moves
to rolling MPC (§4.2) before any strong claim is made about look-ahead value.

---

## 11. References

1. P. Wiesner, I. Behnke, P. Kilian, M. Steinke, O. Kao. **Vessim: A Testbed for Carbon-Aware
   Applications and Systems.** *ACM SIGENERGY Energy Informatics Review* 4(5), 59–66, 2024.
   doi:10.1145/3727200.3727210 · arXiv:2306.09774.
2. P. Wiesner, M. Steinke, H. Nickel, Y. Kitana, O. Kao. **Software-in-the-Loop Simulation for
   Developing and Testing Carbon-Aware Applications.** *Software: Practice and Experience* 53(12),
   2362–2376, 2023. doi:10.1002/spe.3275.
3. T. Brown, J. Hörsch, D. Schlachtberger. **PyPSA: Python for Power System Analysis.** *Journal of
   Open Research Software* 6(1), 4, 2018. doi:10.5334/jors.188.
4. F. Schweppe, M. Caramanis, R. Tabors, R. Bohn. **Spot Pricing of Electricity.** Kluwer, 1988
   (marginal-cost pricing as the dual of the dispatch problem).
5. M. Ellis, H. Durand, P. D. Christofides. **A tutorial review of economic model predictive control
   methods.** *Journal of Process Control* 24(8), 1156–1178, 2014 (why MPC needs a terminal condition).
6. C. Gomes, C. Thule, D. Broman, P. G. Larsen, H. Vangheluwe. **Co-simulation: A Survey.**
   *ACM Computing Surveys* 51(3), 49, 2018 (coupling, synchronisation and step-size issues).
7. L. Kotzur, P. Markewitz, M. Robinius, D. Stolten. **Impact of different time series aggregation
   methods on optimal energy system design.** *Renewable Energy* 117, 474–487, 2018 (why snapshot
   weighting changes results).
8. M. D. Wilkinson et al. **The FAIR Guiding Principles for scientific data management and
   stewardship.** *Scientific Data* 3, 160018, 2016.
9. Reference implementations of the two simulators whose diagram vocabulary and manual methods this
   project re-implements: https://tool.adjiebrotots.com/loadflow-simulator/ and
   https://tool.adjiebrotots.com/ed-uc-simulator/ (GPL-3.0; clean-room Python port, see `NOTICE`).

---

*This document is versioned with the code it describes. If a number here does not match what the app
prints, the app is right and this file is stale — open an issue.*
