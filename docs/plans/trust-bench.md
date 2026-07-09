# Optimisation-Solver Comparison Harness: Project Plan

## 1. Purpose and goals

This project is a test-and-benchmark harness for comparing non-linear
least-squares / unconstrained-minimisation solvers across languages and
implementations. The immediate motivation is the Dyalog APL library
[`yiyus/trust`](https://github.com/yiyus/trust) (trust-region Newton, with BFGS
and Levenberg-Marquardt Hessian models, robust loss functions, and box
constraints), assessed here against the state of the art.

Three co-equal goals drive every design decision:

1. **Capability overview.** Produce a clear, evidence-backed picture of what
   each library can and cannot do, and *where each one shines or breaks*, not
   just pass/fail on easy problems, but behaviour along difficulty axes
   (residual size, conditioning, outlier fraction, scaling, dimensionality,
   constraints, derivative source).

2. **Assess `trust` against the state of the art.** Two distinct questions,
   kept separate throughout:
   - *Same algorithm, different implementation*: e.g. `trust`'s LM vs SciPy's
     LM vs Julia's LM on identical problems, starts, and tolerances. This
     measures implementation quality and is the fair apples-to-apples check.
   - *Different algorithms / feature coverage*: which engine (LM / BFGS /
     Newton) and which features (robust losses, bounds, analytic Hessian) each
     library offers, and how method choice interacts with problem class.

3. **Longitudinal tracking.** The harness must remain useful over time: the
   same questions must be re-answerable as library versions, interpreters
   and compilers, BLAS/LAPACK backends, and hardware change. Every result must
   carry enough provenance to compare a run today against a run next year on a
   different machine.

Non-goals: building a new solver; producing a published paper (though the
outputs should be publication-grade); optimising any single library.

### 1.1 What the prototype already established

A throwaway prototype (kept under `prototype/`) validated the approach on the
large-residual axis: it reproduced the classic result that Gauss-Newton/LM
converges linearly at a rate equal to the spectral radius of `(JᵀJ)⁻¹S` (with
`S` the neglected second-order term), diverges once that radius exceeds 1, and
that trust-region globalisation trades divergence for a rising iteration count.
That prototype's metric functions (order/rate estimation, error traces) and the
large-residual family are the seed for Phases 1 and 6 below, but they will be
**re-derived under TDD**, not lifted verbatim.

---

## 2. Guiding principles

- **TDD, strictly.** Every layer is specified by tests written before the
  implementation. A phase is "done" only when its tests pass and run in CI.
  Red → green → refactor. Tests encode *contracts and shapes*, not incidental
  numbers, so they survive refactors and library updates.
- **Language-agnostic core.** The Python package holds no solver-specific or
  language-specific logic in its core. Solvers enter through a `Backend`
  adapter; problems through a `Problem` registry. Adding APL or Julia means
  implementing an adapter and a problem set plus their parity tests: nothing
  in the core changes.
- **Honest metrics.** Metrics are tiered by how comparable they are across
  languages (Section 6). Claims are only made at the tier a metric supports.
- **Provenance is a first-class citizen.** No result exists without a full
  environment record attached. This is what makes longitudinal comparison
  possible.
- **Reproducible and regenerable.** Raw results are stored append-only; all
  tables and plots are regenerated from stored results by a single command.

---

## 3. Repository layout

Name: `trust-bench`, a **separate repository** from `trust`. The `trust`
repository stays untouched; `trust-bench` treats `trust` as one backend among
several. The distribution/repo is `trust-bench`; the importable Python package
is `trust_bench`; the CLI command is `trust-bench`.

```
trust-bench/
  README.md
  pyproject.toml              # package + dev/test deps, tool config
  Makefile                    # common targets: test, lint, run-study, report, compare

  trust_bench/                # language-agnostic core (pure Python)
    core/
      problem.py              # Problem spec (residual/jac/hess, starts, optima, metadata)
      registry.py             # problem registration + lookup
      backend.py              # Backend ABC, Capabilities descriptor
      result.py               # RunResult, RunStatus, TimingStats dataclasses + (de)serialise
      provenance.py           # environment capture (versions, BLAS, CPU, git, timestamp)
      metrics.py              # pure functions: distance, order, rate, cost gap
      storage.py              # append-only result store (JSONL) + pandas loader
      runner.py               # (problem, backend, method, config) -> RunResult
      config.py               # RunConfig, tolerance mapping, timing policy
    problems/                 # canonical problem definitions (Python reference impls)
      rosenbrock.py beale.py powell.py helical.py expdec.py quadratic.py linear.py
      families/               # parametrised difficulty families
        large_residual.py ill_conditioned.py outliers.py scaling.py dimensionality.py
    backends/
      scipy_backend.py        # scipy.optimize least_squares + minimize
      optimistix_backend.py   # optional 2nd Python impl (JAX), same-algorithm check
      apl_backend.py          # Dyalog via subprocess/Docker (Phase 8)
      julia_backend.py        # Julia via subprocess (Phase 9)
    studies/                  # capability studies (difficulty sweeps)
      baseline.py large_residual.py ill_conditioning.py robust_loss.py
      bounds.py scaling.py dimensionality.py derivative_source.py
    reporting/
      tables.py plots.py capability_matrix.py compare.py
    cli.py                    # entry point: run / report / compare / list

  backends_ext/               # non-Python solver harnesses (thin, protocol-speaking)
    apl/                      # loads `trust`, reads a problem+config, emits a RunResult JSON
    julia/                    # Julia project (Optim.jl, LsqFit.jl, LeastSquaresOptim.jl)

  tests/
    unit/                     # metrics, result (de)serialisation, provenance, config
    problems/                 # parity (analytic vs FD) + known-optimum tests
    backends/                 # backend contract tests (parametrised over all backends)
    studies/                  # study-shape regression tests
    integration/              # end-to-end runner tests
    conftest.py

  results/                    # stored RunResult records (JSONL), one file per run-batch
  reports/                    # generated tables/plots (regenerable, may be git-ignored)
  prototype/                  # the validated throwaway experiment, for reference only
  docs/
    methodology.md            # metric taxonomy, fairness rules, timing policy
    adding_a_backend.md
    adding_a_problem.md
    plans/                    # feature plans; see CLAUDE.md "What lives where"
    prs/                      # per-cycle PR notes; see CLAUDE.md "What lives where"
    bugs/                     # bug investigations; see CLAUDE.md "What lives where"
    reviews/                  # review logs; see CLAUDE.md "What lives where"
```

---

## 4. Core interfaces (the reviewable design decisions)

These sketches define the contracts the tests will pin down. Field lists are
indicative; exact types are settled in the relevant phase.

### 4.1 Problem

The canonical problem is defined once in Python with analytic derivatives and
serves as the reference for all other languages.

```python
@dataclass(frozen=True)
class Problem:
    id: str                      # stable, e.g. "rosenbrock", "expdec:rho=10"
    residual: Callable           # x -> ndarray[m]   (or scalar objective form)
    jacobian: Callable | None    # x -> ndarray[m, n] (analytic; None => FD only)
    hessian:  Callable | None    # x -> ndarray[n, n] full Hessian of 0.5||r||^2
    starts:   dict[str, ndarray] # named starting points, e.g. {"standard": ..., "far": ...}
    optima:   list[Optimum]      # known solutions: x*, cost*, basin notes
    kind:     Literal["residuals", "scalar"]  # dispatch: LM/Newton vs BFGS
    tags:     frozenset[str]     # {"ill-conditioned","singular-jacobian","large-residual",...}
    probe_points: list[ndarray]  # points used by parity tests
    source:   str                # citation (e.g. Moré-Garbow-Hillstrom #)
```

Difficulty *families* are factory functions `make(**params) -> Problem` that
produce a parametrised Problem (e.g. `large_residual.make(rho=10.0)`), enabling
sweeps.

Cross-language rule: each backend implements each problem natively and must pass
**parity tests** against these reference values at `probe_points` (Section 5).
This is the only supported path. A callback bridge (residuals evaluated
in Python across the process boundary) is a possible future fallback for
black-box-only solvers, but it is **not built until a solver actually requires
it**: native problem implementations plus parity tests are faster, simpler,
and exercise each language's own evaluation path.

### 4.2 Backend

```python
@dataclass(frozen=True)
class MethodCapabilities:
    kind: Literal["residuals", "scalar"]  # Problem.kind this method solves
    losses: frozenset[str]                # {"l2","huber",...}; empty for scalar methods
    bounds: bool
    analytic_hessian: bool
    derivative_modes: frozenset[str]      # {"analytic","finite-difference"}

@dataclass(frozen=True)
class Capabilities:
    methods: dict[str, MethodCapabilities]  # e.g. {"lm": ..., "trf": ..., "bfgs": ...}

class Backend(ABC):
    name: str                        # "scipy", "trust-apl", "julia-optim", ...
    def capabilities(self) -> Capabilities: ...
    def environment(self) -> EnvProvenance: ...
    def solve(self, problem: Problem, method: str, start: str,
              config: RunConfig) -> RunResult: ...
```

`bounds` and `analytic_hessian` are keyed per method, not per backend: SciPy's
`least_squares(method="lm")` rejects bounds while `trf`/`dogbox` accept them,
and among `minimize` methods only `L-BFGS-B`/`trust-constr` accept bounds
while plain `BFGS` does not. A single backend-level boolean cannot express
this, and the illustrative capability matrix in Section 9.1 already needs the
per-method distinction (its "yes (trf)", "yes (trust-*)" annotations are
exactly `capabilities().methods[<method>]` lookups under this schema).

In-process backends (SciPy, Optimistix) implement `solve` directly.
Out-of-process backends (APL, Julia, and any future compiled backend) use a
single, general **subprocess** transport: the adapter serialises the request to
their harness in `backends_ext/`, runs it, and parses a `RunResult` JSON back.
Subprocess is chosen over in-process bridges (e.g. PyJulia) precisely because it
is the most general mechanism: it gives clean version isolation and accurate
per-runtime provenance, and it is identical in shape for every non-Python
language. The adapter, not the core, owns that transport.

### 4.3 RunResult and provenance

```python
@dataclass
class RunResult:
    problem_id: str
    backend: str
    method: str
    start: str
    x_final: list[float]
    cost_final: float
    dist_to_opt: float | None        # ||x_final - x*|| against nearest known optimum
    cost_gap: float | None           # cost_final - cost*
    grad_norm_final: float | None    # ||grad(x_final)||; optimality residual, no trace needed
    status: RunStatus                # CONVERGED | MAX_ITER | FAILED | DIVERGED | ERROR
    n_iter: int | None
    n_feval: int | None
    n_jeval: int | None
    n_heval: int | None
    trace: list[list[float]] | None  # iterates, if the backend exposes them
    timing: TimingStats | None       # median, MAD, n_reps, warmup, thread counts
    config: RunConfig
    provenance: EnvProvenance
    harness_git_sha: str
    timestamp: str                   # ISO-8601 UTC

@dataclass
class EnvProvenance:
    backend_name: str
    backend_version: str             # library version (and git sha if available)
    language_runtime: str            # e.g. "CPython 3.13.2", "Dyalog 19.0", "Julia 1.11"
    blas_lapack: str                 # vendor + version + threading (critical for these solvers)
    os: str
    cpu_model: str
    cpu_count: int
    machine_fingerprint: str         # stable hash to group runs by machine
```

Rationale for the emphasised fields: these solvers are dominated by dense linear
algebra, so the **BLAS/LAPACK vendor and thread count** materially affect timing
and occasionally iteration counts (via conditioning of solves). Capturing them
is what lets a future timing regression be attributed to a library change rather
than a machine or BLAS change.

---

## 5. Cross-language problem parity (the correctness backbone)

The single biggest correctness risk in a multi-language benchmark is that the
"same" problem is subtly different across languages. Guards:

- **Analytic-vs-FD parity (per language, Python included):** at every
  `probe_point`, the analytic Jacobian/Hessian must match a high-order
  finite-difference estimate within tolerance. Catches transcription errors in
  derivatives.
- **Cross-language parity:** each non-Python backend's residual/Jacobian/Hessian
  at the `probe_points` must match the Python reference within tolerance. This
  is a required test for every problem a backend implements natively.
- **Known-optimum invariants:** `‖grad(x*)‖ ≈ 0` and `cost(x*)` matches the
  registry value; where documented, second-order sufficiency holds.

These tests are the contract underpinning any downstream comparison.

---

## 6. Metric taxonomy (what each number is allowed to claim)

| Tier | Metrics | Cross-language claim? | Primary use |
|------|---------|-----------------------|-------------|
| **1: intrinsic** | final `dist_to_opt`, `cost_gap`, `grad_norm_final`, convergence success/failure, basin-of-attraction rate (fraction of a problem's registered `starts` reaching a known optimum), empirical convergence order & linear rate (from `trace`, where exposed) | Yes: these are algorithm properties, not environment artefacts | Capability comparison; "shines/fails" boundaries; assessing `trust` |
| **2: semi-comparable** | `n_iter`, `n_feval`, `n_jeval`, `n_heval` | Only within the same algorithm class, and only after normalising the definition of an iteration (rejected steps, inner iterations), documented per backend | Efficiency comparison with explicit caveats |
| **3: environment-dependent** | wall-clock `timing`, memory | No bare "X is faster than Y" across languages | Longitudinal tracking of one backend across versions/hardware; within-machine snapshots |

Consequences baked into the design:
- Headline cross-language findings are built on Tier 1.
- Order and rate are reported as an explicit `unavailable` outcome, not
  omitted or zero, for any backend whose harness does not expose per-iterate
  `trace`. `grad_norm_final` and the basin-of-attraction rate need only the
  final iterate, so they stay comparable across every backend regardless of
  trace support, and keep goal 2's cross-language assessment intact even
  where a backend's trace instrumentation lags.
- Tier 2 is always reported next to the algorithm and its stopping-criterion
  mapping, never as a lone number.
- Tier 3 is always reported as a distribution (median + MAD over N warm
  repetitions) with full provenance, and its main job is *change over time*,
  which is exactly one of the project goals.

---

## 7. Comparability methodology (`docs/methodology.md`)

Decisions to be written up and enforced by the runner/config layer:

- **Tolerance mapping.** Each library exposes different stopping parameters
  (`ftol/xtol/gtol`, cost tolerance, step tolerance, max iterations). `RunConfig`
  defines *intent-level* tolerances and each backend maps them to its native
  parameters; the mapping is documented and tested.
- **Derivative source held constant.** A given comparison fixes whether
  derivatives are analytic or finite-difference for all backends, since FD
  changes eval counts and accuracy. Both modes are studied, never mixed within
  one comparison.
- **Iteration definition.** Record and document how each backend counts
  iterations and evaluations (e.g. whether rejected trust-region steps count).
  Tier-2 comparisons normalise to a common definition where possible.
- **Timing policy.** Warm-up runs before measurement (essential for Julia JIT
  and for a warmed APL interpreter), N repetitions, pinned BLAS thread count,
  report median + MAD. Warm-up and thread counts are stored in `TimingStats`.
- **Starting points and problem scaling** are part of the problem spec, so they
  are identical across backends by construction.

---

## 8. Longitudinal / reproducibility design

- Results are appended to `results/*.jsonl`; JSONL is diff-friendly, streamable,
  and loads to pandas/Parquet on demand. Nothing is overwritten.
- `trust-bench compare <baseline> <candidate>` diffs two result sets:
  - Tier 1 metrics are expected to be **stable** → a change flags a real
    behavioural difference or bug (e.g. a library update changed convergence).
  - Tier 3 metrics are expected to **drift** → the command reports trends
    (speedups/regressions) grouped by `machine_fingerprint` and
    `backend_version`.
- A lightweight CI job runs a small, fast subset on every change and stores its
  results, so the harness's own behaviour is itself tracked over time.
- Because every `RunResult` pins `harness_git_sha`, `backend_version`,
  `language_runtime`, `blas_lapack`, and hardware, any historical result is
  fully attributable.

---

## 9. Capability studies (the experiments)

Each study is an independent module producing standardised `RunResult`s plus a
report section. They map onto the "where does each library shine or fail" goal.

1. **Baseline correctness.** The canonical set (Rosenbrock, Beale, Powell,
   Helical, ExpDec, Quadratic, Linear) at their standard starts. Regression /
   parity across backends; the floor everything must clear. Also reports the
   basin-of-attraction rate across each problem's registered `starts`
   (including deliberately hard ones, e.g. Rosenbrock's `far` start): the
   first Tier-1 metric in the plan that needs no `trace` at all.
2. **Large residual (LM vs Newton).** The prototype's axis: sweep residual
   size; Tier-1 order/rate where the backend exposes `trace`, `grad_norm_final`
   and basin-of-attraction rate regardless of trace support; predicted vs
   measured Gauss-Newton failure boundary.
3. **Ill-conditioning.** Sweep the condition number of `J`/Hessian; record
   success and precision. Separates exact trust-region, CG/Krylov, and dense
   quasi-Newton behaviour.
4. **Robust-loss / outliers.** Sweep outlier fraction across L2 → Huber →
   Cauchy → SoftL1 → Tukey/Welsch/Fair/Arctan. This is a likely `trust`
   *advantage*: SciPy's `least_squares` offers only `linear/soft_l1/huber/
   cauchy/arctan`, with no redescending Tukey/Welsch/Fair, so beyond high
   contamination `trust` (and a hand-rolled IRLS reference) is expected to
   survive where stock SciPy losses cannot.
5. **Bounded / constrained.** Box constraints: `trust`'s Coleman-Li affine
   scaling vs SciPy `trf`/`dogbox`. Inactive, active-at-boundary, and infeasible
   starts.
6. **Variable scaling.** Parameters spanning many orders of magnitude, with and
   without Marquardt/`x_scale` scaling. Tests `trust`'s adaptive diagonal
   scaling against `x_scale` counterparts.
7. **Dimensionality.** Generalised Rosenbrock at n = 2, 10, 100, 1000. A likely
   `trust` *limitation*: dense-Hessian methods blow up where matrix-free
   (Newton-CG, trust-krylov, L-BFGS-B) methods scale.
8. **Derivative source.** Analytic vs finite-difference Jacobian: cost in evals
   and precision.

Deliverables per study: a comparison table (Tier 1 + guarded Tier 2), a plot,
and rows in the capability matrix.

### 9.1 Capability matrix (a key output)

A backend × capability grid combining declared `Capabilities` with measured
outcomes (does the declared feature actually work, and how well), e.g.:

| Feature | trust (APL) | SciPy | Optimistix | Julia |
|---|---|---|---|---|
| LM / Gauss-Newton | yes | yes | yes | yes |
| BFGS | yes | yes | yes | yes |
| True-Newton (analytic H) | yes | yes (trust-*) | yes | yes |
| Redescending losses (Tukey/Welsch) | yes | no | ? | ? |
| Box constraints | yes | yes (trf) | ? | yes |
| Matrix-free / large n | no (dense) | yes | ? | ? |

(Cells filled from tests, not from documentation claims.)

---

## 10. TDD phase plan

Each phase lists the tests written *first*, then the implementation they drive.
No phase closes until its tests are green in CI.

**Phase 0: Scaffolding & CI.**
Tests: repo imports; `provenance.capture()` returns a populated `EnvProvenance`
and round-trips through serialisation. Deliver: package skeleton, pytest, lint
(ruff), coverage, CI workflow.

**Phase 1: Metrics (pure functions).**
Tests first: feed synthetic sequences of *known* behaviour: quadratic
(`e_{k+1}=e_k²`) → order ≈ 2; linear (`e_{k+1}=r·e_k`) → rate ≈ r; guards for
too-few-points → NaN (learned from the prototype, where fp64 quadratic
convergence yields too few iterates to fit an order); basin-of-attraction
rate over a set of `RunResult`s matches a hand-computed fraction. Deliver:
`metrics.py`.

**Phase 2: Problem registry & canonical problems.**
Tests first: analytic-vs-FD parity at probe points; known-optimum invariants
(`‖grad(x*)‖≈0`, `cost(x*)` matches); registry lookup by id/tag. Deliver:
`problem.py`, `registry.py`, the canonical problems and the difficulty families.

**Phase 3: RunResult & storage.**
Tests first: `RunResult` serialise/deserialise round-trip; append-only store
never overwrites; pandas loader reconstructs a DataFrame; provenance is
mandatory (constructing a result without it fails). Deliver: `result.py`,
`storage.py`.

**Phase 4: Backend contract & SciPy backend.**
Tests first (parametrised, so every future backend inherits them): `solve`
returns a well-formed `RunResult`; solves a trivial quadratic to the known
optimum within tolerance; respects `max_iter` (returns `MAX_ITER`, not a
crash); eval counts are non-negative and monotone; `capabilities()` is
consistent with what `solve` accepts (per method: a backend cannot claim
`bounds=True` for a method whose `solve` call rejects a bounded config).
Deliver: `backend.py`, `scipy_backend.py`
(least_squares lm/trf/dogbox, minimize BFGS/L-BFGS-B/Newton-CG/trust-*).

**Phase 5: Runner (orchestration).**
Tests first: `(problem, backend, method, start, config) -> RunResult` end-to-end
on Rosenbrock; tolerance intent is correctly mapped to SciPy params; trace is
captured when available. Deliver: `runner.py`, `config.py`.

**Phase 6: Studies & shape-regression tests.**
Tests first (assert *shape*, not exact numbers): large-residual study: measured
Gauss-Newton rate matches predicted spectral radius within tolerance and GN
diverges past the boundary; monotonic trends where expected. Deliver: the
studies in Section 9, one module at a time, each behind its own shape test.
Port the prototype's large-residual family here under these tests.

**Phase 7: Reporting.**
Tests first: report generation runs headless and produces expected artefacts;
capability matrix cells are derived from measured results and declared
capabilities and agree where both exist. Deliver: `reporting/*`, `cli.py`
`report` command. **Milestone: full Python-only pipeline is shippable here.**

**Phase 8: APL backend (`trust`).**
Spike first (de-risk the environment): stand up Dyalog headless (Docker image +
non-commercial licence), load the `trust` package (Link/Tatin), run one problem,
emit a `RunResult` JSON. Then tests: APL problem impls pass cross-language
parity; the backend passes the Phase-4 contract tests; end-to-end run stored.
Deliver: `backends_ext/apl/`, `apl_backend.py`. (This phase carries the main
schedule risk and is intentionally isolated.)

**Phase 9: Julia backend.**
Same pattern: Julia project with Optim.jl / LsqFit.jl / LeastSquaresOptim.jl;
parity + contract tests; warm-up handling for JIT baked into the timing policy.
Deliver: `backends_ext/julia/`, `julia_backend.py`.

**Phase 10: Longitudinal tooling.**
Tests first: `compare` correctly classifies a synthetic Tier-1 change as a
regression and a synthetic Tier-3 change as drift; grouping by machine/version
is correct. Deliver: `reporting/compare.py`, `trust-bench compare`, and the CI job
that runs a fast subset and stores results.

Ordering rationale: the core, SciPy backend, and studies (Phases 1-7) deliver a
complete, useful Python-only product before either non-Python backend is
touched, so value lands early and the risky APL integration cannot block it.

---

## 11. Risks and mitigations

- **APL environment setup** (Dyalog install, licensing, headless invocation,
  package loading). *Mitigation:* isolated Phase 8 with an explicit spike;
  Docker image; the Python pipeline is fully shippable without it.
- **Problem transcription drift across languages.** *Mitigation:* mandatory
  cross-language parity tests (Section 5); a problem is not "supported" by a
  backend until parity passes.
- **Stopping-criterion mismatch inflating/deflating iteration counts.**
  *Mitigation:* intent-level tolerance mapping; Tier-2 caveats; reliance on
  Tier-1 metrics for headline claims.
- **Timing noise and environment coupling.** *Mitigation:* warm-up, repetitions,
  pinned threads, full provenance; Tier-3 restricted to longitudinal/within-
  machine use.
- **JIT / interpreter warm-state effects (Julia, APL).** *Mitigation:* explicit
  warm-up in the timing policy; recorded in `TimingStats`.
- **Scope creep across studies.** *Mitigation:* studies are independent modules
  behind their own tests; ship Python end-to-end (Phase 7) first, add studies
  and backends incrementally.

---

## 12. Definition of done (per goal)

- *Capability overview:* every study in Section 9 runs across all available
  backends and populates the capability matrix from measured results.
- *Assess `trust`:* same-algorithm comparisons (LM/BFGS/Newton) against SciPy
  (and Julia) on Tier-1 metrics, plus the coverage matrix, with `trust`'s
  advantages (redescending losses, Coleman-Li bounds) and limitations
  (dense-Hessian scaling) quantified.
- *Longitudinal:* `trust-bench compare` distinguishes behavioural regressions from
  environmental drift, and CI stores a tracked result stream over time.

---

## 13. Resolved decisions

Settled during review; folded into the sections above.

1. **Name & location.** `trust-bench`, a separate repository from `trust`
   (package `trust_bench`, CLI `trust-bench`). See Section 3.
2. **Result store.** JSONL with a pandas loader; append-only. No database.
   Parquet remains an optional on-demand export only if run volume ever warrants
   it. See Sections 3 and 8.
3. **Non-Python transport.** Subprocess for all out-of-process backends (APL,
   Julia, future compiled backends), chosen as the most general mechanism, not
   in-process bridges. See Section 4.2.
4. **Ceres Solver.** Out of scope. The architecture supports adding it
   later as a subprocess backend, but the plan defers it. See Section 9.
5. **Callback bridge.** Not built. Native problem implementations plus parity
   tests are the only supported path; a Python-evaluated-residual bridge is
   added only if a black-box-only solver ever makes it necessary. See
   Sections 4.1 and 5.

No open questions remain; the plan is ready to execute from Phase 0.
