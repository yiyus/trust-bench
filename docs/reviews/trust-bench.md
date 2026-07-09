# Review: docs/plans/trust-bench.md - Optimisation-Solver Comparison Harness

## Summary

The plan is well-structured: three goals stay separated throughout, the metric
tiering (Section 6) gives every downstream claim an explicit comparability
bound, and the phase plan sequences risk sensibly (Python-only value lands by
Phase 7 before the two out-of-process backends). The prototype's mathematics
(Gauss-Newton spectral-radius argument in `prototype/large_residual.py`) is
sound, and the claims made about `trust`'s capabilities (Coleman-Li bounds,
eight loss functions including Tukey/Welsch/Fair, dense-only Hessian) match
the `yiyus/trust` repository. SciPy capability claims (loss set, matrix-free
methods) are also accurate.

Two design gaps are significant enough to raise before the phases that depend
on them lock in a contract: trace availability across backends, which the
project's headline Tier-1 metric depends on, and the flat `Capabilities`
schema, which cannot express the per-method distinctions the plan's own
illustrative capability matrix already uses. Neither blocks Phases 0-3.

## Findings

### Major: Tier-1 order/rate metric depends on trace availability, which no backend is required to provide

`RunResult.trace` is optional ("iterates, if the backend exposes them",
Section 4.3), yet Section 6 places empirical convergence order and linear
rate in Tier 1, the only tier the plan permits for cross-language claims and
for assessing `trust` against the state of the art (goal 2). If an
out-of-process backend does not expose iterates through the subprocess
transport, the project's central apples-to-apples metric is unavailable for
that backend, and the plan does not say what happens then: a documented
gap, a NaN with a distinct reason code, or a requirement to instrument the
backend's harness in `backends_ext/` to force trace capture.

This is not hypothetical for the two backends the plan actually schedules.
Julia's Optim.jl supports `store_trace`, but the Dyalog `trust` package's
public interface (per the repository) is not documented as returning
per-iteration iterates, so `backends_ext/apl/` may need extra instrumentation
work that Phase 8 does not currently scope.

Recommendation: state in Section 6 or Section 9.2 (large-residual study)
whether trace exposure is a mandatory backend-contract requirement (Phase 4)
or an optional capability with an explicit metric-unavailable outcome, and
scope any required APL/Julia harness instrumentation into Phase 8/9.

Location: Section 4.3 (`RunResult.trace`), Section 6 (Tier 1), Section 9 item 2.

### Major: `Capabilities` is backend-level, but capability truth is per-method

```python
class Capabilities:
    methods: frozenset[str]
    losses:  frozenset[str]
    bounds:  bool
    analytic_hessian: bool
    derivative_modes: frozenset[str]
```

`bounds` and `analytic_hessian` are single booleans per backend, but SciPy
already falsifies the premise: `least_squares(method="lm")` rejects bounds
while `trf`/`dogbox` accept them, and among `minimize` methods only
`L-BFGS-B`/`trust-constr` accept bounds while plain `BFGS` does not. The
plan's own illustrative capability matrix (Section 9.1) already annotates
this per-method ("yes (trf)", "yes (trust-*)"), so the gap is visible in the
document itself, not just in the underlying libraries.

Phase 4 names a concrete test that this schema cannot satisfy as written:
"`capabilities()` is consistent with what `solve` accepts." With a single
backend-level `bounds: bool`, that consistency check is either wrong (claims
bounds support that only one method has) or untestable at the granularity
the check implies.

Recommendation: key `bounds` and `analytic_hessian` by method (e.g.
`bounds_methods: frozenset[str]`, or restructure `Capabilities` as a mapping
from method name to a small per-method capability record) before Phase 4
pins the contract tests.

Location: Section 4.2 (`Capabilities`), Section 9.1 (capability matrix),
Section 10 Phase 4.

### Minor: no defined location for the Study 4 IRLS reference implementation

Section 9 item 4 relies on "a hand-rolled IRLS reference" alongside `trust`
and SciPy to demonstrate the robust-loss advantage, but Section 3's repo
layout has no module for it (not under `problems/`, `backends/`, or
`studies/`). Since this reference implementation is itself a comparison
baseline whose correctness matters for the headline "trust survives past
high contamination" claim, it should have the same home and scrutiny as a
backend, not live as an inline helper inside `robust_loss.py`.

Location: Section 3 (repo layout), Section 9 item 4.

### Minor: EnvProvenance has two producers with no stated merge rule

`provenance.py`'s `capture()` (Phase 0) is tested to return "a populated
`EnvProvenance`", and `Backend.environment()` (Section 4.2) also returns an
`EnvProvenance`. Some fields are clearly machine-level (`os`, `cpu_model`,
`cpu_count`, `machine_fingerprint`) and some are clearly backend-level
(`backend_version`, `language_runtime`, `blas_lapack`), but the plan never
states whether `Backend.environment()` calls `provenance.capture()` and
overlays backend-specific fields, or the two are independently assembled by
the runner. Worth pinning down before Phase 4, since every backend
implementation will otherwise invent its own answer.

Location: Section 4.2, Section 4.3, Section 10 Phase 0/4.

### Note: multi-year append-only results with no stated retention policy

Section 8 has CI append a result batch to `results/*.jsonl` "on every
change", and Section 3 does not mark `results/` as git-ignored (unlike
`reports/`). Given the project's explicit multi-year longitudinal horizon
(goal 3), continuous per-commit appends are a plausible source of
unbounded repository growth. This may be an intentional trade-off (the
append-only history is itself the audit trail the longitudinal goal wants),
but Section 11's risk list does not mention it, so it reads as unconsidered
rather than accepted.

Location: Section 3, Section 8, Section 11.

## Verification

- Tests: not run (design review; no test surface exists yet).
- Prototype mathematics (Gauss-Newton spectral-radius argument): checked
  against `prototype/large_residual.py`, `harness.py`, `run_experiment.py`;
  consistent with the standard Gauss-Newton/Newton convergence-rate result
  the plan claims.
- `trust` capability claims (Coleman-Li bounds, eight loss functions, dense
  Hessian): checked against the `yiyus/trust` repository; accurate.
- SciPy capability claims (loss set, matrix-free methods): accurate against
  `scipy.optimize` documentation.
- Style: no emojis, em-dashes, narrative markers, or US spellings found in
  the document.

## Recommendation

Approve with minor changes. The two Major findings should be resolved (or
turned into explicit, tracked open questions) before Phase 4 pins the
backend contract and before Phase 1/6 pin the order/rate metric contract;
neither blocks starting Phase 0-3.
