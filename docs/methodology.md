# Comparability methodology

Decisions that keep a cross-backend comparison meaningful, enforced by
the runner/config layer and tested per backend. See Section 7 of
`docs/plans/trust-bench.md` for the full list this document expands.

## Tolerance mapping

`RunConfig.tolerance` is a single intent-level value. Each backend maps
it onto its own native stopping parameters; leaving it `None` means
the backend falls back to its own native default rather than any value
this project chooses, and the two defaults are not comparable (see
below).

### scipy

`least_squares` (`lm`/`trf`/`dogbox`) exposes three independent stopping
criteria - `ftol` (cost change), `xtol` (step size), `gtol` (gradient
projection) - applied simultaneously. A single `RunConfig.tolerance`
value is written to all three (`trust_bench/backends/scipy_backend.py`).
Left `None`, scipy's own default is `ftol=xtol=gtol=1e-8`.

`minimize` methods each expose a different subset of `{ftol, xtol,
gtol}` (`_MINIMIZE_TOLERANCE_PARAMS` in the same file, derived from
`scipy.optimize.show_options`); a single `RunConfig.tolerance` value is
applied to every parameter the chosen method accepts.

### trust-apl

`trust`'s Newton-region solver (`Newton.aplo`) does not expose separate
step/cost/gradient thresholds. Its convergence check (`Newton.aplo`'s
`C` function) stops when either:

- the cost itself drops below an absolute threshold (`tc`) - a genuine
  precision guarantee, or
- a relative change metric `r` - the minimum of relative parameter-space
  change and relative cost change between the last two accepted
  iterates (`Newton.aplo`'s `T` function) - drops below a relative
  threshold (`tr`), once `r` is positive (a negative `r` means the step
  made things worse, not converged). `tr` is a stall detector, not a
  precision selector: it bounds how small the change between accepted
  iterates can get before the solver gives up, regardless of how far
  that point is from the optimum.

`solve.dyalog` maps `RunConfig.tolerance` onto `tc` (`cfg.tolc`) only.
`tr` (`cfg.tolr`) always stays at its tight native default (`⎕CT`,
measured `1e-14` in this environment) regardless of the requested
tolerance. Loosening `tr` in step with `tc` does not trade precision for
speed the way loosening `tc` does - it only lowers the bar for a false
stall. Confirmed directly: mapping a requested `tolerance=0.1` onto both
`tc` and `tr` made `trust-apl`'s `lm` report a stall after a single
iteration (`dist_to_opt≈1.95`), worse than the *tighter* `tolerance=0.01`
(`dist_to_opt≈0.078`) - a looser tolerance produced a worse result, the
opposite of the intended trade-off. Left `None`, `tc` also falls back to
`⎕CT`.

### Why an equal numeric value is not an equal test

scipy's thresholds are independent, absolute-scale comparisons against
the current step/cost/gradient. `trust`'s `tc` is a single absolute cost
threshold, not scipy's three independent step/cost/gradient checks. The
same numeric `tolerance` value therefore does not exercise the same
stopping condition in each backend.

Confirmed directly (`rosenbrock.PROBLEM`, method `"lm"`, `tolerance`
from `1e-2` to `1e-14`): scipy reaches `dist_to_opt == 0.0` at every
value in that range, unaffected by tightening it further. `trust-apl`
degrades smoothly as `tolerance` loosens (`0.078` at `1e-2`, `6.2e-5` at
`1e-6`) and plateaus at a floor of `1.55e-7` from `1e-10` onward (the
`⎕CT` floor its own iteration cannot resolve below). A precision or
iteration-count difference measured under an equal `RunConfig.tolerance`
is not, by itself, evidence of a capability difference between the two
libraries: it may equally reflect this mapping mismatch. Section 9's
capability studies that report precision across backends should be read
with this in mind; `trust_bench.studies.tolerance_comparability` exists
to make the comparison explicit rather than leave it as an unstated
side effect of `tolerance=None`.

## Status classification: CONVERGED vs STALLED

`trust`'s own API never asserts convergence: `Newton.aplo` returns
`iter`/`cost`/`rel`/`dnorm`/`p` and leaves interpretation to the caller.
Of the two conditions in `Newton.aplo`'s own termination check that
aren't iteration-limit or damping-saturation (`FAILED`/`MAX_ITER`), only
one is a genuine near-optimality guarantee: the cost dropping below an
absolute threshold (`tc>c`). The other - the relative-change metric `r`
stalling (`(r>0)∧tr>r`) - only means the last accepted step stopped
changing much; on a problem whose residual doesn't vanish at the true
optimum (e.g. the `large_residual`/`outliers` families, where cost is
large by construction even at `x_star`), it is the *only* criterion that
ever fires, and does so at genuine convergence. At extreme conditioning
in `ill_conditioning` (high `kappa`, both `BFGS` and `trust-exact`), the
same criterion can fire from a point nowhere near the optimum, once
damping has grown large without quite crossing the `FAILED` threshold -
not because any Hessian involved is indefinite (`ill_conditioning`'s is
never indefinite; see the capability boundary sections below), but
because solving the damped-Newton system itself loses numerical
precision at that condition number.

`solve.dyalog` (`backends_ext/apl/solve.dyalog`) resolves this using
vendored `trust`'s own `Result` namespace (`APLSource/Result.apln`), a
set of small predicates for classifying why a solve terminated, called
directly on the namespace `Min` returns:

- `Result.StalledByEscalation` - a relative-change stall with damping
  already escalated past `1` (`Newton.aplo`'s own signal that the last
  rejected step's damping increase, not genuine convergence, is what
  drove the relative-change metric below threshold).
- `Result.StalledByPrecision` - a relative-change stall at *low*
  damping (`≤1`) with a genuinely nonzero final gradient: the
  ill-conditioning-driven precision-loss case `StalledByEscalation`
  can't see, since damping never escalated there at all. Uses the same
  `1e-2` gradient-norm heuristic this project's own harness previously
  computed independently before this predicate existed - not a value
  `trust` itself derives from anything, just a heuristic with a large
  measured margin either side (genuine convergences, including the
  large-residual case above, leave a gradient norm at or below `1.6e-6`;
  the indefinite-Hessian stalls this distinguishes leave one at `4.8`
  or higher).

A relative-change stall is reported as the distinct `RunStatus.STALLED`
when either predicate fires, `CONVERGED` otherwise - which also
correctly covers `Result.Converged`'s own strict `cost<tc` guarantee
and the large-residual/outliers case, both of which leave neither
predicate's signature (no damping escalation, a small final gradient).

`Result.StalledByPrecision` shipped upstream (`da5dfcc`) with its guard
condition inverted - it returned false exactly for the low-damping case
its own name and comment describe, only ever agreeing with
`StalledByEscalation`. Confirmed directly (`ill_conditioned(kappa=1e7)`
`trust-exact`, `dnorm≈0.001`): before the fix, `StalledByPrecision`
returned `0`; fixed upstream (`8ba5494`) before landing here.

A second upstream bug (fixed at `5acffe4`/`bd62a45`/`ae359a0`): neither
`StalledByEscalation` nor `StalledByPrecision` checked `~Converged`
before firing, so a genuinely converged run (`cost` far below `tolc`)
whose gradient norm happened to sit just over the `1e-2` heuristic could
still be misreported `STALLED`. Confirmed directly
(`scaling.make(1e6)`/`lm`, unscaled: `cost_final≈1.95e-16`,
`grad_norm_final≈0.0198`, just over `gtol`): reported `STALLED` before
the fix, `CONVERGED` after. The same investigation surfaced two further
bugs in the bare `Result.Stalled` predicate itself (an `∨`/`∧` slip that
briefly broadened it to fire on any non-convergence, and a missing
`rel≥0` guard against `Newton.aplo`'s own "not yet computed" sentinel,
`rel=-1`) - neither corrupted this project's own report, since
`solve.dyalog` only calls the two derived predicates, both of which
were fixed directly and now delegate to the corrected `Stalled`
internally rather than duplicating the same `tolc`/`tolr`/`rel` logic.

This gradient-norm check does not apply to a bounded request
(`req.bounds` set): at an active-boundary optimum, the unconstrained
gradient is genuinely nonzero by construction (measured, `quadratic`'s
`active_at_boundary` scenario: `grad_norm_final≈0.5`), not a stall.
Distinguishing a real stall from a genuine bounded convergence needs a
KKT-aware (projected) gradient check this harness does not compute. A
bounded request's relative-change stall is reported `CONVERGED`
unconditionally, exactly as before this fix - a known limitation, not a
claim this fix resolves bounded near-optimality detection.

`grad_norm_final` itself no longer costs an extra evaluation: `Min.aplo`
(vendored `trust` commit `d151b8b`) now returns the final gradient it
already computes internally on every accepted step (used to compute the
next search direction) as `r.grad`, read directly instead of the
harness re-evaluating the problem at the final point via a separate
probe.

## Parameter scaling: trust's pscale vs scipy's x_scale

`RunConfig.x_scale` maps to two structurally different mechanisms.

scipy's `x_scale` (`least_squares`-only, `lm`/`trf`/`dogbox`) accepts
either a fixed per-parameter array or the string `"jac"`: an *adaptive*
rescaling recomputed every iteration from the current Jacobian's own
column norms.

`trust`'s `pscale` (vendored commit `118263c`) is a *fixed* vector only
(`x = pscale × y`, a plain linear reparameterisation around the
evaluation function) - there is no adaptive equivalent. This isn't a
gap: `Newton.aplo`'s own damping already scales by `diag(H)`, not
`diag(1)`, giving an adaptive-scaling effect internally regardless of
`pscale`, so a separate adaptive mode would be largely redundant.

`pscale` applies to `lm` and `BFGS` (`apl_backend.py` maps a fixed
`RunConfig.x_scale` to it for both), but not `trust-exact`: the
wrapper (`Min.aplo`'s `PS`) only rescales a 2-item `(value, derivative)`
return - `lm`'s `(residual, jacobian)`, `BFGS`'s `(cost, gradient)` -
not `trust-exact`'s 3-item `(cost, hessian, gradient)`. A Hessian needs
outer-product scaling on both axes, not a column scale, so `trust-exact`
rejects a fixed `x_scale` explicitly rather than silently applying it
wrong. `"jac"` is rejected for every method: no backend-native adaptive
equivalent exists to map it to.

This closes a real capability gap, not just a missing code path.
Confirmed directly (`scaling.make(1e8)`, `lm`, unscaled): `trust-apl`
reports `FAILED` with `cost_final≈2e16`. With a fixed `x_scale=(1.0,
1e-8)` - matching this problem's own known anisotropy (`scaling.make`'s
own docstring: the Hessian's diagonal ratio is exactly `scale**2`) -
the same request reports `CONVERGED` with `cost_final≈1.9e-12`.
`trust_bench.studies.scaling`'s own sweep now exercises this
(`X_SCALES` gains `"fixed"`, a per-`scale` `(1.0, 1.0/scale)` value),
alongside a `trust-apl`-only exercise of `BFGS` with the same fixed
`pscale` - a `trust`-only capability, since scipy's own `BFGS`
(`minimize`-family) has no `x_scale` concept at all to compare against,
matching the `robust_loss` study's own precedent for a backend-specific
capability with no cross-backend comparison point.

## Capability boundary: trust-apl's BFGS under ill-conditioning and scale

`trust-apl`'s BFGS engine is measurably less robust to ill-conditioning
and dimensionality than scipy's BFGS, confirmed directly by sweeping
both. In both cases the underlying Hessian involved is never indefinite
(`ill_conditioning`'s is the constant, positive-definite `a.T@a`;
`dimensionality`'s is discussed below) - this is a numerical-precision
boundary, not the indefinite-Hessian one the next section covers.

- `ill_conditioning` (`kappa` from `1` to `1e8`): `trust-apl BFGS`'s cost
  at the reported solution grows from machine precision at `kappa≤1e3`
  to `~1.1e6` at `kappa=1e4` (`STALLED`, not a genuine convergence -
  see above) and `~2.7e14` by `kappa=1e8` (`FAILED`). scipy's own BFGS
  stays at `cost_final==0.0` from `kappa=1e6` onward across the same
  range.
- `dimensionality` (`n` from `2` to `1000`): `trust-apl BFGS` reaches
  `MAX_ITER` at `n=100` with a worse residual than scipy's own BFGS at
  the same `n` (`1.71` vs `0.07`), and times out entirely at `n=1000`
  (`tests/backends/test_apl_difficulty_families.py::
  test_solve_bfgs_completes_without_a_timeout_at_a_practical_dimension`
  pins the `n=100` case; the harness's own subprocess timeout, not a
  study assertion, is what stops `n=1000` from completing).

A `dimensionality`/`BFGS`/`n=1000` report row reads as a plain `ERROR`,
its `message` field reading `"harness did not complete within 60s"` -
distinguishing it from any other harness-reported `ERROR` (e.g. an
unrecognised `problem_id`'s own `"Unknown problem_id: ..."`) by content,
without a dedicated status of its own (see "Declared-unsupported,
timeout, and genuine-crash reporting" below). Confirmed directly (raw
`run_harness.sh` invocation, no subprocess timeout applied): the solve
is not crashing, just slow - it completes on its own after ~280s,
`MAX_ITER` at 201 iterations, `cost_final≈21.27`. The underlying cause
is architectural, not a bug to fix here: `trust`'s BFGS keeps and
updates a dense `n×n` approximate Hessian every iteration (the same
`O(n²)`-`O(n³)` cost profile as `lm`'s own linear solve), so it neither
exploits residual/Jacobian structure the way `lm` does on this
benchmark's fit-shaped problems, nor scales the way a genuinely
limited-memory BFGS (scipy's `L-BFGS-B`, which converges cleanly at
`n=1000` in the same study) would.

`trust-apl trust-exact`, by contrast, tracks scipy's own trust-exact
closely across both sweeps - `dimensionality` converges at `n=1000` on
both backends
(`tests/backends/test_apl_difficulty_families.py::
test_solve_trust_exact_converges_at_n_1000`), and `ill_conditioning`
only shows the same stall pattern at `kappa=1e7`/`1e8`, well past where
BFGS already fails. The gap is BFGS-specific, not a general weakness of
`trust-apl`'s Newton-region engine.

## Declared-unsupported, timeout, and genuine-crash reporting

`trust_bench.reporting.tables.results_to_dataframe` accepts either a
`RunResult` or a raised exception per sweep entry - a study's own
`except ValueError` block (`scaling.py`'s `x_scale="jac"` probe against
a backend with no adaptive equivalent, `bounded.py`'s infeasible-start
scenario against `lm`/`trf`/`dogbox`) catches exactly a backend's own
declared-unsupported or rejected-input rejection, the expected, passing
outcome of the sweep's own probe, never a genuine uncaught crash (which
would propagate rather than land here). That row's status is
`"UNSUPPORTED"`, with the exception's own message kept rather than
discarded and every other metric field blank.

A completed `RunResult`'s own `status="ERROR"` (a harness-side crash or
the subprocess timeout described above) is a different case entirely:
these already carry real content in `RunResult.message`, threaded
through from each backend's own already-computed termination
explanation (scipy's `OptimizeResult.message`; `trust-apl`'s harness
response, which already includes a `message` key for every `ErrorResult`
- previously discarded entirely by `apl_backend.py`). A timeout's
`"harness did not complete within 60s"` and an unrecognised
`problem_id`'s `"Unknown problem_id: ..."` both report `status="ERROR"`
but are distinguishable by message content alone, without a dedicated
status for either.

Both `trust_bench.cli._check_backend_coverage` and
`trust_bench.reporting.cross_study._pivot_by_backend`'s missing-backend
guards filter on `trust_bench.reporting.tables.NON_RESULT_STATUSES`
(`{"ERROR", "UNSUPPORTED"}`) rather than `"ERROR"` alone, so a backend
whose only rows in a sweep are declared-unsupported rejections is
correctly treated the same as one whose only rows are crashes: neither
represents a genuine, comparable result.

## Trust-exact and Hessian indefiniteness

Unlike `ill_conditioning`'s numerical-precision boundary above, Hessian
indefiniteness is a separate, genuine property of the underlying maths
that `trust-exact`'s Newton-region engine must navigate, confirmed
directly (`tests/problems/test_families.py::
test_dimensionality_hessian_is_indefinite_partway_between_the_start_and_the_optimum`,
`tests/problems/test_typical.py::
test_hessian_indefiniteness_at_the_standard_start_matches_trust_exacts_known_fragility`),
not only inferred from solver behaviour:

- A nonlinear-least-squares problem's true Hessian is `J.T@J` plus a
  residual-weighted correction term that vanishes only where the
  residual itself is zero. Away from a zero-residual point, that
  correction can flip an eigenvalue's sign.
- `dimensionality` (generalised Rosenbrock, zero residual at `x*`):
  indefinite partway between its standard start and the optimum
  (measured, `n=10`: minimum eigenvalue `-194` at the midpoint) - but
  mildly enough that `trust-exact` still converges reliably through
  `n=1000` (see the section above).
- `noisy_expdec`/`gaussian_peak` (the typical study, nonzero residual
  even at `x*` since both fit noisy data): indefinite already at the
  standard start (measured minimum eigenvalue `-82`/`-8`). Before
  vendored `trust` commit `05f9010`, `trust-apl trust-exact` genuinely
  failed here (`MAX_ITER`, `dist_to_opt` in the thousands to billions):
  `Newton.aplo` accepted a step whenever its actual outcome improved
  the cost, even if the quadratic model that produced it had predicted
  a negative error decrement - a sign the damping wasn't yet enough to
  counteract the indefinite Hessian. `05f9010` rejects any such step
  outright, and `trust-apl trust-exact` now converges cleanly on both
  problems (`tests/studies/test_typical_study_apl.py::
  test_trust_apls_trust_exact_converges_despite_an_indefinite_hessian_away_from_the_optimum`).
  `logistic`/`michaelis_menten`, the typical study's other two problems,
  have an always-PSD Hessian (canonical-link-shaped likelihoods) and
  were never affected either way.

`ill_conditioning`'s fragility, despite looking similar in its symptom
(`trust-exact` failing at an extreme parameter value), is a different
mechanism entirely (numerical precision loss, not indefiniteness - see
above) and is not affected by this fix.

## Robust loss tuning constants: matching scipy's f_scale to trust's own

A named robust loss (`huber`, `cauchy`, `soft_l1`, `arctan`) is not a
single fixed function: each has a tuning constant controlling where it
transitions from quadratic to its outlier-downweighting behaviour, and
the two libraries do not default to the same one. `trust`'s `Loss.apln`
(`backends_ext/apl/trust/APLSource/Loss.apln`) bakes in the textbook
~95%-asymptotic-efficiency constants (`huber=1.345`, `cauchy=2.385`,
`softl1=1`, `arctan=1`), while scipy's `least_squares` always uses its own
fixed default, `f_scale=1.0`, unless a caller passes one explicitly -
which `trust_bench.studies.robust_loss` did not.

Confirmed directly this was the actual mechanism behind the study's
measured scipy-vs-`trust-apl` precision gap, not a coincidence: the gap's
size tracked the ratio between each loss's `trust` constant and scipy's
fixed `1.0`, not a general "`trust` is less robust" pattern. `soft_l1`/
`arctan` (both libraries at `1.0`, matching) showed the two backends
landing close together across the fraction sweep; `huber` (`1.345` vs
`1.0`, a modest mismatch) showed a modest, consistent gap; `cauchy`
(`2.385` vs `1.0`, the largest mismatch) showed the largest gap. Read at
face value, the original table looked like a genuine capability
difference; the better-supported reading was an unheld comparability
variable, the same category of pitfall the tolerance-mapping section
above already documents for a different parameter.

`RunConfig.f_scale` (mirroring `x_scale`'s shape) closes this gap on the
scipy side: `trust_bench.studies.robust_loss._F_SCALE_FOR_LOSS` maps each
swept loss to trust's own constant, passed as `f_scale` for the scipy
backend only. `trust-apl` calls are left untouched, since `APLBackend`
has no way to honour an explicit `f_scale` at all: trust's own MAD-based
auto-scaling (`Min.aplo`'s `L` function, `sigma←(A ⍵)÷0.6745`, recomputed
from the current residuals on every call) is a genuine algorithmic
difference from scipy's fixed `f_scale`, not something scipy's side is
made to replicate - `APLBackend.solve` rejects an explicit `f_scale`
outright rather than silently ignoring it, matching `x_scale`'s own
precedent for a parameter one backend has no native equivalent for.

Confirmed directly, re-sweeping `fraction=0.3` after the fix: `huber`
(scipy `7.80`, `trust-apl` `8.27`), `cauchy` (`7.82` vs `7.86`) and
`soft_l1` (`7.89` vs `7.91`) now cluster together, against the original
`cauchy` gap of `1.69` (scipy, mismatched constant) vs `7.86`
(`trust-apl`) the issue itself measured. The two backends still don't
land on identical values - a matched tuning constant controls for the
loss shape, not for the two solvers' different optimisation paths
through it - but the dominant, systematic driver of the original gap is
gone.

## scalar_cost and the parity/frontier pool

`scalar_cost` (`rastrigin`/`cauchy_mle`, genuine Jacobian-free scalar
objectives exercising `BFGS`/`L-BFGS-B` on their own terms) is in
neither `cross_study.parity_frame`'s pooled comparison nor
`frontier_panels`'s small-multiples chart, for two independent reasons.

**Parity scatter**: shape-wise, `scalar_cost` would fit - like
`baseline`/`typical`/`bounded`, its rows are already `dist_to_opt`/
`status` per `(problem_id, method, backend)`, not `robust_loss`'s own
distance-to-true-parameters shape. The blocker is `trust-apl` itself:
it has no evaluator for either problem at all (`APLBackend` doesn't
even declare `"L-BFGS-B"`; every `"BFGS"` row reports `ERROR`,
`"Unknown problem_id"`). Confirmed directly: pooling `scalar_cost`
alongside the other three studies and pivoting it the same way raises
`_pivot_by_backend`'s own missing-backend guard unconditionally
(`ValueError: no results for backend(s) trust-apl`) on any two-backend
call - there is no `trust-apl` side to plot at all, not merely a partial
gap `results_to_dataframe`'s `UNSUPPORTED`/`ERROR` labelling could
paper over.

**Capability frontier**: `scalar_cost` sweeps two fixed problems with no
difficulty parameter at all - unlike every other panel here
(`kappa`/`scale`/`n`/`rho`/`fraction`), there is no natural x-axis for a
small-multiples line, so it doesn't fit this chart's shape regardless of
backend coverage.

Both exclusions are permanent under the current `trust-apl` capability
set, not a gap to close by more report-side wiring: the parity blocker
specifically would need `trust-apl` to gain a Jacobian-free scalar
evaluator first (a real, nontrivial capability gap in its own right, not
a reporting-layer fix) - out of scope here, and not attempted.

## Timing measurement: `RunResult.timing`

Both backends follow the same policy (Section 7 of the design doc):
one discarded warm-up solve, then five measured repetitions, reporting
median and MAD (1.4826-scaled, matching `robust_loss.py`'s own
`irls_tukey` convention) rather than mean/stddev.

**scipy**: each measured repetition is wrapped in `time.perf_counter()`,
all inside one `threadpoolctl.threadpool_limits(limits=1)` context -
pinned to a single BLAS thread so the number reflects single-threaded
algorithmic cost, not a machine-dependent parallel-scaling factor that
would vary between whoever's machine ran the comparison.

**trust-apl**: measured *inside* `solve.dyalog` itself, via `⎕AI[2]`
(confirmed directly: computation time in milliseconds) wrapped tightly
around the `Min` call, not by wall-clocking the round trip from Python.
This matters because of what a round trip includes: since #139, a
repeated call no longer pays interpreter-startup cost, but it still
pays JSON encoding/decoding and the request/response IPC transfer -
wall-clocking the whole `_send_request` call would count that as if it
were solve time. `thread_count` is recorded as `1` without an active
pinning call: Dyalog's interpreter is single-threaded for this workload
(`Newton.aplo`/`Min.aplo` don't parallelise), nothing to pin.
