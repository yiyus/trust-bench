# trust-bench

Optimisation-solver comparison harness. Benchmarks the Dyalog APL library
[`trust`](https://github.com/yiyus/trust) (trust-region Newton, BFGS and
Levenberg-Marquardt, robust losses, box constraints) against SciPy across a
set of difficulty studies (large residuals, ill-conditioning, robust losses,
bounds, scaling, dimensionality, derivative source), producing comparison
tables, plots, and an HTML report.

See `docs/plans/trust-bench.md` for the full design and `docs/methodology.md`
for the comparability rules behind the metrics.

## Setup

```
git clone --recurse-submodules <repo-url>
cd trust-bench
pip install -e .[dev]
```

The APL backend additionally requires `dyalogscript` (Dyalog APL) on `PATH`.
Without it, APL-backed studies and tests are skipped automatically; the SciPy
backend has no extra dependencies.

## Usage

Run the full report (SciPy only):

```
trust-bench report --output-dir reports --html
```

Include the APL backend:

```
trust-bench report --output-dir reports --html --backends scipy trust-apl
```

Run a subset of studies, or skip the slow ones:

```
trust-bench report --only baseline scaling
trust-bench report --skip-slow
```

## Tests

```
make test       # fast subset
make coverage   # full suite with coverage
make lint
```
