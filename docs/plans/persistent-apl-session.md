# Persistent APL session (#139)

## Context

Every `APLBackend.solve()`/`evaluate_problem()` call currently spawns a fresh
`dyalogscript` subprocess (`apl_backend.py`'s `_run_harness`, calling
`backends_ext/apl/run_harness.sh`, which concatenates ~20 small `.dyalog`
source files into one temp script and runs it via
`subprocess.run(..., timeout=60)`, file-based request/response). Interpreter
startup alone costs 3-9s; a real solve costs about the same, so it's virtually
all overhead, not compute. Measured impact: ~15 minutes of pure spawn overhead
across 134 `trust-apl` rows in one real report run; ~3.3s/test average across
the ~90 APL-touching test functions in 18 files. This replaces that with one
long-lived `dyalogscript` process per outer run (one `trust-bench report`
invocation, one pytest session), talking newline-delimited JSON over
stdin/stdout, synchronous one-request-at-a-time - with **no change to how any
caller works**: every `sweep()`, every test file's `APLBackend()` construction,
stays exactly as it is today.

## `run_harness.sh` stays untouched

`tests/backends_ext/apl/test_harness.py` is a 20-test, `@pytest.mark.slow`
suite exercising `run_harness.sh` directly, file-based, one-shot - and this
project's own review history (`docs/reviews/36.md`, `docs/reviews/83.md`,
`docs/methodology.md`) repeatedly uses "a raw `run_harness.sh` invocation" as
an independent, manual verification technique outside pytest entirely. This
is a first-class, load-bearing interface, not incidental plumbing. This
change adds a **new**, parallel entry point for the persistent session; it
does not repurpose or remove the existing one-shot path.

## Design

**1. Session ownership: a module-level singleton in `apl_backend.py`.**
22+ places across `trust_bench/cli.py` and 18 test files construct
`APLBackend()` independently (`Backend`/`APLBackend` have no `__init__`/
teardown/context-manager today). Private module-level state (a `Popen`
handle + a lock), lazily started on first use by *any* `APLBackend` instance
or `evaluate_problem()` call, transparently shared for the life of the
process - matching "no change to any caller" exactly, unlike a pytest-fixture
design that would need threading through every one of those files.

**2. New driver + script pair, reusing the same source files.**
- `backends_ext/apl/session.dyalog` (new): a persistent-loop driver, same
  `:Trap 0` / `Solve`-or-`Evaluate` dispatch and `ErrorResult` fallback as the
  existing one-shot `run.dyalog`'s `Run`, but looping until it reads an empty
  line instead of handling exactly one request and calling `⎕OFF`.
  `run.dyalog` itself is untouched.
- `backends_ext/apl/run_session.sh` (new): mirrors `run_harness.sh`'s
  mktemp+trap-cleanup structure but concatenates `session.dyalog` instead of
  `run.dyalog`, and takes no positional input/output path arguments - I/O is
  stdin/stdout, inherited from the parent `Popen`.
- The shared ~20-file concatenation list is factored out of `run_harness.sh`
  into one sourced fragment (`backends_ext/apl/_sources.sh`, a bash array)
  both scripts read from, so adding a new problem family only means updating
  one place (multiple past commits had to touch this exact list by hand).

**3. Transport - confirmed empirically via a hands-on spike, not assumed:**
- `⍞` reads one line from stdin at a time, correctly stripped of its
  terminator; an empty line is a clean stop signal the loop can check with
  `:If 0=≢line`.
- `⎕←` terminates each line with a bare `\r`, not `\n` - hard-wired
  dyalogscript behaviour, confirmed directly (`⎕←(⎕UCS 10)` still gets a
  trailing `\r` appended after the embedded `\n`). **Python must read stdout
  in binary mode and split on `\r` manually.** Using `text=True`/universal
  newlines deadlocks: the incremental newline decoder can't safely decide a
  bare `\r` isn't the start of `\r\n` without peeking one more byte, and the
  child is meanwhile blocked waiting for the next request - confirmed
  directly by reproducing the hang, then fixing it.
- The child's echo of stdin input goes to **stderr**, not stdout (confirmed
  directly) - keep them separate, never merge.
- **`dyalogscript`'s stdout is fully buffered, not line-buffered, when
  connected to a pipe** (as opposed to a file or a terminal) - without a fix,
  each response sits in the child's internal buffer and is never actually
  written to the pipe until the buffer fills or the process exits, which a
  synchronous one-request-at-a-time protocol can't tolerate. Confirmed
  directly (a live round-trip hung 9+ minutes without this fix, resolved
  immediately with it). Fix: invoke via `stdbuf -oL` (coreutils, forces
  line-buffering via `LD_PRELOAD`; already available in this environment and
  on essentially any Linux box) - `["stdbuf", "-oL", "bash",
  "run_session.sh"]` as the `Popen` argv. Verified this propagates correctly
  through the nested `bash -> dyalogscript (itself a bash wrapper) -> dyalog`
  process chain, since `LD_PRELOAD` is inherited across `exec`.
- End-to-end round-trip timing, measured directly: first request ~2.2s
  (interpreter startup), every subsequent request ~1-2ms - confirms the
  issue's whole premise and payoff in this environment.
- Timeout + crash recovery, measured directly: a request that hangs times
  out cleanly via `select.select(..., timeout)`; `proc.kill()` + `wait()`
  cleanly tears down the stuck session; a freshly-spawned session handles
  the next, unrelated request correctly with no special caller-visible
  behaviour.

**4. Timeout and crash recovery: kill-and-restart, not silent retry.**
- Read times out, or the process is found dead (`proc.poll() is not None`)
  before/after a write: kill the process, discard the handle so the *next*
  call lazily starts a fresh session, and return the *current* call's
  existing `_timeout_result(...)` shape (already exists, unchanged) - a hung
  request still surfaces as today's familiar `ERROR`/timeout-message result.
- No automatic retry of the failed request against a fresh session - the
  failure is real and visible for that one call, matching the issue's own
  accepted tradeoff.
- `atexit.register`-based teardown (no `conftest.py`/pytest hooks exist in
  this project today, and `atexit` needs none): send an empty line (the
  clean stop signal), close stdin, wait briefly, kill if still alive. Works
  identically whether the process is `pytest` or `trust-bench`.

## Files touched

- New: `backends_ext/apl/session.dyalog`, `backends_ext/apl/run_session.sh`,
  `backends_ext/apl/_sources.sh`.
- Changed: `trust_bench/backends/apl_backend.py` (module-level session
  state, `_send_request`, `atexit` teardown; `solve`/`evaluate_problem` call
  it instead of `_run_harness`; `_run_harness`/`_HARNESS` removed once
  nothing calls them).
- Changed: `tests/backends/test_apl_backend_timeout.py` - its current
  mechanism (monkeypatching `subprocess.run` to raise `TimeoutExpired`)
  doesn't apply to a persistent `Popen`+`select` design; rewritten to drive
  a genuinely slow/hanging request against the real session and confirm
  recovery.
- Unchanged by design: every study's `sweep()`, every existing test file's
  `APLBackend()` construction site, `run_harness.sh`,
  `tests/backends_ext/apl/test_harness.py`.

## Out of scope (explicitly, per the issue)

- `#138` (`RunResult.timing` instrumentation).
- Moving `scipy` to an isolated subprocess.
- Julia backend (`#38`-`#40`) - mentioned only as "this pattern generalises."

## Verification

1. New tests for the session machinery: reuse across multiple `APLBackend()`
   instances (one process, one subprocess spawned), timeout triggers
   kill-and-restart and returns the existing timeout-result shape, a
   killed/dead session is transparently replaced on the next call,
   `evaluate_problem` and `solve` both route through it.
2. Rewritten `test_apl_backend_timeout.py`.
3. Full existing APL-touching test suite (all 18 files, ~90 functions) run
   unchanged and green - the real regression gate, since a shared, long-lived
   interpreter session raises a genuine question of whether any request
   leaves global-namespace state that leaks into a later, unrelated request;
   `solve.dyalog`'s own functions all operate on request-local variables
   today, which should make this safe, confirmed by running the whole suite.
4. Quantify the actual speedup: full suite wall-clock time before vs. after,
   and a `trust-bench report --backends trust-apl scipy` run's wall-clock
   time before vs. after.
5. `ruff check`, then the project's standard RED/GREEN cycle.
