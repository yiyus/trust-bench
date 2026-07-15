import numpy as np

from trust_bench.core.metrics import mad
from trust_bench.core.result import TimingStats

# docs/plans/trust-bench.md Section 7's timing policy: warm-up run(s)
# before measurement, N repetitions, median + MAD. Fixed rather than a
# new RunConfig field, since every solve() call now pays this cost.
WARMUP = 1
N_REPS = 5


def summarize(samples, warmup, n_reps, thread_count) -> TimingStats:
    """TimingStats from N measured repetition timings (seconds)."""
    return TimingStats(
        median=float(np.median(samples)),
        mad=mad(samples),
        n_reps=n_reps,
        warmup=warmup,
        thread_count=thread_count,
    )
