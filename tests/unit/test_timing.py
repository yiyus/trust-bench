from trust_bench.core.result import TimingStats
from trust_bench.core.timing import N_REPS, WARMUP, summarize


def test_summarize_reports_median_and_mad_of_the_given_samples():
    samples = [0.10, 0.11, 0.09, 0.30, 0.10]

    stats = summarize(samples, warmup=2, n_reps=5, thread_count=1)

    assert isinstance(stats, TimingStats)
    assert stats.median == 0.10
    assert stats.mad > 0.0
    assert stats.n_reps == 5
    assert stats.warmup == 2
    assert stats.thread_count == 1


def test_summarize_defaults_match_the_projects_own_timing_policy():
    # docs/plans/trust-bench.md Section 7: warm-up run(s) before
    # measurement, N repetitions - small defaults, not the more
    # expensive end of what would still be statistically defensible,
    # since every solve() call now pays this cost.
    assert WARMUP == 1
    assert N_REPS == 5
