from trust_bench.backends import BACKENDS
from trust_bench.core.result import RunStatus
from trust_bench.studies.ill_conditioning import KAPPAS, METHODS, sweep


def test_success_and_precision_are_recorded_for_every_kappa_and_method():
    results = sweep()

    assert len(results) == len(KAPPAS) * len(METHODS) * len(BACKENDS)
    for (kappa, method, backend_name), result in results.items():
        assert result.status is not None, f"kappa={kappa} method={method} on {backend_name}"
        assert result.dist_to_opt is not None, f"kappa={kappa} method={method} on {backend_name}"


def test_all_three_method_families_converge_precisely_at_moderate_conditioning():
    results = sweep(kappas=[100.0])

    for (kappa, method, backend_name), result in results.items():
        assert result.status is RunStatus.CONVERGED, f"{method} on {backend_name} at kappa={kappa}"
        assert result.dist_to_opt < 1e-4, f"{method} on {backend_name} at kappa={kappa}"


def test_trust_exact_precision_degrades_monotonically_with_conditioning():
    # "exact trust-region" (Section 9 item 3): a direct, exact subproblem
    # solve, so conditioning shows up as a precision loss rather than an
    # iteration-count increase.
    results = sweep(methods=["trust-exact"])
    backend_name = BACKENDS[0].name

    distances = [results[(kappa, "trust-exact", backend_name)].dist_to_opt for kappa in KAPPAS]

    assert all(a <= b for a, b in zip(distances, distances[1:]))


def test_newton_cg_and_bfgs_remain_successful_where_trust_exact_fails_at_extreme_conditioning():
    # Distinguishes exact trust-region (trust-exact) from CG/Krylov
    # (Newton-CG) and dense quasi-Newton (BFGS): at extreme conditioning,
    # trust-exact's exact subproblem solve loses too much precision to
    # report success, while the other two remain CONVERGED.
    kappa = 1e8
    results = sweep(kappas=[kappa])
    backend_name = BACKENDS[0].name

    assert results[(kappa, "trust-exact", backend_name)].status is RunStatus.FAILED
    assert results[(kappa, "Newton-CG", backend_name)].status is RunStatus.CONVERGED
    assert results[(kappa, "BFGS", backend_name)].status is RunStatus.CONVERGED
