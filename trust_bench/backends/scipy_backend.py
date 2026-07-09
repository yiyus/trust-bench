from datetime import datetime, timezone

import numpy as np
from scipy.optimize import least_squares

from trust_bench.core.backend import Backend, Capabilities, MethodCapabilities
from trust_bench.core.provenance import capture, harness_git_sha
from trust_bench.core.result import RunResult, RunStatus

_LEAST_SQUARES_LOSSES = frozenset({"linear", "soft_l1", "huber", "cauchy", "arctan"})
_DERIVATIVE_MODES = frozenset({"analytic", "finite-difference"})

_LEAST_SQUARES_STATUS = {
    -1: RunStatus.ERROR,
    0: RunStatus.MAX_ITER,
}


def _map_least_squares_status(status: int) -> RunStatus:
    if status > 0:
        return RunStatus.CONVERGED
    return _LEAST_SQUARES_STATUS.get(status, RunStatus.FAILED)


class SciPyBackend(Backend):
    name = "scipy"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            methods={
                "lm": MethodCapabilities(
                    kind="residuals",
                    losses=frozenset({"linear"}),
                    bounds=False,
                    analytic_hessian=False,
                    derivative_modes=_DERIVATIVE_MODES,
                ),
                "trf": MethodCapabilities(
                    kind="residuals",
                    losses=_LEAST_SQUARES_LOSSES,
                    bounds=True,
                    analytic_hessian=False,
                    derivative_modes=_DERIVATIVE_MODES,
                ),
                "dogbox": MethodCapabilities(
                    kind="residuals",
                    losses=_LEAST_SQUARES_LOSSES,
                    bounds=True,
                    analytic_hessian=False,
                    derivative_modes=_DERIVATIVE_MODES,
                ),
            }
        )

    def environment(self):
        return capture()

    def solve(self, problem, method: str, start: str, config) -> RunResult:
        if method not in self.capabilities().methods:
            raise ValueError(f"{self.name} has no method {method!r}")

        x0 = np.array(problem.starts[start], dtype=float)
        use_fd = config.get("derivative_mode") == "finite-difference" or problem.jacobian is None
        jac = "2-point" if use_fd else problem.jacobian
        kwargs = dict(
            fun=problem.residual, x0=x0, jac=jac, method=method, loss=config.get("loss", "linear")
        )
        bounds = config.get("bounds")
        if bounds is not None:
            kwargs["bounds"] = bounds
        max_iter = config.get("max_iter")
        if max_iter is not None:
            kwargs["max_nfev"] = max_iter

        result = least_squares(**kwargs)

        x_final = result.x
        r_final = np.asarray(problem.residual(x_final), dtype=float)
        # result.jac is scipy's own Jacobian at x_final, analytic or FD
        # depending on what was requested above; using it here (rather than
        # a fresh problem.jacobian(x_final) call) keeps grad_norm_final
        # consistent with what solve() actually used, and avoids calling an
        # analytic Jacobian that a "finite-difference" request, or a
        # problem with no analytic Jacobian at all, must not depend on.
        grad_final = result.jac.T @ r_final

        optimum = problem.optima[0]
        dist_to_opt = float(np.linalg.norm(x_final - optimum.x_star))
        cost_gap = float(result.cost - optimum.cost_star)

        return RunResult(
            problem_id=problem.id,
            backend=self.name,
            method=method,
            start=start,
            x_final=x_final.tolist(),
            cost_final=float(result.cost),
            dist_to_opt=dist_to_opt,
            cost_gap=cost_gap,
            grad_norm_final=float(np.linalg.norm(grad_final)),
            status=_map_least_squares_status(result.status),
            # least_squares' OptimizeResult has no per-iteration count (no
            # nit-equivalent field, unlike minimize's), so n_iter here is
            # n_feval, not an independent count.
            n_iter=result.nfev,
            n_feval=result.nfev,
            n_jeval=result.njev,
            n_heval=0,
            trace=None,
            timing=None,
            config=config,
            provenance=self.environment(),
            harness_git_sha=harness_git_sha(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
