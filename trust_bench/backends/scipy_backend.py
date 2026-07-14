from datetime import datetime, timezone

import numpy as np
from scipy.optimize import BFGS as BFGSHessianUpdate
from scipy.optimize import least_squares, minimize

from trust_bench.core.backend import Backend, Capabilities, MethodCapabilities
from trust_bench.core.config import RunConfig
from trust_bench.core.provenance import capture, harness_git_sha
from trust_bench.core.result import RunResult, RunStatus

_LEAST_SQUARES_METHODS = frozenset({"lm", "trf", "dogbox"})
_MINIMIZE_USES_HESSIAN = frozenset({"Newton-CG", "trust-exact", "trust-constr", "trust-krylov"})

_LEAST_SQUARES_LOSSES = frozenset({"linear", "soft_l1", "huber", "cauchy", "arctan"})
_BOTH_DERIVATIVE_MODES = frozenset({"analytic", "finite-difference"})
_ANALYTIC_ONLY = frozenset({"analytic"})

# Verified via scipy.optimize.show_options(solver="minimize", method=...):
# the subset of {ftol, xtol, gtol} each method's `options` dict accepts. A
# single intent-level tolerance is applied to every native parameter a
# method exposes, rather than picking one and leaving the others at
# scipy's own default.
_MINIMIZE_TOLERANCE_PARAMS = {
    "BFGS": frozenset({"gtol"}),
    "L-BFGS-B": frozenset({"ftol", "gtol"}),
    "Newton-CG": frozenset({"xtol"}),
    "trust-exact": frozenset({"gtol"}),
    "trust-constr": frozenset({"gtol", "xtol"}),
    "trust-krylov": frozenset(),
}

_LEAST_SQUARES_STATUS = {
    -1: RunStatus.ERROR,
    0: RunStatus.MAX_ITER,
}


def _map_least_squares_status(status: int) -> RunStatus:
    if status > 0:
        return RunStatus.CONVERGED
    return _LEAST_SQUARES_STATUS.get(status, RunStatus.FAILED)


def _map_minimize_status(result, max_iter) -> RunStatus:
    if result.success:
        return RunStatus.CONVERGED
    if max_iter is not None and result.nit >= max_iter:
        return RunStatus.MAX_ITER
    return RunStatus.FAILED


def _objective(problem):
    if problem.kind == "scalar":
        # residual() already is the scalar cost; no sum-of-squares
        # wrapping, unlike a "residuals"-kind problem below.
        def f(x):
            return float(problem.residual(x))

        return f

    def f(x):
        r = np.asarray(problem.residual(x), dtype=float)
        return 0.5 * float(r @ r)

    return f


def _gradient(problem):
    def grad(x):
        r = np.asarray(problem.residual(x), dtype=float)
        return problem.jacobian(x).T @ r

    return grad


def _final_gradient(result):
    # trust-constr is a general nonlinear-constrained solver: its .jac is
    # the constraint Jacobian, not the objective gradient, which it
    # reports as .grad instead. Every other method here uses .jac.
    grad = getattr(result, "grad", None)
    if grad is not None and len(grad) > 0:
        return np.asarray(grad, dtype=float)
    return np.asarray(result.jac, dtype=float)


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
                    derivative_modes=_BOTH_DERIVATIVE_MODES,
                ),
                "trf": MethodCapabilities(
                    kind="residuals",
                    losses=_LEAST_SQUARES_LOSSES,
                    bounds=True,
                    analytic_hessian=False,
                    derivative_modes=_BOTH_DERIVATIVE_MODES,
                ),
                "dogbox": MethodCapabilities(
                    kind="residuals",
                    losses=_LEAST_SQUARES_LOSSES,
                    bounds=True,
                    analytic_hessian=False,
                    derivative_modes=_BOTH_DERIVATIVE_MODES,
                ),
                "BFGS": MethodCapabilities(
                    kind="residuals",
                    losses=frozenset(),
                    bounds=False,
                    analytic_hessian=False,
                    derivative_modes=_BOTH_DERIVATIVE_MODES,
                ),
                "L-BFGS-B": MethodCapabilities(
                    kind="residuals",
                    losses=frozenset(),
                    bounds=True,
                    analytic_hessian=False,
                    derivative_modes=_BOTH_DERIVATIVE_MODES,
                ),
                "Newton-CG": MethodCapabilities(
                    kind="residuals",
                    losses=frozenset(),
                    bounds=False,
                    analytic_hessian=True,
                    derivative_modes=_ANALYTIC_ONLY,
                ),
                "trust-exact": MethodCapabilities(
                    kind="residuals",
                    losses=frozenset(),
                    bounds=False,
                    analytic_hessian=True,
                    derivative_modes=_ANALYTIC_ONLY,
                ),
                "trust-constr": MethodCapabilities(
                    kind="residuals",
                    losses=frozenset(),
                    bounds=True,
                    analytic_hessian=True,
                    derivative_modes=_BOTH_DERIVATIVE_MODES,
                ),
                "trust-krylov": MethodCapabilities(
                    kind="residuals",
                    losses=frozenset(),
                    bounds=False,
                    analytic_hessian=True,
                    derivative_modes=_ANALYTIC_ONLY,
                ),
            }
        )

    def environment(self):
        return capture()

    def solve(self, problem, method: str, start: str, config: RunConfig) -> RunResult:
        if method not in self.capabilities().methods:
            raise ValueError(f"{self.name} has no method {method!r}")

        if method in _LEAST_SQUARES_METHODS:
            if problem.kind == "scalar":
                # lm/trf/dogbox need a genuine residual vector (they
                # minimise sum-of-squares internally); a kind="scalar"
                # problem's residual() is already the cost, not a
                # residual, so least_squares would silently drive it
                # toward zero rather than to its true minimum.
                raise ValueError(f"{method} requires a residual vector; {problem.id!r} is kind='scalar'")
            result, status, grad_final = self._solve_least_squares(problem, method, start, config)
        else:
            result, status, grad_final = self._solve_minimize(problem, method, start, config)

        x_final = np.asarray(result.x, dtype=float)
        optimum = problem.optima[0]
        dist_to_opt = float(np.linalg.norm(x_final - optimum.x_star))
        cost_final = float(result.cost) if hasattr(result, "cost") else float(result.fun)
        cost_gap = float(cost_final - optimum.cost_star)

        return RunResult(
            problem_id=problem.id,
            backend=self.name,
            method=method,
            start=start,
            x_final=x_final.tolist(),
            cost_final=cost_final,
            dist_to_opt=dist_to_opt,
            cost_gap=cost_gap,
            grad_norm_final=float(np.linalg.norm(grad_final)),
            status=status,
            n_iter=result.nit if hasattr(result, "nit") else result.nfev,
            n_feval=result.nfev,
            n_jeval=getattr(result, "njev", None),
            n_heval=getattr(result, "nhev", 0),
            trace=None,
            timing=None,
            config=config,
            provenance=self.environment(),
            harness_git_sha=harness_git_sha(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _solve_least_squares(self, problem, method, start, config):
        x0 = np.array(problem.starts[start], dtype=float)
        use_fd = config.derivative_mode == "finite-difference" or problem.jacobian is None
        jac = "2-point" if use_fd else problem.jacobian

        # result.nfev only counts calls scipy's outer algorithm makes
        # itself, not the extra residual calls its own finite-difference
        # Jacobian estimation makes internally; counting calls to the
        # function actually passed to least_squares captures both.
        call_count = 0

        def counted_residual(x):
            nonlocal call_count
            call_count += 1
            return problem.residual(x)

        kwargs = dict(fun=counted_residual, x0=x0, jac=jac, method=method, loss=config.loss)
        bounds = config.bounds
        if bounds is not None:
            kwargs["bounds"] = bounds
        max_iter = config.max_iter
        if max_iter is not None:
            kwargs["max_nfev"] = max_iter
        if config.tolerance is not None:
            kwargs["ftol"] = kwargs["xtol"] = kwargs["gtol"] = config.tolerance
        if config.x_scale is not None:
            kwargs["x_scale"] = config.x_scale
        if config.f_scale is not None:
            kwargs["f_scale"] = config.f_scale

        result = least_squares(**kwargs)
        result.nfev = call_count
        # result.fun is already the residual at result.x; recomputing it
        # via problem.residual would cost one more (uncounted) call.
        grad_final = result.jac.T @ np.asarray(result.fun, dtype=float)
        return result, _map_least_squares_status(result.status), grad_final

    def _solve_minimize(self, problem, method, start, config):
        caps = self.capabilities().methods[method]
        x0 = np.array(problem.starts[start], dtype=float)

        bounds = config.bounds
        if bounds is not None and not caps.bounds:
            raise ValueError(f"{method} does not support bounds")

        wants_fd = config.derivative_mode == "finite-difference"
        needs_fd_gradient = wants_fd or problem.jacobian is None
        needs_fd_hessian = method in _MINIMIZE_USES_HESSIAN and (wants_fd or problem.hessian is None)
        if (needs_fd_gradient or needs_fd_hessian) and "finite-difference" not in caps.derivative_modes:
            raise ValueError(f"{method} does not support finite-difference derivatives")
        jac = "2-point" if needs_fd_gradient else _gradient(problem)

        kwargs = dict(fun=_objective(problem), x0=x0, jac=jac, method=method)
        if bounds is not None:
            kwargs["bounds"] = bounds
        if method in _MINIMIZE_USES_HESSIAN:
            # A quasi-Newton Hessian update strategy, not the string
            # "2-point", both because scipy requires one whenever the
            # gradient is also FD (verified: otherwise raises "we require
            # the Hessian to be estimated using one of the quasi-Newton
            # strategies"), and because problem.hessian is None needs the
            # same deliberate fallback problem.jacobian is None already
            # gets above, rather than passing hess=None through to scipy
            # and relying on whatever undocumented default it falls back
            # to per method.
            kwargs["hess"] = BFGSHessianUpdate() if needs_fd_hessian else problem.hessian
        max_iter = config.max_iter
        options = {}
        if max_iter is not None:
            options["maxiter"] = max_iter
        if config.tolerance is not None:
            for param in _MINIMIZE_TOLERANCE_PARAMS[method]:
                options[param] = config.tolerance
        if options:
            kwargs["options"] = options

        result = minimize(**kwargs)
        return result, _map_minimize_status(result, max_iter), _final_gradient(result)
