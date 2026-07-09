"""A minimal Backend implementation used only to exercise the contract
tests in test_contract.py. Not a real solver: plain gradient descent at a
fixed step size, one method, no bounds, no analytic Hessian use. Real
backends (SciPy, issue #16/#17) are added to BACKENDS alongside this one,
not in place of it: this backend proves the contract itself is testable
before any real backend exists.
"""
import numpy as np

from trust_bench.core.backend import Backend, Capabilities, MethodCapabilities
from trust_bench.core.provenance import capture
from trust_bench.core.result import RunResult, RunStatus

_STEP = 0.3
_GTOL = 1e-6


class ToyGradientDescentBackend(Backend):
    name = "toy-gd"

    def capabilities(self):
        return Capabilities(
            methods={
                "gd": MethodCapabilities(
                    kind="residuals",
                    losses=frozenset({"l2"}),
                    bounds=False,
                    analytic_hessian=False,
                    derivative_modes=frozenset({"analytic"}),
                )
            }
        )

    def environment(self):
        return capture()

    def solve(self, problem, method, start, config):
        if method not in self.capabilities().methods:
            raise ValueError(f"{self.name} has no method {method!r}")

        max_iter = config["max_iter"]
        x = np.array(problem.starts[start], dtype=float)
        n_feval = n_jeval = 0
        status = RunStatus.MAX_ITER
        n_iter = 0
        for n_iter in range(1, max_iter + 1):
            r = np.asarray(problem.residual(x), dtype=float)
            n_feval += 1
            j = np.asarray(problem.jacobian(x), dtype=float)
            n_jeval += 1
            grad = j.T @ r
            if np.linalg.norm(grad) < _GTOL:
                status = RunStatus.CONVERGED
                break
            x = x - _STEP * grad

        r_final = np.asarray(problem.residual(x), dtype=float)
        cost_final = 0.5 * float(r_final @ r_final)

        return RunResult(
            problem_id=problem.id,
            backend=self.name,
            method=method,
            start=start,
            x_final=x.tolist(),
            cost_final=cost_final,
            dist_to_opt=None,
            cost_gap=None,
            grad_norm_final=float(np.linalg.norm(j.T @ r_final)),
            status=status,
            n_iter=n_iter,
            n_feval=n_feval,
            n_jeval=n_jeval,
            n_heval=0,
            trace=None,
            timing=None,
            config=config,
            provenance=self.environment(),
            harness_git_sha="test",
            timestamp="2026-01-01T00:00:00Z",
        )


BACKENDS = [ToyGradientDescentBackend()]
