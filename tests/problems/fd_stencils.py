import numpy as np


def fd_jacobian(residual, x, h=1e-4):
    """High-order (5-point, O(h^4)) central-difference Jacobian of a
    vector-valued residual function at x."""
    x = np.asarray(x, dtype=float)
    r0 = np.asarray(residual(x), dtype=float)
    J = np.zeros((len(r0), len(x)))
    for j in range(len(x)):
        e = np.zeros(len(x))
        e[j] = 1.0
        rp2 = np.asarray(residual(x + 2 * h * e), dtype=float)
        rp1 = np.asarray(residual(x + h * e), dtype=float)
        rm1 = np.asarray(residual(x - h * e), dtype=float)
        rm2 = np.asarray(residual(x - 2 * h * e), dtype=float)
        J[:, j] = (-rp2 + 8 * rp1 - 8 * rm1 + rm2) / (12 * h)
    return J


def fd_hessian(objective, x, h=1e-3):
    """Central-difference Hessian of a scalar-valued objective at x."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n))
    f0 = objective(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        H[i, i] = (objective(x + h * ei) - 2 * f0 + objective(x - h * ei)) / h**2
    for i in range(n):
        for j in range(i + 1, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i] = ej[j] = 1.0
            hij = (
                objective(x + h * ei + h * ej)
                - objective(x + h * ei - h * ej)
                - objective(x - h * ei + h * ej)
                + objective(x - h * ei - h * ej)
            ) / (4 * h**2)
            H[i, j] = H[j, i] = hij
    return H
