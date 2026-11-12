from __future__ import annotations

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve

from .metrics import error_norms
from .problems import EllipticProblem, ParabolicProblem


def solve_poisson_constant(problem: EllipticProblem) -> np.ndarray:
    """Solve -Delta u=f with Dirichlet boundary values on the same grid."""
    n = problem.n
    h2 = problem.h ** 2
    m = n - 2
    if m <= 0:
        raise ValueError("Grid must have at least three points")
    main = 4.0 * np.ones(m)
    off = -1.0 * np.ones(m - 1)
    T = diags([off, main, off], [-1, 0, 1], shape=(m, m), format="csr")
    I = eye(m, format="csr")
    A = kron(I, T, format="csr") + kron(diags([-1.0 * np.ones(m - 1), -1.0 * np.ones(m - 1)], [-1, 1], shape=(m, m), format="csr"), I, format="csr")
    rhs = (h2 * problem.f_source[1:-1, 1:-1]).astype(np.float64).copy()
    u = problem.u_exact.astype(np.float64).copy()
    rhs[:, 0] += u[1:-1, 0]
    rhs[:, -1] += u[1:-1, -1]
    rhs[0, :] += u[0, 1:-1]
    rhs[-1, :] += u[-1, 1:-1]
    sol = spsolve(A, rhs.ravel()).reshape(m, m)
    out = u.copy()
    out[1:-1, 1:-1] = sol
    return out.astype(np.float32)


def solve_backward_euler(problem: ParabolicProblem, final_time: float, tau: float = 0.1) -> np.ndarray:
    """Classical backward Euler finite-difference reference for u_t - Delta u = f."""
    n = problem.n
    m = n - 2
    h2 = problem.h ** 2
    n_steps = int(round(final_time / tau))
    if abs(n_steps * tau - final_time) > 1.0e-8:
        raise ValueError("final_time must be an integer multiple of tau")
    main = (1.0 + 4.0 * tau / h2) * np.ones(m)
    off = (-tau / h2) * np.ones(m - 1)
    T = diags([off, main, off], [-1, 0, 1], shape=(m, m), format="csr")
    I = eye(m, format="csr")
    A = kron(I, T, format="csr") + kron(diags([off, off], [-1, 1], shape=(m, m), format="csr"), I, format="csr")
    u_prev = problem.u0().astype(np.float64)
    for step in range(1, n_steps + 1):
        t = step * tau
        f = problem.source(t).astype(np.float64)
        b = problem.boundary(t)
        u_boundary = problem.exact(t).astype(np.float64)
        rhs = u_prev[1:-1, 1:-1] + tau * f[1:-1, 1:-1]
        rhs[:, 0] += (tau / h2) * u_boundary[1:-1, 0]
        rhs[:, -1] += (tau / h2) * u_boundary[1:-1, -1]
        rhs[0, :] += (tau / h2) * u_boundary[0, 1:-1]
        rhs[-1, :] += (tau / h2) * u_boundary[-1, 1:-1]
        sol = spsolve(A, rhs.ravel()).reshape(m, m)
        u_next = u_boundary.copy()
        u_next[1:-1, 1:-1] = sol
        u_prev = u_next
    return u_prev.astype(np.float32)


def elliptic_fd_metrics(problem: EllipticProblem) -> dict[str, float]:
    pred = solve_poisson_constant(problem)
    return error_norms(problem.u_exact, pred, problem.h)


def parabolic_fd_metrics(problem: ParabolicProblem, final_time: float, tau: float = 0.1) -> dict[str, float]:
    pred = solve_backward_euler(problem, final_time, tau)
    return error_norms(problem.exact(final_time), pred, problem.h)
