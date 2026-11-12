from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

Array = np.ndarray


@dataclass
class EllipticProblem:
    name: str
    n: int
    x: Array
    y: Array
    X: Array
    Y: Array
    h: float
    u_exact: Array
    f_source: Array
    kappa: Optional[Array] = None

    @property
    def boundary(self) -> tuple[Array, Array, Array, Array]:
        return (
            self.u_exact[:, 0],
            self.u_exact[:, -1],
            self.u_exact[0, :],
            self.u_exact[-1, :],
        )


@dataclass
class ParabolicProblem:
    name: str
    n: int
    x: Array
    y: Array
    X: Array
    Y: Array
    h: float
    exact: Callable[[float], Array]
    source: Callable[[float], Array]
    boundary: Callable[[float], tuple[Array, Array, Array, Array]]

    def u0(self) -> Array:
        return self.exact(0.0)


def _unit_grid(n: int) -> tuple[Array, Array, Array, Array, float]:
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y = np.linspace(0.0, 1.0, n, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    h = float(1.0 / (n - 1))
    return x, y, X.astype(np.float32), Y.astype(np.float32), h


def _square_grid(n: int) -> tuple[Array, Array, Array, Array, float]:
    x = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    h = float(2.0 / (n - 1))
    return x, y, X.astype(np.float32), Y.astype(np.float32), h


def bubble_problem(n: int) -> EllipticProblem:
    x, y, X, Y, h = _unit_grid(n)
    u = X * (X - 1.0) * Y * (Y - 1.0)
    f = -(2.0 * Y * (Y - 1.0) + 2.0 * X * (X - 1.0))
    return EllipticProblem("Bubble", n, x, y, X, Y, h, u.astype(np.float32), f.astype(np.float32))


def peak_problem(n: int) -> EllipticProblem:
    x, y, X, Y, h = _unit_grid(n)
    A = X * (X - 1.0) * Y * (Y - 1.0)
    u = 5.0e-4 * (A ** 2) * np.exp(10.0 * X ** 2 + 10.0 * Y)

    # Manufactured source obtained from symbolic differentiation of -Delta u.
    f = -(0.0010 * np.exp(10.0 * X**2 + 10.0 * Y) * (
        200.0 * X**6 * Y**4 - 400.0 * X**6 * Y**3 + 200.0 * X**6 * Y**2
        - 400.0 * X**5 * Y**4 + 800.0 * X**5 * Y**3 - 400.0 * X**5 * Y**2
        + 340.0 * X**4 * Y**4 - 640.0 * X**4 * Y**3 + 286.0 * X**4 * Y**2
        + 14.0 * X**4 * Y + X**4 - 240.0 * X**3 * Y**4 + 400.0 * X**3 * Y**3
        - 132.0 * X**3 * Y**2 - 28.0 * X**3 * Y - 2.0 * X**3
        + 106.0 * X**2 * Y**4 - 172.0 * X**2 * Y**3 + 52.0 * X**2 * Y**2
        + 14.0 * X**2 * Y + X**2 - 6.0 * X * Y**4 + 12.0 * X * Y**3
        - 6.0 * X * Y**2 + Y**4 - 2.0 * Y**3 + Y**2
    ))
    return EllipticProblem("Peak", n, x, y, X, Y, h, u.astype(np.float32), f.astype(np.float32))


def exp_trig_problem(n: int) -> EllipticProblem:
    x, y, X, Y, h = _unit_grid(n)
    pi = np.pi
    E = np.exp(-X**2 - Y**2)
    Sx = np.sin(3.0 * pi * X)
    Sy = np.sin(3.0 * pi * Y)
    Cx = np.cos(3.0 * pi * X)
    Cy = np.cos(3.0 * pi * Y)
    u = E * Sx * Sy + X

    ux_part = E * Sy * ((4.0 * X**2 - 2.0 - 9.0 * pi**2) * Sx - 12.0 * pi * X * Cx)
    uy_part = E * Sx * ((4.0 * Y**2 - 2.0 - 9.0 * pi**2) * Sy - 12.0 * pi * Y * Cy)
    f = -(ux_part + uy_part)
    return EllipticProblem("Exp-Trig", n, x, y, X, Y, h, u.astype(np.float32), f.astype(np.float32))


def kellogg_problem(n: int) -> EllipticProblem:
    x, y, X, Y, h = _square_grid(n)
    beta = 0.53544095
    a = np.array([0.44721360, -0.74535599, -0.94411759, -2.40170264], dtype=np.float32)
    b = np.array([1.00000000, 2.33333333, 0.55555556, -0.48148148], dtype=np.float32)

    theta = np.mod(np.arctan2(Y, X), 2.0 * np.pi).astype(np.float32)
    r = np.sqrt(X**2 + Y**2).astype(np.float32)
    q = np.zeros_like(theta, dtype=np.int32)
    q[(theta >= np.pi / 2.0) & (theta < np.pi)] = 1
    q[(theta >= np.pi) & (theta < 3.0 * np.pi / 2.0)] = 2
    q[(theta >= 3.0 * np.pi / 2.0)] = 3

    u = (r ** beta) * (a[q] * np.sin(beta * theta) + b[q] * np.cos(beta * theta))
    u[r == 0.0] = 0.0

    kappa = np.ones_like(X, dtype=np.float32)
    kappa[((X >= 0.0) & (Y >= 0.0)) | ((X < 0.0) & (Y < 0.0))] = 5.0
    f = np.zeros_like(X, dtype=np.float32)
    return EllipticProblem("Non-Const-Diff", n, x, y, X, Y, h, u.astype(np.float32), f, kappa)


def get_elliptic_problem(name: str, n: int) -> EllipticProblem:
    key = name.lower().replace("_", "-")
    mapping = {
        "bubble": bubble_problem,
        "peak": peak_problem,
        "exp-trig": exp_trig_problem,
        "exptrig": exp_trig_problem,
        "non-const-diff": kellogg_problem,
        "nonconstant": kellogg_problem,
        "kellogg": kellogg_problem,
    }
    if key not in mapping:
        raise ValueError(f"Unknown elliptic problem: {name}")
    return mapping[key](n)


def trig_parabolic_problem(n: int, mode: int) -> ParabolicProblem:
    x, y, X, Y, h = _unit_grid(n)
    pi = np.pi
    spatial = np.sin(mode * pi * X) * np.sin(mode * pi * Y)

    def exact(t: float) -> Array:
        return (np.cos(t) * spatial).astype(np.float32)

    def source(t: float) -> Array:
        return ((-np.sin(t) + 2.0 * (mode * pi) ** 2 * np.cos(t)) * spatial).astype(np.float32)

    def boundary(t: float) -> tuple[Array, Array, Array, Array]:
        u = exact(t)
        return u[:, 0], u[:, -1], u[0, :], u[-1, :]

    return ParabolicProblem(f"Trig-n{mode}", n, x, y, X, Y, h, exact, source, boundary)


def gaussian_parabolic_problem(n: int) -> ParabolicProblem:
    x, y, X, Y, h = _unit_grid(n)
    q = (2.0 * X - 1.0) ** 2 + (2.0 * Y - 1.0) ** 2
    phi = np.exp(-50.0 * q)

    def exact(t: float) -> Array:
        return (np.cos(t) * phi).astype(np.float32)

    def source(t: float) -> Array:
        return (phi * (np.cos(t) * (800.0 - 40000.0 * q) - np.sin(t))).astype(np.float32)

    def boundary(t: float) -> tuple[Array, Array, Array, Array]:
        u = exact(t)
        return u[:, 0], u[:, -1], u[0, :], u[-1, :]

    return ParabolicProblem("Gaussian", n, x, y, X, Y, h, exact, source, boundary)


def get_parabolic_problem(name: str, n: int) -> ParabolicProblem:
    key = name.lower().replace("_", "-")
    if key in {"trig-n1", "trig1", "trigonometric-n1"}:
        return trig_parabolic_problem(n, 1)
    if key in {"trig-n4", "trig4", "trigonometric-n4"}:
        return trig_parabolic_problem(n, 4)
    if key == "gaussian":
        return gaussian_parabolic_problem(n)
    raise ValueError(f"Unknown parabolic problem: {name}")


ELLIPTIC_CASES = ["Bubble", "Peak", "Exp-Trig", "Non-Const-Diff"]
ELLIPTIC_CONSTANT_CASES = ["Bubble", "Peak", "Exp-Trig"]
PARABOLIC_CASES = ["Trig-n1", "Trig-n4", "Gaussian"]
