from __future__ import annotations

import numpy as np


def periodic_derivatives_1d(u, length=1.0):
    n = u.shape[-1]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    u_hat = np.fft.fft(u)
    ux = np.fft.ifft(1j * k * u_hat).real
    uxx = np.fft.ifft(-(k ** 2) * u_hat).real
    return ux, uxx


def _dealias(values):
    n = values.shape[-1]
    spectrum = np.fft.fft(values)
    cutoff = n // 3
    spectrum[..., cutoff:n - cutoff] = 0.0
    return np.fft.ifft(spectrum).real


def burgers_rhs(u, viscosity, length=1.0):
    ux, uxx = periodic_derivatives_1d(u, length)
    return -_dealias(u * ux) + viscosity * uxx


def solve_burgers_periodic(u0, viscosity, final_time, save_dt, cfl=0.15, length=1.0):
    """Fourier pseudospectral Burgers solver with RK4 and 2/3 de-aliasing."""
    u = np.asarray(u0, dtype=np.float64).copy()
    n = u.size
    dx = length / n
    times = [0.0]
    states = [u.copy()]
    next_save = save_dt
    t = 0.0
    while t < final_time - 1.0e-14:
        adv_dt = cfl * dx / max(np.max(np.abs(u)), 1.0e-6)
        diff_dt = 0.15 * dx * dx / max(viscosity, 1.0e-12)
        dt = min(adv_dt, diff_dt, next_save - t, final_time - t)
        k1 = burgers_rhs(u, viscosity, length)
        k2 = burgers_rhs(u + 0.5 * dt * k1, viscosity, length)
        k3 = burgers_rhs(u + 0.5 * dt * k2, viscosity, length)
        k4 = burgers_rhs(u + dt * k3, viscosity, length)
        u = u + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        t += dt
        if t >= next_save - 1.0e-12:
            times.append(float(next_save))
            states.append(u.copy())
            next_save += save_dt
    return np.asarray(times), np.asarray(states)


def solve_allen_cahn_periodic(u0, epsilon, final_time, save_dt, internal_dt=None, length=1.0):
    """Semi-implicit Fourier solver for u_t=eps^2 u_xx+u-u^3."""
    u = np.asarray(u0, dtype=np.float64).copy()
    n = u.size
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    internal_dt = min(save_dt / 10.0, 1.0e-3) if internal_dt is None else internal_dt
    times = [0.0]
    states = [u.copy()]
    next_save = save_dt
    t = 0.0
    while t < final_time - 1.0e-14:
        dt = min(internal_dt, next_save - t, final_time - t)
        nonlinear = u - u ** 3
        u_hat = (np.fft.fft(u) + dt * np.fft.fft(nonlinear)) / (
            1.0 + dt * epsilon * epsilon * k * k
        )
        u = np.fft.ifft(u_hat).real
        t += dt
        if t >= next_save - 1.0e-12:
            times.append(float(next_save))
            states.append(u.copy())
            next_save += save_dt
    return np.asarray(times), np.asarray(states)


def solve_heat_2d_periodic(u0, diffusivity, final_time, save_dt, length=1.0):
    """Exact-in-time Fourier solution of the periodic heat equation."""
    u0 = np.asarray(u0, dtype=np.float64)
    n, m = u0.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    ky = 2.0 * np.pi * np.fft.fftfreq(m, d=length / m)
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    initial_hat = np.fft.fft2(u0)
    times = np.arange(0.0, final_time + 0.5 * save_dt, save_dt)
    states = [np.fft.ifft2(initial_hat * np.exp(-diffusivity * k2 * t)).real for t in times]
    return times, np.asarray(states)


def laplacian_dirichlet(u, h):
    out = np.zeros_like(u, dtype=np.float64)
    out[1:-1, 1:-1] = (
        u[2:, 1:-1] + u[:-2, 1:-1]
        + u[1:-1, 2:] + u[1:-1, :-2]
        - 4.0 * u[1:-1, 1:-1]
    ) / (h * h)
    return out
