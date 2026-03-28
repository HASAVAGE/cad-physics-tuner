"""
Pure-Python / NumPy metrics for measuring simulation-to-real gap.

All functions operate on plain NumPy arrays so they are usable both inside
Isaac Sim and in offline unit tests.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def compute_tracking_error(
    q_sim: np.ndarray,
    q_real: np.ndarray,
    qd_sim: Optional[np.ndarray] = None,
    qd_real: Optional[np.ndarray] = None,
    pos_weight: float = 1.0,
    vel_weight: float = 0.1,
) -> float:
    """Root-mean-square position (and optionally velocity) tracking error.

    Parameters
    ----------
    q_sim:
        Simulated joint positions, shape ``(T, N)`` or ``(N,)``.
    q_real:
        Recorded real joint positions, same shape as *q_sim*.
    qd_sim:
        Simulated joint velocities (optional), same shape as *q_sim*.
    qd_real:
        Recorded real joint velocities (optional), same shape as *q_sim*.
    pos_weight:
        Weighting factor for the position component.
    vel_weight:
        Weighting factor for the velocity component (used only when
        *qd_sim* and *qd_real* are provided).

    Returns
    -------
    float
        Scalar RMSE (lower is better).
    """
    q_sim = np.asarray(q_sim, dtype=float)
    q_real = np.asarray(q_real, dtype=float)

    if q_sim.shape != q_real.shape:
        raise ValueError(
            f"q_sim shape {q_sim.shape} does not match q_real shape {q_real.shape}"
        )

    pos_err = np.sqrt(np.mean((q_sim - q_real) ** 2))
    error = pos_weight * pos_err

    if qd_sim is not None and qd_real is not None:
        qd_sim = np.asarray(qd_sim, dtype=float)
        qd_real = np.asarray(qd_real, dtype=float)
        if qd_sim.shape != qd_real.shape:
            raise ValueError(
                f"qd_sim shape {qd_sim.shape} does not match qd_real shape {qd_real.shape}"
            )
        vel_err = np.sqrt(np.mean((qd_sim - qd_real) ** 2))
        error += vel_weight * vel_err

    return float(error)


def compute_force_residuals(
    f_sim: np.ndarray,
    f_real: np.ndarray,
) -> float:
    """Mean absolute force/torque residual.

    Parameters
    ----------
    f_sim:
        Simulated joint effort / contact forces, shape ``(T, N)`` or ``(N,)``.
    f_real:
        Measured real joint effort / contact forces, same shape.

    Returns
    -------
    float
        Mean absolute error across all time steps and joints (lower is better).
    """
    f_sim = np.asarray(f_sim, dtype=float)
    f_real = np.asarray(f_real, dtype=float)

    if f_sim.shape != f_real.shape:
        raise ValueError(
            f"f_sim shape {f_sim.shape} does not match f_real shape {f_real.shape}"
        )

    return float(np.mean(np.abs(f_sim - f_real)))


def compute_oscillation_metric(q: np.ndarray, dt: float = 0.01) -> float:
    """Estimate oscillation energy as mean squared second derivative of joint positions.

    Parameters
    ----------
    q:
        Joint positions over time, shape ``(T, N)`` or ``(T,)``.
    dt:
        Time step in seconds.

    Returns
    -------
    float
        Oscillation energy (lower is better / less oscillatory).
    """
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[:, np.newaxis]

    if q.shape[0] < 3:
        return 0.0

    acc = np.diff(q, n=2, axis=0) / (dt ** 2)
    return float(np.mean(acc ** 2))


def combined_objective(
    q_sim: np.ndarray,
    q_real: np.ndarray,
    f_sim: Optional[np.ndarray] = None,
    f_real: Optional[np.ndarray] = None,
    qd_sim: Optional[np.ndarray] = None,
    qd_real: Optional[np.ndarray] = None,
    dt: float = 0.01,
    w_tracking: float = 1.0,
    w_force: float = 0.5,
    w_oscillation: float = 0.1,
) -> float:
    """Weighted combination of all metrics into a single scalar objective.

    This is the function that the optimiser minimises.
    """
    obj = w_tracking * compute_tracking_error(
        q_sim, q_real, qd_sim=qd_sim, qd_real=qd_real
    )

    if f_sim is not None and f_real is not None:
        obj += w_force * compute_force_residuals(f_sim, f_real)

    obj += w_oscillation * compute_oscillation_metric(q_sim, dt=dt)

    return obj
