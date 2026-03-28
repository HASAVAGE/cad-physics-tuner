"""
Auto-calibration loop.

This is the core value of cad-physics-tuner (~70–80 % of unique value).

The calibrator:
1. Accepts a *calibration dataset* (recorded real-robot trajectories).
2. Wraps a *simulator* callable that maps physics parameters → simulated
   joint trajectories.
3. Optimises the parameters so that the simulation matches reality using
   ``scipy.optimize`` (``minimize`` with ``L-BFGS-B`` / ``differential_evolution``).
4. Returns the optimised parameter vector and a short report dict.

When Isaac Sim is available the simulator callable is wired up to a real
physics step loop.  Without Isaac Sim the user can supply any callable that
maps parameters → trajectories (e.g. a mock, a MuJoCo model, etc.).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import differential_evolution, minimize  # type: ignore[import]

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False

from cad_physics_tuner.metrics import combined_objective

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset type
# ---------------------------------------------------------------------------

@dataclass
class CalibrationDataset:
    """Container for real-robot calibration data.

    Attributes
    ----------
    q_real:
        Joint positions array of shape ``(T, N)`` where *T* is the number of
        time steps and *N* is the number of joints.
    qd_real:
        Joint velocities, same shape.  Optional.
    f_real:
        Joint effort / contact forces, same shape.  Optional.
    dt:
        Time step in seconds (default 0.01 s = 100 Hz).
    """

    q_real: np.ndarray
    qd_real: Optional[np.ndarray] = None
    f_real: Optional[np.ndarray] = None
    dt: float = 0.01

    def __post_init__(self) -> None:
        self.q_real = np.asarray(self.q_real, dtype=float)
        if self.qd_real is not None:
            self.qd_real = np.asarray(self.qd_real, dtype=float)
        if self.f_real is not None:
            self.f_real = np.asarray(self.f_real, dtype=float)

    @classmethod
    def from_csv(cls, path: str | Path, dt: float = 0.01) -> "CalibrationDataset":
        """Load a dataset from a CSV file.

        Expected columns (in order): joint positions … [joint velocities …]
        [joint efforts …].  The number of columns is inferred automatically:
        - 1× N_joints  → positions only
        - 2× N_joints  → positions + velocities
        - 3× N_joints  → positions + velocities + efforts

        Parameters
        ----------
        path:
            Path to the CSV file.
        dt:
            Sampling period in seconds.
        """
        data = np.loadtxt(str(path), delimiter=",", dtype=float)
        if data.ndim == 1:
            data = data[:, np.newaxis]

        cols = data.shape[1]
        if cols % 3 == 0:
            n = cols // 3
            q, qd, f = data[:, :n], data[:, n : 2 * n], data[:, 2 * n :]
            return cls(q_real=q, qd_real=qd, f_real=f, dt=dt)
        elif cols % 2 == 0:
            n = cols // 2
            q, qd = data[:, :n], data[:, n:]
            return cls(q_real=q, qd_real=qd, dt=dt)
        else:
            return cls(q_real=data, dt=dt)


# ---------------------------------------------------------------------------
# Parameter descriptor
# ---------------------------------------------------------------------------

@dataclass
class PhysicsParam:
    """Description of a single tuneable physics parameter.

    Attributes
    ----------
    name:
        Human-readable name matching a key in ``physics_writer.PARAM_SCHEMA``.
    prim_path:
        USD prim path this parameter lives on.
    initial:
        Starting value for optimisation.
    bounds:
        ``(lower, upper)`` bounds for the optimiser.
    """

    name: str
    prim_path: str
    initial: float
    bounds: Tuple[float, float] = (0.0, 1e6)


# ---------------------------------------------------------------------------
# Calibration result
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Holds the outcome of a calibration run."""

    optimised_params: Dict[str, Dict[str, float]]  # {prim_path: {param_name: value}}
    initial_error: float
    final_error: float
    n_iterations: int
    elapsed_seconds: float
    success: bool
    message: str = ""
    error_history: List[float] = field(default_factory=list)

    @property
    def improvement_pct(self) -> float:
        """Percentage reduction in error (positive = improvement)."""
        if self.initial_error == 0:
            return 0.0
        return 100.0 * (self.initial_error - self.final_error) / self.initial_error

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["improvement_pct"] = self.improvement_pct
        return d

    def save_report(self, path: str | Path) -> Path:
        """Save the calibration report as a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Calibration report saved to %s", path)
        return path


# ---------------------------------------------------------------------------
# Simulator protocol
# ---------------------------------------------------------------------------

#: Type alias for a simulator callable.
#: ``simulator(params_vector, dataset) → (q_sim, qd_sim, f_sim)``
#: where each may be ``None`` if not simulated.
SimulatorFn = Callable[
    [np.ndarray, CalibrationDataset],
    Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]],
]


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class PhysicsCalibrator:
    """Auto-calibrate physics parameters against a real-robot dataset.

    Parameters
    ----------
    params:
        List of :class:`PhysicsParam` objects describing *which* parameters to
        tune, their starting values, and their search bounds.
    simulator:
        A callable ``(params_vector, dataset) → (q_sim, qd_sim, f_sim)`` that
        runs a short simulation episode with the supplied parameter values and
        returns simulated trajectories.
    method:
        Optimisation strategy:

        * ``"L-BFGS-B"`` – fast gradient-free local search (default)
        * ``"Nelder-Mead"`` – simplex local search, no bounds
        * ``"differential_evolution"`` – global stochastic search (slower)
        * ``"grid"`` – brute-force grid search (small parameter spaces only)

    metric_weights:
        Dict with optional keys ``w_tracking``, ``w_force``,
        ``w_oscillation`` to override the default metric weights.
    max_iterations:
        Maximum number of optimiser iterations / function evaluations.
    """

    def __init__(
        self,
        params: List[PhysicsParam],
        simulator: SimulatorFn,
        method: str = "L-BFGS-B",
        metric_weights: Optional[Dict[str, float]] = None,
        max_iterations: int = 200,
    ) -> None:
        if not params:
            raise ValueError("At least one PhysicsParam must be provided.")
        self.params = params
        self.simulator = simulator
        self.method = method.upper()
        self.metric_weights = metric_weights or {}
        self.max_iterations = max_iterations
        self._error_history: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(self, dataset: CalibrationDataset) -> CalibrationResult:
        """Run the calibration loop and return a :class:`CalibrationResult`.

        Parameters
        ----------
        dataset:
            Real-robot calibration data.

        Returns
        -------
        CalibrationResult
        """
        if not _SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for auto-calibration. "
                "Install it with: pip install scipy"
            )

        x0 = np.array([p.initial for p in self.params], dtype=float)
        bounds = [p.bounds for p in self.params]

        self._error_history = []
        t0 = time.monotonic()

        # Compute baseline error
        initial_error = self._objective(x0, dataset)

        if self.method == "DIFFERENTIAL_EVOLUTION":
            result = differential_evolution(
                self._objective,
                bounds=bounds,
                args=(dataset,),
                maxiter=self.max_iterations,
                tol=1e-6,
                seed=42,
            )
        elif self.method == "GRID":
            result = self._grid_search(x0, bounds, dataset)
        else:
            # Map stored uppercase names to the scipy method strings
            _METHOD_MAP = {
                "L-BFGS-B": "L-BFGS-B",
                "NELDER-MEAD": "Nelder-Mead",
                "COBYLA": "COBYLA",
            }
            scipy_method = _METHOD_MAP.get(self.method, self.method)
            opt_bounds = bounds if scipy_method not in ("Nelder-Mead", "COBYLA") else None
            # Nelder-Mead uses 'xatol'/'fatol'; other methods use 'ftol'
            if scipy_method == "Nelder-Mead":
                options = {"maxiter": self.max_iterations, "xatol": 1e-9, "fatol": 1e-9}
            else:
                options = {"maxiter": self.max_iterations, "ftol": 1e-9}
            result = minimize(
                self._objective,
                x0,
                args=(dataset,),
                method=scipy_method,
                bounds=opt_bounds,
                options=options,
            )

        elapsed = time.monotonic() - t0

        optimised_params = self._vector_to_params(result.x)

        return CalibrationResult(
            optimised_params=optimised_params,
            initial_error=initial_error,
            final_error=float(result.fun),
            n_iterations=getattr(result, "nit", len(self._error_history)),
            elapsed_seconds=elapsed,
            success=bool(result.success),
            message=getattr(result, "message", ""),
            error_history=list(self._error_history),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _objective(self, x: np.ndarray, dataset: CalibrationDataset) -> float:
        """Evaluate the combined objective for parameter vector *x*."""
        q_sim, qd_sim, f_sim = self.simulator(x, dataset)
        error = combined_objective(
            q_sim=q_sim,
            q_real=dataset.q_real,
            f_sim=f_sim,
            f_real=dataset.f_real,
            qd_sim=qd_sim,
            qd_real=dataset.qd_real,
            dt=dataset.dt,
            **self.metric_weights,
        )
        self._error_history.append(float(error))
        return float(error)

    def _vector_to_params(
        self, x: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Convert a flat parameter vector back to a nested dict."""
        result: Dict[str, Dict[str, float]] = {}
        for param, value in zip(self.params, x):
            result.setdefault(param.prim_path, {})[param.name] = float(value)
        return result

    def _grid_search(
        self,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        dataset: CalibrationDataset,
        n_points: int = 10,
    ) -> Any:
        """Minimal grid search for very small parameter spaces (≤3 params)."""
        if len(bounds) > 3:
            logger.warning(
                "Grid search with >3 parameters is expensive; consider "
                "switching to 'differential_evolution'."
            )

        grids = [np.linspace(lo, hi, n_points) for lo, hi in bounds]
        best_x = x0.copy()
        best_val = float("inf")

        import itertools

        for values in itertools.product(*[g.tolist() for g in grids]):
            x = np.array(values, dtype=float)
            val = self._objective(x, dataset)
            if val < best_val:
                best_val = val
                best_x = x.copy()

        class _FakeResult:
            fun = best_val
            x = best_x
            success = True
            message = "grid search complete"
            nit = len(self._error_history)

        return _FakeResult()
