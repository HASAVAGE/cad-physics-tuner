"""
Orchestrator – thin glue layer.

Connects:
1. :class:`~cad_physics_tuner.converter.UrdfToUsdConverter`  (CAD → USD)
2. :class:`~cad_physics_tuner.calibrator.PhysicsCalibrator`  (calibration loop)
3. :class:`~cad_physics_tuner.physics_writer.PhysicsWriter`  (write back to USD)

Typical usage::

    from cad_physics_tuner import Orchestrator
    from cad_physics_tuner.calibrator import CalibrationDataset, PhysicsParam

    orch = Orchestrator(input_path="robot.urdf", output_dir="out/")
    orch.convert()
    dataset = CalibrationDataset.from_csv("real_robot_traj.csv")
    params = [PhysicsParam("stiffness", "/robot/joints/j1", initial=1000, bounds=(100, 1e5))]
    result = orch.calibrate(dataset, params, simulator_fn=my_sim)
    orch.save(output_path="out/robot_physics.usd", report_path="out/report.json")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cad_physics_tuner.calibrator import (
    CalibrationDataset,
    CalibrationResult,
    PhysicsCalibrator,
    PhysicsParam,
    SimulatorFn,
)
from cad_physics_tuner.converter import ConversionResult, UrdfToUsdConverter
from cad_physics_tuner.physics_writer import PhysicsWriter

logger = logging.getLogger(__name__)


class Orchestrator:
    """End-to-end pipeline: convert → calibrate → write → report.

    Parameters
    ----------
    input_path:
        Path to a URDF (or stub) file that will be converted to USD.
    output_dir:
        Directory for all outputs (converted USD, tuned USD, report).
        Defaults to the parent directory of *input_path*.
    converter_kwargs:
        Extra keyword arguments forwarded to
        :class:`~cad_physics_tuner.converter.UrdfToUsdConverter`.
    """

    def __init__(
        self,
        input_path: str | Path,
        output_dir: Optional[str | Path] = None,
        **converter_kwargs,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir) if output_dir else self.input_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._converter = UrdfToUsdConverter(
            output_dir=self.output_dir, **converter_kwargs
        )
        self._conversion_result: Optional[ConversionResult] = None
        self._calibration_result: Optional[CalibrationResult] = None
        self._writer: Optional[PhysicsWriter] = None

    # ------------------------------------------------------------------
    # Step 1: Convert
    # ------------------------------------------------------------------

    def convert(self) -> ConversionResult:
        """Convert the input CAD/URDF file to USD.

        Returns
        -------
        ConversionResult
            Contains the path to the generated ``.usd`` file.
        """
        logger.info("Converting %s …", self.input_path)
        self._conversion_result = self._converter.convert(self.input_path)
        self._writer = PhysicsWriter(self._conversion_result.usd_path)
        logger.info("Conversion done → %s", self._conversion_result.usd_path)
        return self._conversion_result

    # ------------------------------------------------------------------
    # Step 2: Calibrate
    # ------------------------------------------------------------------

    def calibrate(
        self,
        dataset: CalibrationDataset,
        params: List[PhysicsParam],
        simulator_fn: SimulatorFn,
        method: str = "L-BFGS-B",
        metric_weights: Optional[Dict[str, float]] = None,
        max_iterations: int = 200,
    ) -> CalibrationResult:
        """Run the auto-calibration loop.

        Parameters
        ----------
        dataset:
            Real-robot calibration data (see :class:`.CalibrationDataset`).
        params:
            Parameters to tune (see :class:`.PhysicsParam`).
        simulator_fn:
            Callable ``(params_vector, dataset) → (q_sim, qd_sim, f_sim)``.
        method:
            Optimisation strategy (``"L-BFGS-B"``, ``"differential_evolution"``,
            ``"grid"``, ``"Nelder-Mead"``).
        metric_weights:
            Optional override for metric weights.
        max_iterations:
            Budget for the optimiser.

        Returns
        -------
        CalibrationResult
        """
        if self._conversion_result is None:
            logger.warning(
                "convert() has not been called; calibrating without a USD stage."
            )

        calibrator = PhysicsCalibrator(
            params=params,
            simulator=simulator_fn,
            method=method,
            metric_weights=metric_weights,
            max_iterations=max_iterations,
        )

        logger.info("Starting calibration with method=%s …", method)
        self._calibration_result = calibrator.calibrate(dataset)
        logger.info(
            "Calibration done. Error %.4f → %.4f (%.1f %% improvement)",
            self._calibration_result.initial_error,
            self._calibration_result.final_error,
            self._calibration_result.improvement_pct,
        )
        return self._calibration_result

    # ------------------------------------------------------------------
    # Step 3: Save
    # ------------------------------------------------------------------

    def save(
        self,
        output_path: Optional[str | Path] = None,
        report_path: Optional[str | Path] = None,
    ) -> Tuple[Path, Optional[Path]]:
        """Write optimised parameters back to USD and (optionally) save report.

        Parameters
        ----------
        output_path:
            Where to write the tuned USD.  Defaults to
            ``<output_dir>/robot_physics.usd``.
        report_path:
            Where to write the JSON report.  Pass ``None`` to skip.

        Returns
        -------
        (usd_path, report_path)
        """
        if self._calibration_result is None:
            raise RuntimeError("calibrate() must be called before save().")

        if self._writer is None:
            self._writer = PhysicsWriter()  # dry-run

        usd_out = Path(output_path) if output_path else self.output_dir / "robot_physics.usd"
        saved_usd = self._writer.apply_optimised_params(
            self._calibration_result.optimised_params,
            output_path=usd_out,
        )

        saved_report: Optional[Path] = None
        if report_path is not None:
            saved_report = self._calibration_result.save_report(report_path)

        logger.info("Pipeline complete. USD → %s  Report → %s", saved_usd, saved_report)
        return saved_usd, saved_report

    # ------------------------------------------------------------------
    # Convenience: run the whole pipeline in one call
    # ------------------------------------------------------------------

    def run(
        self,
        dataset: CalibrationDataset,
        params: List[PhysicsParam],
        simulator_fn: SimulatorFn,
        method: str = "L-BFGS-B",
        output_path: Optional[str | Path] = None,
        report_path: Optional[str | Path] = None,
        metric_weights: Optional[Dict[str, float]] = None,
        max_iterations: int = 200,
    ) -> Tuple[Path, Optional[Path]]:
        """Run the full pipeline: convert → calibrate → save.

        Returns
        -------
        (usd_path, report_path)
        """
        self.convert()
        self.calibrate(
            dataset,
            params,
            simulator_fn,
            method=method,
            metric_weights=metric_weights,
            max_iterations=max_iterations,
        )
        return self.save(output_path=output_path, report_path=report_path)
