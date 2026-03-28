"""Tests for cad_physics_tuner.orchestrator."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cad_physics_tuner.calibrator import CalibrationDataset, PhysicsParam
from cad_physics_tuner.orchestrator import Orchestrator


def _trivial_sim(x: np.ndarray, dataset: CalibrationDataset):
    """Simulator that scales q_real by first param / 1000."""
    scale = x[0] / 1000.0
    return dataset.q_real * scale, None, None


def _make_dataset():
    t = np.linspace(0, 1, 30)
    q = np.column_stack([np.sin(t), np.cos(t)])
    return CalibrationDataset(q_real=q, dt=0.033)


class TestOrchestrator:
    def test_convert_returns_result(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("<robot/>")
        orch = Orchestrator(input_path=urdf, output_dir=tmp_path)
        result = orch.convert()
        assert result.usd_path.exists()

    def test_calibrate_after_convert(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("<robot/>")
        orch = Orchestrator(input_path=urdf, output_dir=tmp_path)
        orch.convert()
        ds = _make_dataset()
        params = [PhysicsParam("stiffness", "/j1", initial=500.0, bounds=(100.0, 2000.0))]
        result = orch.calibrate(ds, params, simulator_fn=_trivial_sim, max_iterations=20)
        assert result.final_error <= result.initial_error

    def test_save_requires_calibrate_first(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("<robot/>")
        orch = Orchestrator(input_path=urdf, output_dir=tmp_path)
        orch.convert()
        with pytest.raises(RuntimeError, match="calibrate\\(\\) must be called"):
            orch.save()

    def test_full_run_pipeline(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("<robot/>")
        ds = _make_dataset()
        params = [PhysicsParam("stiffness", "/j1", initial=500.0, bounds=(100.0, 2000.0))]

        orch = Orchestrator(input_path=urdf, output_dir=tmp_path)
        usd_path, report_path = orch.run(
            dataset=ds,
            params=params,
            simulator_fn=_trivial_sim,
            output_path=tmp_path / "robot_physics.usd",
            report_path=tmp_path / "report.json",
            max_iterations=20,
        )

        assert usd_path == tmp_path / "robot_physics.usd"
        assert report_path == tmp_path / "report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert "improvement_pct" in data

    def test_calibrate_without_convert_logs_warning(self, tmp_path, caplog):
        import logging

        urdf = tmp_path / "robot.urdf"
        urdf.write_text("<robot/>")
        orch = Orchestrator(input_path=urdf, output_dir=tmp_path)
        ds = _make_dataset()
        params = [PhysicsParam("stiffness", "/j1", initial=500.0, bounds=(100.0, 2000.0))]

        with caplog.at_level(logging.WARNING):
            orch.calibrate(ds, params, simulator_fn=_trivial_sim, max_iterations=10)

        assert "convert()" in caplog.text

    def test_output_dir_created(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("<robot/>")
        new_dir = tmp_path / "new" / "out"
        Orchestrator(input_path=urdf, output_dir=new_dir)
        assert new_dir.exists()
