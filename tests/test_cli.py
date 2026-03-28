"""Tests for cad_physics_tuner CLI."""

from __future__ import annotations

import json

import numpy as np

from cad_physics_tuner.cli import main


class TestConvertCommand:
    def test_convert_produces_usd(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("<robot/>")
        rc = main(["convert", "--input", str(urdf), "--output-dir", str(tmp_path)])
        assert rc == 0
        assert (tmp_path / "robot.usd").exists()

    def test_convert_missing_input_returns_nonzero(self, tmp_path):
        rc = main(["convert", "--input", str(tmp_path / "missing.urdf")])
        assert rc != 0


class TestCalibrateCommand:
    def _write_csv(self, tmp_path, n_steps=30, n_joints=2):
        t = np.linspace(0, 1, n_steps)
        data = np.column_stack([np.sin(t), np.cos(t)])[:, :n_joints]
        csv_path = tmp_path / "traj.csv"
        np.savetxt(str(csv_path), data, delimiter=",")
        return csv_path

    def test_calibrate_produces_outputs(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("<robot/>")
        csv = self._write_csv(tmp_path)
        report = tmp_path / "report.json"
        usd_out = tmp_path / "tuned.usd"

        rc = main([
            "calibrate",
            "--input", str(urdf),
            "--dataset", str(csv),
            "--params", "stiffness:/robot/j1:1000:100:5000",
            "--output-dir", str(tmp_path),
            "--output", str(usd_out),
            "--report", str(report),
            "--max-iterations", "10",
        ])

        assert rc == 0
        assert report.exists()
        data = json.loads(report.read_text())
        assert "improvement_pct" in data

    def test_calibrate_bad_param_spec_returns_error(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("<robot/>")
        csv = self._write_csv(tmp_path)
        rc = main([
            "calibrate",
            "--input", str(urdf),
            "--dataset", str(csv),
            "--params", "bad-spec",
            "--output-dir", str(tmp_path),
        ])
        assert rc != 0
