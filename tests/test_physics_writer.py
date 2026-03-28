"""Tests for cad_physics_tuner.physics_writer."""

from __future__ import annotations

import pytest

from cad_physics_tuner.physics_writer import PARAM_SCHEMA, PhysicsWriter


class TestPhysicsWriterDryRun:
    """All tests run in dry-run mode (no pxr required)."""

    def test_write_and_read_single_param(self):
        writer = PhysicsWriter()
        writer.write_param("/robot/joints/j1", "stiffness", 1500.0)
        assert writer.read_param("/robot/joints/j1", "stiffness") == pytest.approx(1500.0)

    def test_write_params_dict(self):
        writer = PhysicsWriter()
        writer.write_params("/robot/mat", {"static_friction": 0.8, "restitution": 0.1})
        assert writer.read_param("/robot/mat", "static_friction") == pytest.approx(0.8)
        assert writer.read_param("/robot/mat", "restitution") == pytest.approx(0.1)

    def test_read_missing_returns_none(self):
        writer = PhysicsWriter()
        assert writer.read_param("/nonexistent", "stiffness") is None

    def test_apply_optimised_params(self, tmp_path):
        writer = PhysicsWriter()
        params_map = {
            "/robot/joints/j1": {"stiffness": 2000.0, "damping": 50.0},
            "/robot/mat": {"static_friction": 0.7},
        }
        out_path = tmp_path / "robot_physics.usd"
        saved = writer.apply_optimised_params(params_map, output_path=out_path)
        assert saved == out_path
        assert writer.read_param("/robot/joints/j1", "stiffness") == pytest.approx(2000.0)
        assert writer.read_param("/robot/joints/j1", "damping") == pytest.approx(50.0)

    def test_save_requires_path_when_no_usd_path(self):
        writer = PhysicsWriter()
        with pytest.raises(ValueError, match="No output path"):
            writer.save()

    def test_save_writes_to_tmp(self, tmp_path):
        writer = PhysicsWriter()
        out = tmp_path / "test.usd"
        saved = writer.save(output_path=out)
        assert saved == out

    def test_dry_run_store_exposed(self):
        writer = PhysicsWriter()
        writer.write_param("/prim", "mass", 5.0)
        store = writer.dry_run_store
        assert "physics:mass" in store["/prim"]

    def test_all_param_schema_keys_writable(self):
        writer = PhysicsWriter()
        for key in PARAM_SCHEMA:
            writer.write_param("/test/prim", key, 1.0)
            val = writer.read_param("/test/prim", key)
            assert val == pytest.approx(1.0), f"Failed for param key: {key}"

    def test_raw_attribute_name(self):
        """Non-schema attribute names are written verbatim."""
        writer = PhysicsWriter()
        writer.write_param("/prim", "custom:attr", 42.0)
        assert writer.read_param("/prim", "custom:attr") == pytest.approx(42.0)
