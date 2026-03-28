"""Tests for cad_physics_tuner.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from cad_physics_tuner.metrics import (
    combined_objective,
    compute_force_residuals,
    compute_oscillation_metric,
    compute_tracking_error,
)


class TestComputeTrackingError:
    def test_zero_error(self):
        q = np.zeros((10, 3))
        assert compute_tracking_error(q, q) == pytest.approx(0.0)

    def test_constant_offset(self):
        q_real = np.ones((10, 3))
        q_sim = np.zeros((10, 3))
        err = compute_tracking_error(q_sim, q_real)
        assert err == pytest.approx(1.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            compute_tracking_error(np.zeros((5, 2)), np.zeros((5, 3)))

    def test_with_velocity(self):
        q = np.zeros((10, 2))
        qd_real = np.ones((10, 2))
        qd_sim = np.zeros((10, 2))
        err_no_vel = compute_tracking_error(q, q)
        err_with_vel = compute_tracking_error(q, q, qd_sim=qd_sim, qd_real=qd_real)
        assert err_with_vel > err_no_vel

    def test_velocity_shape_mismatch_raises(self):
        q = np.zeros((5, 2))
        with pytest.raises(ValueError, match="shape"):
            compute_tracking_error(q, q, qd_sim=np.zeros((5, 2)), qd_real=np.zeros((5, 3)))

    def test_1d_input(self):
        q = np.array([1.0, 2.0, 3.0])
        err = compute_tracking_error(q, q)
        assert err == pytest.approx(0.0)

    def test_weights(self):
        q_real = np.ones((10, 1))
        q_sim = np.zeros((10, 1))
        err_w1 = compute_tracking_error(q_sim, q_real, pos_weight=1.0)
        err_w2 = compute_tracking_error(q_sim, q_real, pos_weight=2.0)
        assert err_w2 == pytest.approx(err_w1 * 2.0)


class TestComputeForceResiduals:
    def test_zero_residual(self):
        f = np.random.rand(10, 4)
        assert compute_force_residuals(f, f) == pytest.approx(0.0)

    def test_known_residual(self):
        f_sim = np.zeros((5, 2))
        f_real = np.ones((5, 2))
        assert compute_force_residuals(f_sim, f_real) == pytest.approx(1.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            compute_force_residuals(np.zeros((3, 2)), np.zeros((3, 3)))


class TestComputeOscillationMetric:
    def test_constant_is_zero(self):
        q = np.ones((20, 3))
        assert compute_oscillation_metric(q) == pytest.approx(0.0)

    def test_short_series_returns_zero(self):
        assert compute_oscillation_metric(np.ones((2, 3))) == pytest.approx(0.0)

    def test_oscillatory_is_positive(self):
        t = np.linspace(0, 2 * np.pi, 100)
        q = np.sin(t)[:, np.newaxis]
        assert compute_oscillation_metric(q) > 0.0

    def test_1d_input(self):
        q = np.linspace(0, 1, 50)
        metric = compute_oscillation_metric(q)
        assert metric >= 0.0


class TestCombinedObjective:
    def test_perfect_match(self):
        q = np.zeros((10, 3))
        assert combined_objective(q, q) == pytest.approx(0.0)

    def test_with_forces(self):
        q = np.zeros((10, 3))
        f_real = np.ones((10, 3))
        f_sim = np.zeros((10, 3))
        obj_no_f = combined_objective(q, q)
        obj_with_f = combined_objective(q, q, f_sim=f_sim, f_real=f_real)
        assert obj_with_f > obj_no_f

    def test_returns_float(self):
        q = np.random.rand(20, 5)
        obj = combined_objective(q, q + 0.01)
        assert isinstance(obj, float)
