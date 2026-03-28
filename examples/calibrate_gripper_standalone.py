"""
Standalone calibration example for Robotiq gripper using synthetic data.

This script runs on Python 3.8+ and demonstrates the calibration pipeline
without requiring package installation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add cad_physics_tuner to path to import modules
cad_physics_tuner_path = Path(__file__).parent.parent
sys.path.insert(0, str(cad_physics_tuner_path))

import csv
import json
import logging

import numpy as np

# Import calibrator modules
from cad_physics_tuner.calibrator import CalibrationDataset, PhysicsCalibrator, PhysicsParam

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
SYNTHETIC_DATA_DIR = HERE / "synthetic_data"
OUTPUT_DIR = HERE / "calibration_output"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_PATH = SYNTHETIC_DATA_DIR / "gripper_ramp_hold.csv"

# ---------------------------------------------------------------------------
# Physics Simulator
# ---------------------------------------------------------------------------


def simulate_finger_joint(x: np.ndarray, dataset) -> tuple:
    """
    Simulate spring-damper dynamics with given parameters.

    Args:
        x: Parameter vector [stiffness, damping, mass]
        dataset: CalibrationDataset containing real trajectories

    Returns:
        Tuple of (q_sim, qd_sim, f_sim) - simulated trajectories
    """
    stiffness = max(x[0], 100.0)  # Prevent negative/zero values
    damping = max(x[1], 0.001)
    mass = max(x[2], 0.001)

    dt = 0.01
    n_steps = len(dataset.q_real)
    
    # Simulated joint trajectories  
    q_sim = np.zeros_like(dataset.q_real, dtype=float)
    qd_sim = np.zeros_like(dataset.q_real, dtype=float)
    f_sim = np.zeros_like(dataset.q_real, dtype=float)

    # Use the real command profile as the desired trajectory
    command = dataset.q_real[:, 0]
    
    for i in range(1, n_steps):
        cmd = command[i - 1]
        error = cmd - q_sim[i - 1, 0]
        
        # Clamp force to prevent numerical overflow
        force = np.clip(stiffness * error - damping * qd_sim[i - 1, 0], -1e6, 1e6)
        acc = force / mass
        
        # Clamp acceleration
        acc = np.clip(acc, -1e6, 1e6)
        
        qd_sim[i, 0] = qd_sim[i - 1, 0] + acc * dt
        q_sim[i, 0] = q_sim[i - 1, 0] + qd_sim[i, 0] * dt
        f_sim[i, 0] = force

    return q_sim, qd_sim, f_sim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info("=" * 70)
    logger.info("Robotiq 2F-85 Gripper Physics Calibration (Synthetic Data)")
    logger.info("=" * 70)

    # Check dataset
    if not DATASET_PATH.exists():
        logger.error(f"Synthetic data not found at {DATASET_PATH}")
        logger.info("Run: python examples/generate_synthetic_gripper_data.py")
        return 1

    # Load dataset
    logger.info(f"\nLoading dataset: {DATASET_PATH}")
    dataset = CalibrationDataset.from_csv(DATASET_PATH)
    logger.info(f"  Loaded {len(dataset.q_real)} time steps")
    logger.info(f"  Positions shape: {dataset.q_real.shape}")
    logger.info(f"  Velocities shape: {dataset.qd_real.shape}")
    logger.info(f"  Forces shape: {dataset.f_real.shape}")

    # Define parameters to calibrate
    logger.info("\nDefining calibration parameters:")
    params = [
        PhysicsParam(
            name="stiffness",
            prim_path="/finger_joint",
            initial=6000.0,
            bounds=(1000.0, 10000.0),
        ),
        PhysicsParam(
            name="damping",
            prim_path="/finger_joint",
            initial=0.2,
            bounds=(0.01, 1.0),
        ),
        PhysicsParam(
            name="mass",
            prim_path="/finger_joint",
            initial=0.08,
            bounds=(0.01, 0.2),
        ),
    ]

    for p in params:
        logger.info(
            f"  - {p.name:12s}: {p.bounds} (initial={p.initial:.4f})"
        )

    # Run calibration
    logger.info("\nRunning calibration...")
    logger.info("Ground truth: stiffness=5000.0, damping=0.15, mass=0.05")

    calibrator = PhysicsCalibrator(
        params=params,
        simulator=simulate_finger_joint,
        method="L-BFGS-B",
        max_iterations=50,
    )

    result = calibrator.calibrate(dataset=dataset)

    # Report results
    logger.info("\n" + "=" * 70)
    logger.info("CALIBRATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Success: {result.success}")
    logger.info(f"Initial error: {result.initial_error:.6f}")
    logger.info(f"Final error:   {result.final_error:.6f}")
    logger.info(f"Iterations:    {result.n_iterations}")
    logger.info(f"Time elapsed:  {result.elapsed_seconds:.2f}s")

    logger.info("\nOptimized parameters:")
    optimized_params = [result.optimised_params.get(p.prim_path, {}).get(p.name) for p in params]
    for i, p in enumerate(params):
        optimized_val = optimized_params[i]
        if optimized_val is None:
            optimized_val = p.initial
        logger.info(f"  {p.name:12s}: {optimized_val:10.4f}")

    # Compare to ground truth
    ground_truth = {"stiffness": 5000.0, "damping": 0.15, "mass": 0.05}

    logger.info("\nComparison to ground truth:")
    for i, p in enumerate(params):
        optimized_val = optimized_params[i]
        if optimized_val is None:
            optimized_val = p.initial
        truth_val = ground_truth[p.name]
        error_pct = abs(optimized_val - truth_val) / truth_val * 100
        logger.info(
            f"  {p.name:12s}: "
            f"recovered={optimized_val:10.4f}, "
            f"truth={truth_val:10.4f}, "
            f"error={error_pct:6.2f}%"
        )

    # Save report
    report_path = OUTPUT_DIR / "gripper_calibration_report.json"
    report = {
        "dataset": str(DATASET_PATH),
        "method": "L-BFGS-B",
        "initial_error": result.initial_error,
        "final_error": result.final_error,
        "n_iterations": result.n_iterations,
        "elapsed_seconds": result.elapsed_seconds,
        "success": result.success,
        "optimized_params": {
            params[i].name: (optimized_params[i] if optimized_params[i] is not None else params[i].initial)
            for i in range(len(params))
        },
        "ground_truth": ground_truth,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved to: {report_path}")

    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
