"""
Example: Calibrate Robotiq gripper physics using synthetic data.

This script demonstrates the end-to-end calibration pipeline:
  1. Load synthetic gripper trajectory (with known ground-truth physics)
  2. Define parameters to calibrate
  3. Set up a physics simulator callback
  4. Run auto-calibration
  5. Check if recovered parameters match ground truth

Ground truth physics in synthetic data:
  - Stiffness: 5000.0 N/rad
  - Damping: 0.15 Ns/rad
  - Mass: 0.05 kg·m²
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from cad_physics_tuner.calibrator import CalibrationDataset, PhysicsCalibrator, PhysicsParam

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
SYNTHETIC_DATA_DIR = HERE / "synthetic_data"
OUTPUT_DIR = HERE / "calibration_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Use synthetic trajectory with known ground truth
DATASET_PATH = SYNTHETIC_DATA_DIR / "gripper_ramp_hold.csv"

# Robotiq URDF
ROBOTIQ_URDF = HERE.parent / "models" / "urdf" / "robotiq_2f_85.urdf"

# ---------------------------------------------------------------------------
# Physics Simulator
# ---------------------------------------------------------------------------


def simulate_finger_joint(params_dict: dict, trajectory_template: np.ndarray) -> np.ndarray:
    """
    Simple simulator: apply spring-damper dynamics with given params.

    This is a placeholder that demonstrates what a real simulator callback
    should do. In practice, you would:
      - Call Isaac Sim physics steps
      - Use MuJoCo simulate() 
      - Or integrate continuous-time dynamics

    Args:
        params_dict: Physics parameters (stiffness, damping, mass)
        trajectory_template: Reference trajectory to use as command

    Returns:
        Simulated trajectory (n_timesteps, 1) matching the command
    """
    stiffness = params_dict.get("stiffness", 5000.0)
    damping = params_dict.get("damping", 0.15)
    mass = params_dict.get("mass", 0.05)

    dt = 0.01
    pos = np.zeros_like(trajectory_template)
    vel = np.zeros_like(trajectory_template)

    for i in range(1, len(trajectory_template)):
        cmd = trajectory_template[i - 1]
        error = cmd - pos[i - 1]
        force = stiffness * error - damping * vel[i - 1]
        acc = force / mass
        vel[i] = vel[i - 1] + acc * dt
        pos[i] = pos[i - 1] + vel[i] * dt

    return pos.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info("=" * 70)
    logger.info("Robotiq 2F-85 Gripper Physics Calibration (Synthetic Data)")
    logger.info("=" * 70)

    # Check files exist
    if not DATASET_PATH.exists():
        logger.error(f"Synthetic data not found at {DATASET_PATH}")
        logger.info("Run: python examples/generate_synthetic_gripper_data.py")
        return

    if not ROBOTIQ_URDF.exists():
        logger.warning(f"URDF not found at {ROBOTIQ_URDF}")
        logger.info("Proceeding without URDF conversion (stub mode)")

    # -----------------------------------------------------------------------
    # Load calibration dataset
    # -----------------------------------------------------------------------
    logger.info(f"\nLoading dataset: {DATASET_PATH}")
    dataset = CalibrationDataset.from_csv(DATASET_PATH)
    logger.info(f"  Loaded {len(dataset.trajectories)} trajectories")
    logger.info(f"  Shape: {dataset.trajectories[0].shape}")

    # -----------------------------------------------------------------------
    # Define parameters to calibrate
    # -----------------------------------------------------------------------
    logger.info("\nDefining calibration parameters:")
    params = [
        PhysicsParam(
            name="stiffness",
            joint="finger_joint",
            min=1000.0,
            max=10000.0,
            initial=6000.0,  # Start away from ground truth
        ),
        PhysicsParam(
            name="damping",
            joint="finger_joint",
            min=0.01,
            max=1.0,
            initial=0.2,  # Start away from ground truth
        ),
        PhysicsParam(
            name="mass",
            joint="finger_joint",
            min=0.01,
            max=0.2,
            initial=0.08,  # Start away from ground truth
        ),
    ]

    for p in params:
        logger.info(
            f"  - {p.name:12s}: [{p.min:8.2f}, {p.max:8.2f}] "
            f"(initial={p.initial:.2f})"
        )

    # -----------------------------------------------------------------------
    # Run calibration
    # -----------------------------------------------------------------------
    logger.info("\nRunning calibration...")
    logger.info("Ground truth: stiffness=5000.0, damping=0.15, mass=0.05")

    calibrator = PhysicsCalibrator(
        dataset=dataset,
        params=params,
    )

    # Define simulator: returns simulated trajectory given parameter dict
    def simulator_fn(param_dict: dict) -> np.ndarray:
        # Use first trajectory's position as command profile
        command_profile = dataset.trajectories[0][:, 0]
        return simulate_finger_joint(param_dict, command_profile)

    result = calibrator.calibrate(
        simulator_fn=simulator_fn,
        method="lbfgsb",
        max_iterations=50,
    )

    # -----------------------------------------------------------------------
    # Report results
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("CALIBRATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Initial error: {result.initial_error:.6f}")
    logger.info(f"Final error:   {result.final_error:.6f}")
    logger.info(f"Improvement:   {result.improvement_pct:.2f}%")
    logger.info(f"Iterations:    {result.iteration_count}")
    logger.info(f"Time elapsed:  {result.elapsed_time:.2f}s")

    logger.info("\nOptimized parameters:")
    for i, p in enumerate(params):
        optimized_val = result.optimized_params[i]
        logger.info(f"  {p.name:12s}: {optimized_val:10.4f}")

    # -----------------------------------------------------------------------
    # Compare to ground truth
    # -----------------------------------------------------------------------
    ground_truth = {"stiffness": 5000.0, "damping": 0.15, "mass": 0.05}

    logger.info("\nComparison to ground truth:")
    for i, p in enumerate(params):
        optimized_val = result.optimized_params[i]
        truth_val = ground_truth[p.name]
        error_pct = abs(optimized_val - truth_val) / truth_val * 100
        logger.info(
            f"  {p.name:12s}: "
            f"recovered={optimized_val:10.4f}, "
            f"truth={truth_val:10.4f}, "
            f"error={error_pct:6.2f}%"
        )

    # -----------------------------------------------------------------------
    # Save report
    # -----------------------------------------------------------------------
    report_path = OUTPUT_DIR / "gripper_calibration_report.json"
    report = {
        "dataset": str(DATASET_PATH),
        "method": "lbfgsb",
        "initial_error": result.initial_error,
        "final_error": result.final_error,
        "improvement_pct": result.improvement_pct,
        "iteration_count": result.iteration_count,
        "elapsed_time": result.elapsed_time,
        "optimized_params": {
            params[i].name: float(result.optimized_params[i])
            for i in range(len(params))
        },
        "ground_truth": ground_truth,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved to: {report_path}")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
