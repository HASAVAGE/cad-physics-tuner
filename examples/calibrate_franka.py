"""
Example: Calibrate a Franka Panda arm physics model against recorded trajectories.

This script demonstrates the full cad-physics-tuner pipeline:
  1. Convert a URDF to USD  (stub when Isaac Sim is absent)
  2. Load a calibration dataset from a CSV
  3. Run the auto-calibration loop (scipy L-BFGS-B)
  4. Save the tuned USD + JSON report

Usage::

    python examples/calibrate_franka.py

The example generates synthetic "real" data from a known simulator and
measures how well the calibrator recovers the ground-truth parameters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from cad_physics_tuner.calibrator import CalibrationDataset, PhysicsParam
from cad_physics_tuner.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "_output"
OUTPUT_DIR.mkdir(exist_ok=True)

STUB_URDF = OUTPUT_DIR / "franka.urdf"
STUB_URDF.write_text(
    '<robot name="franka_panda">'
    "  <link name='panda_link0'/>"
    "  <joint name='panda_joint1' type='revolute'>"
    "    <parent link='panda_link0'/>"
    "    <child link='panda_link0'/>"
    "  </joint>"
    "</robot>",
    encoding="utf-8",
)

# ---------------------------------------------------------------------------
# Synthetic dataset
# Pretend "ground truth" stiffness=2000, damping=80 Ns/m
# Real trajectories are generated with these values.
# ---------------------------------------------------------------------------
GROUND_TRUTH_STIFFNESS = 2000.0
GROUND_TRUTH_DAMPING = 80.0
N_JOINTS = 7
N_STEPS = 100
DT = 0.01  # 100 Hz


def _simulate(stiffness: float, damping: float) -> np.ndarray:
    """Very simplified spring-damper response for N_JOINTS joints."""
    t = np.linspace(0, N_STEPS * DT, N_STEPS)
    # Simple 2nd-order system: overdamped response toward target
    target = np.ones(N_JOINTS) * 0.5
    q = np.zeros((N_STEPS, N_JOINTS))
    qd = np.zeros((N_STEPS, N_JOINTS))
    q[0] = 0.0
    for i in range(1, N_STEPS):
        acc = stiffness * (target - q[i - 1]) - damping * qd[i - 1]
        qd[i] = qd[i - 1] + acc * DT
        q[i] = q[i - 1] + qd[i] * DT
    return q


q_real = _simulate(GROUND_TRUTH_STIFFNESS, GROUND_TRUTH_DAMPING)
noise = np.random.default_rng(0).normal(scale=0.002, size=q_real.shape)
dataset = CalibrationDataset(q_real=q_real + noise, dt=DT)

# ---------------------------------------------------------------------------
# Simulator callable (used by the calibrator)
# ---------------------------------------------------------------------------


def simulator(x: np.ndarray, ds: CalibrationDataset):
    """Map [stiffness, damping] → simulated trajectory."""
    q_sim = _simulate(x[0], x[1])
    return q_sim, None, None


# ---------------------------------------------------------------------------
# Physics parameters to tune
# ---------------------------------------------------------------------------
params = [
    PhysicsParam(
        name="stiffness",
        prim_path="/robot/joints/panda_joint1",
        initial=500.0,           # start far from ground truth
        bounds=(100.0, 5000.0),
    ),
    PhysicsParam(
        name="damping",
        prim_path="/robot/joints/panda_joint1",
        initial=10.0,
        bounds=(1.0, 500.0),
    ),
]

# ---------------------------------------------------------------------------
# Run the full pipeline
# ---------------------------------------------------------------------------
orch = Orchestrator(input_path=STUB_URDF, output_dir=OUTPUT_DIR)

usd_path, report_path = orch.run(
    dataset=dataset,
    params=params,
    simulator_fn=simulator,
    method="L-BFGS-B",
    output_path=OUTPUT_DIR / "franka_physics.usd",
    report_path=OUTPUT_DIR / "calibration_report.json",
    max_iterations=200,
)

report = json.loads(report_path.read_text())
opt = report["optimised_params"]

print("\n" + "=" * 60)
print("  cad-physics-tuner – Franka Calibration Example")
print("=" * 60)
print(f"  Ground-truth stiffness : {GROUND_TRUTH_STIFFNESS}")
print(f"  Ground-truth damping   : {GROUND_TRUTH_DAMPING}")
opt_joint = opt.get("/robot/joints/panda_joint1", {})
print(f"  Optimised stiffness    : {opt_joint.get('stiffness', 'n/a'):.2f}")
print(f"  Optimised damping      : {opt_joint.get('damping', 'n/a'):.2f}")
print(f"  Initial error          : {report['initial_error']:.6f}")
print(f"  Final error            : {report['final_error']:.6f}")
print(f"  Improvement            : {report['improvement_pct']:.1f} %")
print(f"  Tuned USD              : {usd_path}")
print(f"  Report                 : {report_path}")
print("=" * 60)
