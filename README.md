# cad-physics-tuner

Auto-calibrate physics parameters for CAD robots, then write production-ready USD for NVIDIA Isaac Sim.

`cad-physics-tuner` closes the sim-to-real gap by optimising joint stiffness, damping, friction, mass, inertia, and contact offsets so that a simulated robot matches trajectories recorded on real hardware.  It wraps Isaac Lab's URDF converter and exposes a clean Python API and CLI that work even without a full Isaac Sim installation.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Command-line interface](#command-line-interface)
  - [Python API](#python-api)
- [Pipeline Overview](#pipeline-overview)
- [Dataset Format](#dataset-format)
- [Tunable Parameters](#tunable-parameters)
- [Optimisation Methods](#optimisation-methods)
- [Configuration Reference](#configuration-reference)
- [Examples](#examples)
- [Development](#development)
- [License](#license)

---

## Features

- **URDF → USD conversion** via Isaac Lab's `UrdfConverter` (graceful no-op stub when Isaac Sim is absent)
- **Physics auto-calibration** using `scipy.optimize` — RMSE position tracking error, velocity error, and force residuals are minimised simultaneously
- **Flexible simulator interface** — wire in Isaac Sim physics steps, MuJoCo, or any callable that maps parameter vectors to joint trajectories
- **Write-back to USD** — optimised values are persisted directly on the USD stage using `pxr.UsdPhysics` / `PhysxSchema`
- **JSON calibration report** with initial error, final error, improvement percentage, iteration count, and elapsed time
- **Zero Isaac Sim requirement** for unit tests and offline development — all core logic runs on NumPy + SciPy only

---

## Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.10 |
| NumPy | ≥ 1.24 |
| SciPy | ≥ 1.10 |
| Isaac Sim / Isaac Lab | Optional — enables real conversion and USD write-back |

---

## Installation

**Linux / macOS**

```bash
# From source (recommended during development)
git clone https://github.com/your-org/cad-physics-tuner.git
cd cad-physics-tuner
pip install -e .

# With dev extras (pytest, ruff, coverage)
pip install -e ".[dev]"
```

**Windows (PowerShell)**

```powershell
# From source
git clone https://github.com/your-org/cad-physics-tuner.git
cd cad-physics-tuner
pip install -e .

# With dev extras
pip install -e ".[dev]"
```

> **Isaac Sim note**: if `omni.isaac.lab` is on the Python path, the real `UrdfConverter` and USD stage writer are used automatically.  Without it, the package runs in *stub mode*: conversion produces a placeholder `.usd` file and USD writes are stored in memory only.

---

## Quick Start

### Command-line interface

**Convert a URDF to USD**

```bash
cad-physics-tuner convert --input robot.urdf --output-dir out/
# or
python -m cad_physics_tuner convert --input robot.urdf --output-dir out/
```

**Run the full calibration pipeline**

_Linux / macOS_

```bash
cad-physics-tuner calibrate \
    --input        robot.urdf \
    --dataset      real_traj.csv \
    --params       stiffness:/robot/joints/j1:1000:100:1e5 \
                   damping:/robot/joints/j1:50:1:500 \
    --method       L-BFGS-B \
    --max-iterations 300 \
    --output-dir   out/ \
    --report       out/report.json
```

_Windows (PowerShell)_

```powershell
cad-physics-tuner calibrate `
    --input        robot.urdf `
    --dataset      real_traj.csv `
    --params       stiffness:/robot/joints/j1:1000:100:1e5 `
                   damping:/robot/joints/j1:50:1:500 `
    --method       L-BFGS-B `
    --max-iterations 300 `
    --output-dir   out/ `
    --report       out/report.json
```

> **Tip:** PowerShell uses backticks (`` ` ``) for line continuation instead of backslashes.

Each `--params` entry follows the format `NAME:PRIM_PATH:INITIAL:LOWER:UPPER`, for example:

```
stiffness:/robot/joints/shoulder_pan:1000:100:1e5
```

**CLI flags**

| Flag | Default | Description |
|---|---|---|
| `--input` | — | Path to the input URDF file |
| `--dataset` | — | CSV file of real robot trajectories |
| `--params` | — | One or more `NAME:PRIM:INIT:LO:HI` specs |
| `--method` | `L-BFGS-B` | Optimisation algorithm |
| `--max-iterations` | `200` | Optimiser iteration budget |
| `--dt` | `0.01` | Dataset sampling period in seconds |
| `--output-dir` | `.` | Directory for all outputs |
| `--output` | `<output-dir>/robot_physics.usd` | Path for the tuned USD |
| `--report` | *(omit to skip)* | Path for the JSON calibration report |
| `--fix-base` | `false` | Fix the robot base in place (`convert` sub-command) |
| `--no-instanceable` | `false` | Disable instanceable USD output (`convert`) |
| `--log-level` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

### Python API

**One-shot pipeline with `Orchestrator`**

```python
from cad_physics_tuner import Orchestrator
from cad_physics_tuner.calibrator import CalibrationDataset, PhysicsParam

# 1. Describe the parameters to tune
params = [
    PhysicsParam("stiffness", "/robot/joints/j1", initial=1000, bounds=(100, 1e5)),
    PhysicsParam("damping",   "/robot/joints/j1", initial=50,   bounds=(1,   500)),
]

# 2. Load real robot data
dataset = CalibrationDataset.from_csv("real_traj.csv", dt=0.01)

# 3. Provide a simulator callable
def simulator(x, ds):
    """Run your physics engine with the parameter vector x.
    
    x[0] = stiffness, x[1] = damping (in declaration order)
    Returns (q_sim, qd_sim, f_sim) — any may be None.
    """
    q_sim = my_physics_step(x, ds)
    return q_sim, None, None

# 4. Run the pipeline
orch = Orchestrator(input_path="robot.urdf", output_dir="out/")
usd_path, report_path = orch.run(
    dataset=dataset,
    params=params,
    simulator_fn=simulator,
    method="L-BFGS-B",
    output_path="out/robot_physics.usd",
    report_path="out/report.json",
    max_iterations=200,
)

print(f"Tuned USD → {usd_path}")
print(f"Report    → {report_path}")
```

**Step-by-step with individual components**

```python
from cad_physics_tuner.converter import UrdfToUsdConverter
from cad_physics_tuner.calibrator import PhysicsCalibrator, CalibrationDataset, PhysicsParam
from cad_physics_tuner.physics_writer import PhysicsWriter

# Convert
converter = UrdfToUsdConverter(output_dir="out/", fix_base=True)
result = converter.convert("robot.urdf")          # result.usd_path

# Calibrate
params  = [PhysicsParam("stiffness", "/robot/joints/j1", 1000, (100, 1e5))]
dataset = CalibrationDataset.from_csv("real_traj.csv")
cal = PhysicsCalibrator(params=params, simulator=simulator)
report = cal.calibrate(dataset)                   # CalibrationResult

# Write back
writer = PhysicsWriter(result.usd_path)
writer.apply_optimised_params(report.optimised_params, output_path="out/tuned.usd")
```

---

## Pipeline Overview

```
URDF / CAD file
      │
      ▼
┌─────────────────────┐
│  UrdfToUsdConverter │  ← wraps Isaac Lab UrdfConverter
│  (convert step)     │
└─────────┬───────────┘
          │  .usd
          ▼
┌─────────────────────┐     real robot
│  PhysicsCalibrator  │ ◄── trajectory CSV
│  (calibrate step)   │
│                     │
│  scipy.optimize     │
│  ┌───────────────┐  │
│  │ simulator_fn  │  │  ← your physics engine callback
│  └───────────────┘  │
└─────────┬───────────┘
          │  optimised params
          ▼
┌─────────────────────┐
│  PhysicsWriter      │  ← pxr.UsdPhysics / PhysxSchema
│  (save step)        │
└─────────┬───────────┘
          │
     tuned .usd  +  report.json
```

---

## Dataset Format

Pass a CSV file with **no header row**.  Columns are interpreted based on the total column count, which must be a multiple of the number of joints `N`:

| Columns | Interpretation |
|---|---|
| `N` | Joint positions only |
| `2 × N` | Joint positions, then joint velocities |
| `3 × N` | Joint positions, velocities, then joint efforts |

Example for a 7-DOF arm with positions + velocities (14 columns, 100 timesteps):

```
0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60,  0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06
0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61,  0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06
...
```

You can also construct a dataset directly in Python:

```python
import numpy as np
from cad_physics_tuner.calibrator import CalibrationDataset

dataset = CalibrationDataset(
    q_real=np.load("q.npy"),      # shape (T, N)
    qd_real=np.load("qd.npy"),    # optional, shape (T, N)
    f_real=np.load("f.npy"),      # optional, shape (T, N)
    dt=0.01,
)
```

---

## Tunable Parameters

The following parameter names are understood by `PhysicsParam` and `PhysicsWriter`.  Each maps to a USD attribute on the specified prim path.

| Name | USD Attribute | Typical Prim |
|---|---|---|
| `stiffness` | `drive:angular:physics:stiffness` | Joint prim |
| `damping` | `drive:angular:physics:damping` | Joint prim |
| `max_force` | `drive:angular:physics:maxForce` | Joint prim |
| `static_friction` | `physics:staticFriction` | Physics Material prim |
| `dynamic_friction` | `physics:dynamicFriction` | Physics Material prim |
| `restitution` | `physics:restitutionCoefficient` | Physics Material prim |
| `mass` | `physics:mass` | Rigid body prim |
| `inertia_x` | `physics:diagonalInertia_x` | Rigid body prim |
| `inertia_y` | `physics:diagonalInertia_y` | Rigid body prim |
| `inertia_z` | `physics:diagonalInertia_z` | Rigid body prim |
| `contact_offset` | `physxCollision:contactOffset` | Collision prim |
| `rest_offset` | `physxCollision:restOffset` | Collision prim |

You can also pass any raw USD attribute name directly as the `name` field if it is not in the schema above.

---

## Optimisation Methods

| Method | Key | Notes |
|---|---|---|
| L-BFGS-B | `L-BFGS-B` | Gradient-based, best for smooth objectives with bounded params |
| Nelder-Mead | `Nelder-Mead` | Gradient-free simplex method, robust to noisy objectives |
| Differential Evolution | `differential_evolution` | Global search, good for multi-modal landscapes, slower |
| Grid search | `grid` | Exhaustive grid over `[lower, upper]` for each parameter |

---

## Configuration Reference

### `PhysicsParam`

```python
PhysicsParam(
    name="stiffness",            # key from PARAM_SCHEMA or raw USD attribute name
    prim_path="/robot/joints/j1",# USD prim path
    initial=1000.0,              # starting value
    bounds=(100.0, 1e5),         # (lower, upper) search bounds
)
```

### `CalibrationDataset`

```python
CalibrationDataset(
    q_real=...,   # np.ndarray (T, N) — required
    qd_real=...,  # np.ndarray (T, N) — optional
    f_real=...,   # np.ndarray (T, N) — optional
    dt=0.01,      # sampling period in seconds
)
```

### `Orchestrator`

```python
Orchestrator(
    input_path="robot.urdf",
    output_dir="out/",         # defaults to input file's directory
    fix_base=False,            # passed to UrdfToUsdConverter
    make_instanceable=True,    # passed to UrdfToUsdConverter
)
```

### Calibration report (`report.json`)

```json
{
  "optimised_params": {
    "/robot/joints/j1": { "stiffness": 1987.4, "damping": 78.2 }
  },
  "initial_error": 0.043210,
  "final_error": 0.001234,
  "improvement_pct": 97.1,
  "n_iterations": 42,
  "elapsed_seconds": 3.7,
  "success": true,
  "message": "Optimization terminated successfully",
  "error_history": [0.043210, 0.031, ...]
}
```

---

## Examples

### Franka Panda (self-contained)

The `examples/calibrate_franka.py` script demonstrates the full pipeline without Isaac Sim.  It generates synthetic "real" trajectories from known ground-truth stiffness and damping values, then verifies the calibrator recovers them:

```bash
python examples/calibrate_franka.py
```

Expected output:

```
============================================================
  cad-physics-tuner – Franka Calibration Example
============================================================
  Ground-truth stiffness : 2000.0
  Ground-truth damping   : 80.0
  Optimised stiffness    : 1999.87
  Optimised damping      : 80.03
  Initial error          : 0.048312
  Final error            : 0.000041
  Improvement            : 99.9 %
  Tuned USD              : examples/_output/franka_physics.usd
  Report                 : examples/_output/calibration_report.json
============================================================
```

### Using a custom simulator

```python
def my_isaac_sim_fn(x: np.ndarray, dataset):
    """
    x — 1-D array of parameter values in the order params were declared.
    dataset — CalibrationDataset with real robot data.
    Returns (q_sim, qd_sim, f_sim); unused channels may be None.
    """
    set_joint_stiffness(x[0])
    set_joint_damping(x[1])
    q_sim, qd_sim = run_physics_steps(dataset)
    return q_sim, qd_sim, None
```

---

## Development

```bash
# Install dev dependencies (Linux / macOS / PowerShell)
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=cad_physics_tuner --cov-report=term-missing

# Lint
ruff check .
```

All tests pass without Isaac Sim or `pxr` installed.

---

## License

MIT — see [LICENSE](LICENSE).
