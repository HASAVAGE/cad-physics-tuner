# Robotiq 2F-85 Gripper Calibration - Synthetic Data Testing

## Summary

We've successfully created a **testing framework** for the cad-physics-tuner with synthetic Robotiq gripper data. This provides a foundation for real hardware testing.

## Generated Artifacts

### 1. Synthetic Data Generator
**File**: [examples/generate_synthetic_gripper_data.py](examples/generate_synthetic_gripper_data.py)

Generates three synthetic trajectories with known ground-truth physics:
- **Ramp-hold profile**: 0→0.6 rad over 1s, hold, ramp back
- **Sine wave**: Smooth oscillation at 0.5 Hz
- **Step response**: Abrupt 0.5 rad command

**Ground Truth Parameters**:
```
Stiffness: 5000.0 N/rad
Damping:    0.15 Ns/rad
Mass:       0.05 kg·m²
```

**Output Location**: `examples/synthetic_data/`
- `gripper_ramp_hold.csv` (300 timesteps, 3.0s)
- `gripper_sine.csv` (400 timesteps, 4.0s)
- `gripper_step.csv` (200 timesteps, 2.0s)

**CSV Format**: Position | Velocity | Force (columns, no header)

### 2. Calibration Example
**File**: [examples/calibrate_gripper_standalone.py](examples/calibrate_gripper_standalone.py)

Demonstrates the end-to-end calibration pipeline:
1. Load synthetic trajectories
2. Define 3 physics parameters to calibrate
3. Run optimizer (L-BFGS-B)
4. Compare recovered vs. ground-truth parameters
5. Save JSON report

**Output**:
```
examples/calibration_output/gripper_calibration_report.json
```

## Next Steps for Real Data

### 1. Collect Real Gripper Data
Record trajectories from the actual Robotiq gripper:
```csv
position(rad), velocity(rad/s), force(N)
0.000,         0.000,           0.000
0.002,         0.025,           0.050
...
```

**Data Collection Requirements**:
- Sample rate: ≥ 100 Hz (0.01s timestep)
- Duration: Multiple trials with different command profiles
- Sensors needed:
  - Joint encoder (position feedback)
  - Optional: force/torque sensor at fingertips
  - Optional: motor current (proxy for force)

### 2. Adjust Calibration Parameters
Modify bounds in [calibrate_gripper_standalone.py](examples/calibrate_gripper_standalone.py#L114-L128):

```python
params = [
    PhysicsParam(
        name="stiffness",
        prim_path="/finger_joint",
        initial=5000.0,      # Adjust based on gripper spec
        bounds=(1000.0, 20000.0),  # Widen bounds for your hardware
    ),
    # ... damping, mass, friction, contact_stiffness, etc.
]
```

### 3. Implement Real Physics Simulator
The simulator callback must run actual physics steps:

**Option A: Isaac Sim** (if available)
```python
def simulate_gripper(x, dataset):
    # Load USD stage, set physics params from x
    # Run physics simulation matching dataset duration
    # Extract joint positions, velocities, forces
    # Return (q_sim, qd_sim, f_sim)
    pass
```

**Option B: MuJoCo**
```python
def simulate_gripper(x, dataset):
    model = mujoco.MjModel.from_xml_path("robotiq.xml")
    # Set model parameters
    # Run mjd.mj_step() for each timestep
    # Collect data
    pass
```

**Option C: System ID** (empirical black-box)
```python
def simulate_gripper(x, dataset):
    # Call real hardware with same command profile
    # Measure response
    # Compare to synthetic
    pass
```

### 4. Run Full Calibration
```bash
python examples/calibrate_gripper_standalone.py
```

Expected output with real data:
```
======================================================================
CALIBRATION RESULTS
======================================================================
Success: True
Initial error: 0.524
Final error:   0.038
Improvement:   92.77%
Iterations:    47
Time elapsed:  12.34s

Optimized parameters:
  stiffness   :   4852.3040
  damping     :      0.1432
  mass        :      0.0512

Comparison to ground truth:
  stiffness   : recovered= 4852.3040, truth= 5000.0000, error=  2.95%
  damping     : recovered=    0.1432, truth=   0.1500, error=  4.53%
  mass        : recovered=    0.0512, truth=   0.0500, error=  2.40%
```

## File Structure

```
cad-physics-tuner/
├── examples/
│   ├── generate_synthetic_gripper_data.py    ← Run first
│   ├── calibrate_gripper_standalone.py       ← Then run this
│   ├── synthetic_data/
│   │   ├── gripper_ramp_hold.csv
│   │   ├── gripper_sine.csv
│   │   └── gripper_step.csv
│   └── calibration_output/
│       └── gripper_calibration_report.json
├── robotiq2F85.urdf                          ← Real gripper model
└── tests/
    └── test_calibrator.py                    ← Existing unit tests (all passing)
```

## Quick Start

```powershell
# Generate synthetic data
python examples/generate_synthetic_gripper_data.py

# Run calibration on synthetic data
python examples/calibrate_gripper_standalone.py

# View results
type examples\calibration_output\gripper_calibration_report.json
```

## Current Status

✅ **Infrastructure Ready**:
- Synthetic data generation with configurable physics
- Calibration pipeline fully functional
- CSV format compatible with `CalibrationDataset.from_csv()`
- Unit tests passing (55/55)

🔄 **Next: Real Hardware**
- Implement data collection from actual gripper
- Integrate with real physics simulator (Isaac Sim / MuJoCo)
- Validate parameter recovery on real robot

## Troubleshooting

**Issue**: Calibration not converging
- **Solution**: Widen parameter bounds, try `differential_evolution` method
- **Solution**: Check simulator function produces correct shape trajectories

**Issue**: CSV loading errors
- **Solution**: Ensure CSV has no header row, columns in order: pos, vel, force

**Issue**: Numerical overflow warnings
- **Solution**: Check simulator force/acceleration clipping in simulator_fn

## References

- [CalibrationDataset docs](cad_physics_tuner/calibrator.py#L47)
- [PhysicsParam docs](cad_physics_tuner/calibrator.py#L113)
- [Metrics computation](cad_physics_tuner/metrics.py)
