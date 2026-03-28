"""
Generate synthetic Robotiq 2F-85 gripper trajectory data for calibration testing.

This script simulates the gripper dynamics with known "ground truth" physics
parameters and saves the trajectory as CSV. The calibration pipeline should
recover these parameters from the synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def simulate_gripper_dynamics(
    t_end: float = 3.0,
    dt: float = 0.01,
    stiffness: float = 5000.0,
    damping: float = 0.15,
    mass: float = 0.05,
    command_profile: str = "ramp_hold",
) -> dict[str, np.ndarray]:
    """
    Simulate gripper finger_joint dynamics using second-order system.

    Args:
        t_end: Total simulation time (seconds)
        dt: Time step (seconds)
        stiffness: Joint stiffness (N/rad)
        damping: Joint damping (Ns/rad)
        mass: Effective inertia (kg·m²)
        command_profile: Type of command ("ramp_hold", "sine", "step")

    Returns:
        Dictionary with keys: time, position, velocity, force
    """
    times = np.arange(0, t_end, dt)
    n_steps = len(times)

    positions = np.zeros(n_steps)
    velocities = np.zeros(n_steps)
    forces = np.zeros(n_steps)
    commands = np.zeros(n_steps)

    # Generate command profile
    for i, t in enumerate(times):
        if command_profile == "ramp_hold":
            # Ramp to 0.6 rad over 1 second, hold for 1 second, ramp back down
            if t < 1.0:
                commands[i] = 0.6 * (t / 1.0)  # Ramp up
            elif t < 2.0:
                commands[i] = 0.6  # Hold
            else:
                commands[i] = max(0.0, 0.6 * (1.0 - (t - 2.0) / 1.0))  # Ramp down

        elif command_profile == "sine":
            # Oscillate between 0 and 0.6 rad at 0.5 Hz
            commands[i] = 0.3 * (1.0 + np.sin(2 * np.pi * 0.5 * t))

        elif command_profile == "step":
            # Step to 0.5 at t=0.5s
            commands[i] = 0.5 if t > 0.5 else 0.0

    # Simulate dynamics: m*x'' + c*x' + k*(x - cmd) = 0
    # Safe integration with clipping to prevent overflow
    for i in range(1, n_steps):
        # Position error
        error = commands[i - 1] - positions[i - 1]

        # Spring-damper force (clipped to prevent overflow)
        force = np.clip(stiffness * error - damping * velocities[i - 1], -1e4, 1e4)

        # Acceleration (clipped)
        acceleration = np.clip(force / mass, -1e4, 1e4)

        # Euler integration
        velocities[i] = velocities[i - 1] + acceleration * dt
        positions[i] = positions[i - 1] + velocities[i] * dt
        forces[i] = force

    return {
        "time": times,
        "position": positions,
        "velocity": velocities,
        "force": forces,
        "command": commands,
    }


def add_noise(
    data: dict[str, np.ndarray],
    position_noise_std: float = 0.002,
    velocity_noise_std: float = 0.05,
    force_noise_std: float = 0.1,
) -> dict[str, np.ndarray]:
    """
    Add realistic measurement noise to synthetic data.

    Args:
        data: Dictionary with trajectory arrays
        position_noise_std: Position measurement noise (rad)
        velocity_noise_std: Velocity measurement noise (rad/s)
        force_noise_std: Force measurement noise (N)

    Returns:
        Data dictionary with added noise
    """
    noisy_data = {k: v.copy() for k, v in data.items()}

    noisy_data["position"] += np.random.normal(
        0, position_noise_std, len(noisy_data["position"])
    )
    noisy_data["velocity"] += np.random.normal(
        0, velocity_noise_std, len(noisy_data["velocity"])
    )
    noisy_data["force"] += np.random.normal(0, force_noise_std, len(noisy_data["force"]))

    return noisy_data


def save_to_csv(data: dict[str, np.ndarray], output_path: Path) -> None:
    """
    Save synthetic trajectory to CSV in CalibrationDataset format.

    CSV columns (no header): position, velocity, force
    Format expected by CalibrationDataset.from_csv():
      - First 1/3 of columns: positions
      - Second 1/3 of columns: velocities  
      - Last 1/3 of columns: forces

    Args:
        data: Dictionary with trajectory arrays
        output_path: Path to save CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stack columns in order: position, velocity, force (no header)
    stacked = np.column_stack([data["position"], data["velocity"], data["force"]])
    np.savetxt(str(output_path), stacked, delimiter=",")

    print(f"✓ Saved synthetic data to {output_path}")
    print(f"  - {len(data['time'])} timesteps")
    print(f"  - Duration: {data['time'][-1]:.2f}s")


def main():
    """Generate multiple synthetic datasets for different test scenarios."""
    output_dir = Path(__file__).parent / "synthetic_data"
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Generating synthetic Robotiq 2F-85 gripper datasets")
    print("=" * 70)

    # Dataset 1: "Reference" with known physics
    print("\n[1/3] Generating reference trajectory (ramp_hold)")
    data1 = simulate_gripper_dynamics(
        t_end=3.0,
        dt=0.01,
        stiffness=5000.0,
        damping=0.15,
        mass=0.05,
        command_profile="ramp_hold",
    )
    data1_noisy = add_noise(data1, position_noise_std=0.002, velocity_noise_std=0.05)
    save_to_csv(
        data1_noisy,
        output_dir / "gripper_ramp_hold.csv",
    )

    # Dataset 2: Sine wave command
    print("\n[2/3] Generating sine wave trajectory")
    data2 = simulate_gripper_dynamics(
        t_end=4.0,
        dt=0.01,
        stiffness=5000.0,
        damping=0.15,
        mass=0.05,
        command_profile="sine",
    )
    data2_noisy = add_noise(data2, position_noise_std=0.002, velocity_noise_std=0.05)
    save_to_csv(
        data2_noisy,
        output_dir / "gripper_sine.csv",
    )

    # Dataset 3: Step response
    print("\n[3/3] Generating step response trajectory")
    data3 = simulate_gripper_dynamics(
        t_end=2.0,
        dt=0.01,
        stiffness=5000.0,
        damping=0.15,
        mass=0.05,
        command_profile="step",
    )
    data3_noisy = add_noise(data3, position_noise_std=0.002, velocity_noise_std=0.05)
    save_to_csv(
        data3_noisy,
        output_dir / "gripper_step.csv",
    )

    print("\n" + "=" * 70)
    print("Ground truth physics parameters:")
    print(f"  Stiffness: 5000.0 N/rad")
    print(f"  Damping: 0.15 Ns/rad")
    print(f"  Mass: 0.05 kg·m²")
    print("\nCalibration should recover these values from the synthetic data.")
    print("=" * 70)


if __name__ == "__main__":
    main()
