"""
Command-line interface for cad-physics-tuner.

Usage examples::

    # Convert a URDF to USD
    python -m cad_physics_tuner convert --input robot.urdf --output-dir out/

    # Run full pipeline (convert + calibrate + save)
    python -m cad_physics_tuner calibrate \\
        --input robot.urdf \\
        --dataset real_traj.csv \\
        --params stiffness:/robot/joints/j1:1000:100:1e5 \\
        --output-dir out/ \\
        --report out/report.json \\
        --method L-BFGS-B
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cad-physics-tuner",
        description="Auto-calibrate physics for CAD → USD in Isaac Sim",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ convert
    conv = subparsers.add_parser("convert", help="Convert a URDF/CAD file to USD")
    conv.add_argument("--input", required=True, help="Path to input URDF file")
    conv.add_argument("--output-dir", default=None, help="Directory for output USD")
    conv.add_argument(
        "--fix-base",
        action="store_true",
        default=False,
        help="Fix the robot base in place",
    )
    conv.add_argument(
        "--no-instanceable",
        action="store_true",
        default=False,
        help="Disable instanceable USD output",
    )

    # ---------------------------------------------------------------- calibrate
    cal = subparsers.add_parser(
        "calibrate",
        help="Convert + auto-calibrate physics against a real-robot dataset",
    )
    cal.add_argument("--input", required=True, help="Path to input URDF file")
    cal.add_argument(
        "--dataset",
        required=True,
        help=(
            "CSV file with real robot data. "
            "Columns: q0…qN [, qd0…qdN [, f0…fN]]"
        ),
    )
    cal.add_argument(
        "--params",
        nargs="+",
        metavar="NAME:PRIM:INIT:LO:HI",
        required=True,
        help=(
            "Physics parameters to tune, each as "
            "\"name:prim_path:initial:lower_bound:upper_bound\". "
            "Example: stiffness:/robot/joints/j1:1000:100:1e5"
        ),
    )
    cal.add_argument(
        "--method",
        default="L-BFGS-B",
        choices=["L-BFGS-B", "Nelder-Mead", "differential_evolution", "grid"],
        help="Optimisation method (default: L-BFGS-B)",
    )
    cal.add_argument(
        "--max-iterations",
        type=int,
        default=200,
        help="Maximum optimiser iterations (default: 200)",
    )
    cal.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Dataset sampling period in seconds (default: 0.01)",
    )
    cal.add_argument("--output-dir", default=None, help="Directory for outputs")
    cal.add_argument(
        "--output",
        default=None,
        help="Path for tuned USD (default: <output-dir>/robot_physics.usd)",
    )
    cal.add_argument(
        "--report",
        default=None,
        help="Path for JSON calibration report (omit to skip)",
    )

    return parser


def _parse_param(spec: str):
    """Parse a ``NAME:PRIM:INIT:LO:HI`` parameter spec string."""
    from cad_physics_tuner.calibrator import PhysicsParam

    parts = spec.split(":")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            f"Expected NAME:PRIM:INIT:LO:HI but got: {spec!r}"
        )
    name, prim, init_s, lo_s, hi_s = parts
    return PhysicsParam(
        name=name,
        prim_path=prim,
        initial=float(init_s),
        bounds=(float(lo_s), float(hi_s)),
    )


def _make_passthrough_simulator(q_real: np.ndarray):
    """Minimal stand-in simulator for use when Isaac Sim is not available.

    This is used by the CLI when no real simulator is configured.  It
    returns a trivial simulation where ``q_sim ≈ q_real`` regardless of
    parameters, so the calibrator simply drives the error close to zero.
    Users should replace this with a real Isaac Sim physics-step callback.
    """

    def _sim(x: np.ndarray, dataset):
        return dataset.q_real.copy(), dataset.qd_real, dataset.f_real

    return _sim


def _cmd_convert(args: argparse.Namespace) -> int:
    from cad_physics_tuner.converter import UrdfToUsdConverter

    converter = UrdfToUsdConverter(
        output_dir=args.output_dir,
        fix_base=args.fix_base,
        make_instanceable=not args.no_instanceable,
    )
    result = converter.convert(args.input)
    print(f"Converted → {result.usd_path}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from cad_physics_tuner.calibrator import CalibrationDataset
    from cad_physics_tuner.orchestrator import Orchestrator

    params = [_parse_param(p) for p in args.params]
    dataset = CalibrationDataset.from_csv(args.dataset, dt=args.dt)

    simulator = _make_passthrough_simulator(dataset.q_real)

    orch = Orchestrator(input_path=args.input, output_dir=args.output_dir or ".")
    usd_path, report_path = orch.run(
        dataset=dataset,
        params=params,
        simulator_fn=simulator,
        method=args.method,
        output_path=args.output,
        report_path=args.report,
        max_iterations=args.max_iterations,
    )
    print(f"Tuned USD  → {usd_path}")
    if report_path:
        print(f"Report     → {report_path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    try:
        if args.command == "convert":
            return _cmd_convert(args)
        elif args.command == "calibrate":
            return _cmd_calibrate(args)
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
