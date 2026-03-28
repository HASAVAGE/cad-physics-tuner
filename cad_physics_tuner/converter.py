"""
URDF / CAD → USD conversion wrapper.

Wraps Isaac Lab's ``UrdfConverter`` (and optionally ``MeshConverter``) with
a clean, test-friendly interface.  When Isaac Lab is not installed the module
falls back to a no-op stub so that the orchestrator and calibrator remain
importable and unit-testable without a full Isaac Sim installation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Isaac Lab imports
# ---------------------------------------------------------------------------
try:
    from omni.isaac.lab.utils.assets.urdf_converter import (  # type: ignore[import]
        UrdfConverter,
        UrdfConverterCfg,
    )

    _ISAAC_AVAILABLE = True
except ImportError:
    _ISAAC_AVAILABLE = False
    logger.debug(
        "omni.isaac.lab not found; UrdfConverter will run in stub mode."
    )


class ConversionResult:
    """Holds the output of a conversion operation."""

    def __init__(self, usd_path: Path, source_path: Path) -> None:
        self.usd_path = usd_path
        self.source_path = source_path

    def __repr__(self) -> str:  # pragma: no cover
        return f"ConversionResult(usd_path={self.usd_path!r})"


class UrdfToUsdConverter:
    """Convert a URDF (or STEP/OBJ via an intermediate URDF) to USD.

    Parameters
    ----------
    output_dir:
        Directory where the converted ``.usd`` file will be written.
        Defaults to the directory of the input file.
    fix_base:
        Whether to fix the robot's base in place (adds a fixed joint at root).
    make_instanceable:
        Whether to produce instanceable USDs (recommended for Isaac Lab).
    default_drive_type:
        ``"position"`` or ``"velocity"`` drive type for all joints.
    """

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        fix_base: bool = False,
        make_instanceable: bool = True,
        default_drive_type: str = "position",
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else None
        self.fix_base = fix_base
        self.make_instanceable = make_instanceable
        self.default_drive_type = default_drive_type

    def convert(self, input_path: str | Path) -> ConversionResult:
        """Convert *input_path* to a USD file.

        The output path is ``<output_dir>/<stem>.usd``.  When Isaac Lab is
        present the real ``UrdfConverter`` is used; otherwise a stub file is
        created so downstream code can keep running.

        Parameters
        ----------
        input_path:
            Path to a ``.urdf`` file (or a ``.step`` / ``.obj`` that has
            already been converted to URDF upstream).

        Returns
        -------
        ConversionResult
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        out_dir = self.output_dir or input_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        usd_path = out_dir / (input_path.stem + ".usd")

        if _ISAAC_AVAILABLE:
            usd_path = self._convert_with_isaac(input_path, usd_path)
        else:
            usd_path = self._convert_stub(input_path, usd_path)

        return ConversionResult(usd_path=usd_path, source_path=input_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _convert_with_isaac(self, input_path: Path, usd_path: Path) -> Path:  # pragma: no cover
        """Delegate to Isaac Lab's UrdfConverter."""
        cfg = UrdfConverterCfg(
            asset_path=str(input_path),
            usd_dir=str(usd_path.parent),
            usd_file_name=usd_path.name,
            fix_base=self.fix_base,
            make_instanceable=self.make_instanceable,
            default_drive_type=self.default_drive_type,
        )
        converter = UrdfConverter(cfg)
        converter.convert()
        logger.info("Isaac Lab UrdfConverter wrote %s", usd_path)
        return usd_path

    def _convert_stub(self, input_path: Path, usd_path: Path) -> Path:
        """Write a minimal stub USD so downstream code has a file to open."""
        stub_content = (
            '#usda 1.0\n'
            f'# Stub conversion of {input_path.name}\n'
            'def Xform "root" {}\n'
        )
        usd_path.write_text(stub_content, encoding="utf-8")
        logger.warning(
            "Isaac Lab not available – wrote stub USD to %s. "
            "Install Isaac Sim to get a real conversion.",
            usd_path,
        )
        return usd_path
