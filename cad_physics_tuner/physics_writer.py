"""
Physics parameter writer.

Writes optimised physics parameters back into a USD stage using
``pxr.UsdPhysics``, ``PhysxSchema``, and raw USD attribute APIs.

When ``pxr`` (OpenUSD / Isaac Sim) is not available the module falls back to a
lightweight *stub* that records writes in memory so that the rest of the
package (and its tests) can operate without a full Isaac Sim installation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional USD imports – graceful fallback for environments without Isaac Sim
# ---------------------------------------------------------------------------
try:
    from pxr import Sdf, Usd  # type: ignore[import]

    _USD_AVAILABLE = True
except ImportError:  # pragma: no cover – only exercised when pxr is missing
    _USD_AVAILABLE = False
    logger.debug("pxr not found; USD writes will be recorded in-memory only.")

try:
    import PhysxSchema  # type: ignore[import]  # noqa: F401

    _PHYSX_AVAILABLE = True
except ImportError:
    _PHYSX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Parameter schema
# ---------------------------------------------------------------------------

#: Supported parameter names and their USD targets.
PARAM_SCHEMA: Dict[str, str] = {
    # DriveAPI (joint actuator)
    "stiffness": "drive:angular:physics:stiffness",
    "damping": "drive:angular:physics:damping",
    "max_force": "drive:angular:physics:maxForce",
    # Physics Material
    "static_friction": "physics:staticFriction",
    "dynamic_friction": "physics:dynamicFriction",
    "restitution": "physics:restitutionCoefficient",
    # Rigid body mass / inertia
    "mass": "physics:mass",
    "inertia_x": "physics:diagonalInertia_x",
    "inertia_y": "physics:diagonalInertia_y",
    "inertia_z": "physics:diagonalInertia_z",
    # Contact offsets (PhysX-specific)
    "contact_offset": "physxCollision:contactOffset",
    "rest_offset": "physxCollision:restOffset",
}


class PhysicsWriter:
    """Read/write physics parameters on a USD stage.

    Parameters
    ----------
    usd_path:
        Path to the ``.usd`` / ``.usda`` file.  When *None* the writer
        operates in *dry-run* mode (nothing is persisted to disk).
    """

    def __init__(self, usd_path: Optional[str | Path] = None) -> None:
        self.usd_path: Optional[Path] = Path(usd_path) if usd_path else None
        self._stage: Any = None  # pxr.Usd.Stage when available
        self._dry_run_store: Dict[str, Dict[str, float]] = {}  # prim_path → {attr: val}

        if self.usd_path and _USD_AVAILABLE:
            self._stage = Usd.Stage.Open(str(self.usd_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_param(self, prim_path: str, param_name: str) -> Optional[float]:
        """Read a physics parameter from the stage.

        Returns *None* when the prim or attribute does not exist.
        """
        attr_name = PARAM_SCHEMA.get(param_name, param_name)

        if self._stage is not None:
            prim = self._stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return None
            attr = prim.GetAttribute(attr_name)
            if attr.IsValid():
                return float(attr.Get())
            return None

        # Dry-run: return from in-memory store
        return self._dry_run_store.get(prim_path, {}).get(attr_name)

    def write_param(self, prim_path: str, param_name: str, value: float) -> None:
        """Write a single physics parameter to *prim_path* on the stage.

        Parameters
        ----------
        prim_path:
            USD prim path, e.g. ``"/robot/joints/shoulder_pan"``.
        param_name:
            One of the keys in :data:`PARAM_SCHEMA` or a raw attribute name.
        value:
            New numeric value.
        """
        attr_name = PARAM_SCHEMA.get(param_name, param_name)

        if self._stage is not None:
            prim = self._stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                logger.warning("Prim %s not found; skipping write of %s", prim_path, attr_name)
                return
            attr = prim.GetAttribute(attr_name)
            if not attr.IsValid():
                attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Double)
            attr.Set(value)
            logger.debug("USD write: %s.%s = %s", prim_path, attr_name, value)
        else:
            store = self._dry_run_store.setdefault(prim_path, {})
            store[attr_name] = value
            logger.debug("Dry-run write: %s.%s = %s", prim_path, attr_name, value)

    def write_params(self, prim_path: str, params: Dict[str, float]) -> None:
        """Write multiple parameters at once.

        Parameters
        ----------
        prim_path:
            USD prim path.
        params:
            Mapping of *param_name* → *value*.
        """
        for name, value in params.items():
            self.write_param(prim_path, name, value)

    def save(self, output_path: Optional[str | Path] = None) -> Path:
        """Save the stage to *output_path* (or overwrite the original file).

        Returns the path that was written.  In dry-run mode nothing is
        persisted and the method simply returns the requested path.
        """
        target = Path(output_path) if output_path else self.usd_path
        if target is None:
            raise ValueError("No output path specified and no usd_path was provided.")

        if self._stage is not None:
            self._stage.GetRootLayer().Export(str(target))
            logger.info("Saved tuned USD to %s", target)
        else:
            logger.info("Dry-run mode: would save to %s", target)

        return target

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def apply_optimised_params(
        self,
        params_map: Dict[str, Dict[str, float]],
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """Write all optimised parameters and save.

        Parameters
        ----------
        params_map:
            Nested dict ``{prim_path: {param_name: value}}``.
        output_path:
            Where to save the result.  Defaults to the input ``usd_path``
            (overwrite) or raises when neither is set.

        Returns
        -------
        Path
            Path of the saved file.
        """
        for prim_path, params in params_map.items():
            self.write_params(prim_path, params)
        return self.save(output_path)

    @property
    def dry_run_store(self) -> Dict[str, Dict[str, float]]:
        """Expose in-memory writes (useful for testing / dry-run mode)."""
        return self._dry_run_store
