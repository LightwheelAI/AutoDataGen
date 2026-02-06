from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass
from pxr import Usd, UsdGeom

from autosim.core.logger import AutoSimLogger
from autosim.core.types import MapBounds, OccupancyMap

_logger = AutoSimLogger("OccupancyMap")


@configclass
class OccupancyMapCfg:
    """Configuration for the occupancy map."""

    floor_prim_suffix: str = MISSING
    """The suffix of the floor prim."""
    max_world_extent: float = 100.0
    """The maximum extent of the world in meters."""
    max_map_size: int = 2000
    """The maximum size of the map in cells."""
    min_xy_extent: float = 0.01
    """Minimum xy extent to consider as obstacle (1cm by default)."""
    cell_size: float = 0.05
    """The size of the cell in meters."""
    sample_height: float = 0.5
    """The height to sample the occupancy map at, in meters."""
    height_tolerance: float = 0.2
    """The tolerance for the height sampling."""


def _get_prim_bounds(stage, prim_path: str, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Get bounding box of a prim

    Returns:
        min_bound, max_bound
    """

    prim = stage.GetPrimAtPath(prim_path)

    # Get bounding box
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim)

    # Aligned bounding box range
    aligned_box = bbox.ComputeAlignedBox()
    min_point = aligned_box.GetMin()
    max_point = aligned_box.GetMax()

    if verbose:
        _logger.info(f"Prim '{prim_path}' bounds: min={list(min_point)}, max={list(max_point)}")

    return np.array([min_point[0], min_point[1], min_point[2]]), np.array([max_point[0], max_point[1], max_point[2]])


def _collect_collision_prims(
    stage, floor_prim_path: str, sample_height_min: float, sample_height_max: float, min_xy_extent: float = 0.01
) -> list:
    """Collect collision primitives from the scene"""

    collision_prims = []
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])

    for prim in stage.Traverse():
        path_str = str(prim.GetPath())

        # Skip the floor itself and robot
        if floor_prim_path in path_str or "Robot" in path_str or "robot" in path_str.lower():
            continue

        # Skip lights, cameras, and other non-geometry prims
        if any(skip in path_str.lower() for skip in ["light", "camera", "looks", "material"]):
            continue

        # Check if prim has geometry
        has_geometry = (
            prim.IsA(UsdGeom.Mesh)
            or prim.IsA(UsdGeom.Cube)
            or prim.IsA(UsdGeom.Cylinder)
            or prim.IsA(UsdGeom.Sphere)
            or prim.IsA(UsdGeom.Capsule)
        )
        if not has_geometry and prim.IsA(UsdGeom.Xform):
            # Check if it's a group that might contain geometry
            for child in prim.GetChildren():
                if child.IsA(UsdGeom.Mesh) or child.IsA(UsdGeom.Cube):
                    has_geometry = True
                    break

        if has_geometry:
            try:
                # Get bounding box
                bbox = bbox_cache.ComputeWorldBound(prim)
                aligned_box = bbox.ComputeAlignedBox()
                prim_min = aligned_box.GetMin()
                prim_max = aligned_box.GetMax()

                # Check if this prim intersects our sampling height range
                if prim_min[2] <= sample_height_max and prim_max[2] >= sample_height_min:
                    # Only include if it has significant XY extent
                    xy_extent_x = prim_max[0] - prim_min[0]
                    xy_extent_y = prim_max[1] - prim_min[1]
                    if xy_extent_x > min_xy_extent and xy_extent_y > min_xy_extent:
                        collision_prims.append({
                            "path": path_str,
                            "min": np.array([prim_min[0], prim_min[1], prim_min[2]]),
                            "max": np.array([prim_max[0], prim_max[1], prim_max[2]]),
                        })
            except Exception:
                # Skip prims that can't be processed
                continue

    return collision_prims


def get_occupancy_map(env: ManagerBasedEnv, cfg: OccupancyMapCfg) -> OccupancyMap:
    """Generate occupancy map from IsaacLab environment.

    Args:
        env: The IsaacLab environment.
        cfg: The configuration for the occupancy map.

    Returns:
        The occupancy map.
    """

    stage = env.scene.stage

    floor_prim_path = f"/World/envs/env_0/{cfg.floor_prim_suffix}"

    min_bound, max_bound = _get_prim_bounds(stage, floor_prim_path)

    # Validate bounds - check for unreasonable values (inf, nan, or too large)
    world_extent_x = max_bound[0] - min_bound[0]
    world_extent_y = max_bound[1] - min_bound[1]

    bounds_invalid = (
        not np.isfinite(world_extent_x)
        or not np.isfinite(world_extent_y)
        or world_extent_x > cfg.max_world_extent
        or world_extent_y > cfg.max_world_extent
        or world_extent_x <= 0
        or world_extent_y <= 0
    )

    if bounds_invalid:
        raise ValueError(f"Floor bounds invalid or too large: extent_x={world_extent_x}, extent_y={world_extent_y}")

    # Calculate map bounds (use floor bounds)
    map_min_x, map_max_x = min_bound[0], max_bound[0]
    map_min_y, map_max_y = min_bound[1], max_bound[1]

    map_width = int((map_max_x - map_min_x) / cfg.cell_size) + 1
    map_height = int((map_max_y - map_min_y) / cfg.cell_size) + 1

    # Clamp map size to prevent memory issues
    if map_width > cfg.max_map_size or map_height > cfg.max_map_size:
        _logger.warning(f"Map size {map_width}x{map_height} exceeds max {cfg.max_map_size}")
        new_cell_size = max((map_max_x - map_min_x) / cfg.max_map_size, (map_max_y - map_min_y) / cfg.max_map_size)
        _logger.info(f"Adjusting cell_size from {cfg.cell_size:.3f}m to {new_cell_size:.3f}m")
        cfg.cell_size = new_cell_size
        map_width = int((map_max_x - map_min_x) / cfg.cell_size) + 1
        map_height = int((map_max_y - map_min_y) / cfg.cell_size) + 1
    _logger.info(
        f"Generating map: {map_width}x{map_height} cells, bounds: x=[{map_min_x:.2f}, {map_max_x:.2f}],"
        f" y=[{map_min_y:.2f}, {map_max_y:.2f}]"
    )

    # Initialize occupancy map (0 = free, 1 = occupied)
    occupancy_map = np.zeros((map_height, map_width), dtype=np.int8)

    # Calculate height range for sampling
    sample_height_min = min_bound[2] + cfg.sample_height - cfg.height_tolerance
    sample_height_max = min_bound[2] + cfg.sample_height + cfg.height_tolerance
    _logger.info(f"Sampling height range: [{sample_height_min:.2f}, {sample_height_max:.2f}]")

    # Collect collision primitives
    collision_prims = _collect_collision_prims(
        stage, floor_prim_path, sample_height_min, sample_height_max, cfg.min_xy_extent
    )
    _logger.info(f"Found {len(collision_prims)} collision primitives")

    # Mark occupied cells
    for prim_info in collision_prims:
        prim_min = prim_info["min"]
        prim_max = prim_info["max"]

        # Calculate grid indices for this prim's bounding box
        min_i = max(0, int((prim_min[1] - map_min_y) / cfg.cell_size))
        max_i = min(map_height - 1, int((prim_max[1] - map_min_y) / cfg.cell_size) + 1)
        min_j = max(0, int((prim_min[0] - map_min_x) / cfg.cell_size))
        max_j = min(map_width - 1, int((prim_max[0] - map_min_x) / cfg.cell_size) + 1)

        # Mark all cells in this bounding box as occupied
        occupancy_map[min_i : max_i + 1, min_j : max_j + 1] = 1

    return OccupancyMap(
        occupancy_map=torch.from_numpy(occupancy_map).to(env.device),
        origin=(map_min_x, map_min_y),
        resolution=cfg.cell_size,
        map_bounds=MapBounds(min_x=map_min_x, max_x=map_max_x, min_y=map_min_y, max_y=map_max_y),
        floor_bounds=MapBounds(min_x=min_bound[0], max_x=max_bound[0], min_y=min_bound[1], max_y=max_bound[1]),
    )
