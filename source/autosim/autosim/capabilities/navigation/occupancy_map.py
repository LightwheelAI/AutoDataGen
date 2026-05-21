from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass
from pxr import Usd, UsdGeom
from scipy.ndimage import binary_dilation

from autosim.core.logger import AutoSimLogger
from autosim.core.types import MapBounds, OccupancyMap

_logger = AutoSimLogger("OccupancyMap")


# -----------------------------------------------------------------------------
# Public config / API
# -----------------------------------------------------------------------------


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
    mesh_max_points: int = 5000
    """Max number of mesh vertices used for footprint estimation (downsample if larger)."""
    robot_radius: float = 0.25
    """Robot footprint radius in meters used to inflate obstacles. Set to 0.0 to disable inflation
    (point-robot assumption). For non-circular bases, use the bounding-circle radius (conservative)."""
    skip_path_substrings: tuple[str, ...] = ("light", "camera", "looks", "material", "sites")
    """Lowercased substrings; any prim whose path contains one of these is excluded from the occupancy map.."""


# -----------------------------------------------------------------------------
# USD helpers (geometry discovery)
# -----------------------------------------------------------------------------


def _get_prim_bounds(stage, prim_path: str, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Get bounding box of a prim

    Returns:
        min_bound, max_bound
    """

    prim = stage.GetPrimAtPath(prim_path)

    bbox_cache = _make_bbox_cache()
    aabb = _world_aabb_or_none(bbox_cache, prim)
    if aabb is None:
        raise ValueError(f"Prim '{prim_path}' has empty or invalid world bounds")
    min_arr, max_arr = aabb

    if verbose:
        _logger.info(f"Prim '{prim_path}' bounds: min={min_arr.tolist()}, max={max_arr.tolist()}")

    return min_arr, max_arr


def _is_geometry_prim(prim: Usd.Prim) -> bool:
    """Check if a prim is a geometry prim."""
    return (
        prim.IsA(UsdGeom.Mesh)
        or prim.IsA(UsdGeom.Cube)
        or prim.IsA(UsdGeom.Cylinder)
        or prim.IsA(UsdGeom.Sphere)
        or prim.IsA(UsdGeom.Capsule)
    )


def _make_bbox_cache(time_code: Usd.TimeCode | None = None) -> UsdGeom.BBoxCache:
    """Create a BBoxCache that covers all USD purposes.

    Collision-only meshes (e.g. ``*/Collisions/*`` under Isaac Lab assets) are typically authored
    with ``purpose=guide`` or ``proxy`` rather than ``default``. A BBoxCache that includes only
    ``default`` returns an empty range for these prims, which surfaces as a sentinel AABB of
    ``[+FLT_MAX, -FLT_MAX]`` and breaks downstream height / extent filters. Including all four
    purposes keeps the cache valid for both visual and collision geometry.
    """
    if time_code is None:
        time_code = Usd.TimeCode.Default()
    return UsdGeom.BBoxCache(
        time_code,
        includedPurposes=[
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
    )


def _world_aabb_or_none(bbox_cache: UsdGeom.BBoxCache, prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(min_xyz, max_xyz)`` of a prim's world-space AABB, or None if it is empty/invalid.

    USD returns an empty ``Gf.Range3d`` (``min > max``, components saturated to ``±FLT_MAX``) when
    a prim has no extent, is hidden, or its purpose is outside the cache's whitelist. We collapse
    those cases into ``None`` so callers can simply skip the prim.
    """
    bbox = bbox_cache.ComputeWorldBound(prim)
    aligned = bbox.ComputeAlignedBox()
    if aligned.IsEmpty():
        return None
    pmin = aligned.GetMin()
    pmax = aligned.GetMax()
    pmin_arr = np.array([pmin[0], pmin[1], pmin[2]], dtype=np.float64)
    pmax_arr = np.array([pmax[0], pmax[1], pmax[2]], dtype=np.float64)
    if not (np.all(np.isfinite(pmin_arr)) and np.all(np.isfinite(pmax_arr))):
        return None
    return pmin_arr, pmax_arr


# -----------------------------------------------------------------------------
# 2D geometry / rasterization utilities
# -----------------------------------------------------------------------------


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Compute 2D convex hull using monotonic chain. Returns CCW hull vertices."""

    if points.shape[0] == 0:
        return points
    pts = np.unique(points.astype(np.float64), axis=0)
    if pts.shape[0] <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def _points_in_convex_poly(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Test if 2D points are inside a convex polygon (CCW)."""

    if poly.shape[0] < 3:
        return np.zeros((points.shape[0],), dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    inside = np.ones((points.shape[0],), dtype=bool)
    for i in range(poly.shape[0]):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % poly.shape[0]]
        inside &= ((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) >= 0.0
        if not inside.any():
            break
    return inside


def _rasterize_convex_poly(
    occupancy_map: np.ndarray,
    poly_xy: np.ndarray,
    map_min_x: float,
    map_min_y: float,
    cell_size: float,
    map_height: int,
    map_width: int,
) -> None:
    """Rasterize a convex polygon into an occupancy map."""

    poly_min_x = float(poly_xy[:, 0].min())
    poly_max_x = float(poly_xy[:, 0].max())
    poly_min_y = float(poly_xy[:, 1].min())
    poly_max_y = float(poly_xy[:, 1].max())

    min_j = max(0, int((poly_min_x - map_min_x) / cell_size) - 1)
    max_j = min(map_width - 1, int((poly_max_x - map_min_x) / cell_size) + 1)
    min_i = max(0, int((poly_min_y - map_min_y) / cell_size) - 1)
    max_i = min(map_height - 1, int((poly_max_y - map_min_y) / cell_size) + 1)
    if min_j > max_j or min_i > max_i:
        return

    cols = np.arange(min_j, max_j + 1, dtype=np.int64)
    rows = np.arange(min_i, max_i + 1, dtype=np.int64)
    cc, rr = np.meshgrid(cols, rows)

    xs = map_min_x + (cc.astype(np.float64) + 0.5) * cell_size
    ys = map_min_y + (rr.astype(np.float64) + 0.5) * cell_size
    pts = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1)

    inside = _points_in_convex_poly(pts, poly_xy).reshape(rr.shape)
    occupancy_map[min_i : max_i + 1, min_j : max_j + 1][inside] = 1


# -----------------------------------------------------------------------------
# Candidate discovery & expansion
# -----------------------------------------------------------------------------


def _collect_candidate_prim_paths(
    stage,
    floor_prim_path: str,
    sample_height_min: float,
    sample_height_max: float,
    min_xy_extent: float = 0.01,
    skip_path_substrings: tuple[str, ...] = (),
) -> list[str]:
    """Collect candidate obstacle prim paths from the scene (coarse filtering only).

    Only direct geometry prims (Mesh / Cube / Cylinder / Sphere / Capsule) are returned.
    Xform containers are skipped here — ``stage.Traverse()`` already visits the leaf geometry
    prims they hold, so going through containers would just produce duplicates and force a
    second expansion pass downstream.
    """

    candidate_paths: list[str] = []
    bbox_cache = _make_bbox_cache()
    skip_lc = tuple(s.lower() for s in skip_path_substrings)

    for prim in stage.Traverse():
        path_str = str(prim.GetPath())

        if floor_prim_path in path_str or "Robot" in path_str or "robot" in path_str.lower():
            continue
        if skip_lc and any(skip in path_str.lower() for skip in skip_lc):
            continue

        if not _is_geometry_prim(prim):
            continue

        aabb = _world_aabb_or_none(bbox_cache, prim)
        if aabb is None:
            _logger.debug(f"Skipping prim with empty/invalid world bounds: {path_str}")
            continue
        prim_min, prim_max = aabb

        if prim_min[2] > sample_height_max or prim_max[2] < sample_height_min:
            continue

        xy_extent_x = prim_max[0] - prim_min[0]
        xy_extent_y = prim_max[1] - prim_min[1]
        if xy_extent_x <= min_xy_extent or xy_extent_y <= min_xy_extent:
            continue

        candidate_paths.append(path_str)

    return candidate_paths


# -----------------------------------------------------------------------------
# Footprint generation (world XY convex polygons)
# -----------------------------------------------------------------------------


def _transform_points_local_to_world(points_local: np.ndarray, mat) -> np.ndarray:
    """Transform points from local to world coordinates."""

    out = np.empty_like(points_local, dtype=np.float64)
    for i in range(points_local.shape[0]):
        p = points_local[i]
        pw = mat.Transform((float(p[0]), float(p[1]), float(p[2])))
        out[i, 0] = pw[0]
        out[i, 1] = pw[1]
        out[i, 2] = pw[2]
    return out


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """Downsample points."""

    if points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[idx]


def _mesh_footprint_poly_xy(
    mesh_prim: Usd.Prim,
    xform_cache: UsdGeom.XformCache,
    sample_height_min: float,
    sample_height_max: float,
    mesh_max_points: int,
) -> np.ndarray | None:
    """Generate a footprint polygon for a mesh."""

    mesh = UsdGeom.Mesh(mesh_prim)
    pts = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
    if pts is None or len(pts) == 0:
        return None
    points_local = np.asarray(pts, dtype=np.float64)
    points_local = _downsample_points(points_local, mesh_max_points)

    mat = xform_cache.GetLocalToWorldTransform(mesh_prim)
    points_w = _transform_points_local_to_world(points_local, mat)

    z = points_w[:, 2]
    mask = (z >= sample_height_min) & (z <= sample_height_max)
    if not np.any(mask):
        return None
    xy = points_w[mask][:, :2]
    poly = _convex_hull_2d(xy)
    if poly.shape[0] < 3:
        return None
    return poly


def _sample_circle_points(radius: float, num: int) -> np.ndarray:
    """Sample points on a circle."""

    angles = np.linspace(0.0, 2.0 * np.pi, num, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1)


def _cube_footprint_poly_xy(cube_prim: Usd.Prim, xform_cache: UsdGeom.XformCache) -> np.ndarray | None:
    """Generate a footprint polygon for a cube."""

    cube = UsdGeom.Cube(cube_prim)
    size = float(cube.GetSizeAttr().Get(Usd.TimeCode.Default()) or 0.0)
    if size <= 0.0:
        return None
    s = 0.5 * size
    corners_local = np.array([[-s, -s, 0.0], [-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0]], dtype=np.float64)
    mat = xform_cache.GetLocalToWorldTransform(cube_prim)
    corners_w = _transform_points_local_to_world(corners_local, mat)
    poly = _convex_hull_2d(corners_w[:, :2])
    if poly.shape[0] < 3:
        return None
    return poly


def _cylinder_like_footprint_poly_xy(
    prim: Usd.Prim, radius: float, xform_cache: UsdGeom.XformCache, num_circle_points: int = 32
) -> np.ndarray | None:
    """Generate a footprint polygon for a cylinder-like prim."""

    if radius <= 0.0:
        return None
    points_local = _sample_circle_points(radius, num_circle_points)
    mat = xform_cache.GetLocalToWorldTransform(prim)
    points_w = _transform_points_local_to_world(points_local, mat)
    poly = _convex_hull_2d(points_w[:, :2])
    if poly.shape[0] < 3:
        return None
    return poly


def _capsule_footprint_poly_xy(
    capsule_prim: Usd.Prim, xform_cache: UsdGeom.XformCache, num_circle_points: int = 32
) -> np.ndarray | None:
    """Generate a footprint polygon for a capsule."""

    cap = UsdGeom.Capsule(capsule_prim)
    radius = float(cap.GetRadiusAttr().Get(Usd.TimeCode.Default()) or 0.0)
    height = float(cap.GetHeightAttr().Get(Usd.TimeCode.Default()) or 0.0)
    if radius <= 0.0:
        return None

    axis = str(cap.GetAxisAttr().Get(Usd.TimeCode.Default()) or "Z").upper()
    if axis == "Z":
        return _cylinder_like_footprint_poly_xy(capsule_prim, radius, xform_cache, num_circle_points=num_circle_points)

    half_len = 0.5 * max(0.0, height)
    angles = np.linspace(0.0, 2.0 * np.pi, num_circle_points, endpoint=False)
    circle = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    if axis == "X":
        c1 = np.array([-half_len, 0.0], dtype=np.float64)
        c2 = np.array([half_len, 0.0], dtype=np.float64)
    else:  # "Y"
        c1 = np.array([0.0, -half_len], dtype=np.float64)
        c2 = np.array([0.0, half_len], dtype=np.float64)

    pts2 = np.concatenate([circle + c1, circle + c2], axis=0)
    points_local = np.concatenate([pts2, np.zeros((pts2.shape[0], 1), dtype=np.float64)], axis=1)

    mat = xform_cache.GetLocalToWorldTransform(capsule_prim)
    points_w = _transform_points_local_to_world(points_local, mat)
    poly = _convex_hull_2d(points_w[:, :2])
    if poly.shape[0] < 3:
        return None
    return poly


def _fallback_bbox_footprint_poly_xy(
    prim: Usd.Prim, bbox_cache: UsdGeom.BBoxCache, xform_cache: UsdGeom.XformCache
) -> np.ndarray | None:
    """Fallback footprint: projected local bbox corners convex hull in world XY."""

    local_bbox = bbox_cache.ComputeLocalBound(prim)
    box = local_bbox.GetRange()
    if box.IsEmpty():
        return None
    mat = xform_cache.GetLocalToWorldTransform(prim)
    bmin = box.GetMin()
    bmax = box.GetMax()
    corners_local = np.array(
        [
            [bmin[0], bmin[1], bmin[2]],
            [bmin[0], bmin[1], bmax[2]],
            [bmin[0], bmax[1], bmin[2]],
            [bmin[0], bmax[1], bmax[2]],
            [bmax[0], bmin[1], bmin[2]],
            [bmax[0], bmin[1], bmax[2]],
            [bmax[0], bmax[1], bmin[2]],
            [bmax[0], bmax[1], bmax[2]],
        ],
        dtype=np.float64,
    )
    corners_w = _transform_points_local_to_world(corners_local, mat)
    poly = _convex_hull_2d(corners_w[:, :2])
    if poly.shape[0] < 3:
        return None
    return poly


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


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

    candidate_paths = _collect_candidate_prim_paths(
        stage,
        floor_prim_path,
        sample_height_min,
        sample_height_max,
        cfg.min_xy_extent,
        tuple(cfg.skip_path_substrings),
    )
    _logger.info(f"Found {len(candidate_paths)} candidate prims")

    time_code = Usd.TimeCode.Default()
    bbox_cache = _make_bbox_cache(time_code)
    xform_cache = UsdGeom.XformCache(time_code)

    for path_str in candidate_paths:
        prim = stage.GetPrimAtPath(path_str)
        if not prim.IsValid():
            continue

        # Coarse height filter (redundant with the candidate-collection pass for the common case,
        # kept for safety in case bbox cache state diverges between the two traversals).
        aabb = _world_aabb_or_none(bbox_cache, prim)
        if aabb is None:
            continue
        pmin, pmax = aabb
        if pmin[2] > sample_height_max or pmax[2] < sample_height_min:
            continue

        poly: np.ndarray | None = None
        if prim.IsA(UsdGeom.Mesh):
            poly = _mesh_footprint_poly_xy(
                prim,
                xform_cache,
                sample_height_min,
                sample_height_max,
                mesh_max_points=cfg.mesh_max_points,
            )
        elif prim.IsA(UsdGeom.Cube):
            poly = _cube_footprint_poly_xy(prim, xform_cache)
        elif prim.IsA(UsdGeom.Cylinder):
            cyl = UsdGeom.Cylinder(prim)
            radius = float(cyl.GetRadiusAttr().Get(time_code) or 0.0)
            poly = _cylinder_like_footprint_poly_xy(prim, radius, xform_cache)
        elif prim.IsA(UsdGeom.Sphere):
            sph = UsdGeom.Sphere(prim)
            radius = float(sph.GetRadiusAttr().Get(time_code) or 0.0)
            poly = _cylinder_like_footprint_poly_xy(prim, radius, xform_cache)
        elif prim.IsA(UsdGeom.Capsule):
            poly = _capsule_footprint_poly_xy(prim, xform_cache)
        else:
            poly = _fallback_bbox_footprint_poly_xy(prim, bbox_cache, xform_cache)

        if poly is None or poly.shape[0] < 3:
            continue

        _rasterize_convex_poly(occupancy_map, poly, map_min_x, map_min_y, cfg.cell_size, map_height, map_width)

    # Inflate obstacles by the robot footprint radius (+ optional extra margin).
    # Cells produced by inflation are tracked separately so the visualization can distinguish
    # "original geometry" vs. "robot-radius safety buffer", but for planning purposes the combined
    # map treats both as occupied.
    inflation_radius = float(cfg.robot_radius)
    inflation_mask_np: np.ndarray | None = None
    if inflation_radius > 0.0:
        inflation_cells = int(np.ceil(inflation_radius / cfg.cell_size))
        if inflation_cells > 0:
            original = occupancy_map.astype(bool)
            inflated = binary_dilation(original, iterations=inflation_cells)
            inflation_mask_np = np.logical_and(inflated, np.logical_not(original))
            occupancy_map = inflated.astype(np.int8)
            _logger.info(
                f"Inflated obstacles by radius={inflation_radius:.3f}m ({inflation_cells} cells); "
                f"original={int(original.sum())} cells, inflation buffer={int(inflation_mask_np.sum())} cells"
            )

    inflation_mask_t = torch.from_numpy(inflation_mask_np).to(env.device) if inflation_mask_np is not None else None

    return OccupancyMap(
        occupancy_map=torch.from_numpy(occupancy_map).to(env.device),
        origin=(map_min_x, map_min_y),
        resolution=cfg.cell_size,
        map_bounds=MapBounds(min_x=map_min_x, max_x=map_max_x, min_y=map_min_y, max_y=map_max_y),
        floor_bounds=MapBounds(min_x=min_bound[0], max_x=max_bound[0], min_y=min_bound[1], max_y=max_bound[1]),
        inflation_mask=inflation_mask_t,
        inflation_radius=inflation_radius if inflation_mask_np is not None else 0.0,
    )
