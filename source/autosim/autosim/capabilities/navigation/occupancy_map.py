"""Build a 2D occupancy grid from an Isaac Lab scene for base navigation.

Pipeline overview:
  1. Floor prim defines map XY bounds and cell grid.
  2. ``stage.Traverse()`` collects obstacle geometry prims (coarse filter incl. ``sample_height``).
  3. Candidates that also belong to ``env.scene`` use simulation link/root poses; others use USD xforms.
  4. Each prim is projected to a 2D convex footprint and rasterized; obstacles are inflated by ``robot_radius``.

Call after ``env.reset()`` so articulation poses match the current episode. The map is static for the
lifetime of the returned ``OccupancyMap`` (no runtime updates).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import MISSING, dataclass

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass
from pxr import Gf, Usd, UsdGeom
from scipy.ndimage import binary_dilation

from autosim.core.logger import AutoSimLogger
from autosim.core.types import MapBounds, OccupancyMap
from autosim.utils.data_util import as_torch

_logger = AutoSimLogger("OccupancyMap")

# Isaac Lab single-env layout; map build currently assumes env index 0 (see ``OccupancyMapCfg.env_id`` for poses).
ENV_PRIM_PATH = "/World/envs/env_0"
_NUM_CIRCLE_POINTS = 32  # samples for cylinder/sphere/capsule XY footprints


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
    """Height slice center (meters above floor z). Used only in candidate-prim coarse filtering."""
    height_tolerance: float = 0.2
    """Half-width of the height window for candidate-prim coarse filtering (meters)."""
    mesh_max_points: int = 5000
    """Max number of mesh vertices used for footprint estimation (downsample if larger)."""
    robot_radius: float = 0.25
    """Robot footprint radius in meters used to inflate obstacles. Set to 0.0 to disable inflation
    (point-robot assumption). For non-circular bases, use the bounding-circle radius (conservative)."""
    skip_path_substrings: tuple[str, ...] = ("light", "camera", "looks", "material", "sites")
    """Lowercased substrings; any prim whose path contains one of these is excluded from the occupancy map.."""
    env_id: int = 0
    """Environment index used when reading poses from ``env.scene`` articulations / rigid objects."""


# -----------------------------------------------------------------------------
# Internal context
# -----------------------------------------------------------------------------


@dataclass
class _RasterizeContext:
    """Shared state for footprint generation and grid rasterization.

    ``world_mat`` passed per prim (when set) overrides ``xform_cache`` for that prim only — used for
    scene-registered assets whose USD xforms may not reflect ``env.reset()`` joint states.
    """

    stage: Usd.Stage
    """The USD stage."""
    occupancy_map: np.ndarray
    """The occupancy map."""
    xform_cache: UsdGeom.XformCache
    """The USD xform cache."""
    bbox_cache: UsdGeom.BBoxCache
    """The USD bbox cache."""
    time_code: Usd.TimeCode
    """The USD time code."""
    mesh_max_points: int
    """The maximum number of mesh points used for footprint generation."""
    map_min_x: float
    """The minimum x coordinate of the map."""
    map_min_y: float
    """The minimum y coordinate of the map."""
    cell_size: float
    """The size of the cell in meters."""
    cell_size: float
    """The size of the cell in meters."""
    map_height: int
    """The height of the map."""
    map_width: int
    """The width of the map."""


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
    """Fill grid cells whose centers fall inside a world-frame XY convex polygon."""

    # Tight integer bounds around the polygon AABB (+1 cell padding) before per-cell inside test.
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
# Candidate discovery
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

        if (prim_max[0] - prim_min[0]) <= min_xy_extent or (prim_max[1] - prim_min[1]) <= min_xy_extent:
            continue

        candidate_paths.append(path_str)

    return candidate_paths


# -----------------------------------------------------------------------------
# Scene-registered assets (candidate ∩ env.scene): Isaac pose + USD geometry
#
# USD ``XformCache`` often reflects authored / bind poses, not post-reset articulation joints.
# For prims that are both (a) in the candidate list and (b) under an ``env.scene`` asset prefix,
# we keep collision **shape** from USD but apply **body pose** from Isaac Lab at build time.
# -----------------------------------------------------------------------------


def _matrix4d_from_pos_quat(pos: np.ndarray, quat_xyzw: np.ndarray) -> Gf.Matrix4d:
    rot = Gf.Rotation(
        Gf.Quatd(float(quat_xyzw[3]), Gf.Vec3d(float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])))
    )
    return Gf.Matrix4d().SetTransform(rot, Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))


def _build_scene_prim_prefixes(env: ManagerBasedEnv) -> dict[str, str]:
    """Resolve ``env.scene`` asset names to concrete USD prim prefixes."""

    prefixes: dict[str, str] = {}
    for name in env.scene.keys():
        asset = env.scene[name]
        if not hasattr(asset, "cfg") or not hasattr(asset.cfg, "prim_path"):
            continue
        prim = sim_utils.find_first_matching_prim(asset.cfg.prim_path)
        if prim is None or not prim.IsValid():
            _logger.warning(f"Could not resolve prim_path for scene asset '{name}': {asset.cfg.prim_path}")
            continue
        prefixes[name] = prim.GetPath().pathString
    return prefixes


def _match_scene_object(path_str: str, scene_prefixes: dict[str, str]) -> str | None:
    """Map a prim path to a scene asset name via longest matching ``prim_path`` prefix."""

    matched_name: str | None = None
    matched_len = -1
    for name, prefix in scene_prefixes.items():
        prefix_norm = prefix.rstrip("/")
        if path_str == prefix_norm or path_str.startswith(prefix_norm + "/"):
            if len(prefix_norm) > matched_len:
                matched_name = name
                matched_len = len(prefix_norm)
    return matched_name


def _partition_candidates_by_scene(
    candidate_paths: list[str], scene_prefixes: dict[str, str]
) -> tuple[dict[str, list[str]], list[str]]:
    """Split candidates into scene-corrected vs USD-xform groups (no prim appears in both)."""

    scene_paths: dict[str, list[str]] = defaultdict(list)
    usd_only_paths: list[str] = []
    for path_str in candidate_paths:
        obj_name = _match_scene_object(path_str, scene_prefixes)
        if obj_name is None:
            usd_only_paths.append(path_str)
        else:
            scene_paths[obj_name].append(path_str)
    return dict(scene_paths), usd_only_paths


def _prim_world_matrix_from_body_frame(
    prim: Usd.Prim,
    body_prim: Usd.Prim,
    body_pos_w: np.ndarray,
    body_quat_xyzw: np.ndarray,
    xform_cache: UsdGeom.XformCache,
) -> Gf.Matrix4d:
    """World transform: T_world_prim = T_world_body(scene) @ T_body_prim(usd_fixed)."""

    prim_usd = xform_cache.GetLocalToWorldTransform(prim)
    body_usd = xform_cache.GetLocalToWorldTransform(body_prim)
    # Rigid offset of collision prim in link/body frame; invariant to joint angle.
    prim_in_body = prim_usd * body_usd.GetInverse()

    return _matrix4d_from_pos_quat(body_pos_w, body_quat_xyzw) * prim_in_body


def _link_name_under_prefix(path_str: str, prefix_norm: str) -> str:
    """First path segment below the asset root — must match ``Articulation.body_names`` entry."""

    rel = path_str[len(prefix_norm) + 1 :]
    return rel.split("/")[0] if "/" in rel else ""


def _to_pose_numpy(pos_w: torch.Tensor, quat_w: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    return pos_w.detach().cpu().numpy(), quat_w.detach().cpu().numpy()


def _get_prim_world_matrix_for_scene_asset(
    env: ManagerBasedEnv,
    stage,
    obj_name: str,
    path_str: str,
    prim: Usd.Prim,
    asset_prefix: str,
    env_id: int,
    xform_cache: UsdGeom.XformCache,
) -> Gf.Matrix4d | None:
    """Build per-prim world matrix from ``env.scene``; returns ``None`` if mapping fails."""

    asset = env.scene[obj_name]
    prefix_norm = asset_prefix.rstrip("/")

    if isinstance(asset, Articulation):
        link_name = _link_name_under_prefix(path_str, prefix_norm)
        if not link_name or link_name not in asset.body_names:
            _logger.warning(f"Link '{link_name}' invalid for articulation '{obj_name}'; skipping '{path_str}'")
            return None
        body_prim = stage.GetPrimAtPath(f"{prefix_norm}/{link_name}")
        if not body_prim.IsValid():
            _logger.warning(f"USD link prim missing at '{prefix_norm}/{link_name}'")
            return None
        link_idx = asset.body_names.index(link_name)
        body_pos, body_quat = _to_pose_numpy(
            as_torch(asset.data.body_pos_w)[env_id, link_idx], as_torch(asset.data.body_quat_w)[env_id, link_idx]
        )
        return _prim_world_matrix_from_body_frame(prim, body_prim, body_pos, body_quat, xform_cache)

    if isinstance(asset, RigidObject):
        body_prim = stage.GetPrimAtPath(prefix_norm)
        if not body_prim.IsValid():
            _logger.warning(f"USD root prim missing at '{prefix_norm}' for rigid object '{obj_name}'")
            return None
        body_pos, body_quat = _to_pose_numpy(
            as_torch(asset.data.root_pos_w)[env_id], as_torch(asset.data.root_quat_w)[env_id]
        )
        return _prim_world_matrix_from_body_frame(prim, body_prim, body_pos, body_quat, xform_cache)

    _logger.debug(f"Scene asset '{obj_name}' is not Articulation/RigidObject; skipping '{path_str}'")
    return None


# -----------------------------------------------------------------------------
# Footprint generation (world XY convex polygons)
# -----------------------------------------------------------------------------


def _transform_points_local_to_world(points_local: np.ndarray, mat: Gf.Matrix4d) -> np.ndarray:
    """Transform Nx3 local points to world frame (USD row-vector convention)."""

    m = np.array(mat, dtype=np.float64)
    hom = np.concatenate([points_local, np.ones((points_local.shape[0], 1), dtype=np.float64)], axis=1)
    return hom @ m


def _prim_local_to_world_matrix(
    prim: Usd.Prim, xform_cache: UsdGeom.XformCache, world_mat: Gf.Matrix4d | None
) -> Gf.Matrix4d:
    # ``None`` → USD static xform; non-``None`` → scene-corrected matrix from link/root pose.
    return world_mat if world_mat is not None else xform_cache.GetLocalToWorldTransform(prim)


def _footprint_poly_xy_from_world_points(points_w: np.ndarray) -> np.ndarray | None:
    """XY convex hull of world-frame points; requires at least 3 distinct vertices."""

    poly = _convex_hull_2d(points_w[:, :2])
    return poly if poly.shape[0] >= 3 else None


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[idx]


def _local_bbox_corners(box_min, box_max) -> np.ndarray:
    return np.array(
        [
            [box_min[0], box_min[1], box_min[2]],
            [box_min[0], box_min[1], box_max[2]],
            [box_min[0], box_max[1], box_min[2]],
            [box_min[0], box_max[1], box_max[2]],
            [box_max[0], box_min[1], box_min[2]],
            [box_max[0], box_min[1], box_max[2]],
            [box_max[0], box_max[1], box_min[2]],
            [box_max[0], box_max[1], box_max[2]],
        ],
        dtype=np.float64,
    )


def _mesh_footprint_poly_xy(
    mesh_prim: Usd.Prim,
    xform_cache: UsdGeom.XformCache,
    mesh_max_points: int,
    world_mat: Gf.Matrix4d | None = None,
) -> np.ndarray | None:
    mesh = UsdGeom.Mesh(mesh_prim)
    pts = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
    if pts is None or len(pts) == 0:
        return None
    points_local = _downsample_points(np.asarray(pts, dtype=np.float64), mesh_max_points)
    mat = _prim_local_to_world_matrix(mesh_prim, xform_cache, world_mat)
    return _footprint_poly_xy_from_world_points(_transform_points_local_to_world(points_local, mat))


def _sample_circle_points(radius: float, num: int) -> np.ndarray:
    """Sample points on a circle."""

    angles = np.linspace(0.0, 2.0 * np.pi, num, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1)


def _footprint_from_local_points(
    prim: Usd.Prim,
    points_local: np.ndarray,
    xform_cache: UsdGeom.XformCache,
    world_mat: Gf.Matrix4d | None,
) -> np.ndarray | None:
    mat = _prim_local_to_world_matrix(prim, xform_cache, world_mat)
    return _footprint_poly_xy_from_world_points(_transform_points_local_to_world(points_local, mat))


def _cube_footprint_poly_xy(
    cube_prim: Usd.Prim, xform_cache: UsdGeom.XformCache, world_mat: Gf.Matrix4d | None = None
) -> np.ndarray | None:
    size = float(UsdGeom.Cube(cube_prim).GetSizeAttr().Get(Usd.TimeCode.Default()) or 0.0)
    if size <= 0.0:
        return None
    s = 0.5 * size
    corners_local = np.array([[-s, -s, 0.0], [-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0]], dtype=np.float64)
    return _footprint_from_local_points(cube_prim, corners_local, xform_cache, world_mat)


def _cylinder_like_footprint_poly_xy(
    prim: Usd.Prim,
    radius: float,
    xform_cache: UsdGeom.XformCache,
    world_mat: Gf.Matrix4d | None = None,
    num_circle_points: int = _NUM_CIRCLE_POINTS,
) -> np.ndarray | None:
    if radius <= 0.0:
        return None
    return _footprint_from_local_points(prim, _sample_circle_points(radius, num_circle_points), xform_cache, world_mat)


def _capsule_footprint_poly_xy(
    capsule_prim: Usd.Prim,
    xform_cache: UsdGeom.XformCache,
    world_mat: Gf.Matrix4d | None = None,
    num_circle_points: int = _NUM_CIRCLE_POINTS,
) -> np.ndarray | None:
    cap = UsdGeom.Capsule(capsule_prim)
    radius = float(cap.GetRadiusAttr().Get(Usd.TimeCode.Default()) or 0.0)
    height = float(cap.GetHeightAttr().Get(Usd.TimeCode.Default()) or 0.0)
    if radius <= 0.0:
        return None

    axis = str(cap.GetAxisAttr().Get(Usd.TimeCode.Default()) or "Z").upper()
    if axis == "Z":
        return _cylinder_like_footprint_poly_xy(capsule_prim, radius, xform_cache, world_mat, num_circle_points)

    half_len = 0.5 * max(0.0, height)
    angles = np.linspace(0.0, 2.0 * np.pi, num_circle_points, endpoint=False)
    circle = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    if axis == "X":
        c1, c2 = np.array([-half_len, 0.0]), np.array([half_len, 0.0])
    else:
        c1, c2 = np.array([0.0, -half_len]), np.array([0.0, half_len])
    pts2 = np.concatenate([circle + c1, circle + c2], axis=0)
    points_local = np.concatenate([pts2, np.zeros((pts2.shape[0], 1), dtype=np.float64)], axis=1)
    return _footprint_from_local_points(capsule_prim, points_local, xform_cache, world_mat)


def _fallback_bbox_footprint_poly_xy(
    prim: Usd.Prim,
    bbox_cache: UsdGeom.BBoxCache,
    xform_cache: UsdGeom.XformCache,
    world_mat: Gf.Matrix4d | None = None,
) -> np.ndarray | None:
    box = bbox_cache.ComputeLocalBound(prim).GetRange()
    if box.IsEmpty():
        return None
    return _footprint_from_local_points(prim, _local_bbox_corners(box.GetMin(), box.GetMax()), xform_cache, world_mat)


def _footprint_poly_for_prim(
    prim: Usd.Prim,
    ctx: _RasterizeContext,
    world_mat: Gf.Matrix4d | None = None,
) -> np.ndarray | None:
    if prim.IsA(UsdGeom.Mesh):
        return _mesh_footprint_poly_xy(prim, ctx.xform_cache, ctx.mesh_max_points, world_mat)
    if prim.IsA(UsdGeom.Cube):
        return _cube_footprint_poly_xy(prim, ctx.xform_cache, world_mat)
    if prim.IsA(UsdGeom.Cylinder):
        radius = float(UsdGeom.Cylinder(prim).GetRadiusAttr().Get(ctx.time_code) or 0.0)
        return _cylinder_like_footprint_poly_xy(prim, radius, ctx.xform_cache, world_mat)
    if prim.IsA(UsdGeom.Sphere):
        radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get(ctx.time_code) or 0.0)
        return _cylinder_like_footprint_poly_xy(prim, radius, ctx.xform_cache, world_mat)
    if prim.IsA(UsdGeom.Capsule):
        return _capsule_footprint_poly_xy(prim, ctx.xform_cache, world_mat)
    return _fallback_bbox_footprint_poly_xy(prim, ctx.bbox_cache, ctx.xform_cache, world_mat)


def _rasterize_prim(
    ctx: _RasterizeContext,
    prim: Usd.Prim,
    world_mat: Gf.Matrix4d | None = None,
) -> bool:
    poly = _footprint_poly_for_prim(prim, ctx, world_mat)
    if poly is None:
        return False
    _rasterize_convex_poly(
        ctx.occupancy_map, poly, ctx.map_min_x, ctx.map_min_y, ctx.cell_size, ctx.map_height, ctx.map_width
    )
    return True


def _rasterize_prim_paths(
    ctx: _RasterizeContext,
    path_strs: list[str],
    world_mat_for_path: dict[str, Gf.Matrix4d] | None = None,
) -> int:
    """Rasterize prims; return count successfully marked occupied.

    ``world_mat_for_path`` maps prim path → override world matrix (scene group). Missing entries use USD.
    """

    mats = world_mat_for_path or {}
    rasterized = 0
    for path_str in path_strs:
        prim = ctx.stage.GetPrimAtPath(path_str)
        if prim.IsValid() and _rasterize_prim(ctx, prim, mats.get(path_str)):
            rasterized += 1
    return rasterized


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
    floor_prim_path = f"{ENV_PRIM_PATH}/{cfg.floor_prim_suffix}"

    min_bound, max_bound = _get_prim_bounds(stage, floor_prim_path)

    world_extent_x = max_bound[0] - min_bound[0]
    world_extent_y = max_bound[1] - min_bound[1]
    if (
        not np.isfinite(world_extent_x)
        or not np.isfinite(world_extent_y)
        or world_extent_x > cfg.max_world_extent
        or world_extent_y > cfg.max_world_extent
        or world_extent_x <= 0
        or world_extent_y <= 0
    ):
        raise ValueError(f"Floor bounds invalid or too large: extent_x={world_extent_x}, extent_y={world_extent_y}")

    # Calculate map bounds (use floor bounds)
    map_min_x, map_max_x = min_bound[0], max_bound[0]
    map_min_y, map_max_y = min_bound[1], max_bound[1]

    map_width = int((map_max_x - map_min_x) / cfg.cell_size) + 1
    map_height = int((map_max_y - map_min_y) / cfg.cell_size) + 1

    # Grow cell size in-place if the grid would exceed ``max_map_size`` (mutates ``cfg.cell_size``).
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

    scene_prefixes = _build_scene_prim_prefixes(env)
    scene_paths, usd_only_paths = _partition_candidates_by_scene(candidate_paths, scene_prefixes)
    scene_prim_count = sum(len(paths) for paths in scene_paths.values())
    _logger.info(
        f"Partitioned candidates: {len(usd_only_paths)} USD-only, {scene_prim_count} scene-registered "
        f"across {len(scene_paths)} asset(s) {list(scene_paths.keys()) if scene_paths else []}"
    )

    time_code = Usd.TimeCode.Default()
    ctx = _RasterizeContext(
        stage=stage,
        occupancy_map=occupancy_map,
        xform_cache=UsdGeom.XformCache(time_code),
        bbox_cache=_make_bbox_cache(time_code),
        time_code=time_code,
        mesh_max_points=cfg.mesh_max_points,
        map_min_x=map_min_x,
        map_min_y=map_min_y,
        cell_size=cfg.cell_size,
        map_height=map_height,
        map_width=map_width,
    )

    # Static / non-scene obstacles: footprint pose from USD ``XformCache``.
    _rasterize_prim_paths(ctx, usd_only_paths)

    # Scene obstacles: same USD local geometry, world pose from simulation (doors, articulated fixtures).
    scene_world_mats: dict[str, Gf.Matrix4d] = {}
    for obj_name, paths in scene_paths.items():
        prefix_norm = scene_prefixes[obj_name].rstrip("/")
        for path_str in paths:
            prim = ctx.stage.GetPrimAtPath(path_str)
            if not prim.IsValid():
                continue
            world_mat = _get_prim_world_matrix_for_scene_asset(
                env, stage, obj_name, path_str, prim, prefix_norm, cfg.env_id, ctx.xform_cache
            )
            if world_mat is not None:
                scene_world_mats[path_str] = world_mat

    scene_rasterized = _rasterize_prim_paths(ctx, list(scene_world_mats.keys()), scene_world_mats)
    if scene_paths:
        _logger.info(f"Scene pose rasterization: {scene_rasterized}/{scene_prim_count} prims marked occupied")

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
