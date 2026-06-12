"""Visualize cuRobo world obstacles in the Isaac Sim viewport.

This script reads the obstacle data that cuRobo actually feeds into its SDF kernel
(``planner.motion_gen.scene_collision_checker.data``), i.e. the GPU tensors inside
``SceneData.cuboids`` / ``SceneData.meshes``. The CPU-side ``scene_model`` is only the
initial config snapshot — runtime updates from ``update_obstacle_pose`` /
``enable_obstacle`` go to the GPU tensors and never touch ``scene_model``, so reading
from ``scene_model`` would show stale poses and bogus enable flags.

Usage
-----
Run with Isaac Sim UI enabled (do NOT use ``--headless`` if you want to see the viewport):

    python examples/visualization/curobo_world_obstacles.py --pipeline_id <PIPELINE_ID>

Notes
-----
* In curobo v2 ``SceneData`` only stores ``cuboids``/``meshes``/``voxels``; spheres,
  cylinders and capsules from the original SceneCfg are converted to meshes (see
  ``SceneCfg.create_collision_support_world``).
* For meshes, ``MeshData.dims`` is only the local AABB extent — the AABB is **not**
  centered at the mesh's local origin (typical for articulated USD sub-prims whose
  origin is at the parent joint). We therefore pre-compute each mesh's tight OBB via
  :meth:`Mesh.get_cuboid` from ``scene_model`` once and compose it with the live mesh
  pose at draw time, so dynamic articulated obstacles render at their correct location.
* Wireframe drawing uses debug lines; keep the app running to inspect the scene.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize cuRobo collision world obstacles.")
parser.add_argument("--pipeline_id", type=str, default=None, help="Name of the autosim pipeline.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import isaaclab.utils.math as PoseUtils
from curobo.types import Pose as CuRoboPose

import autosim_examples  # noqa: F401
from autosim import make_pipeline
from autosim.utils.data_util import as_torch, convert_quat
from autosim.utils.debug_util import clear_debug_drawing, draw_line


@dataclass(frozen=True)
class _Pose7:
    pos: torch.Tensor  # (3,)
    quat: torch.Tensor  # (4,) wxyz


def _quat_to_rotmat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (wxyz) to rotation matrix. q shape (4,). Returns (3,3)."""
    q = q / (q.norm(p=2) + 1e-12)
    w, x, y, z = q
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    return torch.stack(
        [
            torch.stack([ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=0),
            torch.stack([2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)], dim=0),
            torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz], dim=0),
        ],
        dim=0,
    )


def _transform_points(pose: _Pose7, pts: torch.Tensor) -> torch.Tensor:
    """Apply pose (pos, quat) to points. pts shape (...,3) in local frame."""
    r = _quat_to_rotmat_wxyz(pose.quat)
    return (pts @ r.T) + pose.pos


def _draw_oriented_box(*, pose_w: _Pose7, half_dims_xyz: torch.Tensor, color, thickness: float, z_lift: float) -> None:
    hx, hy, hz = float(half_dims_xyz[0]), float(half_dims_xyz[1]), float(half_dims_xyz[2])
    corners_l = torch.tensor(
        [
            [-hx, -hy, -hz],
            [-hx, -hy, +hz],
            [-hx, +hy, -hz],
            [-hx, +hy, +hz],
            [+hx, -hy, -hz],
            [+hx, -hy, +hz],
            [+hx, +hy, -hz],
            [+hx, +hy, +hz],
        ],
        device=pose_w.pos.device,
        dtype=pose_w.pos.dtype,
    )
    corners_w = _transform_points(pose_w, corners_l).detach().cpu()
    if z_lift != 0.0:
        corners_w[:, 2] += float(z_lift)

    # 12 edges by index pairs
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for a, b in edges:
        pa = tuple(float(v) for v in corners_w[a].tolist())
        pb = tuple(float(v) for v in corners_w[b].tolist())
        draw_line(pa, pb, color=color, size=thickness)


def _compose_robot_to_world(*, robot_root_pose_w: torch.Tensor, pose_r: _Pose7) -> _Pose7:
    """Compose pose in robot root frame to world frame.

    Why we need this at all: cuRobo's "world" is the robot's root link, not the sim
    world. Every pose pulled out of ``data.cuboids`` / ``data.meshes`` is in robot-root
    frame, so we have to chain the live robot root pose on top before handing the box
    to ``draw_line``, which expects sim-world coordinates.
    """
    rr_pos_w = robot_root_pose_w[:3].view(1, 3)
    rr_quat_w = robot_root_pose_w[3:].view(1, 4)  # xyzw (IsaacLab v3.0)
    pos_r = pose_r.pos.view(1, 3)
    quat_r = convert_quat(pose_r.quat.view(1, 4), to="xyzw")  # cuRobo wxyz → xyzw
    pos_w, quat_w = PoseUtils.combine_frame_transforms(rr_pos_w, rr_quat_w, pos_r, quat_r)
    return _Pose7(pos=pos_w.view(3), quat=convert_quat(quat_w.view(4), to="wxyz"))  # xyzw → wxyz


def _forward_pose_from_inv(inv_pose7: torch.Tensor) -> CuRoboPose:
    """Recover the obstacle-in-world forward pose from cuRobo's stored ``inv_pose``.

    cuRobo stores ``inv_pose`` = ``w_obj_pose.inverse()``, where ``w_obj_pose`` is the
    obstacle's pose in cuRobo's world frame (i.e. the robot-root frame). We delegate to
    :class:`curobo.types.Pose` so the inversion uses the exact same Warp-backed math
    that cuRobo itself uses internally.
    """
    p_iw = inv_pose7[:3].contiguous().view(1, 3)
    q_iw = inv_pose7[3:7].contiguous().view(1, 4)  # wxyz
    return CuRoboPose(position=p_iw, quaternion=q_iw).inverse()


def _build_mesh_obb_cache(scene_model) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Compute, per mesh, the OBB offset in mesh-local frame and the OBB dims.

    Why: ``MeshData.dims`` is the AABB extent of the mesh vertices, but the AABB is
    **not** centered at the mesh's local origin — especially for articulated USD sub-
    primitives whose origin sits at the parent joint. Drawing a box at the live mesh
    pose with ``MeshData.dims`` therefore appears to be at the right size but shifted.

    We pre-compute the OBB pose of each mesh relative to the mesh's local origin using
    cuRobo's :meth:`Mesh.get_cuboid` (trimesh OBB), then at draw time compose the live
    mesh pose with this fixed offset to recover the live OBB pose.

    Returns:
        Dict mapping mesh name to ``(offset_pos_meshlocal, offset_quat_meshlocal_wxyz,
        dims_full_extents)``, all torch tensors on ``scene_model``'s device.
    """
    cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    meshes = getattr(scene_model, "mesh", None) or []
    for mesh in meshes:
        try:
            obb_cuboid = mesh.get_cuboid()
        except Exception as exc:  # pragma: no cover - logging only
            print(f"[curobo_world_obstacles] skipping mesh {mesh.name!r}: {exc}")
            continue

        # Key invariant: the OBB's pose RELATIVE TO the mesh's own local frame is a
        # constant of the mesh geometry — it doesn't change when the obstacle is moved,
        # rotated, or attached to an articulated joint. So we can compute it once from
        # the load-time poses (both ``mesh.pose`` and ``obb_cuboid.pose`` are in robot-
        # root frame at load time) and reuse it forever:
        #     offset_in_meshlocal = mesh.pose⁻¹  ·  obb.pose
        # At draw time we recover the live mesh-origin pose from cuRobo's ``inv_pose``
        # and compose with this cached offset to get the live OBB pose.
        base_pose = CuRoboPose.from_list(mesh.pose)
        obb_world_pose = CuRoboPose.from_list(obb_cuboid.pose)
        offset_pose = base_pose.inverse().multiply(obb_world_pose)
        dims = torch.as_tensor(obb_cuboid.dims, device=base_pose.position.device, dtype=base_pose.position.dtype)
        cache[mesh.name] = (
            offset_pose.position.view(3).detach(),
            offset_pose.quaternion.view(4).detach(),  # wxyz
            dims.view(3).detach(),
        )
    return cache


def _draw_cuboid_entries(
    *,
    storage,
    env_id: int,
    robot_root_pose_w: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    enabled_color,
    disabled_color,
    thickness: float,
    z_lift: float,
    show_disabled: bool,
) -> None:
    """Draw OBB wireframes from ``data.cuboids`` — pose + dims map directly to the box.

    Unlike meshes, a :class:`Cuboid` in cuRobo is an OBB by definition: its stored pose
    IS the box centre and ``dims`` are full extents around that centre. No mesh-local
    AABB offset to worry about, so we draw directly without an OBB cache.
    """
    # Iterate only the active slots; ``count`` is authoritative — slots beyond it carry
    # the create_cache defaults (zero pos, identity quat, small dims) that would render
    # as spurious tiny boxes at the origin if we walked the full ``max_n``.
    count = int(storage.count[env_id].item())
    for idx in range(count):
        is_enabled = int(storage.enable[env_id, idx].item()) == 1
        if not is_enabled and not show_disabled:
            continue
        obstacle_color = enabled_color if is_enabled else disabled_color

        # cuRobo layout: inv_pose row is [x, y, z, qw, qx, qy, qz, pad];
        # dims row is [x_extent, y_extent, z_extent, pad]. Slice off the trailing pad.
        inv_pose = storage.inv_pose[env_id, idx, :7].detach()
        fwd = _forward_pose_from_inv(inv_pose)
        pose_r = _Pose7(
            pos=fwd.position.view(3).to(device=device, dtype=dtype),
            quat=fwd.quaternion.view(4).to(device=device, dtype=dtype),
        )
        dims = storage.dims[env_id, idx, :3].detach().to(device=device, dtype=dtype)

        pose_w = _compose_robot_to_world(robot_root_pose_w=robot_root_pose_w, pose_r=pose_r)
        _draw_oriented_box(
            pose_w=pose_w,
            half_dims_xyz=dims * 0.5,
            color=obstacle_color,
            thickness=thickness,
            z_lift=z_lift,
        )


def _draw_mesh_entries(
    *,
    storage,
    obb_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    env_id: int,
    robot_root_pose_w: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    enabled_color,
    disabled_color,
    thickness: float,
    z_lift: float,
    show_disabled: bool,
) -> None:
    """Draw OBB wireframes from ``data.meshes``, applying the OBB-in-meshlocal offset.

    Composition (all in robot-root frame): ``live_obb_pose = live_mesh_pose * offset``.
    The ``offset`` is the cached, geometry-only OBB-relative-to-mesh-origin transform
    built once by :func:`_build_mesh_obb_cache`; ``live_mesh_pose`` is recovered from
    ``inv_pose`` at every call so dynamic and articulated meshes track correctly.
    """
    # See _draw_cuboid_entries: only iterate the active prefix [0, count).
    count = int(storage.count[env_id].item())
    for idx in range(count):
        is_enabled = int(storage.enable[env_id, idx].item()) == 1
        if not is_enabled and not show_disabled:
            continue
        obstacle_color = enabled_color if is_enabled else disabled_color

        # Stay on cuRobo's device for the multiply (CuRoboPose.multiply requires both
        # operands on the same device); only move to drawing device at the very end.
        inv_pose = storage.inv_pose[env_id, idx, :7].detach()
        curobo_device = inv_pose.device
        curobo_dtype = inv_pose.dtype
        mesh_pose_r = _forward_pose_from_inv(inv_pose)

        name = storage.names[env_id][idx]
        cached = obb_cache.get(name)
        if cached is None:
            # No precomputed OBB offset. Two ways this can happen:
            #   (a) the mesh appeared at runtime via ``update_from_warp_id`` (e.g. from
            #       a TSDF/depth-image pipeline) and was never in the load-time scene;
            #   (b) ``scene_model`` was None when the cache was built.
            # We fall back to drawing the AABB centred at the mesh origin — known to
            # be wrong for meshes whose origin is far from their geometric centre, but
            # at least keeps the visualization alive instead of skipping the obstacle.
            offset_pos = torch.zeros(3, device=curobo_device, dtype=curobo_dtype)
            offset_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=curobo_device, dtype=curobo_dtype)  # wxyz identity
            dims_tensor = storage.dims[env_id, idx, :3].detach()
        else:
            cached_pos, cached_quat, cached_dims = cached
            offset_pos = cached_pos.to(device=curobo_device, dtype=curobo_dtype)
            offset_quat = cached_quat.to(device=curobo_device, dtype=curobo_dtype)
            dims_tensor = cached_dims.to(device=curobo_device, dtype=curobo_dtype)

        offset_pose = CuRoboPose(
            position=offset_pos.view(1, 3),
            quaternion=offset_quat.view(1, 4),
        )
        obb_pose_r = mesh_pose_r.multiply(offset_pose)

        pose_r = _Pose7(
            pos=obb_pose_r.position.view(3).to(device=device, dtype=dtype),
            quat=obb_pose_r.quaternion.view(4).to(device=device, dtype=dtype),
        )
        dims = dims_tensor.to(device=device, dtype=dtype)

        pose_w = _compose_robot_to_world(robot_root_pose_w=robot_root_pose_w, pose_r=pose_r)
        _draw_oriented_box(
            pose_w=pose_w,
            half_dims_xyz=dims * 0.5,
            color=obstacle_color,
            thickness=thickness,
            z_lift=z_lift,
        )


def _visualize_world_obstacles(
    *,
    pipeline,
    env_id: int = 0,
    color=(0.95, 0.2, 0.25, 1.0),
    thickness: float = 2.0,
    z_lift: float = 0.0,
    show_disabled: bool = True,
):
    """Visualize world obstacles from cuRobo's live GPU collision data.

    Args:
        pipeline: AutoSim pipeline instance
        env_id: Environment index
        color: RGBA color for enabled obstacles
        thickness: Line thickness
        z_lift: Z-axis offset for visualization
        show_disabled: If True, show disabled obstacles in cyan (default: True)
    """
    planner = pipeline._motion_planner
    # Push the latest IsaacLab state (rigid + articulated dynamic obstacles) into
    # ``data.cuboids.inv_pose`` / ``data.meshes.inv_pose`` BEFORE we read those tensors.
    # Without this call we'd be visualising whatever state was last left in GPU memory,
    # which for articulated joints can be many frames stale.
    planner._refine_curobo_world_collision()

    checker = planner.motion_gen.scene_collision_checker
    data = checker.data
    if data is None:
        raise RuntimeError("cuRobo collision checker has no SceneData loaded.")

    robot_root_pose_w = as_torch(pipeline._robot.data.root_pose_w)[env_id].detach()
    device = robot_root_pose_w.device
    dtype = robot_root_pose_w.dtype

    clear_debug_drawing()

    # Disabled obstacles still exist in cuRobo's tensors but are skipped by the SDF
    # kernel (``is_obs_enabled`` returns False). We still draw them by default so the
    # user can see WHY a planning call ignored a particular obstacle — e.g. when the
    # planner disabled all rigid objects during a lift skill. Cyan is chosen as the
    # complement of the enabled-red so the two states are unambiguous on screen.
    disabled_color = (0.2, 0.8, 0.8, 0.35)

    if data.cuboids is not None:
        _draw_cuboid_entries(
            storage=data.cuboids,
            env_id=env_id,
            robot_root_pose_w=robot_root_pose_w,
            device=device,
            dtype=dtype,
            enabled_color=color,
            disabled_color=disabled_color,
            thickness=thickness,
            z_lift=z_lift,
            show_disabled=show_disabled,
        )

    # ``data.meshes`` includes spheres/cylinders/capsules that ``create_collision_support_world``
    # converts to meshes during ``update_world``. The OBB cache is built lazily here (not at
    # planner init) so we always reflect the latest scene_model — e.g. after future calls to
    # ``update_world`` that swap the loaded geometry.
    if data.meshes is not None:
        obb_cache = _build_mesh_obb_cache(checker.scene_model) if checker.scene_model is not None else {}
        _draw_mesh_entries(
            storage=data.meshes,
            obb_cache=obb_cache,
            env_id=env_id,
            robot_root_pose_w=robot_root_pose_w,
            device=device,
            dtype=dtype,
            enabled_color=color,
            disabled_color=disabled_color,
            thickness=thickness,
            z_lift=z_lift,
            show_disabled=show_disabled,
        )


def main():
    pipeline = make_pipeline(args_cli.pipeline_id)
    pipeline.initialize()
    pipeline.reset_env()

    _visualize_world_obstacles(pipeline=pipeline)

    while simulation_app.is_running():
        pipeline._env.sim.render()


if __name__ == "__main__":
    main()
    simulation_app.close()
