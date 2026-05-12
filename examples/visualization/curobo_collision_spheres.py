"""Visualize cuRobo robot self-collision spheres during pipeline execution.

This script runs an AutoSim pipeline and updates collision sphere + EE frame
visualization after every simulation step.

Usage
-----
Run with Isaac Sim UI enabled (do NOT use ``--headless``):

    python examples/visualization/curobo_collision_spheres.py --pipeline_id <PIPELINE_ID>

Defaults
--------
* Environment ID: 0
* Sphere color: green (0.2, 0.9, 0.2)
* Sphere opacity: 0.4
* EE frame scale: 0.1

Notes
-----
* Pipeline execution logic is inlined so visualization can hook into every step.
* Spheres with radius <= 0 are disabled placeholders and are skipped.
* VisualizationMarkers groups spheres by radius (one USD prototype per unique radius).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize cuRobo collision spheres during pipeline execution.")
parser.add_argument("--pipeline_id", type=str, required=True, help="Name of the autosim pipeline.")
parser.add_argument(
    "--curobo_link_name", type=str, default=None, help="cuRobo link name to query pose in robot-root frame."
)
parser.add_argument(
    "--isaaclab_link_name", type=str, default=None, help="Isaac Lab body name to query pose in robot-root frame."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

import autosim_examples  # noqa: F401
from autosim import make_pipeline
from autosim.core.registration import SkillRegistry


def _build_curobo_q(pipeline, env_id: int) -> torch.Tensor:
    """Build a joint position tensor in cuRobo's joint order from Isaac Lab state.

    Isaac Lab and cuRobo use different joint orderings. We look up each cuRobo
    joint by name in Isaac Lab's joint_names list and reorder accordingly.
    Joints not present in Isaac Lab (e.g. virtual base joints) are set to 0.
    """
    planner = pipeline._motion_planner
    robot = pipeline._robot

    isaaclab_names = list(robot.data.joint_names)
    isaaclab_q = robot.data.joint_pos[env_id]

    q = torch.zeros(len(planner.target_joint_names), dtype=isaaclab_q.dtype, device=isaaclab_q.device)
    for i, name in enumerate(planner.target_joint_names):
        if name in isaaclab_names:
            q[i] = isaaclab_q[isaaclab_names.index(name)]
        # joints missing from Isaac Lab (virtual base joints) stay at 0

    return planner._to_curobo_device(q)


def _get_spheres_world(pipeline, env_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions, radii) for all active collision spheres in world frame."""
    import isaaclab.utils.math as PoseUtils
    from curobo.types.state import JointState

    planner = pipeline._motion_planner
    robot = pipeline._robot

    q_curobo = _build_curobo_q(pipeline, env_id)
    js = JointState(position=q_curobo, joint_names=planner.target_joint_names)
    kin_state = planner.motion_gen.compute_kinematics(js)

    spheres_root = kin_state.robot_spheres[0].detach()  # [N, 4]

    root_pose = robot.data.root_pose_w[env_id].detach()
    robot_root_pos = root_pose[:3]
    robot_root_quat = root_pose[3:]  # wxyz

    device, dtype = root_pose.device, root_pose.dtype
    xyz = spheres_root[:, :3].to(device=device, dtype=dtype)
    radii_t = spheres_root[:, 3].to(device=device, dtype=dtype)

    n = xyz.shape[0]
    robot_root_pos_b = robot_root_pos.unsqueeze(0).expand(n, -1)
    robot_root_quat_b = robot_root_quat.unsqueeze(0).expand(n, -1)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).unsqueeze(0).expand(n, -1)

    centers_w, _ = PoseUtils.combine_frame_transforms(robot_root_pos_b, robot_root_quat_b, xyz, identity)

    mask = radii_t > 0.0
    positions = centers_w[mask].cpu().numpy()
    radii = radii_t[mask].cpu().numpy()
    return positions, radii


def _get_ee_pose_world(pipeline, env_id: int) -> torch.Tensor:
    """Return EE pose in world frame as [x, y, z, qw, qx, qy, qz] via cuRobo FK."""
    import isaaclab.utils.math as PoseUtils

    planner = pipeline._motion_planner
    robot = pipeline._robot

    q_curobo = _build_curobo_q(pipeline, env_id)
    ee_pose_root = planner.get_ee_pose(q_curobo)

    root_pose = robot.data.root_pose_w[env_id].detach()
    rr_pos = root_pose[:3].unsqueeze(0)
    rr_quat = root_pose[3:].unsqueeze(0)  # wxyz

    device, dtype = root_pose.device, root_pose.dtype
    ee_pos_root = ee_pose_root.position.view(1, 3).to(device=device, dtype=dtype)
    ee_quat_root = ee_pose_root.quaternion.view(1, 4).to(device=device, dtype=dtype)  # wxyz

    ee_pos_w, ee_quat_w = PoseUtils.combine_frame_transforms(rr_pos, rr_quat, ee_pos_root, ee_quat_root)
    return torch.cat([ee_pos_w, ee_quat_w], dim=-1).squeeze(0)  # [7]


def _create_ee_marker(scale: float) -> VisualizationMarkers:
    """Create a frame-axis marker for the EE pose."""
    cfg = FRAME_MARKER_CFG.copy()
    cfg.markers["frame"].scale = (scale, scale, scale)
    cfg = cfg.replace(prim_path="/World/debug/ee_frame")
    return VisualizationMarkers(cfg)


def _update_ee_marker(vm: VisualizationMarkers, pose_w: torch.Tensor) -> None:
    pos = pose_w[:3].unsqueeze(0)  # [1, 3]
    quat = pose_w[3:].unsqueeze(0)  # [1, 4] wxyz
    vm.visualize(translations=pos, orientations=quat, marker_indices=[0])


def _create_markers(unique_radii: np.ndarray, color: list[float], alpha: float) -> VisualizationMarkers:
    """Build a VisualizationMarkers with one sphere prototype per unique radius."""
    markers_cfg: dict[str, sim_utils.SphereCfg] = {}
    for i, r in enumerate(unique_radii):
        markers_cfg[f"sphere_{i}"] = sim_utils.SphereCfg(
            radius=float(r),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(color),
                opacity=alpha,
            ),
        )
    cfg = VisualizationMarkersCfg(prim_path="/World/debug/collision_spheres", markers=markers_cfg)
    return VisualizationMarkers(cfg)


def _update_markers(
    vm: VisualizationMarkers,
    positions: np.ndarray,
    radii: np.ndarray,
    unique_radii: np.ndarray,
) -> None:
    radius_to_idx = {float(r): i for i, r in enumerate(unique_radii)}
    marker_indices = np.array([radius_to_idx[float(r)] for r in radii], dtype=np.int32)
    translations = torch.from_numpy(positions).float()
    vm.visualize(translations=translations, marker_indices=marker_indices.tolist())


def _update_visualization(pipeline, env_id, vm_spheres, vm_ee, unique_radii):
    positions, radii = _get_spheres_world(pipeline, env_id)
    _update_markers(vm_spheres, positions, radii, unique_radii)
    _update_ee_marker(vm_ee, _get_ee_pose_world(pipeline, env_id))


def _print_link_pose_in_root_frame(
    pipeline, env_id: int, curobo_link_name: str | None, isaaclab_link_name: str | None
) -> None:
    """Print link pose in robot-root frame from cuRobo FK and/or Isaac Lab body_state_w.

    - cuRobo: FK directly gives the pose in robot-root frame.
    - Isaac Lab: body_state_w (world frame) minus root_pose_w via subtract_frame_transforms.
    """
    import isaaclab.utils.math as PoseUtils
    from curobo.types.state import JointState as CuroboJointState

    planner = pipeline._motion_planner
    robot = pipeline._robot

    # --- cuRobo ---
    if curobo_link_name is not None:
        q_curobo = _build_curobo_q(pipeline, env_id)
        js = CuroboJointState(position=q_curobo, joint_names=planner.target_joint_names)
        kin_state = planner.motion_gen.compute_kinematics(js)
        link_poses_curobo = kin_state.link_poses
        if curobo_link_name not in link_poses_curobo:
            print(f"[cuRobo:{curobo_link_name}] link not found. Available: {list(link_poses_curobo.keys())}")
        else:
            link_pose_root = link_poses_curobo[curobo_link_name]
            pos = link_pose_root.position.view(-1).detach().cpu()
            quat = link_pose_root.quaternion.view(-1).detach().cpu()  # wxyz
            print(f"[cuRobo:{curobo_link_name}] (root frame)  pos={pos.tolist()}  quat(wxyz)={quat.tolist()}")

    # --- Isaac Lab ---
    if isaaclab_link_name is not None:
        body_names = list(robot.data.body_names)
        if isaaclab_link_name not in body_names:
            print(f"[IsaacLab:{isaaclab_link_name}] link not found. Available: {body_names}")
        else:
            idx = body_names.index(isaaclab_link_name)
            body_state = robot.data.body_state_w[env_id, idx].detach()
            root_pos_w = robot.data.root_pos_w[env_id].detach()
            root_quat_w = robot.data.root_quat_w[env_id].detach()  # wxyz
            pos_il, quat_il = PoseUtils.subtract_frame_transforms(
                root_pos_w.unsqueeze(0),
                root_quat_w.unsqueeze(0),
                body_state[:3].unsqueeze(0),
                body_state[3:7].unsqueeze(0),
            )
            pos_il = pos_il.squeeze(0).cpu()
            quat_il = quat_il.squeeze(0).cpu()  # wxyz
            print(f"[IsaacLab:{isaaclab_link_name}] (root frame)  pos={pos_il.tolist()}  quat(wxyz)={quat_il.tolist()}")


def _execute_single_skill_with_viz(
    pipeline, skill, goal, vm_spheres, vm_ee, unique_radii, env_id, curobo_link_name=None, isaaclab_link_name=None
):
    """Inlined from AutoSimPipeline._execute_single_skill with per-step visualization."""
    world_state = pipeline._build_world_state()
    plan_success = skill.plan(world_state, goal)

    steps = 0
    while plan_success and steps < pipeline.cfg.max_steps:
        world_state = pipeline._build_world_state()
        output = skill.step(world_state)

        adapter_result = pipeline._action_adapter.apply(skill, output, pipeline._env)
        action = pipeline._last_action.clone()
        action[pipeline._env_id, : adapter_result.shape[0]] = adapter_result

        _, _, terminated, truncated, _ = pipeline._env.step(action)
        pipeline._last_action = action
        pipeline._generated_actions.append(action)

        pipeline._env.sim.render()
        _update_visualization(pipeline, env_id, vm_spheres, vm_ee, unique_radii)
        if curobo_link_name is not None or isaaclab_link_name is not None:
            _print_link_pose_in_root_frame(pipeline, env_id, curobo_link_name, isaaclab_link_name)

        steps += 1
        if bool((terminated[pipeline._env_id] | truncated[pipeline._env_id]).item()):
            return True, steps, True
        if output.done:
            return True, steps, False

    if steps >= pipeline.cfg.max_steps:
        world_state = pipeline._build_world_state()
        current_pos = world_state.robot_base_pose[:2]
        if goal.target_pose is not None:
            target_pos = goal.target_pose[:2]
            dist = float(torch.linalg.norm(current_pos - target_pos))
            pipeline._logger.warning(
                f"Max steps reached. Current pos: ({current_pos[0]:.3f}, {current_pos[1]:.3f}), "
                f"Target pos: ({target_pos[0]:.3f}, {target_pos[1]:.3f}), Distance: {dist:.3f}m"
            )

    return False, steps, False


def main():
    env_id = 0
    color = [0.2, 0.9, 0.2]
    alpha = 0.4
    ee_scale = 0.1

    pipeline = make_pipeline(args_cli.pipeline_id)
    pipeline.initialize()

    # Build markers using the initial robot pose (before reset)
    positions, radii = _get_spheres_world(pipeline, env_id)
    unique_radii = np.unique(radii)
    vm_spheres = _create_markers(unique_radii, color, alpha)
    vm_ee = _create_ee_marker(scale=ee_scale)

    # Decompose task
    decompose_result = pipeline.decompose()

    # Execute skill sequence with per-step visualization
    pipeline._check_skill_extra_cfg()
    pipeline.reset_env()
    _update_visualization(pipeline, env_id, vm_spheres, vm_ee, unique_radii)

    for subtask in decompose_result.subtasks:
        for skill_info in subtask.skills:
            skill = SkillRegistry.create(
                skill_info.skill_type, pipeline.cfg.skills.get(skill_info.skill_type).extra_cfg
            )

            if pipeline._action_adapter.should_skip_apply(skill):
                pipeline._logger.info(f"Skill {skill_info.skill_type} skipped.")
                continue

            goal = skill.extract_goal_from_info(skill_info, pipeline._env, pipeline._env_extra_info)
            success, steps, episode_done = _execute_single_skill_with_viz(
                pipeline,
                skill,
                goal,
                vm_spheres,
                vm_ee,
                unique_radii,
                env_id,
                curobo_link_name=args_cli.curobo_link_name,
                isaaclab_link_name=args_cli.isaaclab_link_name,
            )

            if not success:
                pipeline._logger.error(f"Skill {skill_info.skill_type} failed after {steps} steps.")
                raise ValueError(f"Skill {skill_info.skill_type} failed after {steps} steps.")
            if episode_done:
                pipeline._logger.info(f"Episode completed during skill {skill_info.skill_type}.({steps} steps)")
                pipeline._logger.info("Pipeline execution completed.")
                while simulation_app.is_running():
                    pipeline._env.sim.render()
                return
            pipeline._logger.info(f"Skill {skill_info.skill_type} done ({steps} steps).")

        pipeline._logger.info(f"Subtask {subtask.subtask_name} completed.")

    pipeline._logger.info("Pipeline execution completed.")

    while simulation_app.is_running():
        pipeline._env.sim.render()


if __name__ == "__main__":
    main()
    simulation_app.close()
