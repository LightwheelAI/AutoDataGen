"""Visualize cuRobo robot self-collision spheres during pipeline execution.

This script runs an AutoSim pipeline and updates collision sphere + EE frame
visualization after every simulation step using debug draw lines (no USD prims created).

Usage
-----
Run with Isaac Sim UI enabled (do NOT use ``--headless``):

    python examples/visualization/curobo_collision_spheres.py --pipeline_id <PIPELINE_ID>

To debug reach planning without executing the planned action trajectory:

    python examples/visualization/curobo_collision_spheres.py \
        --pipeline_id <PIPELINE_ID> \
        --hold_on_reach_plan

Defaults
--------
* Environment ID: 0
* Sphere color: green (0.2, 0.9, 0.2, 0.8)
* EE frame scale: 0.1

Notes
-----
* Pipeline execution logic is inlined so visualization can hook into every step.
* Spheres with radius <= 0 are disabled placeholders and are skipped.
* Each sphere is drawn as a 3-axis cross (X/Y/Z lines) scaled by sphere radius.
* EE frame is drawn as three RGB axis lines (R=X, G=Y, B=Z).
* ``--hold_on_reach_plan`` intercepts only reach skills after planning succeeds:
  it draws the full planned end-effector path, updates debug lines for inspection,
  renders the UI for ``--reach_plan_hold_seconds``, then continues normal execution.
"""

from __future__ import annotations

import argparse
import traceback

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
parser.add_argument(
    "--hold_on_reach_plan",
    action="store_true",
    help="For reach skills, visualize the full planned trajectory before executing it.",
)
parser.add_argument(
    "--reach_plan_hold_seconds",
    type=float,
    default=10.0,
    help="Seconds to render the planned reach trajectory before continuing execution.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app


import autosim_examples  # noqa: F401
from autosim import make_pipeline
from autosim.core.registration import SkillRegistry
from autosim.utils.data_util import as_torch, convert_quat
from autosim.utils.debug_util import clear_debug_drawing, draw_line


def _build_curobo_q(pipeline, env_id: int) -> torch.Tensor:
    """Build a joint position tensor in cuRobo's joint order from Isaac Lab state.

    Isaac Lab and cuRobo use different joint orderings. We look up each cuRobo
    joint by name in Isaac Lab's joint_names list and reorder accordingly.
    Joints not present in Isaac Lab (e.g. virtual base joints) are set to 0.
    """
    planner = pipeline._motion_planner
    robot = pipeline._robot

    isaaclab_names = list(robot.data.joint_names)
    isaaclab_q = as_torch(robot.data.joint_pos)[env_id]

    q = torch.zeros(len(planner.target_joint_names), dtype=isaaclab_q.dtype, device=isaaclab_q.device)
    for i, name in enumerate(planner.target_joint_names):
        if name in isaaclab_names:
            q[i] = isaaclab_q[isaaclab_names.index(name)]
        # joints missing from Isaac Lab (virtual base joints) stay at 0

    return planner._to_curobo_device(q)


def _get_spheres_world(pipeline, env_id: int, q_curobo: torch.Tensor | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions, radii) for all active collision spheres in world frame."""
    import isaaclab.utils.math as PoseUtils
    from curobo.types.state import JointState

    planner = pipeline._motion_planner
    robot = pipeline._robot

    if q_curobo is None:
        q_curobo = _build_curobo_q(pipeline, env_id)
    else:
        q_curobo = planner._to_curobo_device(q_curobo)
    js = JointState(position=q_curobo, joint_names=planner.target_joint_names)
    kin_state = planner.motion_gen.compute_kinematics(js)

    spheres_root = kin_state.robot_spheres[0].detach()  # [N, 4]

    root_pose = as_torch(robot.data.root_pose_w)[env_id].detach()
    robot_root_pos = root_pose[:3]
    robot_root_quat = root_pose[3:]  # xyzw (IsaacLab v3.0+)

    device, dtype = root_pose.device, root_pose.dtype
    xyz = spheres_root[:, :3].to(device=device, dtype=dtype)
    radii_t = spheres_root[:, 3].to(device=device, dtype=dtype)

    n = xyz.shape[0]
    robot_root_pos_b = robot_root_pos.unsqueeze(0).expand(n, -1)
    robot_root_quat_b = robot_root_quat.unsqueeze(0).expand(n, -1)
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype).unsqueeze(0).expand(n, -1)

    centers_w, _ = PoseUtils.combine_frame_transforms(robot_root_pos_b, robot_root_quat_b, xyz, identity)

    mask = radii_t > 0.0
    positions = centers_w[mask].cpu().numpy()
    radii = radii_t[mask].cpu().numpy()
    return positions, radii


def _get_ee_pose_world(pipeline, env_id: int, q_curobo: torch.Tensor | None = None) -> torch.Tensor:
    """Return EE pose in world frame as [x, y, z, qx, qy, qz, qw] via cuRobo FK."""
    import isaaclab.utils.math as PoseUtils

    planner = pipeline._motion_planner
    robot = pipeline._robot

    if q_curobo is None:
        q_curobo = _build_curobo_q(pipeline, env_id)
    else:
        q_curobo = planner._to_curobo_device(q_curobo)
    ee_pose_root = planner.get_ee_pose(q_curobo)

    root_pose = as_torch(robot.data.root_pose_w)[env_id].detach()
    rr_pos = root_pose[:3].unsqueeze(0)
    rr_quat = root_pose[3:].unsqueeze(0)  # xyzw

    device, dtype = root_pose.device, root_pose.dtype
    ee_pos_root = ee_pose_root.position.view(1, 3).to(device=device, dtype=dtype)
    ee_quat_root = convert_quat(ee_pose_root.quaternion.view(1, 4), to="xyzw").to(
        device=device, dtype=dtype
    )  # cuRobo wxyz → xyzw

    ee_pos_w, ee_quat_w = PoseUtils.combine_frame_transforms(rr_pos, rr_quat, ee_pos_root, ee_quat_root)
    return torch.cat([ee_pos_w, ee_quat_w], dim=-1).squeeze(0)


def _draw_ee_frame(pose_w: torch.Tensor, scale: float = 0.1) -> None:
    """Draw EE frame axes using debug lines (R=X, G=Y, B=Z)."""
    import isaaclab.utils.math as PoseUtils

    pos = pose_w[:3]
    quat = pose_w[3:]  # xyzw
    rot = PoseUtils.matrix_from_quat(quat.unsqueeze(0)).squeeze(0)  # (3, 3)

    origin = tuple(pos.cpu().tolist())
    for axis_idx, color in enumerate([(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]):
        axis_dir = rot[:, axis_idx] * scale
        tip = tuple((pos + axis_dir).cpu().tolist())
        draw_line(origin, tip, color=color, size=3.0)


_CIRCLE_PTS = 16  # segments per great-circle arc
_CIRCLE_ANGLES = np.linspace(0, 2 * np.pi, _CIRCLE_PTS + 1)
_COS = np.cos(_CIRCLE_ANGLES).astype(np.float32)
_SIN = np.sin(_CIRCLE_ANGLES).astype(np.float32)


def _draw_sphere_wireframe(cx: float, cy: float, cz: float, r: float, color: tuple, size: float = 2.0) -> None:
    """Draw a sphere as three orthogonal great circles (XY, YZ, XZ planes)."""
    # XY plane
    for j in range(_CIRCLE_PTS):
        draw_line(
            (cx + r * _COS[j], cy + r * _SIN[j], cz),
            (cx + r * _COS[j + 1], cy + r * _SIN[j + 1], cz),
            color=color,
            size=size,
        )
    # YZ plane
    for j in range(_CIRCLE_PTS):
        draw_line(
            (cx, cy + r * _COS[j], cz + r * _SIN[j]),
            (cx, cy + r * _COS[j + 1], cz + r * _SIN[j + 1]),
            color=color,
            size=size,
        )
    # XZ plane
    for j in range(_CIRCLE_PTS):
        draw_line(
            (cx + r * _COS[j], cy, cz + r * _SIN[j]),
            (cx + r * _COS[j + 1], cy, cz + r * _SIN[j + 1]),
            color=color,
            size=size,
        )


def _draw_spheres(positions: np.ndarray, radii: np.ndarray, color: tuple = (0.2, 0.9, 0.2, 0.8)) -> None:
    """Draw collision spheres as three orthogonal wireframe circles."""
    for i in range(len(positions)):
        _draw_sphere_wireframe(
            float(positions[i, 0]),
            float(positions[i, 1]),
            float(positions[i, 2]),
            float(radii[i]),
            color,
        )


def _update_visualization(pipeline, env_id, color: tuple = (0.2, 0.9, 0.2, 0.8), ee_scale: float = 0.1):
    clear_debug_drawing()
    positions, radii = _get_spheres_world(pipeline, env_id)
    _draw_spheres(positions, radii, color=color)
    _draw_ee_frame(_get_ee_pose_world(pipeline, env_id), scale=ee_scale)


def _draw_trajectory_path(pipeline, env_id: int, trajectory, stride: int = 1) -> list[torch.Tensor]:
    """Draw the planned reach end-effector path in the Isaac viewport.

    ``trajectory`` is the full cuRobo joint-space plan returned by ``ReachSkill.plan``.
    Each waypoint is converted back through cuRobo FK, then transformed by the current
    robot root pose so the debug lines are drawn in Isaac world coordinates.
    """
    poses_w = []
    positions = trajectory.position
    for i in range(0, len(positions), max(stride, 1)):
        poses_w.append(_get_ee_pose_world(pipeline, env_id, positions[i]))
    if len(positions) > 0 and (len(positions) - 1) % max(stride, 1) != 0:
        poses_w.append(_get_ee_pose_world(pipeline, env_id, positions[-1]))

    for i in range(len(poses_w) - 1):
        start = tuple(poses_w[i][:3].detach().cpu().tolist())
        end = tuple(poses_w[i + 1][:3].detach().cpu().tolist())
        draw_line(start, end, color=(0.0, 0.8, 1.0, 1.0), size=4.0)
    if poses_w:
        start = tuple(poses_w[0][:3].detach().cpu().tolist())
        goal = tuple(poses_w[-1][:3].detach().cpu().tolist())
        draw_line(
            (start[0] - 0.04, start[1], start[2]),
            (start[0] + 0.04, start[1], start[2]),
            color=(0.0, 1.0, 0.0, 1.0),
            size=6.0,
        )
        draw_line(
            (goal[0] - 0.04, goal[1], goal[2]), (goal[0] + 0.04, goal[1], goal[2]), color=(1.0, 0.0, 0.0, 1.0), size=6.0
        )
    return poses_w


def _visualize_planned_reach(pipeline, env_id, trajectory, duration_s: float):
    """Draw a reach plan, render briefly, then return to normal execution.

    This is a pre-execution debugging path: after the full reach plan is available,
    we pause the skill step loop for a bounded wall-clock duration. During this pause
    the script renders only; it does not call ``skill.step`` or ``env.step``.
    """
    clear_debug_drawing()
    _draw_trajectory_path(pipeline, env_id, trajectory)
    if len(trajectory.position) > 0:
        positions, radii = _get_spheres_world(pipeline, env_id, trajectory.position[0])
        _draw_spheres(positions, radii)
        _draw_ee_frame(_get_ee_pose_world(pipeline, env_id, trajectory.position[-1]))
    print(
        f"Planned reach trajectory with {len(trajectory.position)} waypoints. "
        f"Rendering for {duration_s:.1f}s before execution."
    )
    if duration_s > 0.0:
        import time

        end_time = time.monotonic() + duration_s
        while simulation_app.is_running() and time.monotonic() < end_time:
            pipeline._env.sim.render()


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
            body_state = as_torch(robot.data.body_state_w)[env_id, idx].detach()
            root_pos_w = as_torch(robot.data.root_pos_w)[env_id].detach()
            root_quat_w = as_torch(robot.data.root_quat_w)[env_id].detach()  # xyzw
            pos_il, quat_il = PoseUtils.subtract_frame_transforms(
                root_pos_w.unsqueeze(0),
                root_quat_w.unsqueeze(0),
                body_state[:3].unsqueeze(0),
                body_state[3:7].unsqueeze(0),
            )
            pos_il = pos_il.squeeze(0).cpu()
            quat_il = quat_il.squeeze(0).cpu()  # xyzw
            print(f"[IsaacLab:{isaaclab_link_name}] (root frame)  pos={pos_il.tolist()}  quat(xyzw)={quat_il.tolist()}")


def _execute_single_skill_with_viz(pipeline, skill, goal, env_id, curobo_link_name=None, isaaclab_link_name=None):
    """Execute one skill with visualization hooks.

    Normal mode updates debug-line sphere + EE visualization after each simulation step.
    With ``--hold_on_reach_plan``, reach skills pause briefly after successful planning
    so the planned trajectory can be inspected before execution.
    """
    world_state = pipeline._build_world_state()
    plan_success = skill.plan(world_state, goal)

    if (
        args_cli.hold_on_reach_plan
        and plan_success
        and skill.get_cfg().name == "reach"
        and getattr(skill, "_trajectory", None) is not None
    ):
        _visualize_planned_reach(
            pipeline,
            env_id,
            skill._trajectory,
            args_cli.reach_plan_hold_seconds,
        )

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
        _update_visualization(pipeline, env_id)
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

    pipeline = make_pipeline(args_cli.pipeline_id)
    pipeline.initialize()

    # Decompose the task before reset, matching the normal AutoSimPipeline.run() order.
    decompose_result = pipeline.decompose()

    # After reset, execute the decomposed skills with this script's visualization hooks.
    pipeline._check_skill_extra_cfg()
    pipeline.reset_env()
    _update_visualization(pipeline, env_id)

    try:
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
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        while simulation_app.is_running():
            pipeline._env.sim.render()

    while simulation_app.is_running():
        pipeline._env.sim.render()


if __name__ == "__main__":
    main()
    simulation_app.close()
