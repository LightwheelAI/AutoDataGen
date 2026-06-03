from __future__ import annotations

import math
import re

import isaaclab.utils.math as PoseUtils
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

from autosim import register_skill
from autosim.core.logger import AutoSimLogger
from autosim.core.skill import SkillCfg
from autosim.core.types import (
    EnvExtraInfo,
    SkillGoal,
    SkillInfo,
    SkillOutput,
    WorldState,
)

from .base_skill import CuroboSkillExtraCfg
from .reach import ReachSkill

_AXIS_MAP = {
    "x": torch.tensor([1.0, 0.0, 0.0]),
    "y": torch.tensor([0.0, 1.0, 0.0]),
    "z": torch.tensor([0.0, 0.0, 1.0]),
}


def _parse_axis_vector(rotate_axis: str) -> torch.Tensor:
    """Parse a '+x'/'-z'/'+x+y' axis string into a normalized direction vector."""

    matches = re.findall(r"([+-][xyz])", rotate_axis)
    if not matches:
        raise ValueError(f"Invalid rotate_axis format: '{rotate_axis}'. Expected e.g. '+x', '-z', '+x+y'.")

    vec = torch.zeros(3)
    for match in matches:
        sign = 1.0 if match[0] == "+" else -1.0
        vec += sign * _AXIS_MAP[match[1]]

    norm = torch.linalg.norm(vec)
    if norm < 1e-6:
        raise ValueError(f"rotate_axis '{rotate_axis}' results in a zero vector.")
    return vec / norm


def _axis_angle_to_quat(axis: torch.Tensor, angle: float) -> torch.Tensor:
    """Convert axis-angle to quaternion [qw, qx, qy, qz]."""

    half = torch.as_tensor(angle / 2.0, device=axis.device, dtype=axis.dtype)
    s = torch.sin(half)
    w = torch.cos(half)
    return torch.cat([w.unsqueeze(0), axis * s])


@configclass
class RotateSkillExtraCfg(CuroboSkillExtraCfg):
    """Extra configuration for the rotate skill."""

    rotate_axis: str = "+z"
    """Rotation axis expressed in rotate_frame. Same format as move_axis: '+x', '-y', '+x+z', etc."""

    rotate_angle: float = math.pi / 2
    """Rotation angle in radians (default ~90 degrees)."""

    rotate_frame: str = "ee"
    """Frame in which rotate_axis is expressed: 'ee' (end-effector) or 'object' (target object)."""

    rotate_steps: int = 1
    """Number of incremental planning steps to split the total rotation into.
    More steps reduce EE position drift at the cost of more planning calls."""

    def __post_init__(self) -> None:
        super().__post_init__()
        supported_frames = {"ee", "object"}
        if self.rotate_frame not in supported_frames:
            raise ValueError(f"Unsupported rotate_frame: '{self.rotate_frame}'. Supported: {sorted(supported_frames)}")
        _parse_axis_vector(self.rotate_axis)  # validate axis string at config time


@configclass
class RotateSkillCfg(SkillCfg):
    """Configuration for the rotate skill."""

    extra_cfg: RotateSkillExtraCfg = RotateSkillExtraCfg()


@register_skill(
    name="rotate",
    cfg_type=RotateSkillCfg,
    description="Rotate end-effector in place around an axis (e.g. to turn a microwave knob)",
)
class RotateSkill(ReachSkill):
    """Skill to rotate the end-effector in place around a given axis."""

    def __init__(self, extra_cfg: RotateSkillExtraCfg) -> None:
        super().__init__(extra_cfg)
        self._logger = AutoSimLogger("RotateSkill")

    def extract_goal_from_info(
        self, skill_info: SkillInfo, env: ManagerBasedEnv, env_extra_info: EnvExtraInfo
    ) -> SkillGoal:
        """Pre-compute the rotation axis in robot root frame when using object frame."""

        if self.cfg.extra_cfg.rotate_frame == "object":
            axis_local = _parse_axis_vector(self.cfg.extra_cfg.rotate_axis).to(env.device)

            obj_pose_w = env.scene[skill_info.target_object].data.root_pose_w[0]  # [7]
            obj_quat_w = obj_pose_w[3:].unsqueeze(0)  # [1, 4]

            robot = env.scene[env_extra_info.robot_name]
            robot_quat_w = robot.data.root_pose_w[0, 3:].unsqueeze(0)  # [1, 4]

            # object frame -> world frame -> robot root frame
            axis_in_world = PoseUtils.quat_apply(obj_quat_w, axis_local.unsqueeze(0)).squeeze(0)
            axis_in_root = PoseUtils.quat_apply(PoseUtils.quat_inv(robot_quat_w), axis_in_world.unsqueeze(0)).squeeze(
                0
            )  # [3]
            return SkillGoal(target_object=skill_info.target_object, info=dict(axis_in_root=axis_in_root))

        return SkillGoal(target_object=skill_info.target_object)

    def execute_plan(self, state: WorldState, goal: SkillGoal) -> bool:
        """Plan a motion that rotates the EE by rotate_angle around rotate_axis, keeping position fixed."""

        self._planner.set_target_object(None)

        activate_q, activate_qd = self._build_activate_joint_state(
            state.sim_joint_names, state.robot_joint_pos, state.robot_joint_vel
        )
        if activate_qd is None:
            raise ValueError("activate_qd should not be None when planning rotate trajectories.")

        steps = max(1, self.cfg.extra_cfg.rotate_steps)
        step_angle = self.cfg.extra_cfg.rotate_angle / steps

        # For object frame, axis_in_root is fixed (pre-computed from object pose).
        # For ee frame, we recompute axis_in_root at each step from the updated EE orientation.
        object_axis_in_root = None
        if self.cfg.extra_cfg.rotate_frame == "object":
            if "axis_in_root" not in goal.info:
                raise ValueError("Rotate goal is missing axis_in_root. Call extract_goal_from_info before planning.")
            object_axis_in_root = goal.info["axis_in_root"]

        trajectories = []
        current_q = activate_q
        current_qd = activate_qd

        for i in range(steps):
            ee_pose = self._planner.get_ee_pose(current_q)
            ee_pos = ee_pose.position.clone()
            ee_quat = ee_pose.quaternion.clone()

            if self.cfg.extra_cfg.rotate_frame == "ee":
                axis_local = _parse_axis_vector(self.cfg.extra_cfg.rotate_axis).to(
                    device=ee_quat.device, dtype=ee_quat.dtype
                )
                axis_in_root = PoseUtils.quat_apply(ee_quat, axis_local.unsqueeze(0)).squeeze(0)
            else:
                axis_in_root = object_axis_in_root.to(device=ee_quat.device, dtype=ee_quat.dtype)

            delta_quat = _axis_angle_to_quat(axis_in_root, step_angle)
            target_quat = PoseUtils.quat_mul(delta_quat.unsqueeze(0), ee_quat).squeeze(0)
            target_pos = ee_pos.squeeze(0)

            # keep visualizer happy — store final target pose (last step overwrites, which is fine)
            self._target_poses["target_pose"] = torch.cat([target_pos.unsqueeze(0), target_quat.unsqueeze(0)], dim=-1)

            self._logger.info(
                f"Rotate step {i+1}/{steps} ({self.cfg.extra_cfg.rotate_frame} frame): "
                f"axis_in_root={axis_in_root}, step_angle={step_angle:.4f}"
            )

            traj = self._planner.plan_motion(target_pos, target_quat, current_q, current_qd)
            if traj is None:
                self._logger.warning(f"Rotate planning failed at step {i+1}/{steps}")
                if not trajectories:
                    self._trajectory = None
                    return False
                break

            trajectories.append(traj)
            # Use last waypoint as start of next step
            current_q = traj.position[-1]
            current_qd = traj.velocity[-1] if traj.velocity is not None else torch.zeros_like(current_q)

        if not trajectories:
            self._trajectory = None
            return False

        if len(trajectories) == 1:
            self._trajectory = trajectories[0]
        else:
            # Concatenate all trajectory segments along the time axis
            import dataclasses

            combined_pos = torch.cat([t.position for t in trajectories], dim=0)
            combined_vel = (
                torch.cat([t.velocity for t in trajectories], dim=0) if trajectories[0].velocity is not None else None
            )
            combined_acc = (
                torch.cat([t.acceleration for t in trajectories], dim=0)
                if trajectories[0].acceleration is not None
                else None
            )
            combined_jerk = (
                torch.cat([t.jerk for t in trajectories], dim=0) if trajectories[0].jerk is not None else None
            )
            self._trajectory = dataclasses.replace(
                trajectories[0],
                position=combined_pos,
                velocity=combined_vel,
                acceleration=combined_acc,
                jerk=combined_jerk,
            )

        return True

    def step(self, state: WorldState) -> SkillOutput:
        return super().step(state)

    def reset(self) -> None:
        super().reset()
