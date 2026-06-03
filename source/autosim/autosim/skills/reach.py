from __future__ import annotations

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
from autosim.utils.data_util import as_torch, convert_quat

from .base_skill import CuroboSkillBase, CuroboSkillExtraCfg


@configclass
class ReachSkillExtraCfg(CuroboSkillExtraCfg):
    """Extra configuration for the reach skill."""

    corrective_reach: bool = False
    """Whether to perform corrective reach."""


@configclass
class ReachSkillCfg(SkillCfg):
    """Configuration for the reach skill."""

    extra_cfg: ReachSkillExtraCfg = ReachSkillExtraCfg()
    """Extra configuration for the reach skill."""


@register_skill(
    name="reach",
    cfg_type=ReachSkillCfg,
    description="Extend robot arm to target position (for approaching objects or placement locations)",
)
class ReachSkill(CuroboSkillBase):
    """Skill to reach to a target object or location"""

    def __init__(self, extra_cfg: ReachSkillExtraCfg) -> None:
        super().__init__(extra_cfg)

        self._logger = AutoSimLogger("ReachSkill")

        # variables for the skill execution
        self._trajectory = None
        self._step_idx = 0

        self._corrective_reach_done = False
        self._saved_env = None
        self._saved_target_object = None
        self._saved_reach_offset = None
        self._saved_env_extra_info = None

    def _get_current_primary_and_extra_link_poses(
        self, activate_q: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """Get current primary and extra link poses in robot root frame."""

        current_link_poses = self._planner.get_link_poses(
            activate_q, [self._planner.motion_gen.kinematics.ee_link, *self.cfg.extra_cfg.extra_target_link_names]
        )
        primary_pose_in_robot_root = current_link_poses[self._planner.motion_gen.kinematics.ee_link]
        primary_link_pose_in_robot_root = (
            primary_pose_in_robot_root.position,
            convert_quat(primary_pose_in_robot_root.quaternion, to="xyzw"),  # cuRobo wxyz → xyzw
        )
        extra_link_poses_in_robot_root = {
            link_name: (pose.position, convert_quat(pose.quaternion, to="xyzw"))  # cuRobo wxyz → xyzw
            for link_name, pose in current_link_poses.items()
            if link_name != self._planner.motion_gen.kinematics.ee_link
        }
        return primary_link_pose_in_robot_root, extra_link_poses_in_robot_root

    def _compute_relative_extra_target_poses(
        self,
        primary_link_pose_in_robot_root: tuple[torch.Tensor, torch.Tensor],
        extra_link_offsets_in_primary: dict[str, tuple[torch.Tensor, torch.Tensor]],
        target_pose: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Project cached or current primary-frame offsets onto the target primary pose."""

        primary_target_pos_in_robot_root = target_pose[:3].unsqueeze(0).to(primary_link_pose_in_robot_root[0].device)
        primary_target_quat_in_robot_root = target_pose[3:].unsqueeze(0).to(primary_link_pose_in_robot_root[1].device)

        extra_target_poses = {}
        for link_name, (link_pos_in_primary, link_quat_in_primary) in extra_link_offsets_in_primary.items():
            link_target_pos_in_robot_root, link_target_quat_in_robot_root = PoseUtils.combine_frame_transforms(
                primary_target_pos_in_robot_root,
                primary_target_quat_in_robot_root,
                link_pos_in_primary,
                link_quat_in_primary,
            )
            self._logger.debug(
                f"Relative offset for {link_name} in primary frame: pos={link_pos_in_primary},"
                f" quat={link_quat_in_primary}"
            )
            self._logger.debug(
                f"Target pose for {link_name} in robot root frame: pos={link_target_pos_in_robot_root},"
                f" quat={link_target_quat_in_robot_root}"
            )
            extra_target_poses[link_name] = torch.cat(
                (link_target_pos_in_robot_root, link_target_quat_in_robot_root), dim=-1
            ).squeeze(0)

        return extra_target_poses

    def _compute_relative_offsets_in_primary(
        self,
        primary_link_pose_in_robot_root: tuple[torch.Tensor, torch.Tensor],
        extra_link_poses_in_robot_root: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Compute the current rigid offset from the primary link frame to each extra link."""

        primary_pos_in_robot_root, primary_quat_in_robot_root = primary_link_pose_in_robot_root
        extra_link_offsets_in_primary = {}
        for link_name, (link_pos_in_robot_root, link_quat_in_robot_root) in extra_link_poses_in_robot_root.items():
            link_pos_in_primary, link_quat_in_primary = PoseUtils.subtract_frame_transforms(
                primary_pos_in_robot_root,
                primary_quat_in_robot_root,
                link_pos_in_robot_root,
                link_quat_in_robot_root,
            )
            extra_link_offsets_in_primary[link_name] = (link_pos_in_primary, link_quat_in_primary)

        return extra_link_offsets_in_primary

    def _build_extra_target_poses(
        self,
        activate_q: torch.Tensor,
        target_pose: torch.Tensor,
        env_extra_info: EnvExtraInfo,
    ) -> dict[str, torch.Tensor] | None:
        """Build link-level extra target poses based on configuration.

        This is the dispatcher for `extra_target_mode`. It returns a dict mapping link names to pose
        tensors in `[x, y, z, qx, qy, qz, qw]` (single-sample), used as additional link goals/constraints
        during planning.
        """

        if not self.cfg.extra_cfg.extra_target_link_names:
            return None

        if self.cfg.extra_cfg.extra_target_mode == "keep_current":
            return self._build_keep_current_extra_target_poses(activate_q)
        if self.cfg.extra_cfg.extra_target_mode == "keep_relative_offset":
            return self._build_keep_relative_offset_extra_target_poses(activate_q, target_pose)
        if self.cfg.extra_cfg.extra_target_mode == "keep_initial_relative_offset":
            return self._build_keep_initial_relative_offset_extra_target_poses(activate_q, target_pose, env_extra_info)
        raise ValueError(f"Unsupported extra_target_mode: {self.cfg.extra_cfg.extra_target_mode}")

    def _build_keep_current_extra_target_poses(self, activate_q: torch.Tensor) -> dict[str, torch.Tensor] | None:
        """Build "keep current pose" extra targets for configured links.

        In `keep_current` mode, this computes FK for each link in `extra_target_link_names` and uses
        its current pose as the planning target, effectively constraining those links to remain fixed.
        """

        extra_target_poses = {}
        for link_name, pose in self._planner.get_link_poses(
            activate_q, self.cfg.extra_cfg.extra_target_link_names
        ).items():
            extra_target_poses[link_name] = torch.cat(
                (pose.position, convert_quat(pose.quaternion, to="xyzw")), dim=-1  # cuRobo wxyz → xyzw
            ).squeeze(0)

        return extra_target_poses

    def _build_keep_relative_offset_extra_target_poses(
        self, activate_q: torch.Tensor, target_pose: torch.Tensor
    ) -> dict[str, torch.Tensor] | None:
        """Build extra targets by preserving the current rigid transform from primary EE to each extra link."""

        primary_link_pose_in_robot_root, extra_link_poses_in_robot_root = (
            self._get_current_primary_and_extra_link_poses(activate_q)
        )

        extra_link_offsets_in_primary = self._compute_relative_offsets_in_primary(
            primary_link_pose_in_robot_root, extra_link_poses_in_robot_root
        )
        return self._compute_relative_extra_target_poses(
            primary_link_pose_in_robot_root, extra_link_offsets_in_primary, target_pose
        )

    def _build_keep_initial_relative_offset_extra_target_poses(
        self,
        activate_q: torch.Tensor,
        target_pose: torch.Tensor,
        env_extra_info: EnvExtraInfo,
    ) -> dict[str, torch.Tensor] | None:
        """Build extra targets by preserving the first observed rigid transform from primary EE to each extra link."""

        primary_link_pose_in_robot_root, extra_link_poses_in_robot_root = (
            self._get_current_primary_and_extra_link_poses(activate_q)
        )

        if env_extra_info.cached_initial_extra_target_offsets is None:
            env_extra_info.cached_initial_extra_target_offsets = self._compute_relative_offsets_in_primary(
                primary_link_pose_in_robot_root, extra_link_poses_in_robot_root
            )
            self._logger.debug(
                "Cached initial relative offsets for extra links:"
                f" {list(env_extra_info.cached_initial_extra_target_offsets.keys())}"
            )
        else:
            self._logger.debug(
                "Reusing cached initial relative offsets for extra links:"
                f" {list(env_extra_info.cached_initial_extra_target_offsets.keys())}"
            )

        return self._compute_relative_extra_target_poses(
            primary_link_pose_in_robot_root, env_extra_info.cached_initial_extra_target_offsets, target_pose
        )

    def _compute_goal_from_offset(
        self,
        env: ManagerBasedEnv,
        target_object: str,
        reach_offset: torch.Tensor,
        env_extra_info: EnvExtraInfo,
    ) -> SkillGoal:
        """Compute reach goal by transforming object-frame offsets into robot root frame.

        Args:
            env: The Isaac Lab environment.
            target_object: Name of the target object in the scene.
            reach_offset: [7] tensor (pos + quat) in object frame for the primary EE.
            env_extra_info: Env info for cfg-driven extra target.

        Returns:
            SkillGoal with target poses in robot root frame.
        """

        object_pose_in_env = as_torch(env.scene[target_object].data.root_pose_w)

        object_pos_in_env = object_pose_in_env[:, :3]
        object_quat_in_env = object_pose_in_env[:, 3:]

        offset = reach_offset.to(env.device).unsqueeze(0)
        reach_target_pos_in_env, reach_target_quat_in_env = PoseUtils.combine_frame_transforms(
            object_pos_in_env, object_quat_in_env, offset[:, :3], offset[:, 3:]
        )
        self._logger.debug(f"Reach target position in environment: {reach_target_pos_in_env}")
        self._logger.debug(f"Reach target quaternion in environment: {reach_target_quat_in_env}")
        self._target_poses["target_pose"] = torch.cat((reach_target_pos_in_env, reach_target_quat_in_env), dim=-1)
        self.visualize_debug_target_pose()

        robot = env.scene[env_extra_info.robot_name]
        robot_root_pos_in_env = as_torch(robot.data.root_pose_w)[:, :3]
        robot_root_quat_in_env = as_torch(robot.data.root_pose_w)[:, 3:]

        reach_target_pos_in_root, reach_target_quat_in_root = PoseUtils.subtract_frame_transforms(
            robot_root_pos_in_env, robot_root_quat_in_env, reach_target_pos_in_env, reach_target_quat_in_env
        )
        target_pose = torch.cat((reach_target_pos_in_root, reach_target_quat_in_root), dim=-1).squeeze(0)
        self._logger.debug(
            f"Reach target pose in robot root frame: {reach_target_pos_in_root}, {reach_target_quat_in_root}"
        )

        activate_q, _ = self._build_activate_joint_state(
            robot.data.joint_names, as_torch(robot.data.joint_pos)[0], as_torch(robot.data.joint_vel)[0]
        )
        extra_target_poses = self._build_extra_target_poses(activate_q, target_pose, env_extra_info)

        return SkillGoal(target_object=target_object, target_pose=target_pose, extra_target_poses=extra_target_poses)

    def _select_best_candidate(
        self,
        env: ManagerBasedEnv,
        target_object: str,
        candidates: list[torch.Tensor],
        env_extra_info: EnvExtraInfo,
    ) -> torch.Tensor:
        """Select the closest reach candidate in the target object's frame.

        The current end-effector pose is transformed from world frame into the target
        object's frame and compared against all candidate poses stored in object frame.
        Selection is based on a weighted score consisting of position error plus
        orientation error.

        Args:
            env: The Isaac Lab environment.
            target_object: Name of the target object in the scene.
            candidates: List of K ``[7]`` tensors ``(pos + quat)`` in object frame.
            env_extra_info: Environment extra information.

        Returns:
            The selected ``[7]`` offset tensor in object frame.

        Raises:
            ValueError: If candidates is empty.
        """

        if not candidates:
            raise ValueError(f"No reach candidates provided for object '{target_object}'.")

        poses_oe = torch.stack([c.to(env.device) for c in candidates], dim=0)  # [K, 7]

        robot = env.scene[env_extra_info.robot_name]
        ee_link_idx = robot.data.body_names.index(env_extra_info.ee_link_name)
        ee_pose_w = as_torch(robot.data.body_link_pose_w)[0, ee_link_idx]  # [7]
        obj_pose_w = as_torch(env.scene[target_object].data.root_pose_w)[0]  # [7]

        ee_pos_oe, ee_quat_oe = PoseUtils.subtract_frame_transforms(
            obj_pose_w[:3].unsqueeze(0),
            obj_pose_w[3:].unsqueeze(0),
            ee_pose_w[:3].unsqueeze(0),
            ee_pose_w[3:].unsqueeze(0),
        )
        ee_pos_oe = ee_pos_oe.squeeze(0)  # [3]
        ee_quat_oe = ee_quat_oe.squeeze(0)  # [4]

        pos_err = torch.linalg.norm(poses_oe[:, :3] - ee_pos_oe.unsqueeze(0), dim=-1)
        quat_dot = torch.sum(poses_oe[:, 3:] * ee_quat_oe.unsqueeze(0), dim=-1).abs().clamp(max=1.0)
        rot_err = 1.0 * torch.acos(quat_dot)
        score = pos_err + 1.0 * rot_err

        best_idx = int(torch.argmin(score).item())
        self._logger.debug(
            f"Selected candidate {best_idx}/{len(candidates)} for '{target_object}' "
            f"(position_error={float(pos_err[best_idx]):.4f}, rotation_error={float(rot_err[best_idx]):.4f})"
        )
        return candidates[best_idx]

    def _compute_corrective_goal(self) -> SkillGoal | None:
        """Re-compute reach goal using the object's current actual pose.

        This is called after the first trajectory finishes. The same relative offset (in object
        frame) is re-applied to the object's current pose, so if the object was nudged during
        approach the robot corrects for it.
        """

        goal = self._compute_goal_from_offset(
            self._saved_env,
            self._saved_target_object,
            self._saved_reach_offset,
            self._saved_env_extra_info,
        )
        if goal is not None:
            self._logger.info("corrective_reach: recomputed target from current object pose")
        return goal

    def extract_goal_from_info(
        self, skill_info: SkillInfo, env: ManagerBasedEnv, env_extra_info: EnvExtraInfo
    ) -> SkillGoal:
        """Return the target pose[x, y, z, qx, qy, qz, qw] in the robot root frame.
        IMPORTANT: the robot root frame is not the same as the robot base frame.
        """

        target_object = skill_info.target_object
        candidates = env_extra_info.get_reach_target_poses(target_object)
        reach_offset = self._select_best_candidate(env, target_object, candidates, env_extra_info).to(env.device)

        # Save state needed for corrective reach re-planning
        self._saved_env = env
        self._saved_target_object = target_object
        self._saved_reach_offset = reach_offset
        self._saved_env_extra_info = env_extra_info

        return self._compute_goal_from_offset(env, target_object, reach_offset, env_extra_info)

    def execute_plan(self, state: WorldState, goal: SkillGoal) -> bool:
        """Execute the plan of the reach skill."""

        self._logger.debug(f"Reach from pose in environment: {state.robot_ee_pose}")

        # Set current target object for selective collision checking
        self._planner.set_target_object(goal.target_object)

        target_pose = goal.target_pose  # target pose in the robot root frame
        target_pos, target_quat = target_pose[:3], target_pose[3:]

        activate_q, activate_qd = self._build_activate_joint_state(
            state.sim_joint_names, state.robot_joint_pos, state.robot_joint_vel
        )
        if activate_qd is None:
            raise ValueError("activate_qd should not be None when planning reach trajectories.")

        self._trajectory = self._planner.plan_motion(
            target_pos,
            target_quat,
            activate_q,
            activate_qd,
            link_goals=goal.extra_target_poses,
        )

        return self._trajectory is not None

    def step(self, state: WorldState) -> SkillOutput:
        """Step the reach skill.

        Args:
            state: The current state of the world.

        Returns:
            The output of the skill execution.
                action: The action to be applied to the environment. [joint_positions with isaaclab joint order]
        """

        self.visualize_debug_target_pose()

        traj_positions = self._trajectory.position
        if self._step_idx >= len(self._trajectory.position):
            traj_pos = traj_positions[-1]
            done = True
        else:
            traj_pos = traj_positions[self._step_idx]
            done = False
            self._step_idx += 1

        # Corrective reach: when the first trajectory finishes, re-plan using the object's
        # actual current position in case it was nudged during approach.
        need_corrective_reach = (
            done
            and not self._corrective_reach_done
            and type(self) is ReachSkill
            and self.cfg.extra_cfg.corrective_reach
        )
        if need_corrective_reach:
            self._corrective_reach_done = True  # prevent infinite loop
            new_goal = self._compute_corrective_goal()
            if new_goal is not None:
                self._logger.info("corrective_reach: re-planning to corrected object pose")
                self._step_idx = 0
                plan_success = self.execute_plan(state, new_goal)
                if plan_success:
                    done = False  # continue with corrective trajectory

        curobo_joint_names = self._trajectory.joint_names
        sim_joint_names = state.sim_joint_names
        joint_pos = state.robot_joint_pos.clone()
        for curobo_idx, curobo_joint_name in enumerate(curobo_joint_names):
            sim_idx = sim_joint_names.index(curobo_joint_name)
            joint_pos[sim_idx] = traj_pos[curobo_idx]

        info = {}
        if self.cfg.extra_cfg.return_link_poses_in_robot_root_frame:
            activate_q, _ = self._build_activate_joint_state(state.sim_joint_names, joint_pos, None)
            all_link_poses = self._planner.get_link_poses(activate_q, link_names=None)
            info["link_poses_in_robot_root_frame"] = {
                name: torch.cat(
                    [pose.position.squeeze(0), convert_quat(pose.quaternion.squeeze(0), to="xyzw")]
                )  # cuRobo wxyz → xyzw
                for name, pose in all_link_poses.items()
            }

        return SkillOutput(
            action=joint_pos,
            done=done,
            success=True,
            info=info,
        )

    def reset(self) -> None:
        """Reset the reach skill."""

        super().reset()
        self._step_idx = 0
        self._trajectory = None
        self._corrective_reach_done = False
        self._saved_env = None
        self._saved_target_object = None
        self._saved_reach_offset = None
        self._saved_env_extra_info = None
        self._planner.set_target_object(None)
