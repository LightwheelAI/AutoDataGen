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

from .base_skill import CuroboSkillBase, CuroboSkillExtraCfg


@configclass
class ReachSkillCfg(SkillCfg):
    """Configuration for the reach skill."""

    extra_cfg: CuroboSkillExtraCfg = CuroboSkillExtraCfg()
    """Extra configuration for the reach skill."""


@register_skill(
    name="reach",
    cfg_type=ReachSkillCfg,
    description="Extend robot arm to target position (for approaching objects or placement locations)",
)
class ReachSkill(CuroboSkillBase):
    """Skill to reach to a target object or location"""

    def __init__(self, extra_cfg: CuroboSkillExtraCfg) -> None:
        super().__init__(extra_cfg)

        self._logger = AutoSimLogger("ReachSkill")

        # variables for the skill execution
        self._trajectory = None
        self._step_idx = 0

    def extract_goal_from_info(
        self, skill_info: SkillInfo, env: ManagerBasedEnv, env_extra_info: EnvExtraInfo
    ) -> SkillGoal:
        """Return the target pose[x, y, z, qw, qx, qy, qz] in the robot root frame.
        IMPORTANT: the robot root frame is not the same as the robot base frame.
        """

        target_object = skill_info.target_object
        robot = env.scene[env_extra_info.robot_name]

        object_pose_in_env = env.scene[target_object].data.root_pose_w
        object_pos_in_env, object_quat_in_env = object_pose_in_env[:, :3], object_pose_in_env[:, 3:]

        reach_target_pose = env_extra_info.get_next_reach_target_pose(target_object)
        reach_target_pose = reach_target_pose.to(env.device)
        reach_target_pose_in_object = reach_target_pose.unsqueeze(0)
        reach_target_pos_in_object, reach_target_quat_in_object = (
            reach_target_pose_in_object[:, :3],
            reach_target_pose_in_object[:, 3:],
        )

        reach_target_pos_in_env, reach_target_quat_in_env = PoseUtils.combine_frame_transforms(
            object_pos_in_env, object_quat_in_env, reach_target_pos_in_object, reach_target_quat_in_object
        )
        self._logger.info(f"Reach target position in environment: {reach_target_pos_in_env}")
        self._logger.info(f"Reach target quaternion in environment: {reach_target_quat_in_env}")
        self._target_poses["target_pose"] = torch.cat((reach_target_pos_in_env, reach_target_quat_in_env), dim=-1)

        robot_root_pose_in_env = robot.data.root_pose_w
        robot_root_pos_in_env, robot_root_quat_in_env = robot_root_pose_in_env[:, :3], robot_root_pose_in_env[:, 3:]

        reach_target_pos_in_robot_root, reach_target_quat_in_robot_root = PoseUtils.subtract_frame_transforms(
            robot_root_pos_in_env, robot_root_quat_in_env, reach_target_pos_in_env, reach_target_quat_in_env
        )

        target_pose = torch.cat((reach_target_pos_in_robot_root, reach_target_quat_in_robot_root), dim=-1).squeeze(0)

        if target_object in env_extra_info.object_extra_reach_target_poses.keys():
            extra_target_poses = {}
            for ee_name in env_extra_info.object_extra_reach_target_poses[target_object].keys():
                ee_target_pose = env_extra_info.get_next_extra_reach_target_pose(target_object, ee_name)
                ee_target_pose = torch.as_tensor(ee_target_pose, device=env.device)
                extra_target_pos_in_obj, extra_target_quat_in_obj = ee_target_pose[:3].unsqueeze(0), ee_target_pose[
                    3:
                ].unsqueeze(0)
                extra_target_pos_in_env, extra_target_quat_in_env = PoseUtils.combine_frame_transforms(
                    object_pos_in_env, object_quat_in_env, extra_target_pos_in_obj, extra_target_quat_in_obj
                )
                self._logger.info(f"Extra target position for {ee_name} in environment: {extra_target_pos_in_env}")
                self._logger.info(f"Extra target quaternion for {ee_name} in environment: {extra_target_quat_in_env}")
                extra_target_pos_in_robot_root, extra_target_quat_in_robot_root = PoseUtils.subtract_frame_transforms(
                    robot_root_pos_in_env, robot_root_quat_in_env, extra_target_pos_in_env, extra_target_quat_in_env
                )
                extra_target_poses[ee_name] = torch.cat(
                    (extra_target_pos_in_robot_root, extra_target_quat_in_robot_root), dim=-1
                ).squeeze(0)
        else:
            extra_target_poses = None

        return SkillGoal(target_object=target_object, target_pose=target_pose, extra_target_poses=extra_target_poses)

    def execute_plan(self, state: WorldState, goal: SkillGoal) -> bool:
        """Execute the plan of the reach skill."""

        self._logger.info(f"Reach from pose in environment: {state.robot_ee_pose}")

        target_pose = goal.target_pose  # target pose in the robot root frame
        target_pos, target_quat = target_pose[:3], target_pose[3:]

        full_sim_joint_names = state.sim_joint_names
        full_sim_q = state.robot_joint_pos
        full_sim_qd = state.robot_joint_vel
        planner_activate_joints = self._planner.target_joint_names

        activate_q, activate_qd = [], []
        for joint_name in planner_activate_joints:
            if joint_name in full_sim_joint_names:
                activate_q.append(full_sim_q[full_sim_joint_names.index(joint_name)])
                activate_qd.append(full_sim_qd[full_sim_joint_names.index(joint_name)])
            else:
                raise ValueError(
                    f"Joint {joint_name} in planner activate joints is not in the full simulation joint names."
                )
        activate_q = torch.stack(activate_q, dim=0)
        activate_qd = torch.stack(activate_qd, dim=0)
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

        curobo_joint_names = self._trajectory.joint_names
        sim_joint_names = state.sim_joint_names
        joint_pos = state.robot_joint_pos.clone()
        for curobo_idx, curobo_joint_name in enumerate(curobo_joint_names):
            sim_idx = sim_joint_names.index(curobo_joint_name)
            joint_pos[sim_idx] = traj_pos[curobo_idx]

        return SkillOutput(
            action=joint_pos,
            done=done,
            success=True,
            info={},
        )

    def reset(self) -> None:
        """Reset the reach skill."""

        super().reset()
        self._step_idx = 0
        self._trajectory = None
