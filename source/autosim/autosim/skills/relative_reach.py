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


@configclass
class RelativeReachSkillExtraCfg(CuroboSkillExtraCfg):
    """Extra configuration for the relative reach skill."""

    move_offset: float = 0.1
    """The offset to move the end-effector."""
    move_axis: str = "+z"
    """The axis to move the end-effector, which is in the eef frame.
    Supports single axis (e.g. "+x", "-y") or multi-axis combinations (e.g. "+x+y", "+x-z")."""

    def __post_init__(self):
        """Post-initialize the relative reach skill extra configuration."""

        super().__post_init__()
        self._axis_map = {
            "x": torch.tensor([1.0, 0.0, 0.0]),
            "y": torch.tensor([0.0, 1.0, 0.0]),
            "z": torch.tensor([0.0, 0.0, 1.0]),
        }

    def get_direction_vector(self) -> torch.Tensor:
        """Parse move_axis and compute the normalized direction vector.

        This is computed on-demand to support dynamic modification of move_axis.

        Returns:
            Normalized direction vector in EE frame.
        """
        import re

        pattern = r"([+-][xyz])"
        matches = re.findall(pattern, self.move_axis)

        if not matches:
            raise ValueError(
                f"Invalid move_axis format: '{self.move_axis}'. Expected format: '+x', '-y', '+x+y', '+x-z', etc."
            )

        direction_vector = torch.zeros(3)
        for match in matches:
            sign = 1.0 if match[0] == "+" else -1.0
            axis = match[1]
            if axis not in self._axis_map:
                raise ValueError(f"Invalid axis '{axis}' in move_axis: '{self.move_axis}'")
            direction_vector += sign * self._axis_map[axis]

        norm = torch.linalg.norm(direction_vector)
        if norm < 1e-6:
            raise ValueError(f"move_axis '{self.move_axis}' results in zero direction vector")

        return direction_vector / norm


@configclass
class RelativeReachSkillCfg(SkillCfg):
    """Configuration for the relative reach skill."""

    extra_cfg: RelativeReachSkillExtraCfg = RelativeReachSkillExtraCfg()
    """Extra configuration for the relative reach skill."""


class RelativeReachSkill(ReachSkill):
    """Skill to move the end-effector along a specific axis"""

    def __init__(self, extra_cfg: RelativeReachSkillCfg) -> None:
        super().__init__(extra_cfg)
        self._logger = AutoSimLogger("RelativeReachSkill")

    def extract_goal_from_info(
        self, skill_info: SkillInfo, env: ManagerBasedEnv, env_extra_info: EnvExtraInfo
    ) -> SkillGoal:
        """Return the target object of the relative reach skill."""

        return SkillGoal(target_object=skill_info.target_object)

    def execute_plan(self, state: WorldState, goal: SkillGoal) -> bool:
        """Execute the plan of the relative reach skill."""

        # For lift/push/pull, disable all dynamic object collision checking
        # because the object is already grasped and we don't need to avoid any dynamic objects
        self._planner.set_target_object(None)

        activate_q, activate_qd = self._build_activate_joint_state(
            state.sim_joint_names, state.robot_joint_pos, state.robot_joint_vel
        )
        if activate_qd is None:
            raise ValueError("activate_qd should not be None when planning relative reach trajectories.")

        ee_pose = self._planner.get_ee_pose(activate_q)
        target_pos, target_quat = ee_pose.position.clone(), ee_pose.quaternion.clone()
        self._logger.info(f"ee pos in robot root frame: {target_pos}, ee quat in robot root frame: {target_quat}")

        # move the eef along the move axis by the move offset based on eef frame, and convert to robot root frame to get target pose
        isaaclab_device = state.device
        move_offset_vector = self.cfg.extra_cfg.get_direction_vector() * self.cfg.extra_cfg.move_offset
        offset_pos_in_ee = move_offset_vector.to(isaaclab_device).unsqueeze(0)
        offset_quat_in_ee = torch.tensor([1.0, 0.0, 0.0, 0.0], device=isaaclab_device).unsqueeze(0)
        ee_pos_in_robot_root, ee_quat_in_robot_root = target_pos.to(isaaclab_device), target_quat.to(isaaclab_device)

        offset_pos_in_robot_root, offset_quat_in_robot_root = PoseUtils.combine_frame_transforms(
            ee_pos_in_robot_root, ee_quat_in_robot_root, offset_pos_in_ee, offset_quat_in_ee
        )

        planner_device = self._planner.tensor_args.device
        target_pos = offset_pos_in_robot_root.to(planner_device).squeeze(0)
        target_quat = offset_quat_in_robot_root.to(planner_device).squeeze(0)
        self._logger.info(
            f"target pos in robot root frame: {target_pos}, target quat in robot root frame: {target_quat}"
        )

        reach_target_pos_in_robot_root = offset_pos_in_robot_root.clone()
        reach_target_quat_in_robot_root = offset_quat_in_robot_root.clone()
        robot_root_pos_in_env, robot_root_quat_in_env = state.robot_root_pose[None, :3], state.robot_root_pose[None, 3:]
        reach_target_pos_in_env, reach_target_quat_in_env = PoseUtils.combine_frame_transforms(
            robot_root_pos_in_env,
            robot_root_quat_in_env,
            reach_target_pos_in_robot_root,
            reach_target_quat_in_robot_root,
        )
        self._logger.info(f"reach target pos in environment: {reach_target_pos_in_env}")
        self._logger.info(f"reach target quat in environment: {reach_target_quat_in_env}")
        self._target_poses["target_pose"] = torch.cat((reach_target_pos_in_env, reach_target_quat_in_env), dim=-1)

        self._trajectory = self._planner.plan_motion(
            target_pos,
            target_quat,
            activate_q,
            activate_qd,
        )

        return self._trajectory is not None

    def step(self, state: WorldState) -> SkillOutput:
        """Step the relative reach skill.

        Args:
            state: The current state of the world.

        Returns:
            The output of the skill execution.
                action: The action to be applied to the environment. [joint_positions with isaaclab joint order]
        """

        return super().step(state)


"""LIFT SKILL"""


@configclass
class LiftSkillExtraCfg(RelativeReachSkillExtraCfg):
    """Extra configuration for the lift skill."""

    move_axis: str = "+z"
    """lift the end-effector upward"""


@configclass
class LiftSkillCfg(RelativeReachSkillCfg):
    """Configuration for the lift skill."""

    extra_cfg: LiftSkillExtraCfg = LiftSkillExtraCfg()
    """Extra configuration for the lift skill."""


@register_skill(name="lift", cfg_type=LiftSkillCfg, description="Lift end-effector upward (target: 'up')")
class LiftSkill(RelativeReachSkill):
    """Skill to lift end-effector upward"""

    def __init__(self, extra_cfg: LiftSkillExtraCfg) -> None:
        super().__init__(extra_cfg)


"""PULL SKILL"""


@configclass
class PullSkillExtraCfg(RelativeReachSkillExtraCfg):
    """Extra configuration for the pull skill."""

    move_axis: str = "-x"
    """pull the end-effector backward"""


@configclass
class PullSkillCfg(RelativeReachSkillCfg):
    """Configuration for the pull skill."""

    extra_cfg: PullSkillExtraCfg = PullSkillExtraCfg()
    """Extra configuration for the pull skill."""


@register_skill(name="pull", cfg_type=PullSkillCfg, description="Pull end-effector backward (target: 'backward')")
class PullSkill(RelativeReachSkill):
    """Skill to pull end-effector backward"""

    def __init__(self, extra_cfg: PullSkillExtraCfg) -> None:
        super().__init__(extra_cfg)


"""PUSH SKILL"""


@configclass
class PushSkillExtraCfg(RelativeReachSkillExtraCfg):
    """Extra configuration for the push skill."""

    move_axis: str = "+x"
    """push the end-effector forward"""


@configclass
class PushSkillCfg(RelativeReachSkillCfg):
    """Configuration for the push skill."""

    extra_cfg: PushSkillExtraCfg = PushSkillExtraCfg()
    """Extra configuration for the push skill."""


@register_skill(name="push", cfg_type=PushSkillCfg, description="Push end-effector forward (target: 'forward')")
class PushSkill(RelativeReachSkill):
    """Skill to push end-effector forward"""

    def __init__(self, extra_cfg: PushSkillExtraCfg) -> None:
        super().__init__(extra_cfg)
