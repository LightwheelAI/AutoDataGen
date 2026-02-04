import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

from autosim import register_skill
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
class LiftSkillExtraCfg(CuroboSkillExtraCfg):
    """Extra configuration for the lift skill."""

    lift_offset: float = 0.1
    """The offset to lift the end-effector."""


@configclass
class LiftSkillCfg(SkillCfg):
    """Configuration for the lift skill."""

    extra_cfg: LiftSkillExtraCfg = LiftSkillExtraCfg()
    """Extra configuration for the lift skill."""


@register_skill(name="lift", cfg_type=LiftSkillCfg, description="Lift end-effector upward (target: 'up')")
class LiftSkill(ReachSkill):
    """Skill to lift end-effector upward"""

    def __init__(self, extra_cfg: LiftSkillExtraCfg) -> None:
        super().__init__(extra_cfg)

    def extract_goal_from_info(
        self, skill_info: SkillInfo, env: ManagerBasedEnv, env_extra_info: EnvExtraInfo
    ) -> SkillGoal:
        """Return the target object of the lift skill."""

        return SkillGoal(target_object=skill_info.target_object)

    def execute_plan(self, state: WorldState, goal: SkillGoal) -> bool:
        """Execute the plan of the lift skill."""

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

        ee_pose = self._planner.get_ee_pose(activate_q)
        target_pos, target_quat = ee_pose.position.squeeze(0).clone(), ee_pose.quaternion.squeeze(0).clone()
        # lift the end-effector upward by the lift offset
        target_pos[2] += self.cfg.extra_cfg.lift_offset

        self._trajectory = self._planner.plan_motion(
            target_pos,
            target_quat,
            activate_q,
            activate_qd,
        )

        return self._trajectory is not None

    def step(self, state: WorldState) -> SkillOutput:
        """Step the lift skill.

        Args:
            state: The current state of the world.

        Returns:
            The output of the skill execution.
                action: The action to be applied to the environment. [joint_positions with isaaclab joint order]
        """

        return super().step(state)
