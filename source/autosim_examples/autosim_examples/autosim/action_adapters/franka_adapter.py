from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs import ManagerBasedEnv

from autosim import ActionAdapterBase
from autosim.core.types import SkillOutput

if TYPE_CHECKING:
    from .franka_adapter_cfg import FrankaAbsAdapterCfg


class FrankaAbsAdapter(ActionAdapterBase):
    def __init__(self, cfg: FrankaAbsAdapterCfg):
        super().__init__(cfg)

        self.register_apply_method("reach", self._apply_reach)
        self.register_apply_method("grasp", self._apply_grasp)
        self.register_apply_method("lift", self._apply_reach)

    def _apply_reach(self, skill_output: SkillOutput, env: ManagerBasedEnv) -> torch.Tensor:
        target_joint_pos = skill_output.action  # [joint_positions with isaaclab joint order]

        robot = env.scene["robot"]
        default_joint_pos = robot.data.default_joint_pos[0, :]

        last_action = env.action_manager.action
        action = last_action[0, :].clone()

        arm_action_cfg = env.action_manager.get_term("arm_action").cfg

        arm_action_ids, _ = robot.find_joints(arm_action_cfg.joint_names)

        arm_target_joint_pos = target_joint_pos[arm_action_ids]
        arm_action = arm_target_joint_pos
        if arm_action_cfg.use_default_offset:
            arm_action = arm_action - default_joint_pos[arm_action_ids]
        arm_action = arm_action / arm_action_cfg.scale

        action[0:7] = arm_action

        return action

    def _apply_grasp(self, skill_output: SkillOutput, env: ManagerBasedEnv) -> torch.Tensor:
        last_action = env.action_manager.action
        action = last_action[0, :].clone()
        action[-1] = -1.0
        return action
