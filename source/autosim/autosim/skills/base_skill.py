import torch
from isaaclab.utils import configclass

from autosim.capabilities.motion_planning import CuroboPlanner
from autosim.core.skill import Skill, SkillExtraCfg
from autosim.core.types import SkillGoal, SkillOutput, WorldState


@configclass
class GripperSkillExtraCfg(SkillExtraCfg):
    """Extra configuration for the gripper skill."""

    gripper_value: float = 0.0
    """The value of the gripper."""
    duration: int = 10
    """The duration of the gripper."""


class GripperSkillBase(Skill):
    """Base class for gripper skills open/close skills."""

    def __init__(self, extra_cfg: GripperSkillExtraCfg) -> None:
        super().__init__(extra_cfg)

        self._gripper_value = extra_cfg.gripper_value
        self._duration = extra_cfg.duration
        self._step_count = 0
        self._target_object_name = None

    def plan(self, state: WorldState, goal: SkillGoal) -> bool:
        self._target_object_name = goal.target_object
        self._step_count = 0
        return True

    def step(self, state: WorldState) -> SkillOutput:
        done = self._step_count >= self._duration
        self._step_count += 1

        return SkillOutput(
            action=torch.tensor([self._gripper_value], device=state.device),
            done=done,
            success=done,
            info={"step": self._step_count, "target_object": self._target_object_name},
        )


@configclass
class CuroboSkillExtraCfg(SkillExtraCfg):
    """Extra configuration for the curobo skill."""

    curobo_planner: CuroboPlanner | None = None


class CuroboSkillBase(Skill):
    """Base class for skills dependent on curobo."""

    def __init__(self, extra_cfg: CuroboSkillExtraCfg) -> None:
        super().__init__(extra_cfg)
        self._planner = extra_cfg.curobo_planner
