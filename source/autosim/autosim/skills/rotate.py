import torch
from isaaclab.utils import configclass

from autosim import register_skill
from autosim.core.skill import SkillCfg
from autosim.core.types import SkillGoal, SkillOutput, WorldState

from .base_skill import CuroboSkillBase, CuroboSkillExtraCfg


@configclass
class RotateSkillCfg(SkillCfg):
    """Configuration for the rotate skill."""

    extra_cfg: CuroboSkillExtraCfg = CuroboSkillExtraCfg()
    """Extra configuration for the rotate skill."""


@register_skill(
    name="rotate", cfg_type=RotateSkillCfg, description="Rotate action (for rotating objects or operating knobs)"
)
class RotateSkill(CuroboSkillBase):
    """Skill to reach to a target object or location"""

    def __init__(self, extra_cfg: CuroboSkillExtraCfg) -> None:
        super().__init__(extra_cfg)

    def plan(self, state: WorldState, goal: SkillGoal) -> bool:
        return True

    def step(self, state: WorldState) -> SkillOutput:
        return SkillOutput(action=torch.zeros(6), done=True, success=True)
