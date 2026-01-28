import torch
from isaaclab.utils import configclass

from autosim import register_skill
from autosim.core.skill import SkillCfg
from autosim.core.types import SkillGoal, SkillOutput, WorldState

from .base_skill import CuroboSkillBase, CuroboSkillExtraCfg


@configclass
class PressSkillCfg(SkillCfg):
    """Configuration for the press skill."""

    extra_cfg: CuroboSkillExtraCfg = CuroboSkillExtraCfg()
    """Extra configuration for the press skill."""


@register_skill(
    name="press", cfg_type=PressSkillCfg, description="Press action (for buttons and interactive elements)."
)
class PressSkill(CuroboSkillBase):
    """Skill to press buttons or interactive elements"""

    def __init__(self, extra_cfg: CuroboSkillExtraCfg) -> None:
        super().__init__(extra_cfg)

    def plan(self, state: WorldState, goal: SkillGoal) -> bool:
        return True

    def step(self, state: WorldState) -> SkillOutput:
        return SkillOutput(action=torch.zeros(6), done=True, success=True)
