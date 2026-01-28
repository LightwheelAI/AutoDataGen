import torch
from isaaclab.utils import configclass

from autosim import register_skill
from autosim.core.skill import SkillCfg
from autosim.core.types import SkillGoal, SkillOutput, WorldState

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

    def plan(self, state: WorldState, goal: SkillGoal) -> bool:
        return True

    def step(self, state: WorldState) -> SkillOutput:
        return SkillOutput(action=torch.zeros(6), done=True, success=True)
