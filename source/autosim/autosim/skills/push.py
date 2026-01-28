import torch
from isaaclab.utils import configclass

from autosim import register_skill
from autosim.core.skill import SkillCfg
from autosim.core.types import SkillGoal, SkillOutput, WorldState

from .base_skill import CuroboSkillExtraCfg
from .reach import ReachSkill


@configclass
class PushSkillCfg(SkillCfg):
    """Configuration for the push skill."""

    extra_cfg: CuroboSkillExtraCfg = CuroboSkillExtraCfg()
    """Extra configuration for the push skill."""


@register_skill(
    name="push",
    cfg_type=PushSkillCfg,
    description="Push end-effector forward (target: 'forward')",
)
class PushSkill(ReachSkill):
    """Skill to push end-effector forward"""

    def __init__(self, extra_cfg: CuroboSkillExtraCfg) -> None:
        super().__init__(extra_cfg)

    def plan(self, state: WorldState, goal: SkillGoal) -> bool:
        return True

    def step(self, state: WorldState) -> SkillOutput:
        return SkillOutput(action=torch.zeros(6), done=True, success=True)
