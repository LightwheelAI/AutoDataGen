import torch
from isaaclab.utils import configclass

from autosim import register_skill
from autosim.core.skill import SkillCfg
from autosim.core.types import SkillGoal, SkillOutput, WorldState

from .base_skill import CuroboSkillExtraCfg
from .reach import ReachSkill


@configclass
class PullSkillCfg(SkillCfg):
    """Configuration for the pull skill."""

    extra_cfg: CuroboSkillExtraCfg = CuroboSkillExtraCfg()
    """Extra configuration for the pull skill."""


@register_skill(name="pull", cfg_type=PullSkillCfg, description="Pull end-effector backward (target: 'backward')")
class PullSkill(ReachSkill):
    """Skill to pull end-effector backward"""

    def __init__(self, extra_cfg: CuroboSkillExtraCfg) -> None:
        super().__init__(extra_cfg)

    def plan(self, state: WorldState, goal: SkillGoal) -> bool:
        return True

    def step(self, state: WorldState) -> SkillOutput:
        return SkillOutput(action=torch.zeros(6), done=True, success=True)
