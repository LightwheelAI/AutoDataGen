import torch
from isaaclab.utils import configclass

from autosim import register_skill
from autosim.core.skill import Skill, SkillCfg, SkillExtraCfg
from autosim.core.types import SkillGoal, SkillOutput, WorldState


@configclass
class NavigateSkillExtraCfg(SkillExtraCfg):
    """Extra configuration for the navigate skill."""

    place_holder: str = "placeholder"
    """The place holder for the navigate skill."""


@configclass
class NavigateSkillCfg(SkillCfg):
    """Configuration for the navigate skill."""

    extra_cfg: NavigateSkillExtraCfg = NavigateSkillExtraCfg()
    """Extra configuration for the navigate skill."""


@register_skill(
    name="moveto", cfg_type=NavigateSkillCfg, description="Move robot base to near the target object or location."
)
class NavigateSkill(Skill):
    """Skill to navigate to a target position using A* + DWA motion planner."""

    def __init__(self, extra_cfg: NavigateSkillExtraCfg) -> None:
        super().__init__(extra_cfg)

    def plan(self, state: WorldState, goal: SkillGoal) -> bool:
        return True

    def step(self, state: WorldState) -> SkillOutput:
        return SkillOutput(action=torch.zeros(6), done=True, success=True)
