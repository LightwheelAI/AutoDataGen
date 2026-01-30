from isaaclab.utils import configclass

from .base_skill import CuroboSkillExtraCfg
from .gripper import GraspSkill, GraspSkillCfg, UngraspSkill, UngraspSkillCfg
from .lift import LiftSkill, LiftSkillCfg
from .navigate import NavigateSkill, NavigateSkillCfg, NavigateSkillExtraCfg
from .press import PressSkill, PressSkillCfg
from .pull import PullSkill, PullSkillCfg
from .push import PushSkill, PushSkillCfg
from .reach import ReachSkill, ReachSkillCfg
from .rotate import RotateSkill, RotateSkillCfg


@configclass
class AutoSimSkillsExtraCfg:
    """Extra configuration for the AutoSim skills."""

    grasp: GraspSkillCfg = GraspSkillCfg()
    ungrasp: UngraspSkillCfg = UngraspSkillCfg()
    lift: LiftSkillCfg = LiftSkillCfg()
    moveto: NavigateSkillCfg = NavigateSkillCfg()
    press: PressSkillCfg = PressSkillCfg()
    pull: PullSkillCfg = PullSkillCfg()
    push: PushSkillCfg = PushSkillCfg()
    reach: ReachSkillCfg = ReachSkillCfg()
    rotate: RotateSkillCfg = RotateSkillCfg()

    def get(cls, skill_name: str):
        """Get the skill configuration by name."""
        return getattr(cls, skill_name)
