from isaaclab.utils import configclass

from .base_skill import CuroboSkillExtraCfg
from .gripper import GraspSkill, GraspSkillCfg, UngraspSkill, UngraspSkillCfg
from .navigate import NavigateSkill, NavigateSkillCfg, NavigateSkillExtraCfg
from .reach import ReachSkill, ReachSkillCfg
from .relative_reach import (
    LiftSkill,
    LiftSkillCfg,
    PullSkill,
    PullSkillCfg,
    PushSkill,
    PushSkillCfg,
)


@configclass
class AutoSimSkillsExtraCfg:
    """Extra configuration for the AutoSim skills."""

    grasp: GraspSkillCfg = GraspSkillCfg()
    ungrasp: UngraspSkillCfg = UngraspSkillCfg()
    lift: LiftSkillCfg = LiftSkillCfg()
    moveto: NavigateSkillCfg = NavigateSkillCfg()
    pull: PullSkillCfg = PullSkillCfg()
    push: PushSkillCfg = PushSkillCfg()
    reach: ReachSkillCfg = ReachSkillCfg()

    def get(cls, skill_name: str):
        """Get the skill configuration by name."""

        return getattr(cls, skill_name)

    def debug_target_pose(self):
        """Debug the target pose of the skills."""

        self.lift.extra_cfg.debug_target_pose = True
        self.pull.extra_cfg.debug_target_pose = True
        self.push.extra_cfg.debug_target_pose = True
        self.reach.extra_cfg.debug_target_pose = True
