from isaaclab.utils import configclass

from autosim import register_skill
from autosim.core.skill import SkillCfg

from .base_skill import GripperSkillBase, GripperSkillExtraCfg


@configclass
class GraspSkillCfg(SkillCfg):
    """Configuration for the grasp skill."""

    extra_cfg: GripperSkillExtraCfg = GripperSkillExtraCfg(gripper_value=-1.0)
    """default configuration: close gripper[-1.0] for 10 steps"""


@register_skill(name="grasp", cfg_type=GraspSkillCfg, description="Grasp object (close gripper)")
class GraspSkill(GripperSkillBase):
    """Skill to grasp an object"""

    def __init__(self, extra_cfg: GripperSkillExtraCfg) -> None:
        super().__init__(extra_cfg)


@configclass
class UngraspSkillCfg(SkillCfg):
    """Configuration for the ungrasp skill."""

    extra_cfg: GripperSkillExtraCfg = GripperSkillExtraCfg(gripper_value=1.0)
    """default configuration: open gripper[1.0] for 10 steps"""


@register_skill(name="ungrasp", cfg_type=UngraspSkillCfg, description="Release object (open gripper)")
class UngraspSkill(GripperSkillBase):
    """Skill to release an object"""

    def __init__(self, extra_cfg: GripperSkillExtraCfg) -> None:
        super().__init__(extra_cfg)
