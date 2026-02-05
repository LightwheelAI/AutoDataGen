from abc import ABC, abstractmethod

from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

from .types import (
    EnvExtraInfo,
    SkillGoal,
    SkillInfo,
    SkillOutput,
    SkillStatus,
    WorldState,
)


@configclass
class SkillExtraCfg:
    """Extra configuration for the skill."""


@configclass
class SkillCfg:
    """Configuration for the skill."""

    name: str = "base_skill"
    """The name of the skill."""
    description: str = "Base skill class."
    """The description of the skill (for prompt generation)."""
    extra_cfg: SkillExtraCfg = SkillExtraCfg()
    """The extra configuration for the skill (used in specific skill classes)."""


class Skill(ABC):
    """Base class for all skills."""

    cfg: SkillCfg
    """The configuration of the skill."""

    def __init__(self, extra_cfg: SkillExtraCfg) -> None:
        self._status: SkillStatus = SkillStatus.IDLE
        self.cfg.extra_cfg = extra_cfg

    @classmethod
    def get_cfg(cls) -> SkillCfg:
        """Get the configuration of the skill."""

        return cls.cfg

    @abstractmethod
    def extract_goal_from_info(
        self, skill_info: SkillInfo, env: ManagerBasedEnv, env_extra_info: EnvExtraInfo
    ) -> SkillGoal:
        """Extract the goal from the skill information.

        Args:
            skill_info: The skill information.
            env: The environment.
            env_extra_info: The extra information of the environment.

        Returns:
            The goal of the skill.
        """

        raise NotImplementedError(f"{self.__class__.__name__}.extract_goal_from_info() must be implemented.")

    def plan(self, state: WorldState, goal: SkillGoal) -> bool:
        """Plan the skill.

        Args:
            state: The current state of the world.
            goal: The goal of the skill.

        Returns:
            True if the skill is planned successfully, False otherwise.
        """

        self._status = SkillStatus.PLANNING
        success = self.execute_plan(state, goal)
        if success:
            self._status = SkillStatus.EXECUTING
        else:
            self._status = SkillStatus.FAILED
        return success

    @abstractmethod
    def execute_plan(self, state: WorldState, goal: SkillGoal) -> bool:
        """Execute the plan of the skill.

        Args:
            state: The current state of the world.
            goal: The goal of the skill.

        Returns:
            True if the skill is planned successfully, False otherwise.
        """

        raise NotImplementedError(f"{self.__class__.__name__}.plan() must be implemented.")

    @abstractmethod
    def step(self, state: WorldState) -> SkillOutput:
        """Execute one step of the skill.

        Args:
            state: The current state of the world.

        Returns:
            The output of the skill, containing the action, done, success, info, and trajectory.
        """

        raise NotImplementedError(f"{self.__class__.__name__}.step() must be implemented.")

    def reset(self) -> None:
        """Reset the skill."""

        self._status = SkillStatus.IDLE

    def __repr__(self) -> str:

        return f"{self.__class__.__name__}(status={self._status.value})"
