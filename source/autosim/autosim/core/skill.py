from abc import ABC, abstractmethod
from typing import Any

import torch
from isaaclab.utils import configclass

from .types import SkillGoal, SkillOutput, SkillStatus, WorldState


@configclass
class SkillCfg:
    """Configuration for the skill."""

    name: str = "base_skill"
    """The name of the skill."""
    description: str = "Base skill class."
    """The description of the skill."""
    required_modules: list[str] = []
    """The required modules for the skill."""
    extra_cfg: dict = {}
    """The extra configuration for the skill."""


class Skill(ABC):
    """Base class for all skills."""

    cfg: SkillCfg
    """The configuration of the skill."""

    def __init__(self, extra_cfg: dict = {}) -> None:
        self._status = SkillStatus.IDLE
        self._internal_state: dict[str, Any] = {}
        self.cfg.extra_cfg.update(extra_cfg)

        # initialize dependent modules
        self._init_modules()

    @classmethod
    def get_cfg(cls) -> SkillCfg:
        """Get the configuration of the skill."""
        return cls.cfg

    @abstractmethod
    def plan(self, state: WorldState, goal: SkillGoal) -> bool:
        """Plan the skill.

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
        self._internal_state.clear()

    def _init_modules(self) -> None:
        """Initialize the dependent modules."""
        pass

    def __call__(self, state: WorldState, goal: SkillGoal) -> SkillOutput:
        """
        Convenient call interface
        Automatically handles plan -> execute flow
        """
        if self._status == SkillStatus.IDLE and goal is not None:
            self._status = SkillStatus.PLANNING
            success = self.plan(state, goal)
            if success:
                self._status = SkillStatus.EXECUTING
            else:
                self._status = SkillStatus.FAILED
                return SkillOutput(
                    action=torch.zeros_like(state.robot_joint_pos),
                    done=True,
                    success=False,
                    info={"error": "Failed to plan the skill."},
                )
        return self.step(state)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(status={self._status.value})"
