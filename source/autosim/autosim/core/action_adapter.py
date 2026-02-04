from dataclasses import MISSING
from typing import Protocol

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

from autosim.core.logger import AutoSimLogger
from autosim.core.skill import Skill

from .types import SkillOutput


class ApplyMethodProtocol(Protocol):
    """Protocol for apply methods - fixed signature that users must follow."""

    def __call__(self, skill_output: SkillOutput, env: ManagerBasedEnv) -> torch.Tensor:
        """Apply method signature - fixed parameters."""
        ...


@configclass
class ActionAdapterCfg:
    """Configuration for the action adapter."""

    class_type: type = MISSING
    """The class type of the action adapter."""
    skip_apply_skills: list[str] = []
    """The skills that should be skipped for applying the action. e.g. "moveto" should be skipped if the robot is fixed to the environment"""


class ActionAdapterBase:
    """Base class for all action adapters.

    Users should define instance methods and register them using `register_apply_method()`.
    All apply methods must follow this fixed signature:

        def method_name(self, skill_output: SkillOutput, env: ManagerBasedEnv) -> torch.Tensor

    Example:
        >>> class MyActionAdapter(ActionAdapterBase):
        >>>    def __init__(self, cfg: ActionAdapterCfg) -> None:
        >>>        super().__init__(cfg)
        >>>        self.register_apply_method("grasp", self.handle_grasp)

        >>>    def handle_grasp(self, skill_output: SkillOutput, env: ManagerBasedEnv) -> torch.Tensor:
        >>>        # Your implementation here
        >>>        ...
    """

    cfg: ActionAdapterCfg
    """The configuration of the action adapter."""

    def __init__(self, cfg: ActionAdapterCfg) -> None:
        self.cfg = cfg
        self._apply_map: dict[str, ApplyMethodProtocol] = {}
        self._skip_apply_skills = self.cfg.skip_apply_skills
        self._logger = AutoSimLogger("ActionAdapter")

    def register_apply_method(self, skill_name: str, method: ApplyMethodProtocol) -> None:
        """Register an apply method for a specific skill."""

        self._apply_map[skill_name] = method

    def should_skip_apply(self, skill: Skill) -> bool:
        """
        Check if the skill should be skipped for applying the action.

        Args:
            skill: The skill instance.
        """

        return skill.cfg.name in self._skip_apply_skills

    def apply(self, skill: Skill, skill_output: SkillOutput, env: ManagerBasedEnv) -> torch.Tensor:
        """
        Apply the skill output to the environment.

        Args:
            skill: The skill instance.
            skill_output: The output of the skill.
            env: The environment.

        Returns:
            The applied action. [action_dim]
        """

        skill_type = skill.cfg.name
        if skill_type in self._apply_map:
            return self._apply_map[skill_type](skill_output, env)
        else:
            return self._default_apply(skill_output, env)

    def _default_apply(self, skill_output: SkillOutput, env: ManagerBasedEnv) -> torch.Tensor:
        """
        Default apply method for the skill.

        Args:
            skill_output: The output of the skill.
            env: The environment.
        """

        self._logger.warning("Action adapter for skill not implemented. Using default apply.")
        return skill_output.action
