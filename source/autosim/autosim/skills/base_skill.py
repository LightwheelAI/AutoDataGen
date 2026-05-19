import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

from autosim.capabilities.motion_planning import CuroboPlanner
from autosim.core.skill import Skill, SkillExtraCfg
from autosim.core.types import (
    EnvExtraInfo,
    SkillGoal,
    SkillInfo,
    SkillOutput,
    WorldState,
)
from autosim.utils.debug_util import create_marker, visualize_marker


@configclass
class GripperSkillExtraCfg(SkillExtraCfg):
    """Extra configuration for the gripper skill."""

    gripper_value: float = 0.0
    """The value of the gripper."""
    duration: int = 20
    """The duration of the gripper."""


class GripperSkillBase(Skill):
    """Base class for gripper skills open/close skills."""

    def __init__(self, extra_cfg: GripperSkillExtraCfg) -> None:
        super().__init__(extra_cfg)

        self._gripper_value = extra_cfg.gripper_value
        self._duration = extra_cfg.duration
        self._step_count = 0
        self._target_object_name = None

    def extract_goal_from_info(
        self, skill_info: SkillInfo, env: ManagerBasedEnv, env_extra_info: EnvExtraInfo
    ) -> SkillGoal:
        """Return the target object name."""

        return SkillGoal(target_object=skill_info.target_object)

    def execute_plan(self, state: WorldState, goal: SkillGoal) -> bool:
        """Execute the plan of the gripper skill."""

        self._target_object_name = goal.target_object
        self._step_count = 0
        return True

    def step(self, state: WorldState) -> SkillOutput:
        """Step the gripper skill.

        Args:
            state: The current state of the world.

        Returns:
            The output of the skill execution.
                action: The action to be applied to the environment. [gripper_value]
        """

        done = self._step_count >= self._duration
        self._step_count += 1

        return SkillOutput(
            action=torch.tensor([self._gripper_value], device=state.device),
            done=done,
            success=done,
            info={"step": self._step_count, "target_object": self._target_object_name},
        )

    def reset(self) -> None:
        """Reset the gripper skill."""

        super().reset()
        self._step_count = 0
        self._target_object_name = None


@configclass
class CuroboSkillExtraCfg(SkillExtraCfg):
    """Extra configuration for the curobo skill."""

    curobo_planner: CuroboPlanner | None = None
    """The curobo planner for the skill."""

    debug_target_pose: bool = False
    """Whether to debug the target pose."""

    extra_target_link_names: list[str] = []
    """Additional cuRobo link names constrained during planning."""

    extra_target_mode: str = "keep_current"
    """How additional cuRobo link goals are generated."""

    return_link_poses_in_robot_root_frame: bool = True
    """Whether to return the link poses in the robot root frame."""

    def __post_init__(self) -> None:
        supported_modes = {"keep_current", "keep_relative_offset", "keep_initial_relative_offset"}
        if self.extra_target_mode not in supported_modes:
            raise ValueError(
                f"Unsupported extra_target_mode: {self.extra_target_mode}. Supported modes: {sorted(supported_modes)}"
            )
        if len(self.extra_target_link_names) != len(set(self.extra_target_link_names)):
            raise ValueError("extra_target_link_names must not contain duplicates.")


class CuroboSkillBase(Skill):
    """Base class for skills dependent on curobo."""

    def __init__(self, extra_cfg: CuroboSkillExtraCfg) -> None:
        super().__init__(extra_cfg)
        self._planner = extra_cfg.curobo_planner

        self._target_poses = dict()
        if self.cfg.extra_cfg.debug_target_pose:
            create_marker("target_pose")

    def visualize_debug_target_pose(self):
        """Visualize the debug target pose."""

        if self.cfg.extra_cfg.debug_target_pose:
            visualize_marker("target_pose", self._target_poses["target_pose"])
