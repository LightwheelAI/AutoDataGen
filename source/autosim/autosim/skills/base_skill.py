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

    def _build_activate_joint_state(
        self, full_sim_joint_names: list[str], full_sim_q: torch.Tensor, full_sim_qd: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extract the planner's active joint state from full simulation joint state.

        cuRobo typically plans over a subset of "active joints" (`self._planner.target_joint_names`).
        This helper slices the full simulation joint vectors into that active subset, ordered exactly
        as the planner expects, returning `q` and (optionally) `qd`.

        Args:
            full_sim_joint_names: Joint name list from simulation (index-aligned with `full_sim_q`/`full_sim_qd`).
            full_sim_q: Full simulation joint positions, shape `[num_sim_joints]`.
            full_sim_qd: Optional full simulation joint velocities, shape `[num_sim_joints]`.

        Returns:
            A tuple `(activate_q, activate_qd)` where:
            - `activate_q` is ordered by `self._planner.target_joint_names`, shape `[num_active_joints]`.
            - `activate_qd` is the corresponding velocities if `full_sim_qd` is provided; otherwise `None`.

        Raises:
            ValueError: If any planner target joint is missing from `full_sim_joint_names`.
        """

        activate_q, activate_qd = [], [] if full_sim_qd is not None else None
        for joint_name in self._planner.target_joint_names:
            if joint_name not in full_sim_joint_names:
                raise ValueError(
                    f"Joint {joint_name} in planner activate joints is not in the full simulation joint names."
                )
            sim_joint_idx = full_sim_joint_names.index(joint_name)
            activate_q.append(full_sim_q[sim_joint_idx])
            if full_sim_qd is not None and activate_qd is not None:
                activate_qd.append(full_sim_qd[sim_joint_idx])

        activate_q_tensor = torch.stack(activate_q, dim=0)
        if activate_qd is None:
            return activate_q_tensor, None
        return activate_q_tensor, torch.stack(activate_qd, dim=0)
