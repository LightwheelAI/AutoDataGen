from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

"""PIPELINE RELATED TYPES"""


@dataclass
class PipelineOutput:
    """Output of the pipeline execution."""

    success: bool
    """Whether the pipeline execution was successful."""


"""SKILL RELATED TYPES"""


@dataclass
class SkillStatus(Enum):
    """Status of the skill execution."""

    IDLE = "idle"
    """The skill is idle."""
    PLANNING = "planning"
    """The skill is planning."""
    EXECUTING = "executing"
    """The skill is executing."""
    SUCCESS = "success"
    """The skill execution was successful."""
    FAILED = "failed"
    """The skill execution failed."""


@dataclass
class SkillGoal:
    """Goal of the skill."""

    target_object: str | None = None
    """The target object of the skill."""
    target_pose: torch.Tensor | None = None
    """The target pose of the skill."""
    target_joint_pos: torch.Tensor | None = None
    """The target joint positions of the skill."""
    constraints: dict[str, Any] = field(default_factory=dict)
    """The constraints of the skill."""
    params: dict[str, Any] = field(default_factory=dict)
    """The parameters of the skill."""


@dataclass
class SkillOutput:
    """Output of the skill execution."""

    action: torch.Tensor
    """The action of the skill."""
    done: bool
    """Whether the skill execution is done."""
    success: bool
    """Whether the skill execution was successful."""
    info: dict[str, Any] = field(default_factory=dict)
    """The information of the skill execution."""
    trajectory: torch.Tensor | None = None
    """The trajectory of the skill execution."""


"""ENVIRONMENT RELATED TYPES"""


@dataclass
class EnvExtraInfo:
    """Extra information from the environment."""

    task_name: str
    """The name of the task."""
    objects: list[str] | None = None


@dataclass
class WorldState:
    """The unified state representation of the world."""

    robot_joint_pos: torch.Tensor
    """The joint positions of the robot."""
    robot_joint_vel: torch.Tensor
    """The joint velocities of the robot."""
    robot_ee_pose: torch.Tensor
    """The end-effector pose of the robot. [x, y, z, qw, qx, qy, qz]"""
    sim_joint_names: list[str]
    """The joint names of the robot."""

    objects: dict[str, torch.Tensor] = field(default_factory=dict)
    """The state of the objects in the world."""

    rgb: torch.Tensor | None = None
    """The RGB image of the world."""
    depth: torch.Tensor | None = None
    """The depth image of the world."""
    point_cloud: torch.Tensor | None = None
    """The point cloud of the world."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """The metadata of the world state."""

    @property
    def device(self):
        return self.robot_joint_pos.device

    def to(self, device):
        """Move all tensors to device"""
        self.robot_joint_pos = self.robot_joint_pos.to(device)
        self.robot_joint_vel = self.robot_joint_vel.to(device)
        self.robot_ee_pose = self.robot_ee_pose.to(device)
        self.objects = {k: v.to(device) for k, v in self.objects.items()}
        if self.rgb is not None:
            self.rgb = self.rgb.to(device)
        if self.depth is not None:
            self.depth = self.depth.to(device)
        if self.point_cloud is not None:
            self.point_cloud = self.point_cloud.to(device)
        return self


"""DECOMPOSER RELATED TYPES"""


@dataclass
class DecomposeResult:
    """Result of the task decomposition."""

    task_name: str
    """The name of the task."""
    # subtasks: list[SubtaskResult]s
    """The subtasks of the task."""
