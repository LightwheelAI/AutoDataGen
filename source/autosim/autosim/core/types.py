from collections.abc import Iterator
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
    generated_actions: list[torch.Tensor]
    """The generated actions of the pipeline."""


"""SKILL RELATED TYPES"""


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
    extra_target_poses: dict[str, torch.Tensor] | None = None
    """The target poses of the extra end-effectors. dict[link_name, target_pose]."""


@dataclass
class SkillOutput:
    """Output of the skill execution."""

    action: torch.Tensor
    """The action of the skill. shape: [action_dim]"""
    done: bool
    """Whether the skill execution is done."""
    success: bool
    """Whether the skill execution was successful."""
    info: dict[str, Any] = field(default_factory=dict)
    """The information of the skill execution."""


"""ENVIRONMENT RELATED TYPES"""


@dataclass
class EnvExtraInfo:
    """Extra information from the environment."""

    task_name: str
    """The name of the task, need to be the same as the task name in the gymnasium registry."""
    objects: list[str] | None = None
    """The objects in the environment."""
    additional_prompt_contents: str | None = None
    """The additional prompt contents for the task decomposition."""

    robot_name: str = "robot"
    """The name of the robot in the scene."""
    robot_base_link_name: str = "base_link"
    """The name of the base link of the robot (it is not necessarily the root link of the robot)."""
    ee_link_name: str = "ee_link"
    """The name of the end-effector link."""

    object_reach_target_poses: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    """The reach target poses in the objects frame. each object can have a list of reach target poses [x, y, z, qw, qx, qy, qz] in the order of execution."""
    object_extra_reach_target_poses: dict[str, dict[str, list[torch.Tensor]]] = field(default_factory=dict)
    """The extra reach target poses in the objects frame. each object can have a list of extra reach target poses [x, y, z, qw, qx, qy, qz] with ee_name as the key in the order of execution."""

    def __post_init__(self):
        self._object_reach_target_poses_iterator_dict = {
            object_name: self._build_iterator(reach_target_poses)
            for object_name, reach_target_poses in self.object_reach_target_poses.items()
        }
        self._object_extra_reach_target_poses_iterator_dict = {
            object_name: {
                ee_name: self._build_iterator(extra_reach_target_poses)
                for ee_name, extra_reach_target_poses in extra_reach_target_poses.items()
            }
            for object_name, extra_reach_target_poses in self.object_extra_reach_target_poses.items()
        }

    def _build_iterator(self, value_list: list[torch.Tensor]) -> Iterator[torch.Tensor]:
        yield from value_list

    def get_next_reach_target_pose(self, object_name: str) -> torch.Tensor:
        return next(self._object_reach_target_poses_iterator_dict[object_name])

    def get_next_extra_reach_target_pose(self, object_name: str, ee_name: str) -> torch.Tensor:
        return next(self._object_extra_reach_target_poses_iterator_dict[object_name][ee_name])


@dataclass
class WorldState:
    """The unified state representation of the world."""

    robot_joint_pos: torch.Tensor
    """The joint positions of the robot."""
    robot_joint_vel: torch.Tensor
    """The joint velocities of the robot."""
    robot_ee_pose: torch.Tensor
    """The end - effector pose of the robot in the world frame. [x, y, z, qw, qx, qy, qz]"""
    robot_base_pose: torch.Tensor
    """The base pose of the robot in the world frame. [x, y, yaw]"""
    robot_root_pose: torch.Tensor
    """The root pose of the robot in the world frame. [x, y, z, qw, qx, qy, qz]"""
    sim_joint_names: list[str]
    """The joint names of the robot."""
    objects: dict[str, torch.Tensor] = field(default_factory=dict)
    """The state of the objects in the world."""
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
        return self


"""DECOMPOSER RELATED TYPES"""


@dataclass
class ObjectInfo:
    """Information of the object."""

    name: str
    """The name of the object."""
    type: str
    """The type of the object."""
    graspable: bool
    """Whether the object is graspable."""
    initial_location: str
    """The initial location of the object."""
    target_location: str
    """The target location of the object."""
    role: str
    """The role of the object. "manipulated" (needs operation) or "static" (no operation needed)"""


@dataclass
class FixtureInfo:
    """Information of the fixture."""

    name: str
    """The name of the fixture."""
    type: str
    """The type of the fixture."""
    interactive: bool | None = None
    """Whether the fixture is interactive."""
    interaction_type: str | None = None
    """The type of interaction with the fixture."""


@dataclass
class SkillInfo:
    """Information of the skill."""

    step: int
    """The step of the skill, globally sequential across all subtasks"""
    skill_type: str
    """The type of the skill, must be one of the atomic skills"""
    target_object: str
    """The target object of the skill."""
    target_type: str
    """The type of the target. "object", "fixture", "interactive_element", or "position"."""
    description: str
    """The description of the skill."""


@dataclass
class SubtaskResult:
    """Result of the subtask."""

    subtask_id: int
    """The ID of the subtask."""
    subtask_name: str
    """The name of the subtask."""
    description: str
    """The description of the subtask."""
    skills: list[SkillInfo]
    """The skills of the subtask."""


@dataclass
class DecomposeResult:
    """Result of the task decomposition."""

    task_name: str
    """The name of the task."""
    task_description: str
    """The description of the task."""
    parent_classes: list[str]
    """The parent classes of the task."""
    objects: list[ObjectInfo]
    """The objects of the task."""
    fixtures: list[FixtureInfo]
    """The fixtures of the task."""
    interactive_elements: list[str]
    """The interactive elements of the task."""
    subtasks: list[SubtaskResult]
    """The subtasks of the task."""
    success_conditions: list[str]
    """The success conditions of the task."""
    total_steps: int
    """The total number of steps in the task."""
    skill_sequence: list[str]
    """The sequence of skills in the task."""


"""NAVIGATION RELATED TYPES"""


@dataclass
class MapBounds:
    """Bounds of the map. [min_x, max_x, min_y, max_y]"""

    min_x: float
    """The minimum x - coordinate of the map."""
    max_x: float
    """The maximum x - coordinate of the map."""
    min_y: float
    """The minimum y - coordinate of the map."""
    max_y: float
    """The maximum y - coordinate of the map."""


@dataclass
class OccupancyMap:
    """Occupancy map of the environment."""

    occupancy_map: torch.Tensor
    """The occupancy map of the environment. 2D array of shape[height, width] 0: free, 1: occupied, -1: unknown."""
    resolution: float
    """The resolution of the occupancy map, cell size in meters."""
    origin: tuple[float, float]
    """The origin of the occupancy map, (x, y)."""
    map_bounds: MapBounds
    """The bounds of the occupancy map."""
    floor_bounds: MapBounds
    """The bounds of the floor."""
