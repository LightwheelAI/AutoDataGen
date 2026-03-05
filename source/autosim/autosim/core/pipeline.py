from abc import ABC, abstractmethod
from dataclasses import fields

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

from autosim.capabilities.motion_planning import CuroboPlanner, CuroboPlannerCfg
from autosim.capabilities.navigation import OccupancyMapCfg, get_occupancy_map
from autosim.core.logger import AutoSimLogger
from autosim.core.skill import Skill, SkillGoal
from autosim.skills import (
    AutoSimSkillsExtraCfg,
    CuroboSkillExtraCfg,
    NavigateSkillExtraCfg,
)

from .action_adapter import ActionAdapterBase, ActionAdapterCfg
from .decomposer import Decomposer, DecomposerCfg
from .registration import SkillRegistry
from .types import (
    DecomposeResult,
    EnvExtraInfo,
    OccupancyMap,
    PipelineOutput,
    WorldState,
)


@configclass
class AutoSimPipelineCfg:
    """Configuration for the AutoSim pipeline."""

    decomposer: DecomposerCfg = DecomposerCfg()
    """The decomposer for the AutoSim pipeline."""

    motion_planner: CuroboPlannerCfg = CuroboPlannerCfg()
    """The motion planner for the AutoSim pipeline."""

    occupancy_map: OccupancyMapCfg = OccupancyMapCfg()
    """The occupancy map for the AutoSim pipeline."""

    skills: AutoSimSkillsExtraCfg = AutoSimSkillsExtraCfg()
    """The skills for the AutoSim pipeline."""

    action_adapter: ActionAdapterCfg = ActionAdapterCfg()
    """The action adapter for the AutoSim pipeline."""

    max_steps: int = 500
    """The maximum number of steps to execute one skill."""


class AutoSimPipeline(ABC):
    def __init__(self, cfg: AutoSimPipelineCfg) -> None:
        self.cfg = cfg
        self._logger = AutoSimLogger("AutoSimPipeline")

        self._decomposer: Decomposer = self.cfg.decomposer.class_type(self.cfg.decomposer)

        # load the environment and extra information
        self._env: ManagerBasedEnv = self.load_env()
        self._env_extra_info: EnvExtraInfo = self.get_env_extra_info()
        self._env_id = 0

        # robot and env related information
        self._robot_name = self._env_extra_info.robot_name
        self._robot = self._env.scene[self._robot_name]
        self._eef_link_name = self._env_extra_info.ee_link_name
        self._eef_link_idx = self._robot.data.body_names.index(self._eef_link_name)
        self._robot_base_link_name = self._env_extra_info.robot_base_link_name
        self._robot_base_link_idx = self._robot.data.body_names.index(self._robot_base_link_name)

        # initialize the motion planner
        self._motion_planner: CuroboPlanner = self.cfg.motion_planner.class_type(
            env=self._env,
            robot=self._robot,
            cfg=self.cfg.motion_planner,
            env_id=self._env_id,
        )

        # initialize the occupancy map
        self._occupancy_map: OccupancyMap = get_occupancy_map(self._env, self.cfg.occupancy_map)

        # initialize the action adapter
        self._action_adapter: ActionAdapterBase = self.cfg.action_adapter.class_type(self.cfg.action_adapter)

        # save generated actions
        self._generated_actions = []

    def run(self) -> PipelineOutput:
        """Run the AutoSim pipeline."""

        # decompose the task with cache hit check
        decompose_result: DecomposeResult = self.decompose()

        # execute the pipeline
        pipeline_output = self.execute_skill_sequence(decompose_result)

        return pipeline_output

    @abstractmethod
    def load_env(self) -> ManagerBasedEnv:
        """Load the environment in isaaclab."""

        raise NotImplementedError(f"{self.__class__.__name__}.load_env() must be implemented.")

    @abstractmethod
    def get_env_extra_info(self) -> EnvExtraInfo:
        """Get the extra information from the environment."""

        raise NotImplementedError(f"{self.__class__.__name__}.get_env_extra_info() must be implemented.")

    def reset_env(self) -> None:
        """Reset the environment."""

        self._env.reset()

    def decompose(self) -> DecomposeResult:
        """Decompose the task."""

        if self._decomposer.is_cache_hit(self._env_extra_info.task_name):
            decompose_result: DecomposeResult = self._decomposer.read_cache(self._env_extra_info.task_name)
        else:
            decompose_result: DecomposeResult = self._decomposer.decompose(self._env_extra_info)
            self._decomposer.write_cache(self._env_extra_info.task_name, decompose_result)
        return decompose_result

    def execute_skill_sequence(self, decompose_result: DecomposeResult):
        """Execute the skill sequence."""

        self._check_skill_extra_cfg()
        self.reset_env()

        # TODO: add retry mechanism for skill execution
        for subtask in decompose_result.subtasks:
            for skill_info in subtask.skills:

                skill = SkillRegistry.create(
                    skill_info.skill_type, self.cfg.skills.get(skill_info.skill_type).extra_cfg
                )

                if self._action_adapter.should_skip_apply(skill):
                    self._logger.info(f"Skill {skill_info.skill_type} skipped due to action adapter setting.")
                    continue

                goal = skill.extract_goal_from_info(skill_info, self._env, self._env_extra_info)
                success, steps = self._execute_single_skill(skill, goal)

                if not success:
                    self._logger.error(f"Skill {skill_info.skill_type} execution failed with {steps} steps.")
                    raise ValueError(f"Skill {skill_info.skill_type} execution failed with {steps} steps.")
                self._logger.info(f"Skill {skill_info.skill_type} executed successfully.({steps} steps)")
            self._logger.info(
                f"Subtask {subtask.subtask_name} executed successfully with {len(subtask.skills)} skills."
            )

        self.reset_env()

        # build pipeline output
        return PipelineOutput(success=True, generated_actions=self._generated_actions)

    def _check_skill_extra_cfg(self) -> None:
        """modify the extra configuration of the skills."""

        if self.cfg.skills.moveto.extra_cfg.use_dwa and self.cfg.skills.moveto.extra_cfg.local_planner.dt is None:
            physics_dt = self._env.cfg.sim.dt
            decimation = self._env.cfg.decimation
            self.cfg.skills.moveto.extra_cfg.local_planner.dt = physics_dt * decimation
        for skill_cfg_field in fields(self.cfg.skills):
            skill_cfg = self.cfg.skills.get(skill_cfg_field.name)
            if isinstance(skill_cfg.extra_cfg, CuroboSkillExtraCfg):
                skill_cfg.extra_cfg.curobo_planner = self._motion_planner
            if isinstance(skill_cfg.extra_cfg, NavigateSkillExtraCfg):
                skill_cfg.extra_cfg.occupancy_map = self._occupancy_map

    def _execute_single_skill(self, skill: Skill, goal: SkillGoal) -> tuple[bool, int]:
        """Execute a single skill."""

        world_state: WorldState = self._build_world_state()
        plan_success = skill.plan(world_state, goal)

        steps = 0
        while plan_success and steps < self.cfg.max_steps:
            world_state = self._build_world_state()

            output = skill.step(world_state)

            action = torch.zeros(self._env.action_space.shape, device=self._env.device)
            action[self._env_id, :] = self._action_adapter.apply(skill, output, self._env)

            self._env.step(action)
            self._generated_actions.append(action)

            steps += 1
            if output.done:
                return True, steps

        return False, steps

    def _build_world_state(self) -> WorldState:
        """Build the world state."""

        joint_pos_limits = self._robot.data.joint_pos_limits[self._env_id, :, :]
        lower, upper = joint_pos_limits[:, 0], joint_pos_limits[:, 1]
        robot_joint_pos = torch.clamp(self._robot.data.joint_pos[self._env_id, :], min=lower, max=upper)

        robot_joint_vel = self._robot.data.joint_vel[self._env_id, :]
        robot_ee_pose = self._robot.data.body_link_pose_w[self._env_id, self._eef_link_idx]

        robot_base_pose = self._robot.data.body_link_pose_w[
            self._env_id, self._robot_base_link_idx
        ]  # [x, y, z, qw, qx, qy, qz]
        w, x, y, z = robot_base_pose[3:7]
        sin_yaw = 2 * (w * z + x * y)
        cos_yaw = 1 - 2 * (y**2 + z**2)
        yaw = torch.atan2(sin_yaw, cos_yaw)
        robot_base_pose = torch.stack((robot_base_pose[0], robot_base_pose[1], yaw))  # [x, y, yaw]

        robot_root_pose = self._robot.data.root_pose_w[self._env_id]

        sim_joint_names = self._robot.data.joint_names

        objects_dict = dict()
        for obj_name in self._env.scene.keys():
            obj = self._env.scene[obj_name]
            if hasattr(obj, "data") and hasattr(obj.data, "root_pose_w") and obj_name != self._robot_name:
                objects_dict[obj_name] = obj.data.root_pose_w[self._env_id]

        return WorldState(
            robot_joint_pos=robot_joint_pos,
            robot_joint_vel=robot_joint_vel,
            robot_ee_pose=robot_ee_pose,
            robot_base_pose=robot_base_pose,
            robot_root_pose=robot_root_pose,
            sim_joint_names=sim_joint_names,
            objects=objects_dict,
        )
