from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from curobo.cuda_robot_model.util import load_robot_yaml
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.file_path import ContentPath
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv

from autosim.core.logger import AutoSimLogger

if TYPE_CHECKING:
    from .curobo_planner_cfg import CuroboPlannerCfg


class CuroboPlanner:
    """Motion planner for robot manipulation using cuRobo."""

    def __init__(
        self,
        env: ManagerBasedEnv,
        robot: Articulation,
        cfg: CuroboPlannerCfg,
        env_id: int = 0,
    ) -> None:
        """Initialize the motion planner for a specific environment."""

        self._env = env
        self._robot = robot
        self._env_id = env_id

        self.cfg: CuroboPlannerCfg = cfg

        # Initialize logger
        log_level = logging.DEBUG if self.cfg.debug_planner else logging.INFO
        self.logger = AutoSimLogger("CuroboPlanner", log_level)
        setup_curobo_logger("warn")

        # Configuration operations
        self._refine_config_from_env(env)

        # Load robot configuration
        self.robot_cfg: dict[str, Any] = self._load_robot_config()

        # Create motion generator
        world_cfg = WorldConfig()
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            world_cfg,
            self.tensor_args,
            interpolation_dt=self.cfg.interpolation_dt,
            collision_checker_type=self.cfg.collision_checker_type,
            collision_cache=self.cfg.collision_cache,
            collision_activation_distance=self.cfg.collision_activation_distance,
            num_trajopt_seeds=self.cfg.num_trajopt_seeds,
            num_graph_seeds=self.cfg.num_graph_seeds,
            use_cuda_graph=self.cfg.use_cuda_graph,
            fixed_iters_trajopt=True,
            maximum_trajectory_dt=0.5,
            ik_opt_iters=500,
        )
        self.motion_gen: MotionGen = MotionGen(motion_gen_config)

        self.target_joint_names = self.motion_gen.kinematics.joint_names

        # Create plan configuration with parameters from configuration
        self.plan_config: MotionGenPlanConfig = MotionGenPlanConfig(
            enable_graph=self.cfg.enable_graph,
            enable_graph_attempt=self.cfg.enable_graph_attempt,
            max_attempts=self.cfg.max_planning_attempts,
            time_dilation_factor=self.cfg.time_dilation_factor,
        )

        # Create USD helper
        self.usd_helper = UsdHelper()
        self.usd_helper.load_stage(env.scene.stage)

        # Warm up planner
        self.logger.info("Warming up motion planner...")
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

        # Read static world geometry once
        self._initialize_static_world()

        # Define supported cuRobo primitive types for object discovery and pose synchronization
        self.primitive_types: list[str] = ["mesh", "cuboid", "sphere", "capsule", "cylinder", "voxel", "blox"]

    def _refine_config_from_env(self, env: ManagerBasedEnv):
        """Refine the config from the environment."""

        # Force cuRobo to always use CUDA device regardless of Isaac Lab device
        # This isolates the motion planner from Isaac Lab's device configuration
        if torch.cuda.is_available():
            idx = self.cfg.cuda_device if self.cfg.cuda_device is not None else torch.cuda.current_device()
            self.tensor_args = TensorDeviceType(device=torch.device(f"cuda:{idx}"), dtype=torch.float32)
            self.logger.debug(f"cuRobo motion planner initialized on CUDA device {idx}")
        else:
            self.tensor_args = TensorDeviceType()
            self.logger.warning("CUDA not available, cuRobo using CPU - this may cause device compatibility issues")

        # refine interpolation dt
        self.cfg.interpolation_dt = env.cfg.sim.dt * env.cfg.decimation

    def _load_robot_config(self):
        """Load robot configuration from file or dictionary."""

        if isinstance(self.cfg.robot_config_file, str):
            self.logger.info(f"Loading robot configuration from {self.cfg.robot_config_file}")

            content_path = ContentPath(
                robot_config_root_path=self.cfg.curobo_config_path,
                robot_urdf_root_path=self.cfg.curobo_asset_path,
                robot_asset_root_path=self.cfg.curobo_asset_path,
                robot_config_file=self.cfg.robot_config_file,
            )
            robot_cfg = load_robot_yaml(content_path)
            robot_cfg["robot_cfg"]["kinematics"]["external_asset_path"] = self.cfg.curobo_asset_path

            return robot_cfg
        else:
            self.logger.info("Using custom robot configuration dictionary.")

            return self.cfg.robot_config_file

    def _to_curobo_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to cuRobo device for isolated device management."""

        return tensor.to(device=self.tensor_args.device, dtype=self.tensor_args.dtype)

    def _initialize_static_world(self) -> None:
        """Initialize static world geometry from USD stage (only called once)."""

        env_prim_path = f"/World/envs/env_{self._env_id}"
        robot_prim_path = self.cfg.robot_prim_path or f"{env_prim_path}/Robot"

        only_paths = [f"{env_prim_path}/{sub}" for sub in self.cfg.world_only_subffixes] or [env_prim_path]

        ignore_list = [f"{env_prim_path}/{sub}" for sub in self.cfg.world_ignore_subffixes] or [
            f"{env_prim_path}/target",
            "/World/defaultGroundPlane",
            "/curobo",
        ]
        ignore_list.append(robot_prim_path)

        self._static_world_config = self.usd_helper.get_obstacles_from_stage(
            only_paths=only_paths,
            reference_prim_path=robot_prim_path,
            ignore_substring=ignore_list,
        )
        self._static_world_config = self._static_world_config.get_collision_check_world()
        self.motion_gen.update_world(self._static_world_config)

    def plan_motion(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        current_q: torch.Tensor,
        current_qd: torch.Tensor | None = None,
        link_goals: dict[str, torch.Tensor] | None = None,
    ) -> JointState | None:
        """
        Plan a trajectory to reach a target pose from a current joint state.

        Args:
            target_pos: Target position [x, y, z]
            target_quat: Target quaternion [qw, qx, qy, qz]
            current_q: Current joint positions
            current_qd: Current joint velocities
            link_goals: Optional dictionary mapping link names to target poses for other links

        Returns:
            JointState of the trajectory or None if planning failed
        """

        if current_qd is None:
            current_qd = torch.zeros_like(current_q)
        dof_needed = len(self.target_joint_names)

        # adjust the joint number
        if len(current_q) < dof_needed:
            pad = torch.zeros(dof_needed - len(current_q), dtype=current_q.dtype)
            current_q = torch.concatenate([current_q, pad], axis=0)
            current_qd = torch.concatenate([current_qd, torch.zeros_like(pad)], axis=0)
        elif len(current_q) > dof_needed:
            current_q = current_q[:dof_needed]
            current_qd = current_qd[:dof_needed]

        # build the target pose
        goal = Pose(
            position=self._to_curobo_device(target_pos),
            quaternion=self._to_curobo_device(target_quat),
        )

        # build the current state
        state = JointState(
            position=self._to_curobo_device(current_q),
            velocity=self._to_curobo_device(current_qd) * 0.0,
            acceleration=self._to_curobo_device(current_qd) * 0.0,
            jerk=self._to_curobo_device(current_qd) * 0.0,
            joint_names=self.target_joint_names,
        )

        current_joint_state: JointState = state.get_ordered_joint_state(self.target_joint_names)

        # Prepare link_poses for multi-arm robots
        link_poses = None
        if link_goals is not None:
            # Use provided link goals
            link_poses = {
                link_name: Pose(position=self._to_curobo_device(pose[:3]), quaternion=self._to_curobo_device(pose[3:]))
                for link_name, pose in link_goals.items()
            }
        elif self.cfg.maintain_other_ee_poses:
            # Automatically maintain current poses for other end-effectors
            kin_state = self.motion_gen.compute_kinematics(current_joint_state)
            link_poses = {}
            ee_link = self.motion_gen.kinematics.ee_link
            for link_name in self.motion_gen.kinematics.link_names:
                if link_name != ee_link:
                    link_poses[link_name] = kin_state.link_poses[link_name]
                    self.logger.debug(f"Maintaining current pose for link: {link_name}")

        # execute planning
        result = self.motion_gen.plan_single(
            current_joint_state.unsqueeze(0),
            goal,
            self.plan_config,
            link_poses=link_poses,
        )

        if result.success.item():
            current_plan = result.get_interpolated_plan()
            motion_plan = current_plan.get_ordered_joint_state(self.target_joint_names)

            self.logger.debug(f"planning succeeded with {len(motion_plan.position)} waypoints")
            return motion_plan
        else:
            self.logger.warning(f"planning failed: {result.status}")
            return None

    def reset(self):
        """reset the planner state"""

        self.motion_gen.reset()

    def get_ee_pose(self, current_q: torch.Tensor) -> Pose:
        """Get the end-effector pose of the robot."""

        current_joint_state = JointState(
            position=self._to_curobo_device(current_q), joint_names=self.target_joint_names
        )
        kin_state = self.motion_gen.compute_kinematics(current_joint_state)
        return kin_state.link_poses[self.motion_gen.kinematics.ee_link]
