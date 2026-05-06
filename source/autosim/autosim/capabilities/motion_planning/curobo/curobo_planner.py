from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from curobo.cuda_robot_model.util import load_robot_yaml
from curobo.geom.types import WorldConfig
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.types.base import TensorDeviceType
from curobo.types.file_path import ContentPath
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_assets_path, get_configs_path
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import subtract_frame_transforms

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
        self._logger = AutoSimLogger("CuroboPlanner", log_level)
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
            self_collision_check=self.cfg.self_collision_check,
            self_collision_opt=self.cfg.self_collision_opt,
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
        self._logger.info("Warming up motion planner...")
        self.motion_gen.warmup(enable_graph=self.cfg.use_cuda_graph, warmup_js_trajopt=False)

        # Read static world geometry once
        self._initialize_static_world()

        # Cache for dynamic world synchronization.
        self._cached_object_mappings: dict[str, str] | None = None

        # Define supported cuRobo primitive types for object discovery and pose synchronization
        self.primitive_types: list[str] = ["mesh", "cuboid", "sphere", "capsule", "cylinder", "voxel", "blox"]

    def _refine_config_from_env(self, env: ManagerBasedEnv):
        """Refine the config from the environment."""

        # Force cuRobo to always use CUDA device regardless of Isaac Lab device
        # This isolates the motion planner from Isaac Lab's device configuration
        if torch.cuda.is_available():
            idx = self.cfg.cuda_device if self.cfg.cuda_device is not None else torch.cuda.current_device()
            self.tensor_args = TensorDeviceType(device=torch.device(f"cuda:{idx}"), dtype=torch.float32)
            self._logger.debug(f"cuRobo motion planner initialized on CUDA device {idx}")
        else:
            self.tensor_args = TensorDeviceType()
            self._logger.warning("CUDA not available, cuRobo using CPU - this may cause device compatibility issues")

        # refine interpolation dt
        self.cfg.interpolation_dt = env.cfg.sim.dt * env.cfg.decimation

    def _load_robot_config(self):
        """Load robot configuration from file or dictionary."""

        if isinstance(self.cfg.robot_config_file, str):
            self._logger.info(f"Loading robot configuration from {self.cfg.robot_config_file}")

            curobo_config_path = self.cfg.curobo_config_path or f"{get_configs_path()}/robot"
            curobo_asset_path = self.cfg.curobo_asset_path or get_assets_path()

            content_path = ContentPath(
                robot_config_root_path=curobo_config_path,
                robot_urdf_root_path=curobo_asset_path,
                robot_asset_root_path=curobo_asset_path,
                robot_config_file=self.cfg.robot_config_file,
            )
            robot_cfg = load_robot_yaml(content_path)
            robot_cfg["robot_cfg"]["kinematics"]["external_asset_path"] = curobo_asset_path

            return robot_cfg
        else:
            self._logger.info("Using custom robot configuration dictionary.")

            return self.cfg.robot_config_file

    def _to_curobo_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to cuRobo device for isolated device management."""

        return tensor.to(device=self.tensor_args.device, dtype=self.tensor_args.dtype)

    def _initialize_static_world(self) -> None:
        """Initialize static world geometry from USD stage (only called once)."""

        env_prim_path = f"/World/envs/env_{self._env_id}"
        robot_prim_path = self.cfg.robot_prim_path or f"{env_prim_path}/Robot"

        world_only_subffixes = self.cfg.world_only_subffixes or []
        only_paths = [f"{env_prim_path}/{sub}" for sub in world_only_subffixes]

        world_ignore_subffixes = self.cfg.world_ignore_subffixes or []
        ignore_list = [f"{env_prim_path}/{sub}" for sub in world_ignore_subffixes] or [
            f"{env_prim_path}/target",
            "/World/defaultGroundPlane",
            "/curobo",
        ]
        ignore_list.append(robot_prim_path)

        world_cfg = self.usd_helper.get_obstacles_from_stage(
            only_paths=only_paths,
            reference_prim_path=robot_prim_path,
            ignore_substring=ignore_list,
        )
        self._static_world_config = world_cfg.get_collision_check_world()
        self.motion_gen.update_world(self._static_world_config)
        self._invalidate_object_mapping_cache()

    def _get_object_mappings(self) -> dict[str, list[str]]:
        """Map IsaacLab scene object names to cuRobo world obstacle names.

        Returns:
            Dictionary mapping IsaacLab scene object names to list of cuRobo world obstacle names.
        """

        if self._cached_object_mappings is not None:
            return self._cached_object_mappings

        world_model = self.motion_gen.world_coll_checker.world_model
        rigid_objects = self._env.scene.rigid_objects

        env_prefix = f"/World/envs/env_{self._env_id}/Scene"
        world_object_paths: list[str] = []
        for primitive_type in self.primitive_types:
            primitive_list = getattr(world_model, primitive_type, None)
            if not primitive_list:
                continue
            for primitive in primitive_list:
                primitive_name = primitive.name
                if env_prefix in primitive_name:
                    world_object_paths.append(primitive_name)

        mappings: dict[str, list[str]] = {}
        for object_name in rigid_objects.keys():
            mappings[object_name] = []
            for world_path in world_object_paths:
                if world_path.startswith(f"{env_prefix}/{object_name}"):
                    mappings[object_name].append(world_path)

        self._cached_object_mappings = mappings
        self._logger.debug(f"Object mappings built: {mappings}")
        return mappings

    def _invalidate_object_mapping_cache(self) -> None:
        """Invalidate cached object-name mapping used for dynamic sync."""

        self._cached_object_mappings = None

    def sync_dynamic_objects(self) -> int:
        """Synchronize dynamic object poses into cuRobo world model.

        Returns:
            Number of obstacles whose pose was updated.
        """

        object_mappings = self._get_object_mappings()
        if not object_mappings:
            return 0

        rigid_objects = self._env.scene.rigid_objects
        robot_root_pos_in_world, robot_root_quat_in_world = self._robot.data.root_pos_w, self._robot.data.root_quat_w

        updated_count = 0
        for object_name, world_obstacle_names in object_mappings.items():
            obj = rigid_objects[object_name]
            # NOTE: cuRobo world model is in the robot-root frame
            obj_pos_in_world, obj_quat_in_world = obj.data.root_pos_w, obj.data.root_quat_w
            obj_pos_in_robot_root, obj_quat_in_robot_root = subtract_frame_transforms(
                robot_root_pos_in_world, robot_root_quat_in_world, obj_pos_in_world, obj_quat_in_world
            )
            obj_pose = Pose(
                position=self._to_curobo_device(obj_pos_in_robot_root[self._env_id]),
                quaternion=self._to_curobo_device(obj_quat_in_robot_root[self._env_id]),
            )
            for world_obstacle_name in world_obstacle_names:
                self.motion_gen.world_coll_checker.update_obstacle_pose(
                    world_obstacle_name,
                    obj_pose,
                    env_idx=self._env_id,
                    update_cpu_reference=True,
                )
                updated_count += 1

        return updated_count

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

        # Dynamic object pose sync before planning (cheap, incremental).
        if self.cfg.enable_dynamic_world_sync:
            self.sync_dynamic_objects()

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

        joint_limits = self.motion_gen.kinematics.get_joint_limits()
        current_q = torch.clamp(
            self._to_curobo_device(current_q), joint_limits.position[0], joint_limits.position[1]
        ).to(current_q.device)

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

        # Build per-call plan config: clone only when we need to attach a pose_cost_metric
        # so the shared self.plan_config is never mutated.
        if self.cfg.reach_partial_pose_weight is not None:
            weights = torch.tensor(
                self.cfg.reach_partial_pose_weight,
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            pose_metric = PoseCostMetric(reach_partial_pose=True, reach_vec_weight=weights)
            active_plan_config = self.plan_config.clone()
            active_plan_config.pose_cost_metric = pose_metric
            self._logger.debug(f"reach_partial_pose_weight applied: {self.cfg.reach_partial_pose_weight}")
        else:
            active_plan_config = self.plan_config

        # execute planning
        result = self.motion_gen.plan_single(
            current_joint_state.unsqueeze(0),
            goal,
            active_plan_config,
            link_poses=link_poses,
        )

        if result.success.item():
            current_plan = result.get_interpolated_plan()
            motion_plan = current_plan.get_ordered_joint_state(self.target_joint_names)

            self._logger.debug(f"planning succeeded with {len(motion_plan.position)} waypoints")
            return motion_plan
        else:
            self._logger.warning(f"planning failed: {result.status}")
            return None

    def plan_motion_batch(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        current_q: torch.Tensor,
        current_qd: torch.Tensor | None = None,
        link_goals: dict[str, torch.Tensor] | None = None,
    ):
        """
        Plan trajectories for a batch of target poses from the same start joint state.

        This uses cuRobo's batch API (`MotionGen.plan_batch`) under the hood.

        Args:
            target_pos: Tensor of shape [K, 3], in robot root frame.
            target_quat: Tensor of shape [K, 4] in [qw, qx, qy, qz], in robot root frame.
            current_q: Tensor of shape [dof], current joint positions.
            current_qd: Tensor of shape [dof], current joint velocities. Defaults to zeros.
            link_goals: Optional dict mapping extra link names to tensors of shape [K, 7]
                ([x, y, z, qw, qx, qy, qz], robot root frame) for multi-arm robots. Each entry
                specifies the simultaneous target pose of that link for every sample in the batch.

        Returns:
            MotionGenResult (cuRobo). Check `result.success[k]` for each batch index.

        Note:
            `time_dilation_factor` is always suppressed for batch planning because cuRobo's
            `retime_trajectory` does not support batch results.
        """

        # Dynamic object pose sync before planning (cheap, incremental).
        if self.cfg.enable_dynamic_world_sync:
            self.sync_dynamic_objects()

        if target_pos.ndim != 2 or target_pos.shape[-1] != 3:
            raise ValueError(f"target_pos must have shape [K, 3], got {tuple(target_pos.shape)}")
        if target_quat.ndim != 2 or target_quat.shape[-1] != 4:
            raise ValueError(f"target_quat must have shape [K, 4], got {tuple(target_quat.shape)}")
        if target_pos.shape[0] != target_quat.shape[0]:
            raise ValueError(
                f"Batch size mismatch: target_pos has {target_pos.shape[0]}, target_quat has {target_quat.shape[0]}"
            )
        k = target_pos.shape[0]
        if link_goals is not None:
            for ee_name, poses in link_goals.items():
                if poses.ndim != 2 or poses.shape != (k, 7):
                    raise ValueError(f"link_goals['{ee_name}'] must have shape [{k}, 7], got {tuple(poses.shape)}")

        if current_qd is None:
            current_qd = torch.zeros_like(current_q)

        dof_needed = len(self.target_joint_names)
        if len(current_q) < dof_needed:
            pad = torch.zeros(dof_needed - len(current_q), dtype=current_q.dtype, device=current_q.device)
            current_q = torch.concatenate([current_q, pad], axis=0)
            current_qd = torch.concatenate([current_qd, torch.zeros_like(pad)], axis=0)
        elif len(current_q) > dof_needed:
            current_q = current_q[:dof_needed]
            current_qd = current_qd[:dof_needed]

        goal = Pose(
            position=self._to_curobo_device(target_pos),
            quaternion=self._to_curobo_device(target_quat),
        )

        start_state = JointState(
            position=self._to_curobo_device(current_q).view(1, -1),
            velocity=self._to_curobo_device(current_qd).view(1, -1) * 0.0,
            acceleration=self._to_curobo_device(current_qd).view(1, -1) * 0.0,
            jerk=self._to_curobo_device(current_qd).view(1, -1) * 0.0,
            joint_names=self.target_joint_names,
        ).repeat_seeds(int(target_pos.shape[0]))

        link_poses = None
        if link_goals is not None:
            link_poses = {
                ee_name: Pose(
                    position=self._to_curobo_device(poses[:, :3]),
                    quaternion=self._to_curobo_device(poses[:, 3:]),
                )
                for ee_name, poses in link_goals.items()
            }
        # plan_batch does not support retime_trajectory (batch result); disable time_dilation_factor
        batch_plan_config = self.plan_config.clone()
        batch_plan_config.time_dilation_factor = None
        return self.motion_gen.plan_batch(start_state, goal, batch_plan_config, link_poses=link_poses)

    def solve_ik_batch(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        link_goals: dict[str, torch.Tensor] | None = None,
    ):
        """
        Solve IK for a batch of target poses without trajectory optimization.

        Faster than plan_motion_batch for reachability checking since it skips
        trajectory optimization entirely.

        Args:
            target_pos: Tensor of shape [K, 3], in robot root frame.
            target_quat: Tensor of shape [K, 4] in [qw, qx, qy, qz], in robot root frame.
            link_goals: Optional dict mapping extra link names to tensors of shape [K, 7]
                ([x, y, z, qw, qx, qy, qz], robot root frame) for multi-arm robots.

        Returns:
            IKResult from cuRobo. Check result.success[k], result.position_error[k],
            result.rotation_error[k] for each batch index.
        """

        # Dynamic object pose sync before planning (cheap, incremental).
        if self.cfg.enable_dynamic_world_sync:
            self.sync_dynamic_objects()

        if target_pos.ndim != 2 or target_pos.shape[-1] != 3:
            raise ValueError(f"target_pos must have shape [K, 3], got {tuple(target_pos.shape)}")
        if target_quat.ndim != 2 or target_quat.shape[-1] != 4:
            raise ValueError(f"target_quat must have shape [K, 4], got {tuple(target_quat.shape)}")
        k = target_pos.shape[0]
        if link_goals is not None:
            for ee_name, poses in link_goals.items():
                if poses.ndim != 2 or poses.shape != (k, 7):
                    raise ValueError(f"link_goals['{ee_name}'] must have shape [{k}, 7], got {tuple(poses.shape)}")

        goal = Pose(
            position=self._to_curobo_device(target_pos),
            quaternion=self._to_curobo_device(target_quat),
        )
        link_poses = None
        if link_goals is not None:
            link_poses = {
                ee_name: Pose(
                    position=self._to_curobo_device(poses[:, :3]),
                    quaternion=self._to_curobo_device(poses[:, 3:]),
                )
                for ee_name, poses in link_goals.items()
            }
        return self.motion_gen.ik_solver.solve_batch(goal, link_poses=link_poses)

    def plan_to_joint_config(
        self,
        target_q: torch.Tensor,
        current_q: torch.Tensor,
        current_qd: torch.Tensor | None = None,
    ) -> JointState | None:
        """Plan a joint-space trajectory to a target joint configuration.

        Unlike plan_motion which targets a Cartesian pose, this plans directly
        in joint space using cuRobo's plan_single_js. Used by RetractSkill to
        move the robot to a predefined retract configuration.

        Args:
            target_q: Target joint positions, shape [dof].
            current_q: Current joint positions, shape [dof].
            current_qd: Current joint velocities, shape [dof]. Defaults to zeros.

        Returns:
            JointState of the trajectory or None if planning failed.
        """

        if current_qd is None:
            current_qd = torch.zeros_like(current_q)

        dof_needed = len(self.target_joint_names)
        for name, q in [("current_q", current_q), ("target_q", target_q)]:
            if len(q) < dof_needed:
                pad = torch.zeros(dof_needed - len(q), dtype=q.dtype)
                q = torch.concatenate([q, pad], axis=0)
            elif len(q) > dof_needed:
                q = q[:dof_needed]
            if name == "current_q":
                current_q = q
            else:
                target_q = q

        if len(current_qd) < dof_needed:
            current_qd = torch.concatenate(
                [current_qd, torch.zeros(dof_needed - len(current_qd), dtype=current_qd.dtype)]
            )
        elif len(current_qd) > dof_needed:
            current_qd = current_qd[:dof_needed]

        start_state = JointState(
            position=self._to_curobo_device(current_q),
            velocity=self._to_curobo_device(current_qd) * 0.0,
            acceleration=self._to_curobo_device(current_qd) * 0.0,
            jerk=self._to_curobo_device(current_qd) * 0.0,
            joint_names=self.target_joint_names,
        )
        goal_state = JointState(
            position=self._to_curobo_device(target_q),
            velocity=self._to_curobo_device(torch.zeros_like(target_q)),
            acceleration=self._to_curobo_device(torch.zeros_like(target_q)),
            jerk=self._to_curobo_device(torch.zeros_like(target_q)),
            joint_names=self.target_joint_names,
        )

        start_state = start_state.get_ordered_joint_state(self.target_joint_names)
        goal_state = goal_state.get_ordered_joint_state(self.target_joint_names)

        result = self.motion_gen.plan_single_js(
            start_state.unsqueeze(0),
            goal_state.unsqueeze(0),
            self.plan_config.clone(),
        )

        if result.success.item():
            current_plan = result.get_interpolated_plan()
            motion_plan = current_plan.get_ordered_joint_state(self.target_joint_names)
            self._logger.debug(f"joint-space planning succeeded with {len(motion_plan.position)} waypoints")
            return motion_plan
        else:
            self._logger.warning(f"joint-space planning failed: {result.status}")
            return None

    def reset(self):
        """reset the planner state"""

        self.motion_gen.reset()

    def get_ee_pose(self, current_q: torch.Tensor) -> Pose:
        """Get the end-effector pose of the robot."""

        return self.get_link_pose(current_q, self.motion_gen.kinematics.ee_link)

    def get_link_pose(self, current_q: torch.Tensor, link_name: str) -> Pose:
        """Get the pose of a specific link in the robot root frame."""

        return self.get_link_poses(current_q, [link_name])[link_name]

    def get_link_poses(self, current_q: torch.Tensor, link_names: list[str]) -> dict[str, Pose]:
        """Get the poses of specific links in the robot root frame."""

        current_joint_state = JointState(
            position=self._to_curobo_device(current_q), joint_names=self.target_joint_names
        )
        kin_state = self.motion_gen.compute_kinematics(current_joint_state)

        missing_link_names = [link_name for link_name in link_names if link_name not in kin_state.link_poses]
        if missing_link_names:
            raise ValueError(
                f"Unknown cuRobo link name(s): {missing_link_names}. Available links:"
                f" {list(kin_state.link_poses.keys())}"
            )

        return {link_name: kin_state.link_poses[link_name] for link_name in link_names}
