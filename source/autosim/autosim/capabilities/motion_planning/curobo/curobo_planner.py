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
from curobo.util.usd_helper import UsdHelper, get_prim_world_pose
from curobo.util_file import get_assets_path, get_configs_path
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms
from pxr import UsdGeom, UsdPhysics

from autosim.core.logger import AutoSimLogger
from autosim.utils.data_util import as_torch, convert_quat

if TYPE_CHECKING:
    from .curobo_planner_cfg import CuroboPlannerCfg


class CuroboPlanner:
    """Motion planner for robot manipulation using cuRobo."""

    # Identity offset for primitives that should directly follow link pose
    _IDENTITY_OFFSET_POS = torch.zeros(3)
    _IDENTITY_OFFSET_QUAT = torch.tensor([0.0, 0.0, 0.0, 1.0])  # x, y, z, w (xyzw)

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

        # Cache frequently used paths
        self._env_prim_path = f"/World/envs/env_{self._env_id}"
        self._env_scene_prefix = (
            f"{self._env_prim_path}/{self.cfg.env_scene_prefix}" if self.cfg.env_scene_prefix else self._env_prim_path
        )

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

        # Define supported cuRobo primitive types for object discovery and pose synchronization
        self.primitive_types: list[str] = ["mesh", "cuboid", "sphere", "capsule", "cylinder", "voxel", "blox"]

        # Cache for dynamic world synchronization.
        self._cached_object_mappings: dict[str, str] | None = None

        # Current target object for selective collision checking
        self._current_target_object: str | None = None

        # Cache for articulated primitive offsets: {primitive_name: (link_name, offset_pos, offset_quat)}
        self._articulated_primitive_offsets: dict[str, tuple[str, torch.Tensor, torch.Tensor]] = {}

        # Read static world geometry once
        self._initialize_static_world()

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
        """Initialize static world geometry from USD stage.

        This method is called once during initialization. It:
        1. Reads obstacle geometry from USD stage via cuRobo's UsdHelper
        2. Filters primitives based on collision_enable_substrings (if configured)
        3. Applies ancestor prim scale correction for non-mesh primitives
        4. Loads the filtered world into cuRobo's collision checker
        5. Builds offset cache for articulated object primitives
        """

        robot_prim_path = self.cfg.robot_prim_path or f"{self._env_prim_path}/Robot"

        world_only_subffixes = self.cfg.world_only_subffixes or []
        only_paths = [f"{self._env_prim_path}/{sub}" for sub in world_only_subffixes]

        world_ignore_subffixes = self.cfg.world_ignore_subffixes or []
        ignore_list = [f"{self._env_prim_path}/{sub}" for sub in world_ignore_subffixes] or [
            f"{self._env_prim_path}/target",
            "/World/defaultGroundPlane",
            "/curobo",
        ]
        ignore_list.append(robot_prim_path)

        world_cfg = self.usd_helper.get_obstacles_from_stage(
            only_paths=only_paths,
            reference_prim_path=robot_prim_path,
            ignore_substring=ignore_list,
        )

        # Remove primitives that don't match collision_enable_substrings before loading into cuRobo
        self._filter_world_config(world_cfg)

        # Compensate for ancestor prim scale that cuRobo's USD reader misses on non-mesh primitives
        self._apply_ancestor_scale_to_world_config(world_cfg)

        # Load filtered world geometry into cuRobo collision checker
        self._static_world_config = world_cfg.get_collision_check_world()
        self.motion_gen.update_world(self._static_world_config)
        self._invalidate_object_mapping_cache()

        # Build offset cache for articulated primitives (relative to their parent link)
        # This cache is used later to sync articulated object poses before each planning call
        self._build_articulated_primitive_offsets()

    def _filter_world_config(self, world_cfg) -> None:
        """Remove primitives from WorldConfig that don't match collision_enable_substrings.

        Modifies world_cfg in-place before it is loaded into cuRobo, so filtered
        primitives never enter the collision checker (saves memory and computation).

        Args:
            world_cfg: WorldConfig object from cuRobo's UsdHelper.get_obstacles_from_stage()

        Note:
            If collision_enable_substrings is None, all primitives are kept (no filtering).
        """

        substrings = self.cfg.collision_enable_substrings
        if substrings is None:
            return

        for attr in ["mesh", "cuboid", "sphere", "capsule", "cylinder", "voxel", "blox"]:
            primitive_list = getattr(world_cfg, attr, None)
            if not primitive_list:
                continue
            filtered = [p for p in primitive_list if any(sub in p.name for sub in substrings)]
            setattr(world_cfg, attr, filtered)

    def _apply_ancestor_scale_to_world_config(self, world_cfg) -> None:
        """Apply ancestor prim scale to non-mesh primitives in WorldConfig.

        cuRobo's USD reader handles scale differently per primitive type:
        - Mesh: correctly applies full world scale (t_scale) to vertices — no fix needed
        - Cuboid: uses prim's own xformOp:scale as dims — missing ancestor scale
        - Sphere: uses prim's own xformOp:scale for radius — missing ancestor scale
        - Cylinder/Capsule: ignores ALL scale (own + ancestor) — missing full world scale

        This method reads the full world scale from USD and compensates for what cuRobo missed.
        Modifies world_cfg in-place.
        """

        stage = self.usd_helper.stage
        xform_cache = UsdGeom.XformCache()

        for cuboid in world_cfg.cuboid or []:
            prim = stage.GetPrimAtPath(cuboid.name)
            if not prim.IsValid():
                continue
            _, t_scale = get_prim_world_pose(xform_cache, prim)
            # cuRobo already applied own_scale as dims, so we only need ancestor contribution
            own_scale = prim.GetAttribute("xformOp:scale").Get()
            if own_scale is None:
                continue
            own_scale = list(own_scale)
            ancestor_scale = [t / o if o != 0 else 1.0 for t, o in zip(t_scale, own_scale)]
            if all(abs(s - 1.0) < 1e-6 for s in ancestor_scale):
                continue
            cuboid.dims = [d * s for d, s in zip(cuboid.dims, ancestor_scale)]

        for sphere in world_cfg.sphere or []:
            prim = stage.GetPrimAtPath(sphere.name)
            if not prim.IsValid():
                continue
            _, t_scale = get_prim_world_pose(xform_cache, prim)
            # cuRobo applied max(own_scale) to radius, so divide it out
            own_scale = prim.GetAttribute("xformOp:scale").Get()
            own_max = max(list(own_scale)) if own_scale is not None else 1.0
            ancestor_factor = max(t_scale) / own_max if own_max != 0 else 1.0
            if abs(ancestor_factor - 1.0) < 1e-6:
                continue
            sphere.radius *= ancestor_factor

        for prim_obj in list(world_cfg.cylinder or []) + list(world_cfg.capsule or []):
            prim = stage.GetPrimAtPath(prim_obj.name)
            if not prim.IsValid():
                continue
            _, t_scale = get_prim_world_pose(xform_cache, prim)
            if all(abs(s - 1.0) < 1e-6 for s in t_scale):
                continue
            # cuRobo applied NO scale at all for cylinder/capsule, so full t_scale is the correction
            scale_z = t_scale[2]
            scale_xy = max(t_scale[0], t_scale[1])
            prim_obj.radius *= scale_xy
            if hasattr(prim_obj, "height"):  # cylinder
                prim_obj.height *= scale_z
            if hasattr(prim_obj, "base"):  # capsule
                prim_obj.base = [v * scale_z for v in prim_obj.base]
                prim_obj.tip = [v * scale_z for v in prim_obj.tip]

    def _collect_primitives_by_prefix(self, prefix: str) -> list[str]:
        """Collect all primitive names from cuRobo world model that start with given prefix.

        Args:
            prefix: Prefix to filter primitive names (e.g., "/World/envs/env_0/Scene/Refrigerator050")

        Returns:
            List of primitive names matching the prefix.
        """
        world_model = self.motion_gen.world_coll_checker.world_model
        primitives = []
        for primitive_type in self.primitive_types:
            primitive_list = getattr(world_model, primitive_type, None)
            if not primitive_list:
                continue
            for primitive in primitive_list:
                if primitive.name.startswith(prefix):
                    primitives.append(primitive.name)
        return primitives

    def _get_articulated_link_poses_from_usd(
        self,
    ) -> dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """Get initial link poses for all articulated objects from USD stage in robot root frame.

        This method reads link poses directly from the USD stage, which represents the
        initial/reference geometry. Used during initialization to compute primitive offsets
        that are consistent with cuRobo's own primitive poses (also read from USD).

        Returns:
            Nested dict: {obj_name: {link_name: (pos, quat)}}
            Poses are on cuRobo device and in robot root frame.
        """
        stage = self.usd_helper.stage
        xform_cache = UsdGeom.XformCache()
        world_only_subffixes_paths = [f"{self._env_prim_path}/{sub}" for sub in self.cfg.world_only_subffixes or []]

        robot_root_pos_w = as_torch(self._robot.data.root_pos_w)[self._env_id].unsqueeze(0)  # [1, 3]
        robot_root_quat_w = as_torch(self._robot.data.root_quat_w)[self._env_id].unsqueeze(0)  # [1, 4]

        result = {}

        for obj_path in world_only_subffixes_paths:
            obj_name = obj_path.split("/")[-1]
            obj_prim = stage.GetPrimAtPath(obj_path)
            if not obj_prim.IsValid():
                self._logger.warning(f"Articulated object prim not found at {obj_path}")
                continue
            if not UsdPhysics.ArticulationRootAPI(obj_prim):
                self._logger.warning(f"Prim at {obj_path} is not an articulated object, skipping")
                continue

            link_poses = {}
            for child_prim in obj_prim.GetChildren():
                link_name = child_prim.GetName()
                if not child_prim.IsValid():
                    continue

                link_mat, _ = get_prim_world_pose(xform_cache, child_prim)
                link_pose = Pose.from_matrix(torch.tensor(link_mat, dtype=torch.float32).unsqueeze(0))
                link_pos_w = link_pose.position  # [1, 3]
                link_quat_w = convert_quat(link_pose.quaternion, to="xyzw")  # cuRobo wxyz → xyzw

                link_pos_in_robot, link_quat_in_robot = subtract_frame_transforms(
                    robot_root_pos_w,
                    robot_root_quat_w,
                    link_pos_w,
                    link_quat_w,
                )

                link_poses[link_name] = (
                    self._to_curobo_device(link_pos_in_robot.squeeze(0)),
                    self._to_curobo_device(link_quat_in_robot.squeeze(0)),
                )

            if link_poses:
                result[obj_name] = link_poses

        return result

    def _get_articulated_link_poses(
        self,
    ) -> dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """Get current link poses for all articulated objects from Isaac Lab scene in robot root frame.

        This method reads link poses from the Isaac Lab simulation state, which reflects
        the current joint configuration. Used during runtime to sync articulated obstacles.

        Returns:
            Nested dict: {obj_name: {link_name: (pos, quat)}}
            Poses are on cuRobo device and in robot root frame.
        """
        articulations = self._env.scene.articulations
        world_only_subffixes_paths = [f"{self._env_prim_path}/{sub}" for sub in self.cfg.world_only_subffixes or []]

        robot_root_pos_w = as_torch(self._robot.data.root_pos_w)
        robot_root_quat_w = as_torch(self._robot.data.root_quat_w)

        result = {}

        for obj_name, articulation in articulations.items():
            if f"{self._env_scene_prefix}/{obj_name}" not in world_only_subffixes_paths:
                continue

            body_pos_w, body_quat_w = as_torch(articulation.data.body_pos_w), as_torch(articulation.data.body_quat_w)
            body_count = body_pos_w.shape[1]
            body_pos_in_robot, body_quat_in_robot = subtract_frame_transforms(
                robot_root_pos_w.repeat(1, body_count, 1),
                robot_root_quat_w.repeat(1, body_count, 1),
                body_pos_w,
                body_quat_w,
            )

            link_poses = {}
            for link_idx, link_name in enumerate(articulation.data.body_names):
                link_poses[link_name] = (
                    self._to_curobo_device(body_pos_in_robot[self._env_id][link_idx]),
                    self._to_curobo_device(body_quat_in_robot[self._env_id][link_idx]),
                )

            result[obj_name] = link_poses

        return result

    def _build_articulated_primitive_offsets(self) -> None:
        """Build offset cache for articulated primitives relative to their parent link.

        cuRobo only stores leaf primitives (mesh/cuboid/etc), not link-level prims. We read
        each link's initial pose from USD stage (which matches cuRobo's primitive poses),
        convert to robot root frame, then compute offset from each leaf primitive's cuRobo
        pose. The cached offsets are later used in _sync_articulated_obstacles() to update
        primitive poses based on current link poses from Isaac Lab.
        """

        articulated_link_poses = self._get_articulated_link_poses_from_usd()

        for obj_name, link_poses in articulated_link_poses.items():
            obj_prim_prefix = f"{self._env_scene_prefix}/{obj_name}/"

            # Collect all leaf primitives for this articulated object
            all_primitives = self._collect_primitives_by_prefix(obj_prim_prefix)

            # Group primitives by link name
            link_children: dict[str, list[str]] = {}
            for prim_name in all_primitives:
                relative_path = prim_name.split(obj_prim_prefix)[1]
                link_name = relative_path.split("/")[0]
                if link_name not in link_children:
                    link_children[link_name] = []
                link_children[link_name].append(prim_name)

            # Compute offsets for each primitive
            for link_name, child_prim_names in link_children.items():
                if link_name not in link_poses:
                    self._logger.warning(
                        f"Link {link_name} not found in articulation body names, using identity offset"
                    )
                    for child_prim_name in child_prim_names:
                        self._articulated_primitive_offsets[child_prim_name] = (
                            link_name,
                            self._IDENTITY_OFFSET_POS,
                            self._IDENTITY_OFFSET_QUAT,
                        )
                    continue

                link_pos, link_quat = link_poses[link_name]

                for child_prim_name in child_prim_names:
                    child_obstacle = self.motion_gen.world_coll_checker.world_model.get_obstacle(child_prim_name)
                    child_pose = Pose.from_list(child_obstacle.pose, tensor_args=self.tensor_args)
                    child_pos = child_pose.position.squeeze(0)
                    child_quat = convert_quat(child_pose.quaternion.squeeze(0), to="xyzw")  # cuRobo wxyz → xyzw

                    offset_pos, offset_quat = subtract_frame_transforms(
                        link_pos.unsqueeze(0), link_quat.unsqueeze(0), child_pos.unsqueeze(0), child_quat.unsqueeze(0)
                    )

                    self._articulated_primitive_offsets[child_prim_name] = (
                        link_name,
                        offset_pos.squeeze(0),
                        offset_quat.squeeze(0),
                    )

    def _sync_articulated_obstacles(self) -> None:
        """Sync articulated obstacle poses from Isaac Lab link state using cached offsets.

        For each primitive, computes: primitive_pose = link_pose ⊕ offset
        where link_pose comes from Isaac Lab's current articulation state, and offset
        was computed during initialization in _build_articulated_primitive_offsets().

        This method is called automatically before each planning operation via
        _refine_curobo_world_collision().

        Only processes articulated objects that match world_only_subffixes paths.
        """

        if not self._articulated_primitive_offsets:
            return

        articulated_link_poses = self._get_articulated_link_poses()
        updated_count = 0

        for obj_name, link_poses in articulated_link_poses.items():
            obj_prim_prefix = f"{self._env_scene_prefix}/{obj_name}/"

            # Update each primitive using cached offset
            for primitive_name, (link_name, offset_pos, offset_quat) in self._articulated_primitive_offsets.items():
                if not primitive_name.startswith(obj_prim_prefix):
                    continue

                if link_name not in link_poses:
                    self._logger.warning(f"Link {link_name} not found in current articulation state")
                    continue

                link_pos, link_quat = link_poses[link_name]

                # Compute primitive_pose = link_pose ⊕ offset
                primitive_pos = link_pos + quat_apply(link_quat, offset_pos)
                primitive_quat = quat_mul(link_quat, offset_quat)

                self.motion_gen.world_coll_checker.update_obstacle_pose(
                    primitive_name,
                    Pose(
                        position=self._to_curobo_device(primitive_pos),
                        quaternion=self._to_curobo_device(convert_quat(primitive_quat, to="wxyz")),  # xyzw → wxyz
                    ),
                    env_idx=self._env_id,
                    update_cpu_reference=True,
                )
                updated_count += 1

        if updated_count > 0:
            self._logger.debug(f"Synced {updated_count} articulated obstacle primitives from Isaac Lab.")

    def _get_object_mappings(self) -> dict[str, list[str]]:
        """Map IsaacLab scene object names to cuRobo world obstacle names.

        Returns:
            Dictionary mapping IsaacLab scene object names to list of cuRobo world obstacle names.
        """

        if self._cached_object_mappings is not None:
            return self._cached_object_mappings

        rigid_objects = self._env.scene.rigid_objects

        # Collect all primitives in the scene
        all_scene_primitives = self._collect_primitives_by_prefix(self._env_scene_prefix)

        # Map each rigid object to its primitives
        mappings: dict[str, list[str]] = {}
        for object_name in rigid_objects.keys():
            object_prefix = f"{self._env_scene_prefix}/{object_name}/"
            mappings[object_name] = [p for p in all_scene_primitives if p.startswith(object_prefix)]

        self._cached_object_mappings = mappings
        self._logger.debug(f"Object mappings built: {mappings}")
        return mappings

    def _invalidate_object_mapping_cache(self) -> None:
        """Invalidate cached object-name mapping used for dynamic sync."""

        self._cached_object_mappings = None

    def set_target_object(self, target_object: str | None) -> None:
        """Set the current target object for selective collision checking.

        Args:
            target_object: Name of the target object, or None to disable all dynamic objects.
        """

        self._current_target_object = target_object
        self._logger.debug(f"Target object set to: {target_object}")

    def _refine_curobo_world_collision(self) -> None:
        """Refine cuRobo world collision state before planning.

        This method synchronizes all dynamic collision geometry:
        - Rigid object poses (if enable_dynamic_world_sync is True)
        - Articulated object link poses (if articulated offsets are cached)

        Called automatically before each planning operation.

        NOTE: collision geometry is derived from articulation.data (direct PhysX buffer reads),
        not from the viewport rendering. When use_fabric=False, the rendered visual state may
        be slightly inconsistent with the true physics state due to USD Stage sync uncertainty,
        while articulation.data always reflects the correct current state. It is recommended to
        use use_fabric=True to keep the visual output consistent with the collision geometry used here.
        """

        use_fabric = self._env.cfg.sim.use_fabric
        if not use_fabric:
            self._logger.warning(
                f"use_fabric in your isaaclab env: {use_fabric}. curobo articulated collision may be inaccurate, it's"
                " recommended to use use_fabric=True"
            )

        if self.cfg.enable_dynamic_world_sync:
            self._sync_dynamic_objects()
        self._sync_articulated_obstacles()

    def _sync_dynamic_objects(self) -> int:
        """Synchronize dynamic object poses into cuRobo world model.

        If `only_enable_target_object_in_world_sync` is enabled:
        - If target_object is set: only that object will be enabled
        - If target_object is None: all dynamic objects will be disabled

        Returns:
            Number of obstacles whose pose was updated.
        """

        object_mappings = self._get_object_mappings()
        if not object_mappings:
            return 0

        rigid_objects = self._env.scene.rigid_objects
        robot_root_pos_in_world = as_torch(self._robot.data.root_pos_w)
        robot_root_quat_in_world = as_torch(self._robot.data.root_quat_w)

        updated_count = 0

        # Determine which objects should be enabled
        if self.cfg.only_enable_target_object_in_world_sync:
            if self._current_target_object:
                # Only enable target object (reach scenario)
                objects_to_enable = {self._current_target_object}
                self._logger.debug(
                    f"Selective collision: only enabling '{self._current_target_object}', "
                    f"disabling {len(object_mappings) - 1} others"
                )
            else:
                # Disable all dynamic objects (lift/push/pull scenario)
                objects_to_enable = set()
                self._logger.debug(f"Selective collision: disabling all {len(object_mappings)} dynamic objects")
        else:
            # Config not enabled, enable all objects
            objects_to_enable = set(object_mappings.keys())

        for object_name, world_obstacle_names in object_mappings.items():
            obj = rigid_objects[object_name]
            # NOTE: cuRobo world model is in the robot-root frame
            obj_pos_in_world, obj_quat_in_world = as_torch(obj.data.root_pos_w), as_torch(obj.data.root_quat_w)
            obj_pos_in_robot_root, obj_quat_in_robot_root = subtract_frame_transforms(
                robot_root_pos_in_world, robot_root_quat_in_world, obj_pos_in_world, obj_quat_in_world
            )
            obj_pose = Pose(
                position=self._to_curobo_device(obj_pos_in_robot_root[self._env_id]),
                quaternion=self._to_curobo_device(
                    convert_quat(obj_quat_in_robot_root[self._env_id], to="wxyz")  # xyzw → wxyz
                ),
            )

            # Determine if this object should be enabled
            should_enable = object_name in objects_to_enable

            for world_obstacle_name in world_obstacle_names:
                # Update pose
                self.motion_gen.world_coll_checker.update_obstacle_pose(
                    world_obstacle_name,
                    obj_pose,
                    env_idx=self._env_id,
                    update_cpu_reference=True,
                )
                # Enable or disable obstacle
                self.motion_gen.world_coll_checker.enable_obstacle(
                    world_obstacle_name,
                    enable=should_enable,
                    env_idx=self._env_id,
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
            target_quat: Target quaternion [qx, qy, qz, qw]
            current_q: Current joint positions
            current_qd: Current joint velocities
            link_goals: Optional dictionary mapping link names to target poses for other links

        Returns:
            JointState of the trajectory or None if planning failed
        """

        # Refine collision world before planning
        self._refine_curobo_world_collision()

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
            quaternion=self._to_curobo_device(convert_quat(target_quat, to="wxyz")),  # xyzw → wxyz
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
                link_name: Pose(
                    position=self._to_curobo_device(pose[:3]),
                    quaternion=self._to_curobo_device(convert_quat(pose[3:], to="wxyz")),  # xyzw → wxyz
                )
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
            target_quat: Tensor of shape [K, 4] in [qx, qy, qz, qw], in robot root frame.
            current_q: Tensor of shape [dof], current joint positions.
            current_qd: Tensor of shape [dof], current joint velocities. Defaults to zeros.
            link_goals: Optional dict mapping extra link names to tensors of shape [K, 7]
                ([x, y, z, qx, qy, qz, qw], robot root frame) for multi-arm robots. Each entry
                specifies the simultaneous target pose of that link for every sample in the batch.

        Returns:
            MotionGenResult (cuRobo). Check `result.success[k]` for each batch index.

        Note:
            `time_dilation_factor` is always suppressed for batch planning because cuRobo's
            `retime_trajectory` does not support batch results.
        """

        # Refine collision world before planning
        self._refine_curobo_world_collision()

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
            quaternion=self._to_curobo_device(convert_quat(target_quat, to="wxyz")),  # xyzw → wxyz
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
                    quaternion=self._to_curobo_device(convert_quat(poses[:, 3:], to="wxyz")),  # xyzw → wxyz
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
            target_quat: Tensor of shape [K, 4] in [qx, qy, qz, qw], in robot root frame.
            link_goals: Optional dict mapping extra link names to tensors of shape [K, 7]
                ([x, y, z, qx, qy, qz, qw], robot root frame) for multi-arm robots.

        Returns:
            IKResult from cuRobo. Check result.success[k], result.position_error[k],
            result.rotation_error[k] for each batch index.
        """

        # Refine collision world before planning
        self._refine_curobo_world_collision()

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
            quaternion=self._to_curobo_device(convert_quat(target_quat, to="wxyz")),  # xyzw → wxyz
        )
        link_poses = None
        if link_goals is not None:
            link_poses = {
                ee_name: Pose(
                    position=self._to_curobo_device(poses[:, :3]),
                    quaternion=self._to_curobo_device(convert_quat(poses[:, 3:], to="wxyz")),  # xyzw → wxyz
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

        # Refine collision world before planning
        self._refine_curobo_world_collision()

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

    def get_link_poses(self, current_q: torch.Tensor, link_names: list[str] | None = None) -> dict[str, Pose]:
        """Get the poses of specific links in the robot root frame. Returns all links if link_names is None."""

        current_joint_state = JointState(
            position=self._to_curobo_device(current_q), joint_names=self.target_joint_names
        )
        kin_state = self.motion_gen.compute_kinematics(current_joint_state)

        if link_names is None:
            return kin_state.link_poses
        else:
            missing_link_names = [link_name for link_name in link_names if link_name not in kin_state.link_poses]
            if missing_link_names:
                raise ValueError(
                    f"Unknown cuRobo link name(s): {missing_link_names}. Available links:"
                    f" {list(kin_state.link_poses.keys())}"
                )

            return {link_name: kin_state.link_poses[link_name] for link_name in link_names}
