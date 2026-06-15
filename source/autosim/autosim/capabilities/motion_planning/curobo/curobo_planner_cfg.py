from dataclasses import MISSING

from curobo.geom.sdf.world import CollisionCheckerType
from isaaclab.utils.configclass import configclass

from .curobo_planner import CuroboPlanner


@configclass
class CuroboPlannerCfg:
    """Configuration for the Curobo motion planner."""

    class_type: type = CuroboPlanner
    """The class type of the Curobo motion planner."""

    # Curobo robot configuration
    robot_config_file: str | dict = MISSING
    """cuRobo robot configuration file (path or dictionary)."""
    curobo_config_path: str | None = None
    """Path to the curobo config directory."""
    curobo_asset_path: str | None = None
    """Path to the curobo asset directory."""

    # Motion planning parameters
    collision_checker_type: CollisionCheckerType = CollisionCheckerType.MESH
    """Type of collision checker to use."""
    collision_cache: dict[str, int] = {"obb": 1000, "mesh": 500}
    """Collision cache for different collision types."""
    collision_activation_distance: float = 0.05
    """Distance at which collision constraints are activated."""
    interpolation_dt: float = 0.05
    """Time step for interpolating."""
    num_trajopt_seeds: int = 12
    """Number of seeds for trajectory optimization."""
    num_graph_seeds: int = 12
    """Number of seeds for graph search."""

    # Planning configuration
    enable_graph: bool = True
    """Whether to enable graph-based planning."""
    enable_graph_attempt: int = 4
    """Number of graph planning attempts."""
    use_cuda_graph: bool = True
    """Whether to use CUDA graph for planning."""
    max_planning_attempts: int = 10
    """Maximum number of planning attempts."""
    time_dilation_factor: float = 0.5
    """Time dilation factor for planning."""
    reach_partial_pose_weight: list[float] | None = None
    """Per-axis weights [rx, ry, rz, px, py, pz] for partial-pose reaching via cuRobo PoseCostMetric.
    Setting a weight to 0.0 relaxes that axis (e.g. [0,0,0,1,1,1] for position-only reaching)."""
    # Optional prim path configuration
    robot_prim_path: str | None = None
    """Absolute USD prim path to the robot root for world extraction; None derives it from environment root."""
    world_only_subffixes: list[str] | None = None
    """List of subffixes to only extract world obstacles from."""
    world_ignore_subffixes: list[str] | None = None
    """List of subffixes to ignore when extracting world obstacles."""
    collision_enable_substrings: list[str] | None = None
    """If set, only primitives whose name contains at least one of these substrings will have collision enabled.
    All other primitives will be disabled. Applied globally to all world obstacles (static + articulated)."""

    # Self-collision configuration
    self_collision_check: bool = True
    """Whether to check self-collision during planning."""
    self_collision_opt: bool = True
    """Whether to optimize away self-collisions during planning."""

    # World update strategy
    enable_dynamic_world_sync: bool = True
    """If True, synchronize dynamic object poses into cuRobo world before planning (fast incremental update)."""
    only_enable_target_object_in_world_sync: bool = True
    """If True, only enable the target object in the world sync when executing the reach (grasp) skill. Only valid when enable_dynamic_world_sync is True."""

    # Debug and visualization
    debug_planner: bool = False
    """Enable detailed motion planning debug information."""
    cuda_device: int | None = 0
    """Preferred CUDA device index; None uses torch.cuda.current_device() (respects CUDA_VISIBLE_DEVICES)."""

    # Batch planning
    max_batch_size: int = 10
    """Maximum batch size for plan_motion. Used to warmup CUDA graph with the correct buffer size.
    Must be >= the largest number of reach candidates (K) used at runtime. If runtime K exceeds
    this value, cuRobo will need to reset CUDA graph (requires CUDA >= 12.0 or CUROBO_TORCH_CUDA_GRAPH_RESET=1)."""

    env_scene_prefix: str | None = "Scene"
    """Prefix of the environment scene."""
