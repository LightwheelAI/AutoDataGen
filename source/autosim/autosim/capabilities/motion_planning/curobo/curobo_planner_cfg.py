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

    # Optional prim path configuration
    robot_prim_path: str | None = None
    """Absolute USD prim path to the robot root for world extraction; None derives it from environment root."""
    world_only_subffixes: list[str] | None = None
    """List of subffixes to only extract world obstacles from."""
    world_ignore_subffixes: list[str] | None = None
    """List of subffixes to ignore when extracting world obstacles."""

    # Debug and visualization
    debug_planner: bool = False
    """Enable detailed motion planning debug information."""
    cuda_device: int | None = 0
    """Preferred CUDA device index; None uses torch.cuda.current_device() (respects CUDA_VISIBLE_DEVICES)."""
