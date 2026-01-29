from isaaclab.utils import configclass

from .dwa_planner import DWAPlanner


@configclass
class DWAPlannerCfg:
    """Configuration for the DWA planner."""

    class_type: type = DWAPlanner
    """The class type of the DWA planner."""

    max_linear_velocity: float = 1.0
    """The maximum linear velocity of the robot base."""
    max_angular_velocity: float = 1.0
    """The maximum angular velocity of the robot base."""
    max_linear_acceleration: float = 0.5
    """The maximum linear acceleration of the robot base."""
    max_angular_acceleration: float = 1.0
    """The maximum angular acceleration of the robot base."""
    dt: float | None = None
    """The time step of the navigation. If None, will calculate from the environment."""
    predict_time: float = 2.0
    """The prediction time."""
    v_resolution: float = 0.1
    """The velocity resolution."""
    w_resolution: float = 0.2
    """The angular velocity resolution."""
    yaw_facing_threshold: float = 0.2
    """The threshold of the yaw to face the target (radians)."""
