from isaaclab.utils import configclass

from autosim import ActionAdapterCfg

from .franka_adapter import FrankaAbsAdapter


@configclass
class FrankaAbsAdapterCfg(ActionAdapterCfg):
    """Configuration for the Franka adapter."""

    class_type: type = FrankaAbsAdapter

    skip_apply_skills: list[str] = ["moveto"]
