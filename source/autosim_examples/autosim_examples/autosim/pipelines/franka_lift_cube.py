import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

from autosim.core.pipeline import AutoSimPipeline, AutoSimPipelineCfg
from autosim.core.types import EnvExtraInfo
from autosim.decomposers import LLMDecomposerCfg

from ..action_adapters.franka_adapter_cfg import FrankaAbsAdapterCfg


@configclass
class FrankaCubeLiftPipelineCfg(AutoSimPipelineCfg):
    """Configuration for the Franka cube lift pipeline."""

    decomposer: LLMDecomposerCfg = LLMDecomposerCfg()

    action_adapter: FrankaAbsAdapterCfg = FrankaAbsAdapterCfg()

    def __post_init__(self):
        self.skills.lift.extra_cfg.move_axis = "-z"
        self.skills.lift.extra_cfg.lift_offset = 0.30

        self.occupancy_map.floor_prim_suffix = "Table"

        self.motion_planner.robot_config_file = "franka.yml"
        self.motion_planner.world_ignore_subffixes = []
        self.motion_planner.world_only_subffixes = []


class FrankaCubeLiftPipeline(AutoSimPipeline):
    def __init__(self, cfg: AutoSimPipelineCfg):
        self._task_name = "AutoSimExamples-IsaacLab-FrankaCubeLift-v0"

        super().__init__(cfg)

    def load_env(self) -> ManagerBasedEnv:
        import gymnasium as gym
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(self._task_name, device="cuda:0", num_envs=1, use_fabric=True)
        env_cfg.terminations = None
        env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True

        env = gym.make(self._task_name, cfg=env_cfg).unwrapped
        return env

    def get_env_extra_info(self) -> EnvExtraInfo:
        available_objects = self._env.scene.keys()
        return EnvExtraInfo(
            task_name=self._task_name,
            objects=available_objects,
            additional_prompt_contents=None,
            robot_name="robot",
            robot_base_link_name="panda_link0",
            ee_link_name="panda_hand",
            object_reach_target_poses={
                "cube": [
                    torch.tensor([0.0, 0.0, 0.10, 0.0, 1.0, 0.0, 0.0]),
                ],
            },
        )
