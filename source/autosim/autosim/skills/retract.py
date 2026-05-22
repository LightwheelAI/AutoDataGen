import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

from autosim import register_skill
from autosim.core.logger import AutoSimLogger
from autosim.core.skill import SkillCfg
from autosim.core.types import (
    EnvExtraInfo,
    SkillGoal,
    SkillInfo,
    SkillOutput,
    WorldState,
)

from .base_skill import CuroboSkillBase, CuroboSkillExtraCfg


@configclass
class RetractSkillExtraCfg(CuroboSkillExtraCfg):
    """Extra configuration for the retract skill."""

    pass


@configclass
class RetractSkillCfg(SkillCfg):
    """Configuration for the retract skill."""

    extra_cfg: RetractSkillExtraCfg = RetractSkillExtraCfg()


@register_skill(
    name="retract",
    cfg_type=RetractSkillCfg,
    description="Move robot arm to retract (home) configuration",
)
class RetractSkill(CuroboSkillBase):
    """Skill to move the robot arm to a predefined retract configuration.

    The retract configuration is read from the robot's cuRobo yaml file
    (cspace.retract_config). This is useful for resetting the arm to a safe
    pose between tasks or after a grasp.
    """

    def __init__(self, extra_cfg: RetractSkillExtraCfg) -> None:
        super().__init__(extra_cfg)

        self._logger = AutoSimLogger("RetractSkill")
        self._trajectory = None
        self._step_idx = 0

    def extract_goal_from_info(
        self, skill_info: SkillInfo, env: ManagerBasedEnv, env_extra_info: EnvExtraInfo
    ) -> SkillGoal:
        """Retract has no target object — return an empty goal."""

        return SkillGoal()

    def execute_plan(self, state: WorldState, goal: SkillGoal) -> bool:
        """Plan a joint-space trajectory to the retract configuration."""

        retract_q = self._get_retract_config(state)
        if retract_q is None:
            self._logger.warning("retract_config not found in robot yaml, cannot plan")
            return False

        full_sim_joint_names = state.sim_joint_names
        full_sim_q = state.robot_joint_pos
        full_sim_qd = state.robot_joint_vel
        planner_joint_names = self._planner.target_joint_names

        activate_q, activate_qd = [], []
        for joint_name in planner_joint_names:
            if joint_name in full_sim_joint_names:
                idx = full_sim_joint_names.index(joint_name)
                activate_q.append(full_sim_q[idx])
                activate_qd.append(full_sim_qd[idx])
            else:
                raise ValueError(f"Joint {joint_name} in planner joints is not in simulation joint names.")
        activate_q = torch.stack(activate_q, dim=0)
        activate_qd = torch.stack(activate_qd, dim=0)

        self._trajectory = self._planner.plan_to_joint_config(
            retract_q,
            activate_q,
            activate_qd,
        )

        return self._trajectory is not None

    def _get_retract_config(self, state: WorldState) -> torch.Tensor | None:
        """Extract retract_config from the loaded robot yaml and reorder to planner joint order."""

        robot_cfg = self._planner.robot_cfg
        try:
            cspace = robot_cfg["robot_cfg"]["kinematics"]["cspace"]
            yaml_joint_names = cspace["joint_names"]
            yaml_retract = cspace["retract_config"]
        except (KeyError, TypeError):
            return None

        # Build a lookup from yaml joint name → retract value
        retract_map = dict(zip(yaml_joint_names, yaml_retract))

        # Reorder to match planner's joint order; use 0.0 for any missing joint
        planner_joint_names = self._planner.target_joint_names
        retract_values = [retract_map.get(name, 0.0) for name in planner_joint_names]

        return torch.tensor(retract_values, dtype=torch.float32, device=state.device)

    def step(self, state: WorldState) -> SkillOutput:
        """Step through the planned retract trajectory."""

        traj_positions = self._trajectory.position
        if self._step_idx >= len(traj_positions):
            traj_pos = traj_positions[-1]
            done = True
        else:
            traj_pos = traj_positions[self._step_idx]
            done = False
            self._step_idx += 1

        curobo_joint_names = self._trajectory.joint_names
        sim_joint_names = state.sim_joint_names
        joint_pos = state.robot_joint_pos.clone()
        for curobo_idx, curobo_joint_name in enumerate(curobo_joint_names):
            sim_idx = sim_joint_names.index(curobo_joint_name)
            joint_pos[sim_idx] = traj_pos[curobo_idx]

        info = {}
        if self.cfg.extra_cfg.return_link_poses_in_robot_root_frame:
            activate_q, _ = self._build_activate_joint_state(state.sim_joint_names, joint_pos, None)
            all_link_poses = self._planner.get_link_poses(activate_q, link_names=None)
            info["link_poses_in_robot_root_frame"] = {
                name: torch.cat([pose.position.squeeze(0), pose.quaternion.squeeze(0)])
                for name, pose in all_link_poses.items()
            }

        return SkillOutput(
            action=joint_pos,
            done=done,
            success=True,
            info=info,
        )

    def reset(self) -> None:
        """Reset the retract skill."""

        super().reset()
        self._step_idx = 0
        self._trajectory = None
