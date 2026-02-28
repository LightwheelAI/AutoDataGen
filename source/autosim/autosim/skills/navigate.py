import numpy as np
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass
from scipy.ndimage import distance_transform_edt

from autosim import register_skill
from autosim.capabilities.navigation import AStarPlannerCfg, DWAPlannerCfg
from autosim.core.logger import AutoSimLogger
from autosim.core.skill import Skill, SkillCfg, SkillExtraCfg
from autosim.core.types import (
    EnvExtraInfo,
    OccupancyMap,
    SkillGoal,
    SkillInfo,
    SkillOutput,
    WorldState,
)


@configclass
class NavigateSkillExtraCfg(SkillExtraCfg):
    """Extra configuration for the navigate skill."""

    global_planner: AStarPlannerCfg = AStarPlannerCfg()
    """The configuration for the A* motion planner."""
    local_planner: DWAPlannerCfg = DWAPlannerCfg()
    """The configuration for the DWA motion planner."""
    use_dwa: bool = False
    """Whether to use DWA motion planner."""
    waypoint_tolerance: float = 0.5
    """The tolerance of the waypoint."""
    goal_tolerance: float = 0.25
    """The tolerance of the goal."""
    yaw_tolerance: float = 0.01
    """The tolerance of the yaw (radians)."""
    sampling_radius: float = 0.8
    """The sampling radius for the target position, in meters."""
    num_samples: int = 4
    """The number of samples for the target position."""

    occupancy_map: OccupancyMap | None = None
    """The occupancy map of the environment."""


@configclass
class NavigateSkillCfg(SkillCfg):
    """Configuration for the navigate skill."""

    extra_cfg: NavigateSkillExtraCfg = NavigateSkillExtraCfg()
    """Extra configuration for the navigate skill."""


@register_skill(
    name="moveto", cfg_type=NavigateSkillCfg, description="Move robot base to near the target object or location."
)
class NavigateSkill(Skill):
    """Skill to navigate to a target position using A* + DWA motion planner."""

    def __init__(self, extra_cfg: NavigateSkillExtraCfg) -> None:
        super().__init__(extra_cfg)

        self._logger = AutoSimLogger("NavigateSkill")

        self._occupancy_map = extra_cfg.occupancy_map
        self._global_planner = extra_cfg.global_planner.class_type(extra_cfg.global_planner, self._occupancy_map)
        self._local_planner = extra_cfg.local_planner.class_type(extra_cfg.local_planner, self._occupancy_map)

        # variables for the skill execution
        self._target_object_name = None
        self._target_yaw = None
        self._target_pos = None
        self._global_path = None
        self._current_waypoint_idx = 0

    def extract_goal_from_info(
        self, skill_info: SkillInfo, env: ManagerBasedEnv, env_extra_info: EnvExtraInfo
    ) -> SkillGoal:
        """Return the target pose[x, y, yaw] in the world frame."""

        target_object_name = skill_info.target_object
        if target_object_name not in env.scene.keys():
            raise ValueError(f"Object {target_object_name} not found in scene")
        target_object = env.scene[target_object_name]

        obj_pos_w = target_object.data.root_pos_w[0].cpu().numpy()

        is_free = (self._occupancy_map.occupancy_map == 0).cpu().numpy()
        if np.any(is_free):
            dist_field = distance_transform_edt(is_free)
        else:
            dist_field = np.zeros_like(is_free, dtype=np.float32)

        best_score = -1.0

        angles = np.linspace(0, 2 * np.pi, self.cfg.extra_cfg.num_samples, endpoint=False)

        for angle in angles:
            # calculate the sample point coordinates in the world frame
            cx = obj_pos_w[0] + self.cfg.extra_cfg.sampling_radius * np.cos(angle)
            cy = obj_pos_w[1] + self.cfg.extra_cfg.sampling_radius * np.sin(angle)

            # convert to the grid coordinates in the occupancy map
            gx = int((cx - self._occupancy_map.origin[0]) / self._occupancy_map.resolution)
            gy = int((cy - self._occupancy_map.origin[1]) / self._occupancy_map.resolution)

            # check the boundary
            if (
                0 <= gy < self._occupancy_map.occupancy_map.shape[0]
                and 0 <= gx < self._occupancy_map.occupancy_map.shape[1]
            ):
                # check the collision (must be free space)
                if self._occupancy_map.occupancy_map[gy, gx] == 0:
                    # get the safety score (the farther from the obstacle, the better)
                    score = dist_field[gy, gx]
                    if score > best_score:
                        best_score = score
                        target_pos_candidate = np.array([cx, cy])
                        # calculate the yaw (facing the object)
                        dx = obj_pos_w[0] - cx
                        dy = obj_pos_w[1] - cy
                        target_yaw = np.arctan2(dy, dx)

        # if no target position is found, use the default fallback position
        if target_pos_candidate is None:
            self._logger.warning(f"Map sampling failed for {target_object_name}. Using default offset.")

            target_x = obj_pos_w[0]
            target_y = obj_pos_w[1] - 1.0
            target_pos_candidate = np.array([target_x, target_y])

            dx = obj_pos_w[0] - target_x
            dy = obj_pos_w[1] - target_y
            target_yaw = np.arctan2(dy, dx)

        target_pose = torch.tensor(
            [target_pos_candidate[0], target_pos_candidate[1], target_yaw], device=env.device, dtype=torch.float32
        )

        return SkillGoal(target_object=target_object_name, target_pose=target_pose)

    def execute_plan(self, state: WorldState, goal: SkillGoal) -> bool:
        """Global path planning using A*"""

        # Extract start position from metadata (robot root pose in world frame)
        robot_base_pose = state.robot_base_pose

        start_pos = robot_base_pose[:2]  # [x, y]
        goal_pos = goal.target_pose[:2]  # [x, y]

        target_yaw = goal.target_pose[2].item()

        self._target_object_name = goal.target_object
        self._target_yaw = target_yaw
        self._target_pos = goal_pos

        self._logger.info(
            f"Planning from ({start_pos[0]:.2f}, {start_pos[1]:.2f}) to ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}),"
            f" target_yaw={target_yaw:.2f}."
        )

        self._global_path = self._global_planner.plan(start_pos, goal_pos)

        if self._global_path is None:
            self._logger.error("Global planning failed.")
            return False

        self._logger.info(f"Global path planned: {len(self._global_path)} waypoints.")
        return True

    def step(self, state: WorldState) -> SkillOutput:
        """Step the skill execution.

        Args:
            state: The current state of the world.

        Returns:
            The output of the skill execution.
                action: The action to be applied to the environment. [vx, vy, vyaw] in the world frame.
        """

        current_pose = state.robot_base_pose  # [x, y, yaw]

        # Check if reached goal
        goal_pos = self._global_path[-1]
        dist_to_goal = float(torch.linalg.norm(current_pose[:2] - goal_pos))

        desired_yaw = self._target_yaw
        is_final_approach = dist_to_goal < self.cfg.extra_cfg.goal_tolerance

        # If we are not in the final approach, try to face the object
        if not is_final_approach:
            obj_tensor = state.objects[self._target_object_name]  # [x, y, z, qw, qx, qy, qz]
            obj_pos = obj_tensor[:2]  # [x, y] in world frame

            dx_obj = obj_pos[0] - current_pose[0]
            dy_obj = obj_pos[1] - current_pose[1]
            yaw_to_obj = float(torch.arctan2(dy_obj, dx_obj).item())
            desired_yaw = yaw_to_obj

        # Calculate Yaw Error
        yaw_error = self._normalize_angle(desired_yaw - current_pose[2])

        # Check success condition
        if is_final_approach and abs(yaw_error) < self.cfg.extra_cfg.yaw_tolerance:
            # successfully reached the goal
            return SkillOutput(
                action=torch.zeros(3),
                done=True,
                success=True,
                info={"distance_to_goal": dist_to_goal, "yaw_error": yaw_error},
            )

        # Get current target waypoint (go ahead if close enough)
        while self._current_waypoint_idx < len(self._global_path) - 1:
            waypoint = self._global_path[self._current_waypoint_idx]
            dist = float(torch.linalg.norm(current_pose[:2] - waypoint))
            if dist < self.cfg.extra_cfg.waypoint_tolerance:
                self._current_waypoint_idx += 1
            else:
                break

        target_waypoint = self._global_path[self._current_waypoint_idx]

        # Simple proportional control or DWA-based local planning
        # ------------------------------------------------------------------
        # 1) Waypoint-based P controller (original implementation, kept for backward compatibility)
        dx = target_waypoint[0] - current_pose[0]
        dy = target_waypoint[1] - current_pose[1]
        dist_to_waypoint = float(torch.sqrt(dx * dx + dy * dy))

        vx, vy, vyaw = 0.0, 0.0, 0.0

        if not self.cfg.extra_cfg.use_dwa:
            # ---------- Original local planner: P controller ----------
            if dist_to_waypoint > 0.01:
                # Normalize and scale by max velocity
                speed = min(
                    self.cfg.extra_cfg.local_planner.max_linear_velocity, dist_to_waypoint * 2.0
                )  # Proportional gain
                vx = speed * dx / dist_to_waypoint
                vy = speed * dy / dist_to_waypoint

            # Always rotate towards desired_yaw (either object or goal)
            max_w = self.cfg.extra_cfg.local_planner.max_angular_velocity
            vyaw = np.clip(yaw_error * 2.0, -max_w, max_w)

            # If we are far from goal but not facing the object yet, stop linear movement.
            if not is_final_approach and abs(yaw_error) > self.cfg.extra_cfg.local_planner.yaw_facing_threshold:
                vx = 0.0
                vy = 0.0
        else:
            # ---------- Use DWA as local planner ----------
            # Note: DWA logic typically handles collision and velocity profiles better,
            # but getting it to "Face Object" while moving requires modifying the DWA cost function
            # or pre-rotating input. For now, sticking to the requested modification on standard behavior.
            # If DWA is enabled, we assume it handles the movement.
            # However, enforcing "Face Object" in DWA requires tricking the DWA or overriding yaw.

            # Use current waypoint position as DWA target
            dwa_target = target_waypoint[:2].cpu().numpy()
            v_lin, v_yaw = self._local_planner.compute_velocity(
                current_pose=current_pose.cpu().numpy(),  # [x, y, yaw]
                target=dwa_target,  # [x, y]
            )
            # Project body-frame forward speed v_lin into world-frame (vx, vy)
            yaw = current_pose[2].item()
            vx = v_lin * np.cos(yaw)
            vy = v_lin * np.sin(yaw)
            vyaw = v_yaw

        # Create action [vx, vy, vyaw] in world frame (can't be applied to the environment directly yet)
        action = torch.tensor([vx, vy, vyaw])

        return SkillOutput(
            action=action,
            done=False,
            success=None,
            info={
                "waypoint_idx": self._current_waypoint_idx,
                "total_waypoints": len(self._global_path),
                "distance_to_goal": dist_to_goal,
                "yaw_error": yaw_error,
                "velocity_world": [vx, vy, vyaw],
            },
        )

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""

        angle_tensor = torch.as_tensor(angle)
        return float(torch.remainder(angle_tensor + np.pi, 2 * np.pi) - np.pi)
