from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from autosim.core.types import OccupancyMap

    from .dwa_planner_cfg import DWAPlannerCfg


class DWAPlanner:
    """Dynamic Window Approach (DWA) planner for local obstacle avoidance."""

    # TODO: Standardize the use of torch.Tensor and np.ndarray
    # TODO: move magic number to cfg

    def __init__(self, cfg: DWAPlannerCfg, occupancy_map: OccupancyMap) -> None:
        self._cfg = cfg
        self._occupancy_map = occupancy_map

    def compute_velocity(
        self,
        current_pose: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Compute optimal velocity using simplified DWA

        Args:
            current_pose: [x, y, yaw]
            target: [x, y]

        Returns:
            [v, w]: optimal velocity
        """

        # Sample full velocity range (simplified, no dynamic window constraint)
        v_samples = np.arange(0, self._cfg.max_linear_velocity + self._cfg.v_resolution, self._cfg.v_resolution)
        w_samples = np.arange(
            -self._cfg.max_angular_velocity,
            self._cfg.max_angular_velocity + self._cfg.w_resolution,
            self._cfg.w_resolution,
        )

        best_v, best_w = 0.0, 0.0
        best_score = -float("inf")

        for v in v_samples:
            for w in w_samples:
                # Predict trajectory
                trajectory = self._predict_trajectory(current_pose, v, w)
                # Evaluate trajectory
                score = self._evaluate_trajectory(trajectory, target)

                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w

        return np.array([best_v, best_w])

    def _predict_trajectory(self, pose: np.ndarray, v: float, w: float) -> np.ndarray:
        """Predict trajectory given velocity"""

        trajectory = []
        x, y, yaw = pose

        steps = int(self._cfg.predict_time / self._cfg.dt)
        for _ in range(steps):
            x += v * np.cos(yaw) * self._cfg.dt
            y += v * np.sin(yaw) * self._cfg.dt
            yaw += w * self._cfg.dt
            trajectory.append([x, y, yaw])

        return np.array(trajectory)

    def _evaluate_trajectory(self, trajectory: np.ndarray, target: np.ndarray) -> float:
        """Evaluate trajectory quality using grid-based collision check"""

        # Distance to target (minimize)
        final_pos = trajectory[-1, :2]
        dist_to_target = np.linalg.norm(final_pos - target)
        target_score = 1.0 / (dist_to_target + 0.1)

        # Collision check using occupancy grid (O(1) per point)
        # Skip first few points to allow escaping from near-obstacle positions
        check_start = min(3, len(trajectory))
        for point in trajectory[check_start:]:
            if self._is_collision(point[0], point[1]):
                return -1000.0  # Collision, reject trajectory

        # Velocity score (prefer higher velocity toward target)
        velocity_score = np.linalg.norm(trajectory[-1, :2] - trajectory[0, :2]) if len(trajectory) > 1 else 0

        # Combined score
        return target_score * 1.5 + velocity_score * 0.5

    def _is_collision(self, x: float, y: float) -> bool:
        """Check collision using occupancy grid (O(1) lookup)"""

        col = int((x - self._occupancy_map.origin[0]) / self._occupancy_map.resolution)
        row = int((y - self._occupancy_map.origin[1]) / self._occupancy_map.resolution)
        if (
            0 <= row < self._occupancy_map.occupancy_map.shape[0]
            and 0 <= col < self._occupancy_map.occupancy_map.shape[1]
        ):
            return self._occupancy_map.occupancy_map[row, col] == 1
        # Out of bounds - assume free if close to boundary, collision if far
        return False  # Allow movement outside map
