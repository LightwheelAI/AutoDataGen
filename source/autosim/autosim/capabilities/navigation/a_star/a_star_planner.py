from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from autosim.core.logger import AutoSimLogger

if TYPE_CHECKING:
    from autosim.core.types import OccupancyMap

    from .a_star_planner_cfg import AStarPlannerCfg


class AStarPlanner:
    """A* global path planner with distance field guidance"""

    # TODO: refine this class to be more efficient and robust

    def __init__(self, cfg: AStarPlannerCfg, occupancy_map: OccupancyMap) -> None:
        self._cfg = cfg
        self._occupancy_map = occupancy_map
        self._device = self._occupancy_map.occupancy_map.device

        self._logger = AutoSimLogger("AStarPlanner")

        free_space = (self._occupancy_map.occupancy_map == 0).cpu().numpy()
        self._distance_field = distance_transform_edt(free_space) * self._occupancy_map.resolution

    def plan(self, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor | None:
        """
        A* planning from start to goal

        Args:
            start: Tensor of shape [2] with world-frame [x, y].
            goal: Tensor of shape [2] with world-frame [x, y].

        Returns:
            path: Tensor of shape [N, 2] with world-frame waypoints, or None if planning failed.
        """

        # All internal computation is done in numpy
        grid = self._occupancy_map.occupancy_map.cpu().numpy()

        start_np = start.detach().cpu().numpy()
        goal_np = goal.detach().cpu().numpy()

        # convert world-frame numpy inputs to grid indices
        start_grid = self._world_to_grid(start_np)
        goal_grid = self._world_to_grid(goal_np)

        # Check bounds
        if not self._is_valid_grid_pos(start_grid, grid.shape):
            self._logger.warning(f"Start position {start[:2]} -> grid {start_grid} is out of bounds {grid.shape}")
            return None
        if not self._is_valid_grid_pos(goal_grid, grid.shape):
            self._logger.warning(f"Goal position {goal[:2]} -> grid {goal_grid} is out of bounds {grid.shape}")
            return None

        # A* search
        path_grid = self._astar_search(grid, start_grid, goal_grid)
        if path_grid is None:
            return None

        # Convert back to world coordinates
        path_points = [self._grid_to_world(p) for p in path_grid]
        # Simplify path (remove redundant waypoints)
        path_points = self._simplify_path(path_points)

        if not path_points:
            return None

        path_np = np.stack(path_points, axis=0)
        return torch.as_tensor(path_np, device=self._device, dtype=torch.float32)

    def _world_to_grid(self, pos: np.ndarray) -> np.ndarray:
        """Convert world coordinates to grid coordinates (row, col)

        Args:
            pos: [x, y] in world frame

        Returns:
            grid_pos: numpy array [row, col] in grid frame.
        """

        x, y = float(pos[0]), float(pos[1])
        col = int((x - self._occupancy_map.origin[0]) / self._occupancy_map.resolution)
        row = int((y - self._occupancy_map.origin[1]) / self._occupancy_map.resolution)
        return np.array([row, col], dtype=np.int64)

    def _grid_to_world(self, pos: np.ndarray) -> np.ndarray:
        """Convert grid coordinates to world coordinates

        Args:
            pos: [row, col] in grid frame

        Returns:
            world_pos: numpy array [x, y] in world frame.
        """

        row, col = int(pos[0]), int(pos[1])
        x = self._occupancy_map.origin[0] + (col + 0.5) * self._occupancy_map.resolution
        y = self._occupancy_map.origin[1] + (row + 0.5) * self._occupancy_map.resolution
        return np.array([x, y], dtype=np.float32)

    def _is_valid_grid_pos(self, pos: np.ndarray, shape: tuple[int, int]) -> bool:
        """Check if grid position is within bounds (internal numpy representation)"""
        row, col = int(pos[0]), int(pos[1])
        return 0 <= row < shape[0] and 0 <= col < shape[1]

    def _astar_search(self, grid: np.ndarray, start: np.ndarray, goal: np.ndarray) -> list[np.ndarray] | None:
        """A* search algorithm with distance field guidance"""

        start_tuple = tuple(start)
        goal_tuple = tuple(goal)

        # Check if start or goal is in obstacle
        if grid[start_tuple[0], start_tuple[1]] == 1:
            self._logger.warning(f"Start position is in obstacle {start_tuple}")
            return None
        if grid[goal_tuple[0], goal_tuple[1]] == 1:
            self._logger.warning(f"Goal position is in obstacle {goal_tuple}")
            return None

        # Priority queue: (f_score, counter, position, g_score)
        counter = 0
        heap = [(0, counter, start_tuple, 0)]
        visited = set()
        came_from = {}  # For path reconstruction
        g_scores = {start_tuple: 0}

        while heap:
            f_score, _, current_tuple, current_g = heapq.heappop(heap)
            current = np.array(current_tuple)

            if current_tuple in visited:
                continue
            visited.add(current_tuple)

            # Check if reached goal
            if np.linalg.norm(current - goal) < self._cfg.goal_tolerance:
                # Reconstruct path
                path = [goal]
                node = current_tuple
                while node in came_from:
                    path.append(np.array(node))
                    node = came_from[node]
                path.append(start)
                return path[::-1]

            # Explore neighbors (8-connected)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                next_pos = current + np.array([dx, dy])
                next_tuple = tuple(next_pos)

                # Check bounds
                if not self._is_valid_grid_pos(next_pos, grid.shape):
                    continue
                # Check obstacle
                if grid[next_tuple[0], next_tuple[1]] == 1:
                    continue
                if next_tuple in visited:
                    continue

                # Calculate move cost (diagonal moves cost sqrt(2))
                move_cost = 1.414 if (dx != 0 and dy != 0) else 1.0

                # Add proximity penalty (except for start and goal)
                proximity_penalty = 0.0
                if self._distance_field is not None and next_tuple != goal_tuple:
                    dist_to_obstacle = self._distance_field[next_tuple[0], next_tuple[1]]
                    if dist_to_obstacle < self._cfg.safety_distance:
                        # Penalty increases as we get closer to obstacles
                        proximity_penalty = self._cfg.proximity_weight * (self._cfg.safety_distance - dist_to_obstacle)

                tentative_g = current_g + move_cost + proximity_penalty

                if next_tuple not in g_scores or tentative_g < g_scores[next_tuple]:
                    g_scores[next_tuple] = tentative_g
                    came_from[next_tuple] = current_tuple
                    h_score = np.linalg.norm(next_pos - goal)
                    f_score = tentative_g + h_score

                    counter += 1
                    heapq.heappush(heap, (f_score, counter, next_tuple, tentative_g))

        self._logger.warning(f"No path found from {start} to {goal}")
        return None  # No path found

    def _simplify_path(self, path: list[np.ndarray]) -> list[np.ndarray]:
        """Simplify path by removing redundant waypoints"""

        if len(path) <= 2:
            return path

        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            # Check if current point is on the line between previous and next
            v1 = path[i] - path[i - 1]
            v2 = path[i + 1] - path[i]
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 0 and v2_norm > 0:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                if cos_angle < 1.0 - self._cfg.angle_threshold:  # Significant direction change
                    simplified.append(path[i])

        simplified.append(path[-1])
        return simplified
