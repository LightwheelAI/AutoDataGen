import importlib.util

import numpy as np
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import combine_frame_transforms

from autosim.core.types import OccupancyMap

markers: dict[str, VisualizationMarkers] = {}


def create_marker(marker_name: str):
    if marker_name in markers.keys():
        return
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg = frame_marker_cfg.replace(prim_path=f"/World/Visuals/replay_marker_{marker_name}")
    marker = VisualizationMarkers(marker_cfg)
    markers[marker_name] = marker


def visualize_marker(marker_name: str, pose: torch.Tensor):
    pos, quat = pose[:, :3], pose[:, 3:]
    markers[marker_name].visualize(translations=pos, orientations=quat, marker_indices=[0] * pos.shape[0])


def _collect_world_poses(obj_poses: dict[str, list[torch.Tensor]], env: ManagerBasedEnv) -> torch.Tensor | None:
    """Transform a dict of object-frame poses to world frame and stack them.

    Args:
        obj_poses: dict mapping object name -> list of [7] tensors in object frame.
        env: The Isaac Lab environment (used to look up live object poses).

    Returns:
        Stacked world-frame poses of shape [N, 7], or None if there are no poses.
    """
    world_poses = []
    for obj_name, pose_list in obj_poses.items():
        obj_pose_w = env.scene[obj_name].data.root_pose_w[0]  # [7]
        obj_pos_w = obj_pose_w[:3].unsqueeze(0)  # [1, 3]
        obj_quat_w = obj_pose_w[3:].unsqueeze(0)  # [1, 4]
        for pose in pose_list:
            p = pose.unsqueeze(0)  # [1, 7]
            pos_w, quat_w = combine_frame_transforms(obj_pos_w, obj_quat_w, p[:, :3], p[:, 3:])
            world_poses.append(torch.cat([pos_w, quat_w], dim=-1))  # [1, 7]
    if not world_poses:
        return None
    return torch.cat(world_poses, dim=0)  # [N, 7]


def visualize_reach_target_poses(env_extra_info, env: ManagerBasedEnv) -> None:
    """Visualize all reach target poses from env_extra_info as frame markers.

    Creates markers for:
    - ``env_extra_info.object_reach_target_poses`` under the marker name
      ``"reach_target_poses"``.

    Must be called after the environment has been reset so that object poses are
    at their initial positions.
    """
    primary_poses_w = _collect_world_poses(env_extra_info.object_reach_target_poses, env)
    if primary_poses_w is not None:
        create_marker("reach_target_poses")
        visualize_marker("reach_target_poses", primary_poses_w)


def debug_visualize_goal_sampling(
    occupancy_map: OccupancyMap,
    obj_pos_w: np.ndarray,
    robot_pos_w: np.ndarray | None,
    sample_range: tuple[float, float],
    sampling_radius: float,
    num_samples: int,
    target_pos_candidate: np.ndarray | None,
) -> None:
    """
    Visualize the occupancy map, object position, robot position, and sampling around the object.
    Args:
        occupancy_map: The occupancy map of the environment.
        obj_pos_w: The position of the object in the world frame.
        robot_pos_w: The position of the robot in the world frame.
        sample_range: The range of the sampling angles.
        sampling_radius: The radius of the sampling.
        num_samples: The number of samples.
        target_pos_candidate: The candidate position of the target.

    Returns:
        None
    """
    # skip if matplotlib is not installed
    if importlib.util.find_spec("matplotlib") is None:
        return

    import matplotlib.pyplot as plt  # type: ignore[import]
    from matplotlib.colors import ListedColormap  # type: ignore[import]
    from matplotlib.patches import Patch  # type: ignore[import]

    occ = occupancy_map.occupancy_map.cpu().numpy()
    origin_x, origin_y = occupancy_map.origin
    resolution = float(occupancy_map.resolution)

    # Build a 3-class display map: 0 = free, 1 = original obstacle, 2 = inflation buffer.
    display_map = occ.astype(np.int16).copy()
    inflation_mask_np: np.ndarray | None = None
    if occupancy_map.inflation_mask is not None:
        inflation_mask_np = occupancy_map.inflation_mask.cpu().numpy().astype(bool)
        display_map[inflation_mask_np] = 2

    # object position in world frame -> grid coordinates
    ox = float(obj_pos_w[0])
    oy = float(obj_pos_w[1])
    ogx = int((ox - origin_x) / resolution)
    ogy = int((oy - origin_y) / resolution)

    # robot position in world frame -> grid coordinates (if provided)
    rgx, rgy = None, None
    if robot_pos_w is not None:
        rx = float(robot_pos_w[0])
        ry = float(robot_pos_w[1])
        rgx = int((rx - origin_x) / resolution)
        rgy = int((ry - origin_y) / resolution)

    # sampling angles
    angles = np.linspace(sample_range[0], sample_range[1], num_samples, endpoint=False)

    # Sampling-point classification by display map.
    free_x, free_y = [], []
    original_x, original_y = [], []
    inflated_x, inflated_y = [], []
    oob_x, oob_y = [], []

    for angle in angles:
        cx = ox + sampling_radius * np.cos(angle)
        cy = oy + sampling_radius * np.sin(angle)

        gx = int((cx - origin_x) / resolution)
        gy = int((cy - origin_y) / resolution)

        if 0 <= gy < occ.shape[0] and 0 <= gx < occ.shape[1]:
            cell_value = display_map[gy, gx]
            if cell_value == 0:
                free_x.append(gx)
                free_y.append(gy)
            elif cell_value == 2:
                inflated_x.append(gx)
                inflated_y.append(gy)
            else:
                original_x.append(gx)
                original_y.append(gy)
        else:
            oob_x.append(gx)
            oob_y.append(gy)

    # Reuse the same window across calls and clear it to avoid drawing overlaps.
    fig = plt.figure("NavigateSkill Goal Sampling", figsize=(6, 6))
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)
    # 0=free (white), 1=original obstacle (black), 2=inflation buffer (orange)
    cmap = ListedColormap(["#ffffff", "#202020", "#ff8c1a"])
    ax.imshow(display_map, origin="lower", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")

    legend_handles = [
        Patch(facecolor="#202020", label="original obstacle"),
    ]
    if inflation_mask_np is not None:
        radius_str = f" (r={occupancy_map.inflation_radius:.2f}m)"
        legend_handles.append(Patch(facecolor="#ff8c1a", label=f"robot inflation{radius_str}"))

    # object
    if 0 <= ogy < occ.shape[0] and 0 <= ogx < occ.shape[1]:
        ax.scatter(ogx, ogy, c="red", marker="*", s=80, label="object")

    # robot position
    if rgx is not None and rgy is not None:
        if 0 <= rgy < occ.shape[0] and 0 <= rgx < occ.shape[1]:
            ax.scatter(rgx, rgy, c="blue", marker="o", s=60, label="robot")

    # sampling points
    if free_x:
        ax.scatter(free_x, free_y, c="green", s=30, label="free samples")
    if inflated_x:
        ax.scatter(
            inflated_x,
            inflated_y,
            c="orange",
            marker="x",
            s=40,
            linewidths=2,
            label="blocked by inflation",
        )
    if original_x:
        ax.scatter(original_x, original_y, c="red", marker="x", s=40, linewidths=2, label="blocked by obstacle")
    if oob_x:
        ax.scatter(oob_x, oob_y, c="gray", marker="x", s=30, label="out-of-bounds samples")

    # final chosen candidate (if any)
    if target_pos_candidate is not None:
        cx = float(target_pos_candidate[0])
        cy = float(target_pos_candidate[1])
        cgx = int((cx - origin_x) / resolution)
        cgy = int((cy - origin_y) / resolution)
        if 0 <= cgy < occ.shape[0] and 0 <= cgx < occ.shape[1]:
            ax.scatter(cgx, cgy, c="yellow", edgecolors="black", s=80, label="chosen candidate")

    ax.set_title("Goal Sampling around Object")
    ax.set_xlabel("x (grid index)")
    ax.set_ylabel("y (grid index)")
    # Combine map-class legend (patches) with scatter legend.
    scatter_handles, scatter_labels = ax.get_legend_handles_labels()
    ax.legend(handles=legend_handles + list(scatter_handles), loc="upper right")
    fig.tight_layout()
    fig.show()
    # Let the GUI event loop process draw/update events even if the main thread
    # quickly goes back to heavy simulation work.
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.02)


def draw_line(start, end, color=(1.0, 0.0, 0.0, 1.0), size=1.0):
    """
    Draws a single line between two points.
    """
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

    draw = omni_debug_draw.acquire_debug_draw_interface()
    draw.draw_lines([start], [end], [color], [size])


def clear_debug_drawing():
    """
    Clears all debug drawings.
    """
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

    draw = omni_debug_draw.acquire_debug_draw_interface()
    draw.clear_lines()
