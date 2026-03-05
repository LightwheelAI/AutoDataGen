import torch
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

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
    markers[marker_name].visualize(translations=pos, orientations=quat, marker_indices=[0])
