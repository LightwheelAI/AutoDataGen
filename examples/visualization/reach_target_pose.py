"""Visualize all reach target poses for an autosim pipeline.

Usage
-----
1) Start the visualization (it will export a debug JSON once after `pipeline.reset_env()`):
   python examples/visualization/reach_target_pose.py --pipeline_id <PIPELINE_ID> \
     --debug_poses_path /abs/path/reach_target_poses_debug.json

2) Edit and save that JSON file. The script polls its mtime and reloads markers automatically.

`--debug_poses_path` JSON format
---------------------------------
The script expects this payload:
{
  "object_reach_target_poses": {
    "<object_name>": [
      [x, y, z, qx, qy, qz, qw],
      ...
    ],
    ...
  }
}

Notes
-----
* Poses are in the object frame: [x, y, z, qx, qy, qz, qw].
* `--live_poll_interval_s` controls how often the file is checked (default: 0.2s).
"""

import argparse
import json
import os
import time

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize reach target poses for an autosim pipeline.")
parser.add_argument("--pipeline_id", type=str, default=None, help="Name of the autosim pipeline.")
parser.add_argument(
    "--debug_poses_path",
    type=str,
    default="reach_target_poses_debug.json",
    help=(
        "If provided, the script will export the current `object_reach_target_poses` "
        "to this JSON file after `reset_env()`, and reload it on every file change."
    ),
)
parser.add_argument(
    "--live_poll_interval_s",
    type=float,
    default=0.2,
    help="Polling interval (seconds) for checking `--debug_poses_path` updates.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app


from isaaclab.utils.math import (
    compute_pose_error,
    euler_xyz_from_quat,
    subtract_frame_transforms,
)

import autosim_examples  # noqa: F401
from autosim import make_pipeline
from autosim.utils.data_util import as_torch
from autosim.utils.debug_util import visualize_reach_target_poses


def _load_env_extra_poses_json(path: str) -> dict[str, list[list[float]]]:
    """Load reach target poses from the exported debug JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    object_reach_target_poses: dict[str, list[list[float]]] = {}

    if not isinstance(data, dict):
        raise ValueError("Debug JSON root must be an object.")

    reach = data.get("object_reach_target_poses", {})
    if not isinstance(reach, dict):
        raise ValueError("`object_reach_target_poses` must be an object mapping.")
    for obj_name, pose_list in reach.items():
        if not isinstance(obj_name, str):
            raise ValueError("Reach: object names must be strings.")
        if not isinstance(pose_list, list):
            raise ValueError(f"Reach: `{obj_name}` must map to a list of poses.")
        normalized: list[list[float]] = []
        for pose in pose_list:
            if not (isinstance(pose, list) and len(pose) == 7):
                raise ValueError(f"Reach: each pose for `{obj_name}` must be list length 7.")
            normalized.append([float(v) for v in pose])
        object_reach_target_poses[obj_name] = normalized

    return object_reach_target_poses


def _apply_live_poses(*, poses_path: str, pipeline) -> None:
    """Update `pipeline._env_extra_info.object_reach_target_poses` from JSON."""
    env = pipeline._env
    env_extra_info = pipeline._env_extra_info
    object_reach_target_poses = _load_env_extra_poses_json(poses_path)

    env_extra_info.object_reach_target_poses = {}

    for obj_name, pose_list in object_reach_target_poses.items():
        if obj_name not in env.scene.keys():
            continue
        obj_pose_w = as_torch(env.scene[obj_name].data.root_pose_w)[0]  # [7]
        device = obj_pose_w.device
        dtype = obj_pose_w.dtype
        env_extra_info.object_reach_target_poses[obj_name] = [
            torch.tensor(pose, device=device, dtype=dtype) for pose in pose_list
        ]


def _target_object_names(pipeline) -> list[str]:
    return list(pipeline._env_extra_info.object_reach_target_poses.keys())


def _snapshot_object_poses_w(*, env, object_names: list[str]) -> dict[str, list[float]]:
    poses_w: dict[str, list[float]] = {}
    for obj_name in object_names:
        if obj_name not in env.scene.keys():
            print(f"[reach_target_pose] Skip missing scene object: {obj_name}")
            continue
        pose_w = as_torch(env.scene[obj_name].data.root_pose_w)[0]
        poses_w[obj_name] = [float(v) for v in pose_w.detach().cpu().tolist()]
    return poses_w


def _report_pose_drift(*, poses_before: dict[str, list[float]], poses_after: dict[str, list[float]]) -> None:
    """Print world-frame and object-frame relative pose change for each target object."""
    print("[reach_target_pose] Object pose drift after 20 zero-action steps:")
    for obj_name in poses_before:
        if obj_name not in poses_after:
            print(f"  - {obj_name}: missing pose after steps")
            continue

        pose_before = poses_before[obj_name]
        pose_after = poses_after[obj_name]
        pos_b = torch.tensor(pose_before[:3]).view(1, 3)
        quat_b = torch.tensor(pose_before[3:]).view(1, 4)
        pos_a = torch.tensor(pose_after[:3]).view(1, 3)
        quat_a = torch.tensor(pose_after[3:]).view(1, 4)

        world_pos_delta = (pos_a - pos_b).squeeze(0)
        world_pos_norm = float(torch.linalg.norm(world_pos_delta))
        world_pos_err, world_rot_err = compute_pose_error(pos_b, quat_b, pos_a, quat_a)
        world_rot_deg = float(torch.rad2deg(torch.linalg.norm(world_rot_err)).item())

        rel_pos, rel_quat = subtract_frame_transforms(pos_b, quat_b, pos_a, quat_a)
        rel_pos_norm = float(torch.linalg.norm(rel_pos).item())
        _, rel_rot_axis_angle = compute_pose_error(
            torch.zeros_like(pos_b),
            torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 4),
            rel_pos,
            rel_quat,
        )
        rel_rot_deg = float(torch.rad2deg(torch.linalg.norm(rel_rot_axis_angle)).item())
        rel_roll, rel_pitch, rel_yaw = euler_xyz_from_quat(rel_quat)
        rel_roll_deg = float(torch.rad2deg(rel_roll).item())
        rel_pitch_deg = float(torch.rad2deg(rel_pitch).item())
        rel_yaw_deg = float(torch.rad2deg(rel_yaw).item())

        print(f"  - {obj_name}:")
        print(f"      pose_before_w: {pose_before}")
        print(f"      pose_after_w:  {pose_after}")
        print(
            "      world_delta: "
            f"pos=[{world_pos_delta[0]:.6f}, {world_pos_delta[1]:.6f}, {world_pos_delta[2]:.6f}] "
            f"(norm={world_pos_norm:.6f} m), rot={world_rot_deg:.4f} deg"
        )
        print(
            "      relative_delta (in pose_before / object frame): "
            f"pos=[{rel_pos[0, 0]:.6f}, {rel_pos[0, 1]:.6f}, {rel_pos[0, 2]:.6f}] "
            f"(norm={rel_pos_norm:.6f} m), rot={rel_rot_deg:.4f} deg"
        )
        print(
            "      relative_rot_object_frame (XYZ Euler, deg): "
            f"x(roll)={rel_roll_deg:.4f}, y(pitch)={rel_pitch_deg:.4f}, z(yaw)={rel_yaw_deg:.4f}"
        )


def _export_env_extra_poses_to_json(*, out_path: str, pipeline) -> None:
    """Export current env_extra_info reach targets to JSON."""
    env_extra_info = pipeline._env_extra_info

    def _tensor_pose_to_list(p: list) -> list[float]:
        return [float(x) for x in p]

    object_reach_target_poses: dict[str, list[list[float]]] = {}
    for obj_name, pose_list in env_extra_info.object_reach_target_poses.items():
        object_reach_target_poses[obj_name] = [_tensor_pose_to_list(pose.tolist()) for pose in pose_list]

    payload = {
        "object_reach_target_poses": object_reach_target_poses,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    pipeline = make_pipeline(args_cli.pipeline_id)
    pipeline.initialize()
    pipeline.reset_env()

    debug_path = os.path.abspath(args_cli.debug_poses_path)

    _export_env_extra_poses_to_json(out_path=debug_path, pipeline=pipeline)
    print(f"[reach_target_pose] Exported debug poses to: {debug_path}")

    try:
        _apply_live_poses(poses_path=debug_path, pipeline=pipeline)
    except Exception as e:
        print(f"[reach_target_pose] Failed to apply exported debug poses: {e}")

    target_objects = _target_object_names(pipeline)
    poses_before_step = _snapshot_object_poses_w(env=pipeline._env, object_names=target_objects)

    last_mtime = os.path.getmtime(debug_path)
    last_poll_t = 0.0

    for _ in range(20):
        pipeline._env.step(torch.zeros(pipeline._env.action_space.shape, device=pipeline._env.device))

    poses_after_step = _snapshot_object_poses_w(env=pipeline._env, object_names=target_objects)
    _report_pose_drift(poses_before=poses_before_step, poses_after=poses_after_step)

    visualize_reach_target_poses(pipeline._env_extra_info, pipeline._env)

    while simulation_app.is_running():
        pipeline._env.sim.render()

        now = time.time()
        if now - last_poll_t < args_cli.live_poll_interval_s:
            continue
        last_poll_t = now

        try:
            mtime = os.path.getmtime(debug_path)
        except OSError:
            continue

        if mtime > last_mtime:
            last_mtime = mtime
            try:
                _apply_live_poses(poses_path=debug_path, pipeline=pipeline)
                visualize_reach_target_poses(pipeline._env_extra_info, pipeline._env)
                print(f"[reach_target_pose] Reloaded markers from: {debug_path}")
            except Exception as e:
                print(f"[reach_target_pose] Failed to reload poses: {e}")


if __name__ == "__main__":
    main()
    simulation_app.close()
