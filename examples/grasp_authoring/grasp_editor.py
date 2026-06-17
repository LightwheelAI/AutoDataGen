"""Open Isaac Sim's Grasp Editor for an AutoSim pipeline scene.

The script loads an AutoSim pipeline and extracts the gripper described by
``--gripper_cfg`` into a standalone articulated gripper USD at
``/World/GraspEditor/Gripper``.

Usage
-----
1) Start Isaac Sim with the target pipeline:
   python examples/grasp_authoring/grasp_editor.py \
     --pipeline_id <PIPELINE_ID> \
     --gripper_cfg <GRIPPER_CFG_YAML> \
     --viz kit

   Example:
   python examples/grasp_authoring/grasp_editor.py \
     --pipeline_id AutoSimPipeline-FrankaCubeLift-v0 \
     --gripper_cfg examples/grasp_authoring/gripper_configs/franka_panda.yml \
     --viz kit

2) In the Grasp Editor window, confirm the auto-filled gripper, rigid body, and export path,
   then click Ready.

3) Click Mask to mask the colliders and then adjust the pose of the gripper to grasp the object, click Simulate to validate,
   then export `grasps.yaml`.

4) Repeat step 3 to export multiple grasps to the same yaml file if desired.

5) The script automatically appends confidence=1.0 poses converted to the pipeline
   planner frame to the same yaml file after each export.

6) Use the converted grasp poses in your pipeline.

Notes
-----
* The gripper profile is selected entirely by `--gripper_cfg`.
* Pipeline pose tensors use [x, y, z, qx, qy, qz, qw].
* The default exported files are written to `/tmp/autosim_grasp_editor_<pipeline>_<gripper>_*/`.
  Auto conversion watches the current Export File Path shown in Grasp Editor.
* Auto conversion rewrites only the marked converted section, preserving the original
  Isaac Grasp export above it.
"""

import argparse
import math
import os
import re
import sys
import tempfile
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Use Isaac Sim Grasp Editor with an AutoSim pipeline scene.")
parser.add_argument("--pipeline_id", type=str, required=True, help="Name of the autosim pipeline.")
parser.add_argument(
    "--gripper_cfg",
    type=str,
    required=True,
    help="Path to the robot-specific gripper config.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.visualizer is None:
    args_cli.visualizer = ["kit"]

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import carb
import omni.usd
import yaml
from isaacsim.core.utils.extensions import enable_extension, get_extension_id

import autosim_examples  # noqa: F401


def _enable_ui_extensions() -> None:
    """Load Kit UI extensions that IsaacLab's slim app does not enable by default."""
    for ext_name in (
        "omni.kit.uiapp",
        "omni.ui",
        "omni.kit.actions.core",
        "omni.kit.viewport.utility",
        "omni.kit.viewport.window",
    ):
        enable_extension(ext_name)
    for _ in range(3):
        simulation_app.update()


_enable_ui_extensions()

import omni.kit.actions.core
import omni.ui as ui
from grasp_editor_helper import (
    add_world_fixed_joint,
    asset_prim_path,
    configure_grasp_editor_selection,
    current_grasp_editor_export_path,
    extract_gripper_usd,
    first_reach_target_pose_w,
    load_gripper_profile,
    object_rigid_body_path,
    offset_pose_w,
    patch_grasp_editor_runtime,
    reference_gripper_usd,
    resolve_gripper_link_paths,
    robot_prim_path,
    set_prim_pose_w,
    set_selection,
    target_object_name,
)
from pxr import Usd, UsdGeom, UsdPhysics

from autosim import make_pipeline


def _log(message: str = "") -> None:
    print(message, file=sys.__stderr__, flush=True)


def _force_cpu_pipeline_for_grasp_authoring() -> None:
    """Load IsaacLab environments on CPU for Grasp Editor authoring.

    Grasp Editor is an interactive GUI tool and does not need IsaacLab's GPU tensor pipeline.
    Keeping the authoring scene on CPU avoids mixing pipeline-owned GPU tensor views with the
    standalone gripper articulation created by Grasp Editor.
    """
    import isaaclab_tasks.utils as task_utils

    original_parse_env_cfg = task_utils.parse_env_cfg
    if getattr(original_parse_env_cfg, "_autosim_grasp_editor_cpu_patch", False):
        return

    def parse_env_cfg_cpu(
        task_name: str,
        device: str = "cuda:0",
        num_envs: int | None = None,
        use_fabric: bool | None = None,
    ):
        return original_parse_env_cfg(
            task_name=task_name,
            device="cpu",
            num_envs=num_envs,
            use_fabric=False,
        )

    parse_env_cfg_cpu._autosim_grasp_editor_cpu_patch = True
    task_utils.parse_env_cfg = parse_env_cfg_cpu
    _log("[grasp_editor] Forcing IsaacLab pipeline env to device=cpu, use_fabric=False.")


GRIPPER_PRIM_PATH = "/World/GraspEditor/Gripper"
GRIPPER_INITIAL_WORLD_OFFSET = (0.0, 0.0, 0.30)


def _enable_grasp_editor(gripper_profile) -> str | None:
    ext_name = "isaacsim.robot_setup.grasp_editor"
    try:
        enable_extension(ext_name)
        patch_grasp_editor_runtime(gripper_profile)
        for _ in range(5):
            simulation_app.update()
        ext_id = get_extension_id(ext_name)
        if ext_id:
            _log(f"[grasp_editor] Enabled extension: {ext_name}")
            return ext_id
    except Exception as exc:
        _log(f"[grasp_editor] Extension not available ({ext_name}): {exc}")
    _log("[grasp_editor] WARNING: Grasp Editor extension could not be enabled automatically.")
    return None


def _open_grasp_editor_window(ext_id: str | None) -> None:
    if ext_id is None:
        return
    action_name = "CreateUIExtension:Grasp Editor"
    for _ in range(10):
        simulation_app.update()
    try:
        omni.kit.actions.core.get_action_registry().execute_action(ext_id, action_name)
    except Exception as exc:
        carb.log_warn(f"[grasp_editor] Failed to execute Grasp Editor action: {exc}")
    for _ in range(10):
        simulation_app.update()

    window = ui.Workspace.get_window("Grasp Editor")
    if window:
        window.visible = True
        _log("[grasp_editor] Opened Grasp Editor window.")
    else:
        _log("[grasp_editor] Open Grasp Editor from Tools > Robotics > Grasp Editor.")


def _print_scene_summary(
    *,
    gripper_profile,
    robot_path: str,
    link_paths: list[str],
    gripper_usd_path: str,
    target_name: str | None,
    object_path: str | None,
    select_object_path: str | None,
) -> None:
    carb.log_info(f"[grasp_editor] Standalone gripper USD: {gripper_usd_path}")
    _log("\n[grasp_editor] Scene is ready.")
    _log(f"    Pipeline          : {args_cli.pipeline_id}")
    _log(f"    Gripper profile  : {gripper_profile.profile_id} ({gripper_profile.label})")
    _log(f"    Robot prim        : {robot_path}")
    _log(f"    Source links      : {link_paths}")
    _log(f"    Standalone USD    : {gripper_usd_path}")
    _log(f"    Select Gripper    : {GRIPPER_PRIM_PATH}")
    _log(f"    Gripper offset    : {GRIPPER_INITIAL_WORLD_OFFSET}")
    _log(f"    Grasp object      : {target_name or '<not found>'}")
    _log(f"    Object root prim  : {object_path or '<not found>'}")
    _log(f"    Select Object     : {select_object_path or '<select manually>'}")
    _log("\n[grasp_editor] Menu path: Tools > Robotics > Grasp Editor\n")


def _load_pipeline_scene():
    _force_cpu_pipeline_for_grasp_authoring()
    _log(f"[grasp_editor] Loading pipeline: {args_cli.pipeline_id}")
    _log("[grasp_editor] Creating pipeline object...")
    pipeline = make_pipeline(args_cli.pipeline_id)
    _log("[grasp_editor] Initializing pipeline...")
    pipeline.initialize()
    _log("[grasp_editor] Resetting pipeline env...")
    pipeline.reset_env()
    _log("[grasp_editor] Pipeline scene loaded.")
    return pipeline


def _pipeline_core_name(pipeline_id: str) -> str:
    name = pipeline_id.removeprefix("Robofinals-Autosim-")
    name = re.sub(r"Pipeline-v\d+$", "", name)
    name = re.sub(r"-v\d+$", "", name)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _make_work_dir(gripper_profile) -> str:
    prefix = f"autosim_grasp_editor_{_pipeline_core_name(args_cli.pipeline_id)}_{gripper_profile.profile_id}_"
    return tempfile.mkdtemp(prefix=prefix)


def _target_object_paths(pipeline, stage, env_id: int) -> tuple[str | None, str | None, str | None]:
    target_name = target_object_name(pipeline)
    if target_name is None:
        return None, None, None

    object_path = asset_prim_path(pipeline._env.scene[target_name], env_id)
    rigid_body_path = object_rigid_body_path(stage, object_path)
    return target_name, object_path, rigid_body_path


def _suppress_source_robot(stage, robot_path: str) -> None:
    """Hide the pipeline robot and disable its colliders without deleting prims.

    IsaacLab keeps Python-side handles to the robot articulation after env reset.
    Removing the robot prim can leave those handles stale, while hiding and
    disabling collisions is enough for Grasp Editor authoring.
    """
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim or not robot_prim.IsValid():
        carb.log_warn(f"[grasp_editor] Source robot prim not found: {robot_path}")
        return

    UsdGeom.Imageable(robot_prim).MakeInvisible()
    disabled_colliders = 0
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
            disabled_colliders += 1
    _log(f"[grasp_editor] Hid source robot and disabled {disabled_colliders} robot colliders: {robot_path}")


def _make_auto_convert_state() -> dict[str, float | str | None]:
    return {
        "last_checked_at": None,
        "last_handled_mtime": None,
        "export_path": None,
    }


def _maybe_auto_convert_exported_grasps(
    auto_convert_state: dict[str, float | str | None],
    *,
    export_path: str,
    gripper_profile,
    target_name: str | None,
) -> None:
    poll_interval_s = 0.5
    min_file_age_s = 0.5
    now = time.monotonic()
    last_checked_at = auto_convert_state["last_checked_at"]
    if last_checked_at is not None and now - last_checked_at < poll_interval_s:
        return
    auto_convert_state["last_checked_at"] = now

    try:
        export_mtime = os.path.getmtime(export_path)
    except FileNotFoundError:
        return

    if auto_convert_state["export_path"] != export_path:
        auto_convert_state["export_path"] = export_path
        auto_convert_state["last_handled_mtime"] = None
        _log(f"[grasp_editor] Auto-convert watching export path: {export_path}")

    last_handled_mtime = auto_convert_state["last_handled_mtime"]
    if last_handled_mtime is not None and export_mtime <= last_handled_mtime:
        return
    if time.time() - export_mtime < min_file_age_s:
        return

    try:
        handled = _append_converted_grasp_poses(
            export_path,
            gripper_profile=gripper_profile,
            target_name=target_name,
        )
    except Exception as exc:
        _log(f"[grasp_editor] Failed to auto-convert grasp poses: {exc}")
        traceback.print_exc(file=sys.__stderr__)
        return

    if handled:
        try:
            auto_convert_state["last_handled_mtime"] = os.path.getmtime(export_path)
        except FileNotFoundError:
            auto_convert_state["last_handled_mtime"] = export_mtime


def _append_converted_grasp_poses(
    yaml_path: str,
    *,
    gripper_profile,
    target_name: str | None,
) -> bool:
    try:
        with open(yaml_path, encoding="utf-8") as f:
            yaml_text = f.read()
        source_yaml_text = _strip_existing_converted_grasp_section(yaml_text)
        grasp_data = yaml.safe_load(source_yaml_text)
    except FileNotFoundError:
        _log(f"[grasp_editor] Grasp yaml does not exist yet: {yaml_path}")
        return True
    except Exception as exc:
        _log(f"[grasp_editor] Failed to load grasp yaml '{yaml_path}': {exc}")
        return False

    if not isinstance(grasp_data, dict):
        _log(f"[grasp_editor] Cannot convert grasp yaml with non-mapping root: {yaml_path}")
        return True

    converted_poses = _converted_confident_grasp_poses(grasp_data, gripper_profile)
    if not converted_poses:
        _log(f"[grasp_editor] No confidence=1.0 grasps found in: {yaml_path}")
        return True

    converted_section = {
        "source_confidence": 1.0,
        "robot_profile": gripper_profile.robot_profile,
        "target_object_name": target_name,
        "source_object_frame": grasp_data.get("object_frame"),
        "source_gripper_frame": grasp_data.get("gripper_frame"),
        "planner_frame": gripper_profile.planner_frame,
        "pose_format": "[x, y, z, qx, qy, qz, qw]",
        "poses": converted_poses,
    }
    converted_yaml = _render_converted_grasp_section(converted_section, converted_poses)
    stripped_text = source_yaml_text.rstrip()
    new_text = (
        f"{stripped_text}\n\n{_converted_grasp_begin_marker()}\n{converted_yaml}{_converted_grasp_end_marker()}\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(new_text)

    _log(f"[grasp_editor] Appended {len(converted_poses)} converted planner-frame grasp poses to: {yaml_path}")
    return True


def _render_converted_grasp_section(converted_section: dict, converted_poses: dict[str, list[float]]) -> str:
    converted_yaml = yaml.safe_dump(
        {"autosim_converted_grasp_poses": converted_section},
        sort_keys=False,
        default_flow_style=False,
        width=4096,
    )
    return f"{converted_yaml.rstrip()}\n  torch_tensors:\n{_torch_tensor_list_body(converted_poses.values())}\n"


def _converted_confident_grasp_poses(grasp_data: dict, gripper_profile) -> dict[str, list[float]]:
    grasps = grasp_data.get("grasps")
    if not isinstance(grasps, dict):
        return {}

    converted_poses = {}
    gripper_frame_to_planner_pose = gripper_profile.grasp_editor_frame_to_planner_pose
    for grasp_name, grasp in grasps.items():
        if not isinstance(grasp, dict) or not _is_confident_grasp(grasp):
            continue
        gripper_pose = _grasp_yaml_pose_xyzw(grasp)
        converted_poses[str(grasp_name)] = _rounded_pose(
            _compose_poses_xyzw(gripper_pose, gripper_frame_to_planner_pose)
        )
    return converted_poses


def _is_confident_grasp(grasp: dict) -> bool:
    try:
        return math.isclose(float(grasp.get("confidence", 0.0)), 1.0, rel_tol=0.0, abs_tol=1.0e-9)
    except (TypeError, ValueError):
        return False


def _grasp_yaml_pose_xyzw(grasp: dict) -> tuple[float, float, float, float, float, float, float]:
    position = grasp.get("position")
    orientation = grasp.get("orientation")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError(f"Expected grasp position with 3 values, got: {position}")
    if not isinstance(orientation, dict):
        raise ValueError(f"Expected grasp orientation mapping, got: {orientation}")
    xyz = orientation.get("xyz")
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3 or "w" not in orientation:
        raise ValueError(f"Expected orientation as {{w, xyz[3]}}, got: {orientation}")
    return (
        float(position[0]),
        float(position[1]),
        float(position[2]),
        float(xyz[0]),
        float(xyz[1]),
        float(xyz[2]),
        float(orientation["w"]),
    )


def _compose_poses_xyzw(
    parent_to_frame: tuple[float, float, float, float, float, float, float],
    frame_to_child: tuple[float, float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float, float]:
    first_pos, first_quat = parent_to_frame[:3], _normalize_quat_xyzw(parent_to_frame[3:])
    second_pos, second_quat = frame_to_child[:3], _normalize_quat_xyzw(frame_to_child[3:])

    child_pos = _vec_add(first_pos, _quat_rotate_xyzw(first_quat, second_pos))
    child_quat = _normalize_quat_xyzw(_quat_multiply_xyzw(first_quat, second_quat))
    return (*child_pos, *child_quat)


def _quat_multiply_xyzw(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _quat_rotate_xyzw(
    quat: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    quat_vec = quat[:3]
    uv = _cross(quat_vec, vector)
    uuv = _cross(quat_vec, uv)
    return _vec_add(vector, _vec_scale(_vec_add(_vec_scale(uv, quat[3]), uuv), 2.0))


def _normalize_quat_xyzw(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quat))
    if math.isclose(norm, 0.0):
        raise ValueError("Cannot normalize a zero quaternion.")
    return tuple(value / norm for value in quat)


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec_add(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(a_i + b_i for a_i, b_i in zip(a, b, strict=True))


def _vec_scale(vector: tuple[float, ...], scale: float) -> tuple[float, ...]:
    return tuple(value * scale for value in vector)


def _rounded_pose(pose: tuple[float, ...]) -> list[float]:
    return [round(value, 12) for value in pose]


def _torch_tensor_literal(pose: list[float]) -> str:
    values = ", ".join(f"{value:.12g}" for value in pose)
    return f"torch.tensor([{values}])"


def _torch_tensor_list_body(poses) -> str:
    return "\n".join(f"    {_torch_tensor_literal(pose)}," for pose in poses)


def _strip_existing_converted_grasp_section(yaml_text: str) -> str:
    pattern = re.compile(
        rf"\n?{re.escape(_converted_grasp_begin_marker())}\n.*?{re.escape(_converted_grasp_end_marker())}\n?",
        flags=re.DOTALL,
    )
    return pattern.sub("\n", yaml_text)


def _converted_grasp_begin_marker() -> str:
    return "# autosim_converted_grasp_poses_begin"


def _converted_grasp_end_marker() -> str:
    return "# autosim_converted_grasp_poses_end"


def main():
    env_id = 0
    pipeline = _load_pipeline_scene()
    stage = omni.usd.get_context().get_stage()
    gripper_profile = load_gripper_profile(args_cli.gripper_cfg)
    ext_id = _enable_grasp_editor(gripper_profile)
    # Extension startup can finish after the first app updates; the patch is idempotent.
    patch_grasp_editor_runtime(gripper_profile)

    robot_path = robot_prim_path(pipeline, env_id)
    target_name, object_path, rigid_body_path = _target_object_paths(pipeline, stage, env_id)

    link_paths = resolve_gripper_link_paths(stage, robot_path, gripper_profile)
    tmp_dir = _make_work_dir(gripper_profile)
    gripper_usd_path = extract_gripper_usd(stage, robot_path, link_paths, tmp_dir, gripper_profile)
    export_path = f"{tmp_dir}/grasps.yaml"
    reference_gripper_usd(stage, gripper_usd_path, GRIPPER_PRIM_PATH)

    gripper_pose_w = offset_pose_w(
        first_reach_target_pose_w(pipeline, target_name, env_id),
        xyz=GRIPPER_INITIAL_WORLD_OFFSET,
    )
    set_prim_pose_w(stage, GRIPPER_PRIM_PATH, gripper_pose_w)
    add_world_fixed_joint(stage, GRIPPER_PRIM_PATH, gripper_profile)
    _suppress_source_robot(stage, robot_path)
    for _ in range(3):
        simulation_app.update()

    select_object_path = rigid_body_path or object_path
    set_selection(stage, [GRIPPER_PRIM_PATH, select_object_path])
    _open_grasp_editor_window(ext_id)
    configure_grasp_editor_selection(
        gripper_prim_path=GRIPPER_PRIM_PATH,
        object_prim_path=select_object_path,
        export_path=export_path,
    )

    _print_scene_summary(
        gripper_profile=gripper_profile,
        robot_path=robot_path,
        link_paths=link_paths,
        gripper_usd_path=gripper_usd_path,
        target_name=target_name,
        object_path=object_path,
        select_object_path=select_object_path,
    )

    _log(f"[grasp_editor] Auto-convert enabled for exported grasps: {export_path}")
    auto_convert_state = _make_auto_convert_state()
    while simulation_app.is_running():
        active_export_path = current_grasp_editor_export_path(export_path)
        _maybe_auto_convert_exported_grasps(
            auto_convert_state,
            export_path=active_export_path,
            gripper_profile=gripper_profile,
            target_name=target_name,
        )
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        _log(f"[grasp_editor] SystemExit while running Grasp Editor: {exc!r}")
        traceback.print_exc(file=sys.__stderr__)
        raise
    except BaseException as exc:
        _log(f"[grasp_editor] Unhandled error while running Grasp Editor: {exc!r}")
        traceback.print_exc(file=sys.__stderr__)
        raise
    finally:
        simulation_app.close()
