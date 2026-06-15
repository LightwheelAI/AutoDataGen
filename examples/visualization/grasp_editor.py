"""Open Isaac Sim's Grasp Editor for an AutoSim pipeline scene.

The script loads an AutoSim pipeline and extracts the task robot's supported
left-hand profile into a standalone articulated gripper USD at
``/World/GraspEditor/Gripper``.

Usage
-----
1) Start Isaac Sim with the target pipeline:
   python examples/visualization/grasp_editor.py --pipeline_id <PIPELINE_ID> --viz kit

   Example:
   python examples/visualization/grasp_editor.py \
     --pipeline_id Robofinals-Autosim-KettleBoilingPipeline-v0 \
     --viz kit

2) In the Grasp Editor window, confirm the auto-filled gripper, rigid body, and export path,
   then click Ready.

3) Click Mask to mask the colliders and then adjust the pose of the gripper to grasp the object, click Simulate to validate,
   then export `grasps.yaml`.

4) Repeat step 3 to export multiple grasps to the same yaml file if desired.

5) Use the exported grasp poses in your pipeline.

Notes
-----
* Supported pipeline robot profiles: `x7s_joint_left`, `g1_wbc_left`.
* Grasp Editor poses are authored relative to the selected gripper base frame:
  `left_hand_link` for X7S, `left_wrist_yaw_link` for G1 WBC.
* Pipeline pose tensors use [x, y, z, qx, qy, qz, qw].
* The default exported files are written to `/tmp/autosim_grasp_editor_<pipeline>_<gripper>_*/`, but you are encouraged to specify your own export path in the Grasp Editor UI.
"""

import argparse
import re
import sys
import tempfile
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Use Isaac Sim Grasp Editor with an AutoSim pipeline scene.")
parser.add_argument("--pipeline_id", type=str, required=True, help="Name of the autosim pipeline.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.visualizer is None:
    args_cli.visualizer = ["kit"]

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import carb
import omni.usd
from isaacsim.core.utils.extensions import enable_extension, get_extension_id

# post-launch imports
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
    extract_gripper_usd,
    first_reach_target_pose_w,
    get_gripper_profile,
    object_rigid_body_path,
    offset_pose_w,
    patch_grasp_editor_runtime,
    reference_gripper_usd,
    resolve_gripper_link_paths,
    robot_prim_path,
    robot_profile_id,
    set_prim_pose_w,
    set_selection,
    target_object_name,
)
from pxr import Sdf

from autosim import make_pipeline


def _log(message: str = "") -> None:
    print(message, file=sys.__stderr__, flush=True)


ENV_ID = 0
GRIPPER_PRIM_PATH = "/World/GraspEditor/Gripper"
GRIPPER_INITIAL_WORLD_OFFSET = (0.0, 0.0, 0.30)
GRASP_EDITOR_EXTENSIONS = ("isaacsim.robot_setup.grasp_editor",)


def _enable_grasp_editor(gripper_profile) -> str | None:
    for ext_name in GRASP_EDITOR_EXTENSIONS:
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


def _target_object_paths(pipeline, stage) -> tuple[str | None, str | None, str | None]:
    target_name = target_object_name(pipeline)
    if target_name is None:
        return None, None, None

    object_path = asset_prim_path(pipeline._env.scene[target_name], ENV_ID)
    rigid_body_path = object_rigid_body_path(stage, object_path)
    return target_name, object_path, rigid_body_path


def _remove_source_robot(stage, robot_path: str) -> None:
    if stage.RemovePrim(Sdf.Path(robot_path)):
        _log(f"[grasp_editor] Removed source robot from stage: {robot_path}")
    else:
        carb.log_warn(f"[grasp_editor] Failed to remove source robot from stage: {robot_path}")


def main():
    pipeline = _load_pipeline_scene()
    stage = omni.usd.get_context().get_stage()
    gripper_profile = get_gripper_profile(robot_profile_id(pipeline))
    ext_id = _enable_grasp_editor(gripper_profile)
    # Extension startup can finish after the first app updates; the patch is idempotent.
    patch_grasp_editor_runtime(gripper_profile)

    robot_path = robot_prim_path(pipeline, ENV_ID)
    target_name, object_path, rigid_body_path = _target_object_paths(pipeline, stage)

    link_paths = resolve_gripper_link_paths(stage, robot_path, gripper_profile)
    tmp_dir = _make_work_dir(gripper_profile)
    gripper_usd_path = extract_gripper_usd(stage, robot_path, link_paths, tmp_dir, gripper_profile)
    export_path = f"{tmp_dir}/grasps.yaml"
    reference_gripper_usd(stage, gripper_usd_path, GRIPPER_PRIM_PATH)

    gripper_pose_w = offset_pose_w(
        first_reach_target_pose_w(pipeline, target_name, ENV_ID),
        xyz=GRIPPER_INITIAL_WORLD_OFFSET,
    )
    set_prim_pose_w(stage, GRIPPER_PRIM_PATH, gripper_pose_w)
    add_world_fixed_joint(stage, GRIPPER_PRIM_PATH, gripper_profile)
    _remove_source_robot(stage, robot_path)
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

    while simulation_app.is_running():
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
