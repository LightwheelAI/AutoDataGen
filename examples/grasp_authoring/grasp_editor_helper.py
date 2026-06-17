"""Helpers for ``grasp_editor.py``.

This module is imported only after Isaac Sim has been launched by
``AppLauncher``. Keep Isaac/Omniverse imports here post-launch only.
"""

import os
from dataclasses import dataclass

import carb
import carb.settings
import numpy as np
import omni.usd
import torch
import yaml
from isaaclab.utils.math import combine_frame_transforms
from isaacsim.core.utils.xforms import reset_and_set_xform_ops
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from autosim.utils.data_util import as_torch


@dataclass(frozen=True)
class GripperProfile:
    robot_profile: str
    profile_id: str
    label: str
    base_link_name: str
    link_names: tuple[str, ...]
    dof_defaults: dict[str, tuple[float, float]]
    max_effort: float
    output_usd_name: str
    planner_frame: str
    grasp_editor_frame_to_planner_pose: tuple[float, float, float, float, float, float, float]


_LAST_GRASP_EDITOR_UI_BUILDER = None
_PENDING_GRASP_EDITOR_SELECTION: dict[str, str | None] = {}


def load_gripper_profile(cfg_path: str | None = None) -> GripperProfile:
    if not cfg_path:
        raise RuntimeError("Pass --gripper_cfg with a matching gripper config file.")

    cfg_path = os.path.abspath(os.path.expanduser(cfg_path))
    if not os.path.exists(cfg_path):
        raise RuntimeError(f"Gripper config not found: {cfg_path}. Pass --gripper_cfg with a matching config file.")

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Gripper config must be a YAML mapping: {cfg_path}")

    cfg_robot_profile = cfg.get("robot_profile")
    if not cfg_robot_profile:
        raise RuntimeError(f"Gripper config must declare a non-empty robot_profile: {cfg_path}")

    gripper_cfg = _required_mapping(cfg, "gripper", cfg_path)
    planner_cfg = _required_mapping(cfg, "planner", cfg_path)
    return GripperProfile(
        robot_profile=cfg_robot_profile,
        profile_id=_required_str(gripper_cfg, "profile_id", cfg_path),
        label=_required_str(gripper_cfg, "label", cfg_path),
        base_link_name=_required_str(gripper_cfg, "base_link_name", cfg_path),
        link_names=tuple(_required_str_list(gripper_cfg, "link_names", cfg_path)),
        dof_defaults=_parse_dof_defaults(_required_mapping(gripper_cfg, "dof_defaults", cfg_path), cfg_path),
        max_effort=float(_required_value(gripper_cfg, "max_effort", cfg_path)),
        output_usd_name=_required_str(gripper_cfg, "output_usd_name", cfg_path),
        planner_frame=_required_str(planner_cfg, "frame", cfg_path),
        grasp_editor_frame_to_planner_pose=_required_float_tuple(
            planner_cfg, "grasp_editor_frame_to_planner_pose", 7, cfg_path
        ),
    )


def _required_value(cfg: dict, key: str, cfg_path: str):
    if key not in cfg:
        raise RuntimeError(f"Missing '{key}' in gripper config: {cfg_path}")
    return cfg[key]


def _required_mapping(cfg: dict, key: str, cfg_path: str) -> dict:
    value = _required_value(cfg, key, cfg_path)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected '{key}' to be a mapping in gripper config: {cfg_path}")
    return value


def _required_str(cfg: dict, key: str, cfg_path: str) -> str:
    value = _required_value(cfg, key, cfg_path)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Expected '{key}' to be a non-empty string in gripper config: {cfg_path}")
    return value


def _required_str_list(cfg: dict, key: str, cfg_path: str) -> list[str]:
    value = _required_value(cfg, key, cfg_path)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise RuntimeError(f"Expected '{key}' to be a non-empty string list in gripper config: {cfg_path}")
    return value


def _required_float_tuple(cfg: dict, key: str, length: int, cfg_path: str) -> tuple[float, ...]:
    value = _required_value(cfg, key, cfg_path)
    if not isinstance(value, list) or len(value) != length:
        raise RuntimeError(f"Expected '{key}' to contain {length} numbers in gripper config: {cfg_path}")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Expected '{key}' to contain only numbers in gripper config: {cfg_path}") from exc


def _parse_dof_defaults(dof_cfg: dict, cfg_path: str) -> dict[str, tuple[float, float]]:
    dof_defaults = {}
    for dof_name, values in dof_cfg.items():
        if not isinstance(dof_name, str) or not dof_name:
            raise RuntimeError(f"DOF names must be non-empty strings in gripper config: {cfg_path}")
        if not isinstance(values, dict):
            raise RuntimeError(f"Expected DOF '{dof_name}' values to be a mapping in gripper config: {cfg_path}")
        open_position = float(_required_value(values, "open_position", cfg_path))
        close_position = float(_required_value(values, "close_position", cfg_path))
        dof_defaults[dof_name] = (open_position, close_position)
    return dof_defaults


def patch_grasp_editor_runtime(gripper_profile: GripperProfile) -> None:
    try:
        from isaacsim.core.api.articulations import ArticulationSubset
        from isaacsim.core.api.controllers.articulation_controller import (
            ArticulationController,
        )
        from isaacsim.core.prims import RigidPrim, SingleArticulation
        from isaacsim.core.simulation_manager.impl.simulation_manager import (
            SimulationManager,
        )
        from isaacsim.robot_setup.grasp_editor import (
            ui_builder as grasp_editor_ui_builder,
        )
        from isaacsim.robot_setup.grasp_editor import util as grasp_editor_util
        from isaacsim.robot_setup.grasp_editor.ui_builder import UIBuilder
    except Exception as exc:
        carb.log_warn(f"[grasp_editor] Grasp Editor runtime patch skipped: {exc}")
        return

    _patch_simulation_manager_runtime(SimulationManager)

    if getattr(SingleArticulation.initialize, "_autosim_grasp_patch", False):
        return

    ensure_physics_sim_view, coerce_backend_data, to_numpy = _make_physics_helpers(SimulationManager)
    art_patches = _make_articulation_patches(
        _articulation_originals(
            SingleArticulation,
            ArticulationController,
            ArticulationSubset,
            RigidPrim,
        ),
        ensure_physics_sim_view,
        coerce_backend_data,
        to_numpy,
    )
    ui_patches = _make_ui_patches(
        gripper_profile,
        {
            "build_selection_frame": UIBuilder.build_selection_frame,
            "build_reference_frame": UIBuilder.build_reference_frame,
            "finalize_reference_frame_selection": UIBuilder._finalize_reference_frame_selection,
            "on_stage_event": getattr(UIBuilder, "on_stage_event", None),
            "populate_settings_frame": UIBuilder._populate_settings_frame,
        },
    )
    util_patches = _make_util_patches(grasp_editor_util)

    SingleArticulation.__init__ = art_patches["articulation_constructor"]
    SingleArticulation.initialize = art_patches["articulation_initialize"]
    SingleArticulation.apply_action = art_patches["apply_action"]
    SingleArticulation.set_joint_efforts = art_patches["set_joint_efforts"]
    SingleArticulation.set_joint_positions = art_patches["set_joint_positions"]
    SingleArticulation.set_joint_velocities = art_patches["set_joint_velocities"]
    ArticulationController.apply_action = art_patches["controller_apply_action"]
    ArticulationController.set_max_efforts = art_patches["controller_set_max_efforts"]
    ArticulationSubset.get_joint_efforts = art_patches["subset_get_joint_efforts"]
    ArticulationSubset.get_joint_positions = art_patches["subset_get_joint_positions"]
    ArticulationSubset.get_joint_velocities = art_patches["subset_get_joint_velocities"]
    RigidPrim.apply_forces_and_torques_at_pos = art_patches["rigid_apply_forces_and_torques_at_pos"]
    RigidPrim.initialize = art_patches["rigid_initialize"]
    RigidPrim.set_velocities = art_patches["rigid_set_velocities"]
    grasp_editor_util.convert_prim_to_collidable_rigid_body = art_patches["convert_prim_to_rigid_body"]
    grasp_editor_ui_builder.convert_prim_to_collidable_rigid_body = art_patches["convert_prim_to_rigid_body"]
    grasp_editor_util.find_all_articulations = util_patches["find_all_articulations"]
    grasp_editor_ui_builder.find_all_articulations = util_patches["find_all_articulations"]
    grasp_editor_util.move_rb_subframe_to_position = util_patches["move_rb_subframe_to_position"]
    grasp_editor_ui_builder.move_rb_subframe_to_position = util_patches["move_rb_subframe_to_position"]
    UIBuilder.build_selection_frame = ui_patches["build_selection_frame"]
    UIBuilder.build_reference_frame = ui_patches["build_reference_frame"]
    UIBuilder.stop_rigid_body = ui_patches["stop_rigid_body"]
    UIBuilder._finalize_reference_frame_selection = ui_patches["finalize_reference_frame_selection"]
    if ui_patches["on_stage_event"] is not None:
        UIBuilder.on_stage_event = ui_patches["on_stage_event"]
    UIBuilder._populate_settings_frame = ui_patches["populate_settings_frame"]
    print("[grasp_editor] Patched Grasp Editor runtime initialization.")


def _patch_simulation_manager_runtime(SimulationManager) -> None:
    """Point Isaac Sim core prim wrappers at the full SimulationManager implementation.

    In Isaac Sim 6.0, ``isaacsim.core.simulation_manager.SimulationManager`` may resolve to the
    lower-level PhysxManager export inside some extension contexts. Legacy core wrappers still
    expect helper methods such as ``_get_backend_utils()``, so Grasp Editor needs the full Python
    implementation while it constructs SingleArticulation/RigidPrim objects.
    """
    import isaacsim.core.simulation_manager as simulation_manager_module

    _configure_grasp_editor_physics_sync(SimulationManager)
    simulation_manager_module.SimulationManager = SimulationManager
    for module_name in (
        "isaacsim.core.prims.impl.prim",
        "isaacsim.core.prims.impl.xform_prim",
        "isaacsim.core.prims.impl.articulation",
        "isaacsim.core.prims.impl.rigid_prim",
    ):
        try:
            module = __import__(module_name, fromlist=["SimulationManager"])
        except Exception as exc:
            carb.log_warn(f"[grasp_editor] Failed to patch {module_name}.SimulationManager: {exc}")
            continue
        if hasattr(module, "SimulationManager"):
            module.SimulationManager = SimulationManager


def _configure_grasp_editor_physics_sync(SimulationManager) -> None:
    """Use USD transform syncing instead of PhysX Fabric for Grasp Editor.

    Grasp Editor reads and displays ordinary USD prim transforms. With Fabric enabled, Isaac Sim 6 can update
    articulation joint state without writing the resulting link transforms back to USD, which makes the gripper
    appear static while the DOF values change.
    """
    settings = carb.settings.get_settings()
    try:
        SimulationManager.enable_fabric(False)
    except Exception as exc:
        carb.log_warn(f"[grasp_editor] Failed to disable PhysX Fabric: {exc}")
    settings.set_bool("/physics/updateToUsd", True)
    settings.set_bool("/physics/updateVelocitiesToUsd", True)


def configure_grasp_editor_selection(
    *,
    gripper_prim_path: str,
    object_prim_path: str | None,
    export_path: str | None,
) -> None:
    global _PENDING_GRASP_EDITOR_SELECTION
    _PENDING_GRASP_EDITOR_SELECTION = {
        "gripper_prim_path": gripper_prim_path,
        "object_prim_path": object_prim_path,
        "export_path": export_path,
    }
    if _LAST_GRASP_EDITOR_UI_BUILDER is None:
        carb.log_warn("[grasp_editor] Grasp Editor UI is not built yet; selection will be applied when it opens.")
        return

    _apply_pending_grasp_editor_selection(_LAST_GRASP_EDITOR_UI_BUILDER)


def current_grasp_editor_export_path(default_export_path: str) -> str:
    """Return the export path currently shown in Grasp Editor's UI."""

    if _LAST_GRASP_EDITOR_UI_BUILDER is None:
        return default_export_path

    export_field = getattr(_LAST_GRASP_EDITOR_UI_BUILDER, "_export_path", None)
    if export_field is None:
        return default_export_path

    for getter_name in ("get_value_as_string", "get_value"):
        getter = getattr(export_field, getter_name, None)
        if getter is None:
            continue
        try:
            value = getter()
        except Exception:
            continue
        if value:
            return str(value)

    return default_export_path


def robot_prim_path(pipeline, env_id: int) -> str:
    robot = pipeline._robot
    if hasattr(robot, "prim_paths") and len(robot.prim_paths) > env_id:
        return robot.prim_paths[env_id]
    if hasattr(robot, "prim_path"):
        return robot.prim_path
    return f"/World/envs/env_{env_id}/Robot"


def asset_prim_path(asset, env_id: int) -> str | None:
    if hasattr(asset, "prim_paths") and len(asset.prim_paths) > env_id:
        return asset.prim_paths[env_id]
    if hasattr(asset, "prim_path"):
        return asset.prim_path
    cfg_prim_path = getattr(getattr(asset, "cfg", None), "prim_path", None)
    if cfg_prim_path:
        env_ns = f"/World/envs/env_{env_id}"
        return (
            cfg_prim_path.replace("{ENV_REGEX_NS}", env_ns)
            .replace("{ENV_NS}", env_ns)
            .replace("/World/envs/env_.*/", f"{env_ns}/")
        )
    return None


def target_object_name(pipeline) -> str | None:
    env = pipeline._env
    for name in pipeline._env_extra_info.object_reach_target_poses.keys():
        if name in env.scene.keys():
            return name
    return None


def object_rigid_body_path(stage, object_prim_path: str | None) -> str | None:
    if object_prim_path is None:
        return None
    object_prim = stage.GetPrimAtPath(object_prim_path)
    if not object_prim or not object_prim.IsValid():
        return None
    if object_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return object_prim_path

    rigid_bodies = [
        prim
        for prim in Usd.PrimRange(object_prim)
        if prim.GetPath() != object_prim.GetPath() and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(rigid_bodies) == 1:
        return str(rigid_bodies[0].GetPath())
    if len(rigid_bodies) > 1:
        return str(sorted(rigid_bodies, key=lambda prim: len(str(prim.GetPath())))[0].GetPath())
    return None


def first_reach_target_pose_w(pipeline, object_name: str | None, env_id: int) -> torch.Tensor | None:
    if object_name is None:
        return None
    pose_list = pipeline._env_extra_info.object_reach_target_poses.get(object_name)
    if not pose_list:
        return None

    obj_pose_w = as_torch(pipeline._env.scene[object_name].data.root_pose_w)[env_id]
    target_pose = pose_list[0].to(device=obj_pose_w.device, dtype=obj_pose_w.dtype).unsqueeze(0)
    target_pos_w, target_quat_w = combine_frame_transforms(
        obj_pose_w[:3].unsqueeze(0),
        obj_pose_w[3:].unsqueeze(0),
        target_pose[:, :3],
        target_pose[:, 3:],
    )
    return torch.cat([target_pos_w, target_quat_w], dim=-1).squeeze(0)


def offset_pose_w(pose_w: torch.Tensor | None, *, xyz: tuple[float, float, float]) -> torch.Tensor | None:
    if pose_w is None:
        return None
    offset = pose_w.new_tensor(xyz)
    pose_w = pose_w.clone()
    pose_w[:3] += offset
    return pose_w


def set_prim_pose_w(stage, prim_path: str, pose_w: torch.Tensor | None) -> None:
    if pose_w is None:
        return
    pose = [float(v) for v in pose_w.detach().cpu().tolist()]
    prim = stage.GetPrimAtPath(prim_path)
    reset_and_set_xform_ops(
        prim,
        translation=Gf.Vec3d(*pose[:3]),
        orientation=Gf.Quatd(pose[6], Gf.Vec3d(*pose[3:6])),
    )


def resolve_gripper_link_paths(stage, robot_prim_path: str, gripper_profile: GripperProfile) -> list[str]:
    link_paths = [f"{robot_prim_path}/{name}" for name in gripper_profile.link_names]
    missing = [path for path in link_paths if not stage.GetPrimAtPath(path).IsValid()]
    if missing:
        raise RuntimeError(f"{gripper_profile.label} links not found: {missing}")

    print(f"[grasp_editor] {gripper_profile.label} links: {link_paths}")
    return link_paths


def extract_gripper_usd(
    stage,
    robot_prim_path: str,
    link_paths: list[str],
    tmp_dir: str,
    gripper_profile: GripperProfile,
) -> str:
    src_link_paths = [Sdf.Path(path) for path in link_paths]
    dst_root_path = Sdf.Path("/Gripper")

    flat_path = os.path.join(tmp_dir, "flat_scene.usda")
    stage.Flatten().Export(flat_path)
    flat_stage = Usd.Stage.Open(flat_path)

    missing = [str(path) for path in src_link_paths if not flat_stage.GetPrimAtPath(path).IsValid()]
    if missing:
        raise RuntimeError(f"Gripper links not found in flattened stage: {missing}")

    out_path = os.path.join(tmp_dir, gripper_profile.output_usd_name)
    dst_stage = Usd.Stage.CreateNew(out_path)
    dst_stage.DefinePrim(dst_root_path, "Xform")
    dst_stage.SetDefaultPrim(dst_stage.GetPrimAtPath(dst_root_path))

    base_world_tf = UsdGeom.Xformable(flat_stage.GetPrimAtPath(src_link_paths[0])).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    base_world_tf_inv = base_world_tf.GetInverse()

    for src_link_path in src_link_paths:
        dst_link_path = dst_root_path.AppendChild(src_link_path.name)
        _copy_link_subtree_without_selected_children(
            flat_stage=flat_stage,
            dst_stage=dst_stage,
            src_link_path=src_link_path,
            src_link_paths=src_link_paths,
            dst_link_path=dst_link_path,
        )
        src_link_prim = flat_stage.GetPrimAtPath(src_link_path)
        dst_link_prim = dst_stage.GetPrimAtPath(dst_link_path)
        link_world_tf = UsdGeom.Xformable(src_link_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        _set_local_transform_matrix(dst_link_prim, link_world_tf * base_world_tf_inv)

    _copy_flattened_prototypes_into_gripper(flat_stage, dst_stage, dst_root_path)

    joints_root_path = dst_root_path.AppendChild("joints")
    dst_stage.DefinePrim(joints_root_path, "Scope")
    copied_joints: list[str] = []
    robot_prim = flat_stage.GetPrimAtPath(robot_prim_path)
    for prim in Usd.PrimRange(robot_prim):
        if not _is_joint_prim(prim):
            continue
        body0_targets, body1_targets = _joint_body_targets(prim)
        if not (
            _targets_are_inside_links(body0_targets, src_link_paths)
            and _targets_are_inside_links(body1_targets, src_link_paths)
        ):
            continue
        dst_joint_path = joints_root_path.AppendChild(prim.GetName())
        _copy_joint_with_retargeted_bodies(
            src_stage=flat_stage,
            dst_stage=dst_stage,
            src_joint_path=prim.GetPath(),
            dst_joint_path=dst_joint_path,
            src_link_paths=src_link_paths,
            dst_root_path=dst_root_path,
        )
        copied_joints.append(str(dst_joint_path))

    if not copied_joints:
        raise RuntimeError(f"No internal gripper joints were copied for links: {link_paths}")

    _clear_external_material_bindings(dst_stage, dst_root_path)
    _initialize_gripper_physics_state(dst_stage, dst_root_path, gripper_profile)

    for prim in Usd.PrimRange(dst_stage.GetPrimAtPath(dst_root_path)):
        if prim.GetPath() != dst_root_path and (
            prim.HasAPI(UsdPhysics.ArticulationRootAPI) or prim.HasAPI(PhysxSchema.PhysxArticulationAPI)
        ):
            _remove_articulation_root(prim)

    _apply_articulation_root(dst_stage.GetPrimAtPath(dst_root_path))
    dst_stage.Save()

    print(f"[grasp_editor] Standalone gripper USD: {out_path}")
    print(f"[grasp_editor] Copied gripper joints: {copied_joints}")
    return out_path


def reference_gripper_usd(stage, gripper_usd_path: str, gripper_prim_path: str) -> None:
    parent_path = Sdf.Path(gripper_prim_path).GetParentPath()
    if str(parent_path) != ".":
        stage.DefinePrim(parent_path, "Xform")
    gripper_prim = stage.DefinePrim(gripper_prim_path, "Xform")
    gripper_prim.GetReferences().ClearReferences()
    gripper_prim.GetReferences().AddReference(gripper_usd_path)
    _apply_articulation_root(gripper_prim)


def add_world_fixed_joint(stage, gripper_prim_path: str, gripper_profile: GripperProfile) -> None:
    base_link_path = Sdf.Path(f"{gripper_prim_path}/{gripper_profile.base_link_name}")
    if not stage.GetPrimAtPath(base_link_path).IsValid():
        raise RuntimeError(f"Cannot add fixed joint; gripper base link not found: {base_link_path}")

    fixed_joint_path = Sdf.Path(f"{gripper_prim_path}/FixedJoint")
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, fixed_joint_path)
    fixed_joint.GetBody0Rel().ClearTargets(False)
    fixed_joint.GetBody1Rel().SetTargets([base_link_path])
    fixed_joint.GetJointEnabledAttr().Set(True)
    fixed_joint.GetExcludeFromArticulationAttr().Set(False)
    print(f"[grasp_editor] Added world fixed joint: {fixed_joint_path} -> {base_link_path}")


def set_selection(stage, paths: list[str]) -> None:
    valid_paths = [path for path in paths if path and stage.GetPrimAtPath(path).IsValid()]
    omni.usd.get_context().get_selection().set_selected_prim_paths(valid_paths, True)


def _apply_pending_grasp_editor_selection(ui_builder) -> None:
    if not _PENDING_GRASP_EDITOR_SELECTION:
        return

    gripper_prim_path = _PENDING_GRASP_EDITOR_SELECTION["gripper_prim_path"]
    object_prim_path = _PENDING_GRASP_EDITOR_SELECTION["object_prim_path"]
    export_path = _PENDING_GRASP_EDITOR_SELECTION["export_path"]

    gripper_dropdown = getattr(ui_builder, "_gripper_selection_dropdown", None)
    if gripper_dropdown is not None:
        gripper_dropdown.repopulate()
        if gripper_prim_path in gripper_dropdown.get_items():
            gripper_dropdown.set_selection(gripper_prim_path)
        else:
            carb.log_warn(f"[grasp_editor] Gripper is not present in Grasp Editor dropdown: {gripper_prim_path}")

    rb_field = getattr(ui_builder, "_rb_conversion_stringfield", None)
    if rb_field is not None and object_prim_path:
        rb_field.set_value(object_prim_path)

    export_field = getattr(ui_builder, "_export_path", None)
    if export_field is not None and export_path:
        export_field.set_value(export_path)

    ready_btn = getattr(ui_builder, "_selection_ready_btn", None)
    if ready_btn is not None and object_prim_path and export_path:
        ready_btn.enabled = True

    print(
        "[grasp_editor] Auto-selected Grasp Editor inputs: "
        f"gripper={gripper_prim_path}, object={object_prim_path or '<none>'}, export={export_path or '<none>'}"
    )


def _make_physics_helpers(SimulationManager):
    def get_physics_sim_view():
        if hasattr(SimulationManager, "get_physics_sim_view"):
            return SimulationManager.get_physics_sim_view()
        if hasattr(SimulationManager, "get_physics_simulation_view"):
            return SimulationManager.get_physics_simulation_view()
        return None

    def ensure_physics_sim_view() -> None:
        _configure_grasp_editor_physics_sync(SimulationManager)
        if get_physics_sim_view() is not None:
            return
        if hasattr(SimulationManager, "initialize_physics"):
            SimulationManager.initialize_physics()
        if get_physics_sim_view() is None and hasattr(SimulationManager, "_create_simulation_view"):
            SimulationManager._create_simulation_view(None)

    def coerce_backend_data(backend_utils, device, data, dtype="float32"):
        if data is None:
            return None
        if dtype == "long":
            dtype = "int64"
        return backend_utils.convert(data, device=device, dtype=dtype)

    def to_numpy(data):
        if data is None:
            return None
        if hasattr(data, "detach"):
            return data.detach().cpu().numpy()
        return data

    return ensure_physics_sim_view, coerce_backend_data, to_numpy


def _make_util_patches(grasp_editor_util):
    def as_numpy(data):
        if hasattr(data, "detach"):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    def as_backend_data(view, data, dtype="float32"):
        if not hasattr(view, "_backend_utils"):
            return data
        return view._backend_utils.convert(data, device=view._device, dtype=dtype)

    def move_rb_subframe_to_position(rb_xform_view, rb_subframe, desired_translation, desired_orientation):
        a_trans, a_orient = rb_xform_view.get_world_poses()
        a_trans = as_numpy(a_trans)[0]
        a_orient = as_numpy(a_orient)[0]

        b_trans, b_orient = grasp_editor_util.get_world_pose(rb_subframe)
        b_trans = as_numpy(b_trans)
        b_orient = as_numpy(b_orient)

        c_trans = as_numpy(desired_translation)
        c_orient = as_numpy(desired_orientation)

        a_rot, b_rot, c_rot = grasp_editor_util.quats_to_rot_matrices(np.vstack([a_orient, b_orient, c_orient]))
        a_rot_cmd = c_rot @ b_rot.T @ a_rot
        a_trans_cmd = c_trans + c_rot @ b_rot.T @ (a_trans - b_trans)
        a_orient_cmd = grasp_editor_util.rot_matrices_to_quats(a_rot_cmd)

        rb_xform_view.set_world_poses(
            as_backend_data(rb_xform_view, a_trans_cmd[np.newaxis, :]),
            as_backend_data(rb_xform_view, a_orient_cmd[np.newaxis, :]),
        )

    def find_all_articulations():
        art_root_paths = []
        articulation_candidates = set()
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return art_root_paths

        for prim in Usd.PrimRange(stage.GetPrimAtPath("/")):
            if (
                prim.HasAPI(UsdPhysics.ArticulationRootAPI)
                and prim.GetProperty("physxArticulation:articulationEnabled").IsValid()
                and prim.GetProperty("physxArticulation:articulationEnabled").Get()
            ):
                art_root_paths.append(tuple(str(prim.GetPath()).split("/")[1:]))
                continue

            joint = UsdPhysics.Joint(prim)
            if not joint:
                continue
            bodies = joint.GetBody0Rel().GetTargets()
            bodies.extend(joint.GetBody1Rel().GetTargets())
            if len(bodies) < 2:
                continue

            base_path_split = str(bodies[0]).split("/")[1:]
            for body in bodies[1:]:
                body_path_split = str(body).split("/")[1:]
                for i in range(len(base_path_split)):
                    if i >= len(body_path_split) or base_path_split[i] != body_path_split[i]:
                        base_path_split = base_path_split[:i]
                        break
            if base_path_split:
                articulation_candidates.add(tuple(base_path_split))

        unique_candidates = []
        for c1 in articulation_candidates:
            is_unique = True
            for c2 in articulation_candidates:
                if c1 == c2:
                    continue
                if c2[: len(c1)] == c1:
                    is_unique = False
                    break
            if is_unique:
                unique_candidates.append(c1)

        art_base_paths = []
        for candidate in unique_candidates:
            subset_count = 0
            for root in art_root_paths:
                if root[: len(candidate)] == candidate:
                    subset_count += 1
            if subset_count == 1:
                art_base_paths.append("/" + "/".join(candidate))

        return art_base_paths

    return {
        "find_all_articulations": find_all_articulations,
        "move_rb_subframe_to_position": move_rb_subframe_to_position,
    }


def _articulation_originals(SingleArticulation, ArticulationController, ArticulationSubset, RigidPrim) -> dict:
    return {
        "articulation_constructor": SingleArticulation.__init__,
        "articulation_initialize": SingleArticulation.initialize,
        "apply_action": SingleArticulation.apply_action,
        "controller_apply_action": ArticulationController.apply_action,
        "controller_set_max_efforts": ArticulationController.set_max_efforts,
        "rigid_apply_forces_and_torques_at_pos": RigidPrim.apply_forces_and_torques_at_pos,
        "rigid_initialize": RigidPrim.initialize,
        "rigid_set_velocities": RigidPrim.set_velocities,
        "set_joint_efforts": SingleArticulation.set_joint_efforts,
        "set_joint_positions": SingleArticulation.set_joint_positions,
        "set_joint_velocities": SingleArticulation.set_joint_velocities,
        "subset_get_joint_efforts": ArticulationSubset.get_joint_efforts,
        "subset_get_joint_positions": ArticulationSubset.get_joint_positions,
        "subset_get_joint_velocities": ArticulationSubset.get_joint_velocities,
    }


def _make_articulation_patches(originals, ensure_physics_sim_view, coerce_backend_data, to_numpy):
    original_articulation_constructor = originals["articulation_constructor"]
    original_articulation_initialize = originals["articulation_initialize"]
    original_apply_action = originals["apply_action"]
    original_controller_apply_action = originals["controller_apply_action"]
    original_controller_set_max_efforts = originals["controller_set_max_efforts"]
    original_rigid_apply_forces_and_torques_at_pos = originals["rigid_apply_forces_and_torques_at_pos"]
    original_rigid_initialize = originals["rigid_initialize"]
    original_rigid_set_velocities = originals["rigid_set_velocities"]
    original_set_joint_efforts = originals["set_joint_efforts"]
    original_set_joint_positions = originals["set_joint_positions"]
    original_set_joint_velocities = originals["set_joint_velocities"]
    original_subset_get_joint_efforts = originals["subset_get_joint_efforts"]
    original_subset_get_joint_positions = originals["subset_get_joint_positions"]
    original_subset_get_joint_velocities = originals["subset_get_joint_velocities"]

    def coerce_articulation_data(self, data, dtype="float32"):
        return coerce_backend_data(self._backend_utils, self._device, data, dtype)

    def coerce_articulation_action(articulation_view, control_actions):
        backend_utils = articulation_view._backend_utils
        dev = articulation_view._device
        control_actions.joint_positions = coerce_backend_data(backend_utils, dev, control_actions.joint_positions)
        control_actions.joint_velocities = coerce_backend_data(backend_utils, dev, control_actions.joint_velocities)
        control_actions.joint_efforts = coerce_backend_data(backend_utils, dev, control_actions.joint_efforts)
        control_actions.joint_indices = coerce_backend_data(
            backend_utils, dev, control_actions.joint_indices, dtype="int64"
        )
        return control_actions

    def articulation_constructor(self, *args, **kwargs):
        ensure_physics_sim_view()
        return original_articulation_constructor(self, *args, **kwargs)

    def articulation_initialize(self, *args, **kwargs):
        ensure_physics_sim_view()
        result = original_articulation_initialize(self, *args, **kwargs)
        try:
            self.disable_gravity()
        except Exception as exc:
            carb.log_warn(f"[grasp_editor] Failed to disable gripper gravity: {exc}")
        return result

    def rigid_initialize(self, *args, **kwargs):
        ensure_physics_sim_view()
        return original_rigid_initialize(self, *args, **kwargs)

    def set_joint_positions(self, positions, joint_indices=None):
        positions = coerce_articulation_data(self, positions)
        joint_indices = coerce_articulation_data(self, joint_indices, dtype="int64")
        return original_set_joint_positions(self, positions, joint_indices)

    def set_joint_velocities(self, velocities, joint_indices=None):
        velocities = coerce_articulation_data(self, velocities)
        joint_indices = coerce_articulation_data(self, joint_indices, dtype="int64")
        return original_set_joint_velocities(self, velocities, joint_indices)

    def set_joint_efforts(self, efforts, joint_indices=None):
        efforts = coerce_articulation_data(self, efforts)
        joint_indices = coerce_articulation_data(self, joint_indices, dtype="int64")
        return original_set_joint_efforts(self, efforts, joint_indices)

    def apply_action(self, control_actions):
        coerce_articulation_action(self._articulation_view, control_actions)
        result = original_apply_action(self, control_actions)
        try:
            if (
                control_actions.joint_positions is not None
                and control_actions.joint_indices is not None
                and len(control_actions.joint_positions) == 1
                and len(control_actions.joint_indices) == 1
            ):
                original_set_joint_positions(self, control_actions.joint_positions, control_actions.joint_indices)
        except Exception as exc:
            carb.log_warn(f"[grasp_editor] Failed to mirror single-joint action to joint position: {exc}")
        return result

    def controller_apply_action(self, control_actions):
        coerce_articulation_action(self._articulation_view, control_actions)
        return original_controller_apply_action(self, control_actions)

    def controller_set_max_efforts(self, values, joint_indices=None):
        joint_indices = coerce_backend_data(
            self._articulation_view._backend_utils,
            self._articulation_view._device,
            joint_indices,
            dtype="int64",
        )
        return original_controller_set_max_efforts(self, values, joint_indices)

    def rigid_set_velocities(self, velocities, indices=None):
        velocities = coerce_backend_data(self._backend_utils, self._device, velocities)
        indices = coerce_backend_data(self._backend_utils, self._device, indices, dtype="int64")
        return original_rigid_set_velocities(self, velocities, indices)

    def rigid_apply_forces_and_torques_at_pos(
        self, forces=None, torques=None, positions=None, indices=None, is_global=True
    ):
        forces = coerce_backend_data(self._backend_utils, self._device, forces)
        torques = coerce_backend_data(self._backend_utils, self._device, torques)
        positions = coerce_backend_data(self._backend_utils, self._device, positions)
        indices = coerce_backend_data(self._backend_utils, self._device, indices, dtype="int64")
        return original_rigid_apply_forces_and_torques_at_pos(
            self, forces=forces, torques=torques, positions=positions, indices=indices, is_global=is_global
        )

    def subset_get_joint_positions(self):
        return to_numpy(original_subset_get_joint_positions(self))

    def subset_get_joint_velocities(self):
        return to_numpy(original_subset_get_joint_velocities(self))

    def subset_get_joint_efforts(self):
        return to_numpy(original_subset_get_joint_efforts(self))

    def convert_prim_to_rigid_body_without_extra_collider(prim_path: str, articulation_paths: list[str]):
        stage = omni.usd.get_context().get_stage()
        prim_to_convert = stage.GetPrimAtPath(prim_path)
        for art_path in articulation_paths:
            if prim_path[: len(art_path)] == art_path:
                return "Cannot convert a part of an Articulation to Rigid Body"
        if not prim_to_convert.IsValid():
            return f"No prim can be found at path {prim_path}"
        for prim in Usd.PrimRange(prim_to_convert):
            path = str(prim.GetPath())
            if path == prim_path:
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return (
                    "One or more prims nested under the selected path are already Rigid Bodies.  "
                    "Select one of these nested rigid body prims instead."
                )
        has_collision = any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(prim_to_convert))
        if not has_collision:
            carb.log_warn(
                f"[grasp_editor] {prim_path} has no authored CollisionAPI. "
                "Skipping Grasp Editor's automatic collider generation by request."
            )
        rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim_to_convert)
        rb_api.GetDisableGravityAttr().Set(True)
        UsdPhysics.RigidBodyAPI.Apply(prim_to_convert)
        return None

    articulation_initialize._autosim_grasp_patch = True
    return {
        "articulation_constructor": articulation_constructor,
        "articulation_initialize": articulation_initialize,
        "apply_action": apply_action,
        "controller_apply_action": controller_apply_action,
        "controller_set_max_efforts": controller_set_max_efforts,
        "convert_prim_to_rigid_body": convert_prim_to_rigid_body_without_extra_collider,
        "rigid_apply_forces_and_torques_at_pos": rigid_apply_forces_and_torques_at_pos,
        "rigid_initialize": rigid_initialize,
        "rigid_set_velocities": rigid_set_velocities,
        "set_joint_efforts": set_joint_efforts,
        "set_joint_positions": set_joint_positions,
        "set_joint_velocities": set_joint_velocities,
        "subset_get_joint_efforts": subset_get_joint_efforts,
        "subset_get_joint_positions": subset_get_joint_positions,
        "subset_get_joint_velocities": subset_get_joint_velocities,
    }


def _make_ui_patches(gripper_profile: GripperProfile, originals):
    original_build_selection_frame = originals["build_selection_frame"]
    original_build_reference_frame = originals["build_reference_frame"]
    original_finalize_reference_frame_selection = originals["finalize_reference_frame_selection"]
    original_on_stage_event = originals["on_stage_event"]
    original_populate_settings_frame = originals["populate_settings_frame"]

    def activate_profile_gripper_dofs(self) -> None:
        if self._joint_settings_ui_state is None:
            return
        for dof_name in self._articulation.dof_names:
            open_position, close_position = gripper_profile.dof_defaults.get(dof_name, (None, None))
            self._joint_settings_ui_state.set_active_dof(
                self._articulation,
                dof_name,
                open_position=open_position,
                close_position=close_position,
                max_effort=gripper_profile.max_effort if dof_name in gripper_profile.dof_defaults else None,
            )
            if dof_name in gripper_profile.dof_defaults:
                dof_index = self._articulation.get_dof_index(dof_name)
                self._articulation.get_articulation_controller().set_max_efforts(
                    [gripper_profile.max_effort], [dof_index]
                )
        if hasattr(self, "_robot_joint_frames"):
            for joint_frame in self._robot_joint_frames:
                joint_frame.rebuild()
        self._test_frame.rebuild()

    def stop_rigid_body(self):
        rigid_body = self._rigid_body
        zeros6 = rigid_body._backend_utils.create_zeros_tensor([1, 6], device=rigid_body._device, dtype="float32")
        zeros3 = rigid_body._backend_utils.create_zeros_tensor([1, 3], device=rigid_body._device, dtype="float32")
        try:
            rigid_body.set_velocities(zeros6)
            rigid_body.apply_forces_and_torques_at_pos(zeros3, zeros3)
        except Exception as exc:
            carb.log_warn(f"[grasp_editor] Failed to stop rigid body cleanly: {exc}")

    def build_selection_frame(self):
        global _LAST_GRASP_EDITOR_UI_BUILDER
        result = original_build_selection_frame(self)
        _LAST_GRASP_EDITOR_UI_BUILDER = self
        _apply_pending_grasp_editor_selection(self)
        return result

    def build_reference_frame(self):
        result = original_build_reference_frame(self)
        if self._articulation is None or self._rigid_body is None:
            return result
        gripper_frame = f"{self._articulation.prim_path}/{gripper_profile.base_link_name}"
        if hasattr(self, "_gripper_subframe") and gripper_frame in self._gripper_subframe.get_items():
            self._gripper_subframe.set_selection(gripper_frame)
        rigid_body_frame = self._rigid_body.prim_paths[0]
        if hasattr(self, "_rb_subframe") and rigid_body_frame in self._rb_subframe.get_items():
            self._rb_subframe.set_selection(rigid_body_frame)
        if hasattr(self, "_finalize_frame_btn"):
            self._finalize_frame_btn.enabled = True
        return result

    def finalize_reference_frame_selection(self):
        result = original_finalize_reference_frame_selection(self)
        activate_profile_gripper_dofs(self)
        return result

    def populate_settings_frame(self):
        result = original_populate_settings_frame(self)
        activate_profile_gripper_dofs(self)
        return result

    def on_stage_event(self, event):
        if event.type == int(omni.usd.StageEventType.SIMULATION_STOP_PLAY):
            return None
        return original_on_stage_event(self, event)

    return {
        "build_selection_frame": build_selection_frame,
        "build_reference_frame": build_reference_frame,
        "finalize_reference_frame_selection": finalize_reference_frame_selection,
        "on_stage_event": on_stage_event if original_on_stage_event is not None else None,
        "populate_settings_frame": populate_settings_frame,
        "stop_rigid_body": stop_rigid_body,
    }


def _joint_body_targets(joint_prim) -> tuple[list[Sdf.Path], list[Sdf.Path]]:
    joint = UsdPhysics.Joint(joint_prim)
    return joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()


def _is_joint_prim(prim) -> bool:
    type_name = prim.GetTypeName()
    return (
        prim.IsA(UsdPhysics.Joint)
        or prim.HasAPI(PhysxSchema.PhysxJointAPI)
        or (type_name.startswith("Physics") and type_name.endswith("Joint"))
    )


def _is_descendant_or_self(path: Sdf.Path, root_path: Sdf.Path) -> bool:
    return path == root_path or path.HasPrefix(root_path)


def _relative_to_any_link(path: Sdf.Path, link_paths: list[Sdf.Path]) -> tuple[Sdf.Path, Sdf.Path] | None:
    for link_path in link_paths:
        if _is_descendant_or_self(path, link_path):
            return link_path, path.MakeRelativePath(link_path)
    return None


def _dst_path_for_src(src_path: Sdf.Path, src_link_paths: list[Sdf.Path], dst_root_path: Sdf.Path) -> Sdf.Path:
    match = _relative_to_any_link(src_path, src_link_paths)
    if match is None:
        raise RuntimeError(f"Path is outside extracted gripper links: {src_path}")
    src_link_path, rel_path = match
    dst_link_path = dst_root_path.AppendChild(src_link_path.name)
    return dst_link_path if str(rel_path) == "." else dst_link_path.AppendPath(str(rel_path))


def _set_local_transform_matrix(prim, matrix: Gf.Matrix4d) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xform_op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    xform_op.Set(matrix)


def _apply_articulation_root(prim) -> None:
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    physx_articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
    physx_articulation_api.GetArticulationEnabledAttr().Set(True)
    physx_articulation_api.GetEnabledSelfCollisionsAttr().Set(False)


def _remove_articulation_root(prim) -> None:
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
        prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)


def _targets_are_inside_links(targets: list[Sdf.Path], link_paths: list[Sdf.Path]) -> bool:
    return bool(targets) and all(_relative_to_any_link(target, link_paths) is not None for target in targets)


def _copy_joint_with_retargeted_bodies(
    *,
    src_stage,
    dst_stage,
    src_joint_path: Sdf.Path,
    dst_joint_path: Sdf.Path,
    src_link_paths: list[Sdf.Path],
    dst_root_path: Sdf.Path,
) -> None:
    Sdf.CopySpec(src_stage.GetRootLayer(), src_joint_path, dst_stage.GetRootLayer(), dst_joint_path)

    dst_joint = UsdPhysics.Joint(dst_stage.GetPrimAtPath(dst_joint_path))
    src_joint = UsdPhysics.Joint(src_stage.GetPrimAtPath(src_joint_path))
    for src_rel_getter, dst_rel_getter in (
        (src_joint.GetBody0Rel, dst_joint.GetBody0Rel),
        (src_joint.GetBody1Rel, dst_joint.GetBody1Rel),
    ):
        retargeted = [_dst_path_for_src(path, src_link_paths, dst_root_path) for path in src_rel_getter().GetTargets()]
        rel = dst_rel_getter()
        rel.ClearTargets(False)
        for path in retargeted:
            rel.AddTarget(path)

    dst_joint.GetExcludeFromArticulationAttr().Set(False)
    dst_joint.GetJointEnabledAttr().Set(True)


def _copy_link_subtree_without_selected_children(
    *,
    flat_stage,
    dst_stage,
    src_link_path: Sdf.Path,
    src_link_paths: list[Sdf.Path],
    dst_link_path: Sdf.Path,
) -> None:
    flat_layer = flat_stage.GetRootLayer()
    dst_layer = dst_stage.GetRootLayer()
    for prim in Usd.PrimRange(flat_stage.GetPrimAtPath(src_link_path)):
        if _is_joint_prim(prim):
            continue
        if any(other != src_link_path and _is_descendant_or_self(prim.GetPath(), other) for other in src_link_paths):
            continue
        rel_path = prim.GetPath().MakeRelativePath(src_link_path)
        dst_path = dst_link_path if str(rel_path) == "." else dst_link_path.AppendPath(str(rel_path))
        Sdf.CopySpec(flat_layer, prim.GetPath(), dst_layer, dst_path)
    _remove_copied_excluded_link_children(
        dst_stage=dst_stage,
        src_link_path=src_link_path,
        src_link_paths=src_link_paths,
        dst_link_path=dst_link_path,
    )
    _remove_copied_joint_prims(dst_stage, dst_link_path)


def _remove_copied_excluded_link_children(
    *,
    dst_stage,
    src_link_path: Sdf.Path,
    src_link_paths: list[Sdf.Path],
    dst_link_path: Sdf.Path,
) -> None:
    for other in src_link_paths:
        if other == src_link_path or not _is_descendant_or_self(other, src_link_path):
            continue
        rel_path = other.MakeRelativePath(src_link_path)
        dst_stage.RemovePrim(dst_link_path.AppendPath(str(rel_path)))


def _remove_copied_joint_prims(dst_stage, root_path: Sdf.Path) -> None:
    root_prim = dst_stage.GetPrimAtPath(root_path)
    joint_paths = [
        prim.GetPath() for prim in Usd.PrimRange(root_prim) if prim.GetPath() != root_path and _is_joint_prim(prim)
    ]
    for joint_path in sorted(joint_paths, key=lambda path: len(str(path)), reverse=True):
        dst_stage.RemovePrim(joint_path)


def _reference_list_items(reference_list_op) -> list[Sdf.Reference]:
    if reference_list_op is None:
        return []

    items: list[Sdf.Reference] = []
    for attr_name in ("explicitItems", "prependedItems", "appendedItems", "addedItems"):
        items.extend(getattr(reference_list_op, attr_name, []))
    return items


def _flattened_prototype_reference_targets(prim) -> list[Sdf.Path]:
    targets: list[Sdf.Path] = []
    for reference in _reference_list_items(prim.GetMetadata("references")):
        prim_path = reference.primPath
        if reference.assetPath or not prim_path or str(prim_path) == ".":
            continue
        if str(prim_path).startswith("/Flattened_Prototype_"):
            targets.append(prim_path)
    return targets


def _add_internal_reference(references, prim_path: Sdf.Path) -> None:
    if hasattr(references, "AddInternalReference"):
        references.AddInternalReference(prim_path)
    else:
        references.AddReference("", prim_path)


def _retarget_flattened_prototype_references(prim, prototype_path_map: dict[Sdf.Path, Sdf.Path]) -> None:
    reference_list_op = prim.GetMetadata("references")
    references = _reference_list_items(reference_list_op)
    if not references:
        return

    changed = False
    prim_references = prim.GetReferences()
    prim_references.ClearReferences()
    for reference in references:
        target_path = prototype_path_map.get(reference.primPath)
        if target_path is not None and not reference.assetPath:
            _add_internal_reference(prim_references, target_path)
            changed = True
        elif reference.assetPath:
            prim_references.AddReference(reference.assetPath, reference.primPath)
        else:
            _add_internal_reference(prim_references, reference.primPath)

    if changed:
        prim.SetInstanceable(False)


def _copy_flattened_prototypes_into_gripper(flat_stage, dst_stage, dst_root_path: Sdf.Path) -> None:
    prototype_targets: set[Sdf.Path] = set()
    for prim in Usd.PrimRange(dst_stage.GetPrimAtPath(dst_root_path)):
        prototype_targets.update(_flattened_prototype_reference_targets(prim))

    if not prototype_targets:
        return

    prototypes_root_path = dst_root_path.AppendChild("_Prototypes")
    dst_stage.DefinePrim(prototypes_root_path, "Scope")

    flat_layer = flat_stage.GetRootLayer()
    dst_layer = dst_stage.GetRootLayer()
    prototype_path_map: dict[Sdf.Path, Sdf.Path] = {}

    pending = set(prototype_targets)
    while pending:
        src_prototype_path = sorted(pending, key=str)[0]
        pending.remove(src_prototype_path)
        if src_prototype_path in prototype_path_map:
            continue
        if not flat_stage.GetPrimAtPath(src_prototype_path).IsValid():
            carb.log_warn(f"[grasp_editor] Missing flattened prototype: {src_prototype_path}")
            continue
        dst_prototype_path = prototypes_root_path.AppendChild(src_prototype_path.name)
        Sdf.CopySpec(flat_layer, src_prototype_path, dst_layer, dst_prototype_path)
        prototype_path_map[src_prototype_path] = dst_prototype_path
        for prim in Usd.PrimRange(dst_stage.GetPrimAtPath(dst_prototype_path)):
            pending.update(_flattened_prototype_reference_targets(prim))

    for prim in Usd.PrimRange(dst_stage.GetPrimAtPath(dst_root_path)):
        _retarget_flattened_prototype_references(prim, prototype_path_map)

    print(f"[grasp_editor] Copied flattened prototypes: {len(prototype_path_map)}")


def _clear_external_material_bindings(stage, root_path: Sdf.Path) -> None:
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        relationship = prim.GetRelationship("material:binding")
        if not relationship:
            continue
        targets = relationship.GetTargets()
        if any(not _is_descendant_or_self(target, root_path) for target in targets):
            prim.RemoveProperty("material:binding")


def _initialize_gripper_physics_state(stage, root_path: Sdf.Path, gripper_profile: GripperProfile) -> None:
    zero_vec = Gf.Vec3f(0.0, 0.0, 0.0)
    for link_name in gripper_profile.link_names:
        link_prim = stage.GetPrimAtPath(root_path.AppendChild(link_name))
        for attr_name in ("physics:velocity", "physics:angularVelocity"):
            attr = link_prim.GetAttribute(attr_name)
            if attr:
                attr.Set(zero_vec)

    joints_root_path = root_path.AppendChild("joints")
    for dof_name, (open_position, _) in gripper_profile.dof_defaults.items():
        joint_prim = stage.GetPrimAtPath(joints_root_path.AppendChild(dof_name))
        for prefix in ("linear", "angular"):
            for attr_name, value in (
                (f"state:{prefix}:physics:position", open_position),
                (f"state:{prefix}:physics:velocity", 0.0),
                (f"drive:{prefix}:physics:maxForce", gripper_profile.max_effort),
                (f"drive:{prefix}:physics:targetPosition", open_position),
            ):
                attr = joint_prim.GetAttribute(attr_name)
                if attr:
                    attr.Set(float(value))
