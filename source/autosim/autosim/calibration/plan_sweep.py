import math
import time
from dataclasses import dataclass, field
from typing import Any

import isaaclab.utils.math as PoseUtils
import torch

from autosim.core.pipeline import AutoSimPipeline
from autosim.core.registration import SkillRegistry
from autosim.skills.reach import ReachSkill
from autosim.utils.data_util import as_torch

from .pose_sampler import OffsetSampler, PoseSampler


@dataclass
class ReachPlanSweepCfg:
    reach_skill_index: int = 0
    """Which reach skill to sweep at (0-based, globally across all subtasks)."""
    sampling: PoseSampler = field(default_factory=OffsetSampler)
    """Sampler used to generate candidate poses around the base pose."""
    top_k: int = 10
    """Number of top poses to print."""
    ik_only: bool = False
    """If True, use IK-only solving instead of full motion planning. Much faster for reachability checking; does not produce trajectories."""
    num_object_rotations: int = 4
    """Number of uniformly sampled object yaw rotations in [0, 2π)."""


def _tensor_to_list(x: torch.Tensor) -> list[float]:
    """Convert a tensor to a flat Python float list for reporting/serialization."""
    return [float(v) for v in x.detach().cpu().flatten().tolist()]


def _fmt_pose(vals: list[float]) -> str:
    """Format a 7-value pose as a Python list literal, ready to copy-paste into code."""
    return "[" + ", ".join(f"{v:.4f}" for v in vals) + "]"


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions in xyzw format elementwise over the leading dimensions."""
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dim=-1,
    )


def _uniform_yaw_rotations(device: torch.device, dtype: torch.dtype, num_rotations: int) -> list[torch.Tensor]:
    """Generate uniformly spaced yaw-only quaternions over a full 360° rotation."""
    if num_rotations <= 0:
        return []
    rotations = []
    for idx in range(num_rotations):
        yaw = (2.0 * math.pi * idx) / num_rotations
        half = yaw * 0.5
        rotations.append(torch.tensor([0.0, 0.0, math.sin(half), math.cos(half)], device=device, dtype=dtype))
    return rotations


def _row_sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    """Rank sampled poses by success first, then shorter trajectories, then lower position error."""
    return (
        0.0 if row["plan_success"] else 1.0,
        float(row["traj_len"]) if row["traj_len"] is not None else 10**8,
        float(row["position_error"]) if row["position_error"] is not None else 10**8,
    )


def _build_extra_target_link_goals(
    reach_skill: ReachSkill,
    activate_q: torch.Tensor,
    target_poses_r: torch.Tensor,
    env_extra_info,
) -> dict[str, torch.Tensor] | None:
    """Build batched extra link goals by reusing ReachSkill logic."""
    if not reach_skill.cfg.extra_cfg.extra_target_link_names:
        return None

    goal_list = []
    for target_pose in target_poses_r:
        goal_list.append(reach_skill._build_extra_target_poses(activate_q, target_pose, env_extra_info))  # noqa: SLF001

    first_goal = goal_list[0]
    if first_goal is None:
        return None

    return {link_name: torch.stack([goal[link_name] for goal in goal_list], dim=0) for link_name in first_goal.keys()}


def reach_plan_sweep(pipeline: AutoSimPipeline, cfg: ReachPlanSweepCfg) -> list[dict[str, Any]]:
    """
    Execute the pipeline step by step. When the reach_skill_index-th reach skill
    is encountered, capture the live robot and object state and sweep around the
    reach target pose using cuRobo batch planning.

    All skills before the target reach skill are executed normally, so the
    environment reflects the actual state at the point of interest.

    The offline tuning flow uniformly samples object yaw rotations, applies each
    rotation to the target object, reuses the runtime reach candidate selection
    logic to choose a base pose, and then runs the local sweep around that pose.

    Returns:
        One result block per sampled object rotation.
    """

    pipeline.initialize()

    decompose_result = pipeline.decompose()
    pipeline._check_skill_extra_cfg()
    pipeline.reset_env()

    reach_skill_counter = 0

    for subtask in decompose_result.subtasks:
        for skill_info in subtask.skills:
            skill = SkillRegistry.create(
                skill_info.skill_type,
                pipeline.cfg.skills.get(skill_info.skill_type).extra_cfg,
            )

            if pipeline._action_adapter.should_skip_apply(skill):
                continue

            is_reach = skill_info.skill_type == "reach"

            if is_reach and reach_skill_counter == cfg.reach_skill_index:
                obj_name = skill_info.target_object
                return _sweep_all_rotations(pipeline, cfg, obj_name, skill)

            goal = skill.extract_goal_from_info(skill_info, pipeline._env, pipeline._env_extra_info)
            if is_reach:
                reach_skill_counter += 1

            success, _, episode_done = pipeline._execute_single_skill(skill, goal)
            if episode_done:
                raise ValueError(
                    f"Episode completed during skill '{skill_info.skill_type}' (step {skill_info.step}) before reaching"
                    f" target reach skill (index {cfg.reach_skill_index})."
                )
            if not success:
                raise ValueError(
                    f"Skill '{skill_info.skill_type}' (step {skill_info.step}) failed before reaching target reach"
                    f" skill (index {cfg.reach_skill_index})."
                )

    raise ValueError(
        f"reach_skill_index={cfg.reach_skill_index} is out of range: only {reach_skill_counter} reach skill(s) found in"
        " the decompose result."
    )


def _sweep_all_rotations(
    pipeline: AutoSimPipeline,
    cfg: ReachPlanSweepCfg,
    obj_name: str,
    reach_skill: ReachSkill,
) -> list[dict[str, Any]]:
    """Evaluate the target reach step across uniformly sampled object yaw rotations.

    For each sampled object rotation, this function temporarily updates the object's
    world pose, reuses the runtime reach candidate selector to choose a base pose,
    runs the local sweep around that base pose, and finally restores the original
    object pose.
    """
    env = pipeline._env
    env_extra_info = pipeline._env_extra_info
    obj = env.scene[obj_name]

    original_pose_w = as_torch(obj.data.root_pose_w)[pipeline._env_id].clone()
    base_quat_w = original_pose_w[3:].clone()
    env_ids = torch.tensor([pipeline._env_id], device=original_pose_w.device, dtype=torch.int32)
    results: list[dict[str, Any]] = []

    try:
        for rotation_idx, yaw_quat_w in enumerate(
            _uniform_yaw_rotations(original_pose_w.device, original_pose_w.dtype, cfg.num_object_rotations)
        ):
            rotated_pose_w = original_pose_w.clone().unsqueeze(0)
            rotated_pose_w[0, 3:] = _quat_mul(base_quat_w.unsqueeze(0), yaw_quat_w.unsqueeze(0)).squeeze(0)
            obj.write_root_pose_to_sim(rotated_pose_w, env_ids=env_ids)

            candidates = env_extra_info.get_reach_target_poses(obj_name)
            selected_pose_oe = reach_skill._select_best_candidate(  # noqa: SLF001
                env, obj_name, candidates, env_extra_info
            ).to(env.device)
            selected_idx = next(
                idx for idx, pose in enumerate(candidates) if torch.allclose(pose.to(env.device), selected_pose_oe)
            )

            result_block = _sweep(
                pipeline=pipeline,
                cfg=cfg,
                obj_name=obj_name,
                base_pose_oe=selected_pose_oe,
                reach_skill=reach_skill,
                rotation_idx=rotation_idx,
                object_pose_w=rotated_pose_w.squeeze(0),
                selected_candidate_idx=selected_idx,
            )
            results.append(result_block)
    finally:
        obj.write_root_pose_to_sim(original_pose_w.unsqueeze(0), env_ids=env_ids)

    _print_summary(obj_name, cfg, results)
    return results


def _print_summary(obj_name: str, cfg: ReachPlanSweepCfg, results: list[dict[str, Any]]) -> None:
    """Print the final per-rotation report and per-selected-candidate aggregate summary."""
    if not results:
        return

    _SEP = "═" * 100
    print()
    print(_SEP)
    print(f"  reach_plan_sweep summary  │  object='{obj_name}'  reach_skill_index={cfg.reach_skill_index}")
    print(_SEP)

    candidate_summary: dict[int, dict[str, Any]] = {}

    for block in results:
        rotation_idx = block["rotation_index"]
        success_count = block["success_count"]
        num_samples = block["num_samples"]
        elapsed_ms = block["elapsed_ms"]
        selected_candidate_idx = block["selected_candidate_idx"]
        selected_base_pose_oe = block["selected_base_pose_oe"]
        print(
            f"  rotation={rotation_idx:02d}  selected_candidate={selected_candidate_idx}  "
            f"success={success_count}/{num_samples} ({success_count / num_samples:.1%})  time={elapsed_ms:.0f} ms"
        )
        print(f"    base_pose={_fmt_pose(selected_base_pose_oe)}")
        for rank, row in enumerate(block["top_k"]):
            mark = "✓" if row["plan_success"] else "✗"
            metric = (
                f"traj_len={row['traj_len']}" if row["traj_len"] is not None else f"pos_err={row['position_error']:.4f}"
            )
            print(f"    [{rank}] {mark}  {_fmt_pose(row['pose_oe'])}  # {metric}")
        print("─" * 100)

        summary = candidate_summary.setdefault(
            selected_candidate_idx,
            {
                "count": 0,
                "total_success": 0,
                "total_samples": 0,
                "total_time_ms": 0.0,
                "base_pose": selected_base_pose_oe,
                "recommended_pose": None,
                "recommended_row": None,
            },
        )
        summary["count"] += 1
        summary["total_success"] += success_count
        summary["total_samples"] += num_samples
        summary["total_time_ms"] += elapsed_ms

        candidate_best_row = min(block["top_k"], key=_row_sort_key) if block["top_k"] else None
        if candidate_best_row is not None:
            if summary["recommended_row"] is None or _row_sort_key(candidate_best_row) < _row_sort_key(
                summary["recommended_row"]
            ):
                summary["recommended_row"] = candidate_best_row
                summary["recommended_pose"] = candidate_best_row["pose_oe"]

    print()
    print(_SEP)
    print("  selected_candidate aggregate")
    print(_SEP)
    for candidate_idx in sorted(candidate_summary):
        summary = candidate_summary[candidate_idx]
        total_samples = summary["total_samples"]
        success_rate = summary["total_success"] / total_samples if total_samples > 0 else 0.0
        avg_time_ms = summary["total_time_ms"] / summary["count"] if summary["count"] > 0 else 0.0
        print(
            f"  candidate={candidate_idx}  selected_in={summary['count']} rotation(s)  "
            f"success={summary['total_success']}/{total_samples} ({success_rate:.1%})  avg_time={avg_time_ms:.0f} ms"
        )
        print(f"    base_pose={_fmt_pose(summary['base_pose'])}")
        if summary["recommended_pose"] is not None:
            print(f"    recommended_pose={_fmt_pose(summary['recommended_pose'])}")
    print(_SEP)


def _sweep(
    pipeline: AutoSimPipeline,
    cfg: ReachPlanSweepCfg,
    obj_name: str,
    base_pose_oe: torch.Tensor,
    reach_skill: ReachSkill,
    rotation_idx: int,
    object_pose_w: torch.Tensor,
    selected_candidate_idx: int,
) -> dict[str, Any]:
    """
    Core sweep logic. Called once the environment is in the correct pre-reach state.

    Samples K candidate poses around the selected base reach target (object frame),
    transforms them to robot root frame, then batch-plans with cuRobo. When
    configured, extra link goals are generated from the live joint state using the
    same extra-target strategy as `ReachSkill.extract_goal_from_info()`.
    """

    env = pipeline._env
    env_id = pipeline._env_id
    env_extra_info = pipeline._env_extra_info
    planner = pipeline._motion_planner
    robot = pipeline._robot

    base_pose_oe = torch.as_tensor(base_pose_oe, device=env.device, dtype=torch.float32).view(7)
    poses_oe = cfg.sampling.sample(base_pose_oe)
    k = int(poses_oe.shape[0])

    obj_pose_w = as_torch(env.scene[obj_name].data.root_pose_w)[env_id]
    obj_pos_w = obj_pose_w[:3].view(1, 3).repeat(k, 1)
    obj_quat_w = obj_pose_w[3:].view(1, 4).repeat(k, 1)

    target_pos_w, target_quat_w = PoseUtils.combine_frame_transforms(
        obj_pos_w, obj_quat_w, poses_oe[:, :3], poses_oe[:, 3:]
    )
    robot_root_pose_w = as_torch(robot.data.root_pose_w)[env_id]
    rr_pos_w = robot_root_pose_w[:3].view(1, 3).repeat(k, 1)
    rr_quat_w = robot_root_pose_w[3:].view(1, 4).repeat(k, 1)
    target_pos_r, target_quat_r = PoseUtils.subtract_frame_transforms(rr_pos_w, rr_quat_w, target_pos_w, target_quat_w)

    state = pipeline._build_world_state()
    activate_q, activate_qd = reach_skill._build_activate_joint_state(  # noqa: SLF001
        state.sim_joint_names, state.robot_joint_pos, state.robot_joint_vel
    )
    if activate_qd is None:
        raise ValueError("activate_qd should not be None when sweep planning reach trajectories.")

    target_poses_r = torch.cat((target_pos_r, target_quat_r), dim=-1)
    link_goals = _build_extra_target_link_goals(reach_skill, activate_q, target_poses_r, env_extra_info)

    t0 = time.time()
    if cfg.ik_only:
        result = planner.solve_ik_batch(target_pos_r, target_quat_r, link_goals=link_goals)
    else:
        result = planner.plan_motion_batch(target_pos_r, target_quat_r, activate_q, activate_qd, link_goals=link_goals)
    dt_ms = (time.time() - t0) * 1000.0

    success = (
        result.success.detach().cpu().bool().reshape(-1)
        if result.success is not None
        else torch.zeros((k,), dtype=torch.bool)
    )
    pos_err = result.position_error.detach().cpu().reshape(-1) if result.position_error is not None else None
    traj_last = (
        torch.as_tensor(result.path_buffer_last_tstep).reshape(-1)
        if (not cfg.ik_only and result.path_buffer_last_tstep is not None)
        else None
    )

    rows = []
    for i in range(k):
        rows.append({
            "pose_oe": _tensor_to_list(poses_oe[i]),
            "plan_success": bool(success[i].item()),
            "traj_len": int(traj_last[i]) if traj_last is not None else None,
            "position_error": float(pos_err[i].item()) if pos_err is not None else None,
        })

    top_k = sorted(rows, key=_row_sort_key)[: cfg.top_k]
    return {
        "rotation_index": rotation_idx,
        "object_pose_w": _tensor_to_list(object_pose_w),
        "selected_candidate_idx": selected_candidate_idx,
        "selected_base_pose_oe": _tensor_to_list(base_pose_oe),
        "success_count": int(success.sum().item()),
        "num_samples": k,
        "elapsed_ms": dt_ms,
        "top_k": top_k,
    }
