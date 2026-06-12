"""Visualize a cuRobo robot configuration in a browser.

Loads an existing ``.yml`` or ``.xrdf`` robot config and opens an interactive
Viser viewer so you can inspect collision spheres and link meshes.

Usage
-----
Run directly (no Isaac Sim required):

    python examples/visualization/curobo_robot_model.py --config franka.yml

    # Also overlay the original collision meshes
    python examples/visualization/curobo_robot_model.py --config franka.yml --show_meshes

    # Use an absolute path to a custom config
    python examples/visualization/curobo_robot_model.py --config /path/to/my_robot.yml

Open ``http://localhost:8080`` in a browser. Press Ctrl-C to stop.

Notes
-----
* ``--config`` accepts a filename relative to the cuRobo robot-configs directory
  (e.g. ``franka.yml``), an absolute path, or a path relative to the CWD.
* ``--show_meshes`` overlays the original collision meshes in semi-transparent grey.
* Sphere counts per link are printed to stdout on load.
"""

from __future__ import annotations

import argparse
import os

from curobo._src.robot.loader.util import load_robot_yaml
from curobo._src.util_file import join_path, load_yaml
from curobo.content import get_assets_path, get_robot_configs_path
from curobo.logging import setup_logger
from curobo.robot_builder import RobotBuilder
from curobo.types import ContentPath

parser = argparse.ArgumentParser(description="Visualize a cuRobo robot configuration in a browser.")
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help=(
        "Robot config file (.yml or .xrdf). Accepts a filename relative to the cuRobo "
        "robot-configs directory (e.g. 'franka.yml'), an absolute path, or a CWD-relative path."
    ),
)
parser.add_argument(
    "--config_path",
    type=str,
    default=None,
    help="Path to the robot config directory. If not provided, the default cuRobo config path will be used.",
)
parser.add_argument(
    "--asset_path",
    type=str,
    default=None,
    help="Path to the robot asset directory. If not provided, the default cuRobo asset path will be used.",
)
parser.add_argument(
    "--show_meshes",
    action="store_true",
    help="Overlay the original collision meshes (semi-transparent) on top of spheres.",
)
parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Viser server port (default: 8080).",
)
parser.add_argument(
    "--log_level",
    type=str,
    default="warning",
    choices=["debug", "info", "warning", "error"],
    help="Logging level (default: warning).",
)
args_cli = parser.parse_args()


def _build_robot_builder(config: str, config_path: str, asset_path: str) -> RobotBuilder:
    """Build a robot builder from a config file."""

    curobo_config_path = config_path or get_robot_configs_path()
    curobo_asset_path = asset_path or get_assets_path()

    content_path = ContentPath(
        robot_config_root_path=curobo_config_path,
        robot_urdf_root_path=curobo_asset_path,
        robot_asset_root_path=curobo_asset_path,
        robot_config_file=config,
    )

    # Load YAML config
    config_data = load_robot_yaml(content_path)
    config_data = config_data.get("robot_cfg")
    kinematics_data = config_data["kinematics"]

    # Create instance from existing config
    urdf_path = join_path(curobo_asset_path, kinematics_data["urdf_path"])
    # Use the URDF's parent directory as mesh_root so that relative paths like
    # "../meshes/link.STL" in the URDF resolve correctly (yourdfpy joins mesh_root
    # with the relative path verbatim, so mesh_root must be the URDF's directory).
    urdf_dir = os.path.dirname(os.path.realpath(urdf_path))

    tool_frames = kinematics_data.get("tool_frames")
    instance = RobotBuilder(urdf_path=urdf_path, asset_path=urdf_dir, tool_frames=tool_frames)

    # Load existing collision data
    if "collision_spheres" in kinematics_data:
        if isinstance(kinematics_data["collision_spheres"], str):
            loaded = load_yaml(kinematics_data["collision_spheres"])
            # collision sphere YAML files wrap data under a "collision_spheres" key
            kinematics_data["collision_spheres"] = loaded.get("collision_spheres", loaded)
        instance._collision_spheres = kinematics_data["collision_spheres"].copy()

    if "self_collision_ignore" in kinematics_data:
        instance._self_collision_ignore = kinematics_data["self_collision_ignore"].copy()

    if "self_collision_buffer" in kinematics_data:
        instance._self_collision_buffer = kinematics_data["self_collision_buffer"].copy()

    if "cspace" in kinematics_data:
        instance._cspace_config = kinematics_data["cspace"].copy()

    return instance


def _print_summary(builder: RobotBuilder) -> None:
    """Print per-link sphere counts and total."""
    print(f"\nLoaded robot: {len(builder.collision_link_names)} collision links")

    if builder.collision_spheres:
        print(f"\n  {'Link':<30s}  Spheres")
        print(f"  {'-' * 40}")
        total = 0
        for link_name, spheres in builder.collision_spheres.items():
            n = len(spheres)
            total += n
            print(f"  {link_name:<30s}  {n:4d}")
        print(f"  {'-' * 40}")
        print(f"  {'TOTAL':<30s}  {total:4d}")
    else:
        print("  (no collision spheres found in config)")


def main():
    setup_logger(args_cli.log_level)

    builder = _build_robot_builder(args_cli.config, args_cli.config_path, args_cli.asset_path)
    _print_summary(builder)

    config = builder.build()

    try:
        builder.visualize(
            config=config,
            port=args_cli.port,
            show_meshes=args_cli.show_meshes,
            show_spheres=True,
            timeout_sec=-1,
        )
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
