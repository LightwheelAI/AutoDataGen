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
import sys
from pathlib import Path

from curobo.content import get_robot_configs_path
from curobo.logging import setup_logger
from curobo.robot_builder import RobotBuilder

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


def _resolve_config_path(config: str) -> str:
    """Return an absolute path to the robot config file.

    Search order: absolute path → cuRobo robot-configs directory → CWD.
    """
    p = Path(config)
    if p.is_absolute() and p.exists():
        return str(p)

    robot_cfg_path = Path(get_robot_configs_path()) / config
    if robot_cfg_path.exists():
        return str(robot_cfg_path)

    cwd_path = Path.cwd() / config
    if cwd_path.exists():
        return str(cwd_path)

    return config  # let RobotBuilder raise a clear FileNotFoundError


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

    config_path = _resolve_config_path(args_cli.config)
    print(f"Loading config: {config_path}")

    try:
        builder = RobotBuilder.from_config(config_path)
    except FileNotFoundError:
        print(
            f"Error: config file not found: {config_path}\n"
            f"Available built-in configs are in: {get_robot_configs_path()}",
            file=sys.stderr,
        )
        sys.exit(1)

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
