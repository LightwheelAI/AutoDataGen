"""Script to run autosim example."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="run autosim example pipeline.")
parser.add_argument(
    "--pipeline_id", type=str, default="AutoSimPipeline-FrankaCubeLift-v0", help="Name of the autosim pipeline."
)
parser.add_argument("--num_runs", type=int, default=10, help="Number of times to run the pipeline.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


import autosim_examples  # noqa: F401
from autosim import make_pipeline


def main():
    pipeline = make_pipeline(args_cli.pipeline_id)

    for i in range(args_cli.num_runs):
        print(f"====== run {i + 1} times =======")
        pipeline.run()


if __name__ == "__main__":
    main()
