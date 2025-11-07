"""Launch Isaac Sim Simulator first."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL Agent.")
parser.add_argument(
    "--experiment_workflow",
    type=str,
    default="solo",
    choices=["solo", "orchestrated"],
    help="experiment type, solo, orchestrated.",
)

parser.add_argument(
    "--rl_framework",
    type=str,
    default="rslrl",
    choices=["skrl", "rslrl"],
    help="RL framework, skrl, rsl_rl, stable_baselines3, etc.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.workflows.rl_workflow import RlWorkflow, build_experiment_manager
from isaaclab_tasks.workflows.rl_framework_factory import RlFrameworkFactory


def main(args_cli: argparse.Namespace):
    exp_mgr = build_experiment_manager(args_cli)
    rl_framework = RlFrameworkFactory(args_cli.rl_framework).create()

    rl_workflow = RlWorkflow(
        simulation_app=simulation_app,
        algo_class=rl_framework,
        experiment_manager=exp_mgr,
    )

    args_cli = rl_workflow.add_args(parser)
    # now that experiment args are parsed, update experiment manager config
    exp_mgr.update_experiment_cfg(
        args_cli,
        rl_framework=args_cli.rl_framework,
        agent_cfg_entry_point="rsl_rl_cfg_entry_point",
    )
    rl_workflow.make_config(args_cli)
    rl_workflow.launch()


if __name__ == "__main__":
    main(args_cli)
    simulation_app.close()
    exit(0)
