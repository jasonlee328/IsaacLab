"""Check observation dimensions."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.manager_based.manipulation.push.config.franka_robotiq_2f85 import FrankaRobotiq2f85CustomOmniFlipEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

env_cfg = FrankaRobotiq2f85CustomOmniFlipEnvCfg()
env_cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg=env_cfg)
env.reset()
obs = env.observation_manager.compute_group("policy", env)
print(obs.shape[-1])
env.close()
simulation_app.close()
