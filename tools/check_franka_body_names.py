#!/usr/bin/env python3
"""Check what body names the Franka robot asset reports."""

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

# Import the task
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.reset_states.config.franka_robotiq_2f85 import reset_states_cfg

# Create the environment
env_cfg = reset_states_cfg.ObjectAnywhereEEAnywhereResetStatesCfg()
env_cfg.scene.num_envs = 2
env = gym.make("OmniReset-FrankaRobotiq2f85-ObjectAnywhereEEAnywhere-v0", cfg=env_cfg)

# Get the robot
robot = env.unwrapped.scene["robot"]

print(f"\n{'='*80}")
print(f"Franka Robot Body Names")
print(f"{'='*80}\n")
print(f"Total bodies: {len(robot.body_names)}")
print(f"\nBody names:")
for i, name in enumerate(robot.body_names):
    print(f"  {i}: {name}")
print(f"\n{'='*80}\n")

# Close
env.close()
app.close()

