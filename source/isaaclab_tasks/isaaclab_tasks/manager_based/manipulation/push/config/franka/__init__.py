# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from isaaclab.utils import configclass

from . import agents, push_joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Default configuration
##
gym.register(
    id="Isaac-Push-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": push_joint_pos_env_cfg.FrankaPushCubeEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

##
# Easy variant: Large goal region (8cm), close target spawn (10cm)
##
@configclass
class FrankaPushCubeEasyEnvCfg(push_joint_pos_env_cfg.FrankaPushCubeEnvCfg):
    def __post_init__(self):
        # Set parameters BEFORE calling super so they're used in event configuration
        # self.target_spawn_radius = 0.20  # 50cm - target spawns closer
        # self.target_goal_radius = 0.08   # 8cm - bigger succwhess region
        
        super().__post_init__()
        
        # Update the visual target size and reward threshold
        # self.scene.target.spawn.radius = self.target_goal_radius
        # self.rewards.reaching_goal.params["threshold"] = self.target_goal_radius

gym.register(
    id="Isaac-Push-Cube-Franka-Easy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaPushCubeEasyEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushCubeEasyPPORunnerCfg",
    },
    disable_env_checker=True,
)

##
# Hard variant: Small goal region (3cm), far target spawn (25cm)
##
@configclass
class FrankaPushCubeHardEnvCfg(push_joint_pos_env_cfg.FrankaPushCubeEnvCfg):
    def __post_init__(self):
        # Set parameters BEFORE calling super so they're used in event configuration  
        self.target_spawn_radius = 0.25  # 25cm - target spawns farther
        self.target_goal_radius = 0.03   # 3cm - smaller success region
        
        super().__post_init__()
        
        # Update the visual target size and reward threshold
        self.scene.target.spawn.radius = self.target_goal_radius
        self.rewards.reaching_goal.params["threshold"] = self.target_goal_radius

gym.register(
    id="Isaac-Push-Cube-Franka-Hard-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaPushCubeHardEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushCubeHardPPORunnerCfg",
    },
    disable_env_checker=True,
)
