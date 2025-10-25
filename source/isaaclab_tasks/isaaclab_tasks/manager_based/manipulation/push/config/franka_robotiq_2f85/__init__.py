# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, robotiq2f85_joint_rel_env_cfg



gym.register(
    id="Blocks-Robotiq2f85-Push-Cube",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": robotiq2f85_joint_rel_env_cfg.FrankaRobotiq2f85CustomRelTrainCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Blocks-Robotiq2f85-Custom-Push-Cube",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": robotiq2f85_joint_rel_env_cfg.FrankaRobotiq2f85CustomRelTrainCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Blocks-Robotiq2f85-CustomOmni-Push-Cube",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": robotiq2f85_joint_rel_env_cfg.FrankaRobotiq2f85CustomOmniRelTrainCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)




gym.register(
    id="Blocks-Robotiq2f85-CustomOmni-Reorient",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": robotiq2f85_joint_rel_env_cfg.FrankaRobotiq2f85CustomOmniReorientEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Blocks-Robotiq2f85-CustomOmni-Push",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": robotiq2f85_joint_rel_env_cfg.FrankaRobotiq2f85CustomOmniPushEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)
