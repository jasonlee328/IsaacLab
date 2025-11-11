# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from typing import Sequence

from isaaclab.envs import ManagerBasedRLEnv
from . import agents, robotiq2f85_joint_rel_env_cfg
from isaaclab_tasks.manager_based.manipulation.push import mdp as push_mdp


class ReorientWithDistractorsEnv(ManagerBasedRLEnv):
    """Custom environment that positions distractors after command reset."""
    
    def _reset_idx(self, env_ids: Sequence[int]):
        """Override reset to position distractors AFTER command manager resets."""
        # Call parent reset (includes command_manager.reset())
        super()._reset_idx(env_ids)
        
        # Now position distractors based on the freshly reset commands
        if hasattr(self.cfg, 'distractor_config'):
            push_mdp.position_distractors_adjacent_to_target(
                self, 
                env_ids,
                **self.cfg.distractor_config
            )


# gym.register(
#     id="Blocks-Robotiq2f85-Push-Cube",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": robotiq2f85_joint_rel_env_cfg.FrankaRobotiq2f85CustomRelTrainCfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Base_PPORunnerCfg",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Blocks-Robotiq2f85-Custom-Push-Cube",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": robotiq2f85_joint_rel_env_cfg.FrankaRobotiq2f85CustomRelTrainCfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Base_PPORunnerCfg",
#     },
#     disable_env_checker=True,
# )

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
    entry_point=f"{__name__}:ReorientWithDistractorsEnv",  # Use custom env for distractor positioning
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

gym.register(
    id="Blocks-Robotiq2f85-CustomOmni-Push-Distractor",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": robotiq2f85_joint_rel_env_cfg.FrankaRobotiq2f85CustomOmniPushDistractorEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)
