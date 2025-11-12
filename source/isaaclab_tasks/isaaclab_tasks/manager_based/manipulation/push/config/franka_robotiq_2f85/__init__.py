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
    """Custom environment that positions distractors after command reset and tracks their initial poses."""
    
    def __init__(self, cfg, render_mode=None, **kwargs):
        """Initialize environment and setup distractor tracking."""
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize dictionary to store initial distractor poses (in world coordinates)
        # Format: {distractor_name: (initial_pos_w, initial_quat_w)}
        self.distractor_initial_poses_w = {}
    
    def _reset_idx(self, env_ids: Sequence[int]):
        """Override reset to position distractors AFTER command manager resets and store their initial poses."""
        # Call parent reset (includes command_manager.reset())
        super()._reset_idx(env_ids)
        
        # Now position distractors based on the freshly reset commands
        if hasattr(self.cfg, 'distractor_config'):
            push_mdp.position_distractors_adjacent_to_target(
                self, 
                env_ids,
                **self.cfg.distractor_config
            )
            
            # Store initial poses for all distractors after they've been positioned
            # This allows the reward function to check if distractors have been moved
            distractor_cfgs = self.cfg.distractor_config.get("distractor_1_cfg"), self.cfg.distractor_config.get("distractor_2_cfg")
            
            for distractor_cfg in distractor_cfgs:
                if distractor_cfg is None:
                    continue
                    
                distractor_name = distractor_cfg.name
                
                # Skip if distractor not in scene
                if distractor_name not in self.scene.keys():
                    continue
                
                # Get the distractor asset
                distractor_asset = self.scene[distractor_name]
                
                # Store or update initial poses for the reset environments
                if distractor_name not in self.distractor_initial_poses_w:
                    # First time - create tensors for all environments
                    self.distractor_initial_poses_w[distractor_name] = (
                        distractor_asset.data.root_pos_w[:, :3].clone(),
                        distractor_asset.data.root_quat_w.clone(),
                    )
                else:
                    # Update only the environments being reset
                    self.distractor_initial_poses_w[distractor_name][0][env_ids] = distractor_asset.data.root_pos_w[env_ids, :3].clone()
                    self.distractor_initial_poses_w[distractor_name][1][env_ids] = distractor_asset.data.root_quat_w[env_ids].clone()


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
