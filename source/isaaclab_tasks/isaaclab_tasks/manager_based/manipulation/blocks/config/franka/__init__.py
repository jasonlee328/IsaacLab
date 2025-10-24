# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from isaaclab.utils import configclass

from . import agents, push_joint_pos_env_cfg

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RwdTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

import isaaclab.envs.mdp as isaaclab_mdp


##
# Register Gym environments
##

# Default/Base configuration
gym.register(
    id="Blocks-Push-Cube",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": push_joint_pos_env_cfg.FrankaPushCubeEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

# Easy configuration - smaller ranges, closer targets
gym.register(
    id="Blocks-Push-Cube-Easy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.blocks.config.franka:FrankaPushCubeEasyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

# Medium configuration - moderate ranges
gym.register(
    id="Blocks-Push-Cube-Medium",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.blocks.config.franka:FrankaPushCubeMediumEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

# Hard configuration - large ranges, can enable orientation
gym.register(
    id="Blocks-Push-Cube-Hard",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.blocks.config.franka:FrankaPushCubeHardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)


@configclass
class FrankaPushCubeEasyEnvCfg(push_joint_pos_env_cfg.FrankaPushCubeEnvCfg):
    """Easy configuration for push task with smaller ranges and closer targets."""
    
    def __post_init__(self):
        # Call parent first to set up base configuration
        super().__post_init__()
        
        # Cube spawn configuration
        cube_x_range = (0.60, 0.70)  # Forward distance from robot
        cube_y_range = (-0.05, 0.05)  # Lateral offset from robot center
        
        # Target offset ranges (relative to cube position)
        target_x_range = (-0.10, 0.10)  # Forward/backward from cube
        target_y_range = (-0.10, 0.10)  # Left/right from cube
        target_z_range = (0.0, 0.0)     # Same height as cube
        
        # Exclusion zone (targets can't spawn too close to cube)
        exclusion_x = (-0.03, 0.03)  # 3cm exclusion in x
        exclusion_y = (-0.03, 0.03)  # 3cm exclusion in y
        
        # Success thresholds
        position_threshold = 0.03  # 3cm for easy task
        orientation_threshold = 0.2
        
        # Position only flag
        position_only = True
        
        # Apply cube spawn configuration
        self.events.randomize_cube_position.params["pose_range"]["x"] = cube_x_range
        self.events.randomize_cube_position.params["pose_range"]["y"] = cube_y_range
        
        # Apply command configuration
        self.commands.ee_pose.position_only = position_only
        self.commands.ee_pose.success_position_threshold = position_threshold
        self.commands.ee_pose.success_orientation_threshold = orientation_threshold
        
        # Apply target ranges
        self.commands.ee_pose.ranges.pos_x = target_x_range
        self.commands.ee_pose.ranges.pos_y = target_y_range
        self.commands.ee_pose.ranges.pos_z = target_z_range
        
        # Apply exclusion ranges
        self.commands.ee_pose.exclusion_ranges.pos_x = exclusion_x
        self.commands.ee_pose.exclusion_ranges.pos_y = exclusion_y


@configclass
class FrankaPushCubeMediumEnvCfg(push_joint_pos_env_cfg.FrankaPushCubeEnvCfg):
    """Medium difficulty configuration with larger ranges."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Cube spawn configuration
        cube_x_range = (0.55, 0.65)
        cube_y_range = (-0.5, 0.5)
        
        # Target offset ranges
        target_x_range = (-0.10, 0.10)
        target_y_range = (-0.10, 0.10)
        target_z_range = (0.0, 0.0)
        
        # Exclusion zone
        exclusion_x = (-0.03, 0.03)
        exclusion_y = (-0.03, 0.03)
        
        # Success thresholds
        position_threshold = 0.01
        orientation_threshold = 0.2
        position_only = True
        
        # Apply configurations
        self.events.randomize_cube_position.params["pose_range"]["x"] = cube_x_range
        self.events.randomize_cube_position.params["pose_range"]["y"] = cube_y_range
        
        self.commands.ee_pose.position_only = position_only
        self.commands.ee_pose.success_position_threshold = position_threshold
        self.commands.ee_pose.success_orientation_threshold = orientation_threshold
        
        self.commands.ee_pose.ranges.pos_x = target_x_range
        self.commands.ee_pose.ranges.pos_y = target_y_range
        self.commands.ee_pose.ranges.pos_z = target_z_range
        
        self.commands.ee_pose.exclusion_ranges.pos_x = exclusion_x
        self.commands.ee_pose.exclusion_ranges.pos_y = exclusion_y


@configclass
class FrankaPushCubeHardEnvCfg(push_joint_pos_env_cfg.FrankaPushCubeEnvCfg):
    """Hard configuration with maximum ranges and orientation requirements."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Cube spawn configuration
        cube_x_range = (0.55, 0.65)
        cube_y_range = (-0.5, 0.5)
        
        # Target offset ranges
        target_x_range = (-0.10, 0.10)
        target_y_range = (-0.10, 0.10)
        target_z_range = (0.0, 0.0)
        
        # Exclusion zone
        exclusion_x = (-0.03, 0.03)
        exclusion_y = (-0.03, 0.03)
        
        # Success thresholds
        position_threshold = 0.01
        orientation_threshold = 0.01
        position_only = False
        
        # Apply configurations
        self.events.randomize_cube_position.params["pose_range"]["x"] = cube_x_range
        self.events.randomize_cube_position.params["pose_range"]["y"] = cube_y_range
        
        self.commands.ee_pose.position_only = position_only
        self.commands.ee_pose.success_position_threshold = position_threshold
        self.commands.ee_pose.success_orientation_threshold = orientation_threshold
        
        self.commands.ee_pose.ranges.pos_x = target_x_range
        self.commands.ee_pose.ranges.pos_y = target_y_range
        self.commands.ee_pose.ranges.pos_z = target_z_range
        
        self.commands.ee_pose.exclusion_ranges.pos_x = exclusion_x
        self.commands.ee_pose.exclusion_ranges.pos_y = exclusion_y

