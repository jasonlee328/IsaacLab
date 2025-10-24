# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from isaaclab.utils import configclass

from . import agents, push_joint_pos_env_cfg
# Replace reward with distance_orientation_goal
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RwdTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab_tasks.manager_based.manipulation.push import mdp as push_mdp
from isaaclab_tasks.manager_based.manipulation.push.mdp import observations as push_observations
import isaaclab.envs.mdp as isaaclab_mdp

from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp
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
# Easy variant - Customize all parameters here
##
@configclass
class FrankaPushCubeEasyEnvCfg(push_joint_pos_env_cfg.FrankaPushCubeEnvCfg):
    def __post_init__(self):
        # ============================================================
        # CONFIGURABLE PARAMETERS - Edit these values as needed
        # ============================================================
        
        # Success threshold (how close cube needs to be to target)
        self.threshold = 0.03
        
        # Cube spawn position range (in meters, relative to robot base)
        cube_x_range = (0.45, 0.55)  # Forward distance from robot
        cube_y_range = (0.03, 0.07)  # Lateral offset from robot center
        
        # Target command range (in meters, RELATIVE to cube position)
        # These are OFFSETS - target will spawn at cube_pos + offset
        target_x_range = (-0.15, 0.15)  # Forward/backward from cube
        target_y_range = (-0.15, 0.15)  # Left/right from cube
        
        # ============================================================
        # Apply configuration (don't edit below unless you know what you're doing)
        # ============================================================
        
        # Set threshold before calling parent __post_init__
        super().__post_init__()
        
        # Update cube spawn ranges
        self.events.randomize_cube_position.params["pose_range"]["x"] = cube_x_range
        self.events.randomize_cube_position.params["pose_range"]["y"] = cube_y_range
        
        # Update target command ranges (relative to cube)
        self.commands.ee_pose.ranges.pos_x = target_x_range
        self.commands.ee_pose.ranges.pos_y = target_y_range
        
        # Re-sync command success threshold with reward threshold
        self.commands.ee_pose.success_threshold = self.threshold

gym.register(
    id="Isaac-Push-Cube-Franka-Easy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaPushCubeEasyEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushCubeEasyPPORunnerCfg",
    },
    disable_env_checker=True,
)


@configclass
class FrankaPushCubeHardEnvCfg(push_joint_pos_env_cfg.FrankaPushCubeEnvCfg):
    def __post_init__(self):
  
        self.threshold = 0.03  
        cube_x_range = (0.50, 0.60)  # Forward distance from robot
        cube_y_range = (0.03, 0.07)  # Lateral offset from robot center
        target_x_range = (-0.25, 0.25)  # Forward/backward from cube (harder: farther targets)
        target_y_range = (-0.25, 0.25)  # Left/right from cube
        

        super().__post_init__()
        
        self.events.randomize_cube_position.params["pose_range"]["x"] = cube_x_range
        self.events.randomize_cube_position.params["pose_range"]["y"] = cube_y_range
        self.commands.ee_pose.ranges.pos_x = target_x_range
        self.commands.ee_pose.ranges.pos_y = target_y_range
        self.commands.ee_pose.success_threshold = self.threshold

gym.register(
    id="Isaac-Push-Cube-Franka-Hard-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaPushCubeHardEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushCubeHardPPORunnerCfg",
    },
    disable_env_checker=True,
)


@configclass
class FrankaPushCubeCube5cmTarget20cmThreshold1cmEnvCfg(push_joint_pos_env_cfg.FrankaPushCubeEnvCfg):
    def __post_init__(self):
  
        self.threshold = 0.03
        cube_x_range = (0.70, 0.70)  # Forward distance from robot
        cube_y_range = (0.0, 0.0)  # Lateral offset from robot center
        target_x_range = (-0.20, 0.20)  # Forward/backward from cube (harder: farther targets)
        target_y_range = (-0.20, 0.20)  # Left/right from cube
        

        super().__post_init__()
        
        self.events.randomize_cube_position.params["pose_range"]["x"] = cube_x_range
        self.events.randomize_cube_position.params["pose_range"]["y"] = cube_y_range
        self.commands.ee_pose.ranges.pos_x = target_x_range
        self.commands.ee_pose.ranges.pos_y = target_y_range
        self.commands.ee_pose.success_threshold = self.threshold
        self.commands.ee_pose.min_distance = 0.30
        
        
        self.rewards.reaching_goal = RwdTerm(
                func=push_mdp.object_reached_goal,
                params={
                    "object_cfg": SceneEntityCfg("cube"),
                    "goal_cfg": "ee_pose",
                    "threshold": self.threshold,  # Will be synchronized with commands.threshold in __post_init__
                },
                weight=1.0,  # Sparse reward: +1 for success
            )
                

gym.register(
    id="Isaac-Push-Cube-Franka-cube5cm-target20cm-threshold30cm",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaPushCubeCube5cmTarget20cmThreshold1cmEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)



@configclass
class FrankaPushCubeCube5cmTarget20cmThreshold1cmReorientYaw1degEnvCfg(push_joint_pos_env_cfg.FrankaPushCubeEnvCfg):
    def __post_init__(self):
  
        self.threshold = 0.01  # Position threshold: 1cm
        self.orientation_threshold = 0.01746  # Orientation threshold: 1 degrees in radians
        cube_x_range = (0.55, 0.55)  # Forward distance from robot
        cube_y_range = (0.05, 0.05)  # Lateral offset from robot center
        target_x_range = (-0.10, 0.10)  # Forward/backward from cube (harder: farther targets)
        target_y_range = (-0.10, 0.10)  # Left/right from cube
        yaw_range = (-0.1745, 0.1745)  # ±10 degrees in radians
        

        super().__post_init__()
        
        # Set cube spawn position
        self.events.randomize_cube_position.params["pose_range"]["x"] = cube_x_range
        self.events.randomize_cube_position.params["pose_range"]["y"] = cube_y_range
        
        # Set command ranges (target offsets from cube)
        self.commands.ee_pose.ranges.pos_x = target_x_range
        self.commands.ee_pose.ranges.pos_y = target_y_range
        self.commands.ee_pose.ranges.yaw = yaw_range  # Sample yaw from -10 to +10 degrees
        self.commands.ee_pose.position_only = False  # Enable orientation commands
        self.commands.ee_pose.success_threshold = self.threshold
        
        
        # self.rewards.reaching_goal = RwdTerm(
        #     func=push_mdp.object_reached_goal,
        #     params={
        #         "object_cfg": SceneEntityCfg("cube"),
        #         "goal_cfg": "ee_pose",
        #         "threshold": self.threshold,  # 1cm position threshold
        #     },
        #     weight=1.0,  # Sparse reward: +1 for success (position)
        # )
        
        # self.rewards.orientation_goal = RwdTerm(
        #     func=push_mdp.orientation_goal,
        #     params={
        #         "object_cfg": SceneEntityCfg("cube"),
        #         "goal_cfg": "ee_pose",
        #         "orientation_threshold": self.orientation_threshold,  # 5 degree orientation threshold
        #     },
        #     weight=1.0,  # Sparse reward: +1 for success (orientation)
        # )
        
        self.rewards.distance_orientation_goal = RwdTerm(
            func=push_mdp.distance_orientation_goal,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "goal_cfg": "ee_pose",
                "distance_threshold": self.threshold,  # 1cm position threshold
                "orientation_threshold": self.orientation_threshold,  # 1 degree orientation threshold
            },
            weight=1.0,  # Sparse reward: +1 for success (both position and orientation)
        )
     
     
gym.register(
    id="Isaac-Push-Cube-Franka-cube5cm-target20cm-threshold1cm-reorient-yaw1deg",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaPushCubeCube5cmTarget20cmThreshold1cmReorientYaw1degEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)


##
# Custom Observation Configuration for Reorientation Task
##

@configclass
class ReorientObservationsCfg:
    """Custom observation specifications for the reorientation task.
    
    This includes yaw angle and orientation delta observations that are more
    suitable for learning rotation tasks compared to full quaternions.
    """
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group - optimized for reorientation."""
        
        # Robot observations
        # joint_pos = ObsTerm(func=isaaclab_mdp.joint_pos)
        
        joint_pos = ObsTerm(func=isaaclab_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=isaaclab_mdp.joint_vel_rel)
        # actions = ObsTerm(func=isaaclab_mdp.last_action)
        gripper_pos = ObsTerm(func=stack_mdp.gripper_pos) 
        
        
        ee_pos = ObsTerm(func=push_observations.ee_frame_pos_rel)
        ee_quat = ObsTerm(func=push_observations.ee_frame_quat_rel)
        
        # Cube observations
        cube_pos = ObsTerm(func=push_observations.cube_pos_rel, params={"asset_cfg": SceneEntityCfg("cube")})
        cube_yaw = ObsTerm(func=push_observations.cube_yaw_angle, params={"asset_cfg": SceneEntityCfg("cube")})
        
        # Target observations
        target_pos = ObsTerm(func=push_observations.target_pos_rel, params={"command_name": "ee_pose"})
        target_yaw = ObsTerm(func=push_observations.target_yaw_angle, params={"command_name": "ee_pose"})
        
        # Key observation: signed angular difference (most important for learning!)
        orientation_delta = ObsTerm(
            func=push_observations.orientation_delta,
            params={"asset_cfg": SceneEntityCfg("cube"), "command_name": "ee_pose"}
        )
        
        # Cube position relative to goal (frame-invariant)
        cube_pos_goal = ObsTerm(
            func=push_observations.cube_in_target_frame,
            params={"command_name": "ee_pose", "asset_cfg": SceneEntityCfg("cube")}
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    

    
    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaReorientYaw90degEnvCfg(push_joint_pos_env_cfg.FrankaPushCubeEnvCfg):
    def __post_init__(self):
  
        self.threshold = 0.03  # Position threshold: 1cm
        self.orientation_threshold = 0.052  # Orientation threshold: 1 degrees in radians
        cube_x_range = (0.70, 0.70)  # Forward distance from robot
        cube_y_range = (0.0, 0.0)  # Lateral offset from robot center
        target_x_range = (-0.04, 0.04)  # Forward/backward from cube (harder: farther targets)
        target_y_range = (-0.04, 0.04)  # Left/right from cube
        yaw_range = (-3.14, 3.14)  # ±90 degrees in radians
        
    

        super().__post_init__()
        
        # Override with custom observation configuration for reorientations
        self.observations = ReorientObservationsCfg()
        
        # Set cube spawn position
        self.events.randomize_cube_position.params["pose_range"]["x"] = cube_x_range
        self.events.randomize_cube_position.params["pose_range"]["y"] = cube_y_range
        
        # Set command ranges (target offsets from cube)
        self.commands.ee_pose.ranges.pos_x = target_x_range
        self.commands.ee_pose.ranges.pos_y = target_y_range
        self.commands.ee_pose.ranges.yaw = yaw_range  # Sample yaw from -90 to +90 degrees
        self.commands.ee_pose.position_only = False  # Enable orientation commands
        self.commands.ee_pose.success_threshold = self.threshold
        self.commands.ee_pose.min_distance = 0.07
        
        
        self.rewards.distance_orientation_goal = RwdTerm(
            func=push_mdp.distance_orientation_goal,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "goal_cfg": "ee_pose",
                "distance_threshold": self.threshold,  # 1cm position threshold
                "orientation_threshold": self.orientation_threshold,  # 1 degree orientation threshold
            },
            weight=1.0,  # Sparse reward: +1 for success (both position and orientation)
        )

gym.register(
    id="Isaac-reorient-yaw90deg",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaReorientYaw90degEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)
