# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RwdTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Import MDP functions from isaaclab core and stack task
import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp
from isaaclab_tasks.manager_based.manipulation.blocks import mdp as blocks_mdp
from . import mdp
from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp import commands as dex_cmd



@configclass
class PushSceneCfg(InteractiveSceneCfg):
    """Configuration for the push scene with a robot, cube, and target.
    
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot, cube, target, and end-effector frames.
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # cube to push: will be populated by agent env cfg
    cube: AssetBaseCfg = MISSING
    # target location: will be populated by agent env cfg  
    # target: AssetBaseCfg = MISSING
    

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: stack_mdp.JointPositionActionCfg = MISSING
    gripper_action: stack_mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the push MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot observations
        joint_pos = ObsTerm(func=isaaclab_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=isaaclab_mdp.joint_vel_rel)
        # actions = ObsTerm(func=isaaclab_mdp.last_action, params={"action_name": "gripper_action"})  # Add 1D gripper action
        gripper_pos = ObsTerm(func=stack_mdp.gripper_pos) 
        
        ee_pos = ObsTerm(func=blocks_mdp.ee_frame_pos_w)    # End-effector position
        ee_quat = ObsTerm(func=blocks_mdp.ee_frame_quat_w)  # End-effector quaternion
        
        # Cube observations (relative to robot base)
        cube_pos = ObsTerm(func=blocks_mdp.cube_pos_rel, params={"asset_cfg": SceneEntityCfg("cube")})
        cube_rot = ObsTerm(func=isaaclab_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube")})
        
        # Target observations (relative to robot base)
        target_pos = ObsTerm(func=blocks_mdp.target_pos_rel, params={"command_name": "ee_pose"})
        target_rot = ObsTerm(func=blocks_mdp.target_rot_rel, params={"command_name": "ee_pose"})
        
        

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    """Command terms for the MDP.
    
    Note: The success_threshold will be automatically set in __post_init__ 
    based on the reward threshold to maintain consistency.
    """

    # Target position for the cube to reach using the new BlocksPoseCommand
    ee_pose = blocks_mdp.BlocksPoseCommandCfg(
        asset_name="robot",  # Reference frame (robot base)
        object_name="cube",  # The object to generate commands for
        resampling_time_range=(10e9, 10e9),  # Never resample during episode
        debug_vis=True,  # Enable visualization of target and current positions
        position_only=True,  # Only generate position commands (no orientation)
        make_quat_unique=False,
        success_position_threshold=0.1,  # 10cm threshold for success
        success_orientation_threshold=0.2,  # Not used when position_only=True
        ranges=blocks_mdp.BlocksPoseCommandCfg.Ranges(
            # These are OFFSETS from cube's current position (not absolute positions!)
            pos_x=(-0.30, 0.30),  # Target can be ±30cm from cube
            pos_y=(-0.30, 0.30),  # Target can be ±30cm from cube
            pos_z=(0.0, 0.0),     # Same height as cube (no vertical offset)
            roll=(0, 0),
            pitch=(0, 0),
            yaw=(0.0, 0),
        ),
        exclusion_ranges=blocks_mdp.BlocksPoseCommandCfg.ExclusionRanges(
            # Exclude targets too close to the cube (within ±10cm in x and y)
            pos_x=(-0.10, 0.10),  # Can't be within 10cm in x direction
            pos_y=(-0.10, 0.10),  # Can't be within 10cm in y direction
            pos_z=(0.0, 0.0),     # No z exclusion
            roll=(0, 0),
            pitch=(0, 0),
            yaw=(0, 0),
        ),
        # Optional: Enable per-axis slicing for curriculum learning
        slice_counts=blocks_mdp.BlocksPoseCommandCfg.SliceCounts(
            # pos_x=5,  # Would create 5 discrete x position options
            # pos_y=5,  # Would create 5 discrete y position options
            # pos_z=1,  # Keep z fixed (only 1 option)
            # roll=1, pitch=1, yaw=3,  # For orientation if position_only=False
        ),
    )

@configclass
class RewardsCfg:
    """Reward specifications for the push MDP."""
    # # safety rewards

    action_magnitude = RwdTerm(func=blocks_mdp.action_l2_clamped, weight=-1e-4)

    action_rate = RwdTerm(func=blocks_mdp.action_rate_l2_clamped, weight=-1e-4)

    joint_vel = RwdTerm(
        func=blocks_mdp.joint_vel_l2_clamped,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )

    abnormal_robot = RwdTerm(func=blocks_mdp.abnormal_robot_state, weight=-100.0)

    # Dense reward based on exponential decay of cube-to-target distance
    dense_success_reward = RwdTerm(
        func=blocks_mdp.dense_success_reward, 
        weight=0.1,
        params={
            "std": 0.1, 
            "command_name": "ee_pose",
            
        }
    )
    
    ee_asset_distance = RwdTerm(
        func=blocks_mdp.ee_asset_distance_tanh,
        weight=0.1,
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_asset_cfg": SceneEntityCfg("cube"),
            "std": 0.1,  
        },
    )
    
    # Sparse reward for reaching goal using command's success tracking
    success_reward = RwdTerm(
        func=blocks_mdp.success_reward,
        params={
            "command_name": "ee_pose",
        },
        weight=1.0,  
    )


@configclass
class TerminationsCfg:
    """Termination terms for the push MDP."""

    # Episode timeout
    time_out = DoneTerm(func=isaaclab_mdp.time_out, time_out=True)

    # Cube falling off table
    cube_falling = DoneTerm(
        func=isaaclab_mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube")}
    )
    
    abnormal_robot = DoneTerm(func=blocks_mdp.abnormal_robot_termination)


@configclass
class PushEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for cube pushing tasks."""

    # Scene settings
    scene: PushSceneCfg = PushSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    # Camera/Viewer settings for video recording
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.2, 0.0, 0.5),  # Camera position - right side, zoomed out
        lookat=(0.3, 0.0, 0.3),  # Camera target position (looking at robot/cube area)
        origin_type="env",  # Camera frame: "world", "env", "asset_root"
        resolution=(2560, 1440),  # Video resolution (width, height)
    )
    
    
    # Unused managers
    commands = CommandsCfg()
    curriculum = None
    events = None  # Will be set by derived classes
    
    # Task-specific settings
    position_only: bool = True  # Whether to only track position (not orientation) for success

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10  # 10Hz control
        self.episode_length_s = 10.0  # 10 second episodes
        
        # Update command settings based on task configuration
        self.commands.ee_pose.position_only = self.position_only
        
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation
        
        # physics settings
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        
        # self.sim.render.enable_dlssg = True
        # self.sim.render.enable_ambient_occlusion = True
        # self.sim.render.enable_reflections = True
        # self.sim.render.enable_dl_denoiser = True
