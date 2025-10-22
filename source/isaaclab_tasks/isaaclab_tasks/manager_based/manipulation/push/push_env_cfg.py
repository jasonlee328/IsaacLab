# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from isaaclab_tasks.manager_based.manipulation.push import mdp as push_mdp
from isaaclab_tasks.manager_based.manipulation.push.mdp import observations as push_observations
from isaaclab_tasks.manager_based.manipulation.push.mdp import commands as push_commands
from . import mdp
from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp import commands as dex_cmd


##
# Scene definition
##
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
        # actions = ObsTerm(func=isaaclab_mdp.last_action)
        gripper_pos = ObsTerm(func=stack_mdp.gripper_pos) 
        
        ee_pos = ObsTerm(func=push_observations.ee_frame_pos_w)    # End-effector position
        ee_quat = ObsTerm(func=push_observations.ee_frame_quat_w)  # End-effector quaternion
        
        # Cube observations (relative to robot base)
        cube_pos = ObsTerm(func=push_observations.cube_pos_rel, params={"asset_cfg": SceneEntityCfg("cube")})
        cube_rot = ObsTerm(func=isaaclab_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube")})
        
        # Target observations (relative to robot base)
        target_pos = ObsTerm(func=push_observations.target_pos_rel, params={"command_name": "ee_pose"})
        target_rot = ObsTerm(func=push_observations.target_quat_rel, params={"command_name": "ee_pose"})
        

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    # @configclass
    # class PolicyCfg(ObsGroup):
    #     """Observations for policy group."""

    #     # Robot observations
        
    #     # prev_actions = ObsTerm(func=isaaclab_mdp.last_action)
    #     joint_pos = ObsTerm(func=isaaclab_mdp.joint_pos)
        
    #     # joint_pos = ObsTerm(func=isaaclab_mdp.joint_pos_rel)
    #     # joint_vel = ObsTerm(func=isaaclab_mdp.joint_vel_rel)

    #     gripper_pos = ObsTerm(func=stack_mdp.gripper_pos) 
        
    #     ee_pos = ObsTerm(func=push_observations.ee_frame_pos_rel)    # End-effector position
    #     ee_quat = ObsTerm(func=push_observations.ee_frame_quat_rel)  # End-effector quaternion
        
        
    #     # cube_pos_w = ObsTerm(func=push_observations.cube_pos_w, params={"asset_cfg": SceneEntityCfg("cube")})
    #     # cube_rot_w = ObsTerm(func=push_observations.cube_rot_w, params={"asset_cfg": SceneEntityCfg("cube")})
        
    #     cube_pos = ObsTerm(func=push_observations.cube_pos_rel, params={"asset_cfg": SceneEntityCfg("cube")})
    #     cube_rot = ObsTerm(func=isaaclab_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube")})
    #     # target_pos_w = ObsTerm(func=push_observations.target_pos_w, params={"command_name": "ee_pose"})
    #     # target_rot_w = ObsTerm(func=push_observations.target_quat_w, params={"command_name": "ee_pose"})


    #     # Target observations (relative to robot base)
    #     target_pos = ObsTerm(func=push_observations.target_pos_rel, params={"command_name": "ee_pose"})
    #     target_rot = ObsTerm(func=push_observations.target_quat_rel, params={"command_name": "ee_pose"})
        
        
    #     # Cube observations (relative to goal frame)
    #     # cube_pos_goal = ObsTerm(func=push_observations.cube_in_target_frame, params={"command_name": "ee_pose", "asset_cfg": SceneEntityCfg("cube")})
    #     # cube_pos = ObsTerm(func=push_observations.cube_pos_rel, params={"asset_cfg": SceneEntityCfg("cube")})
    #     # cube_rot = ObsTerm(func=isaaclab_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube")})
        
        
        
    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = True
    #         self.history_length = 1

    # @configclass
    # class CriticCfg(ObsGroup):
    #     """Critic observations for asymmetric actor-critic."""

    #     # Robot observations
    #     # prev_actions = ObsTerm(func=isaaclab_mdp.last_action)
    #     joint_pos = ObsTerm(func=isaaclab_mdp.joint_pos)

    #     gripper_pos = ObsTerm(func=stack_mdp.gripper_pos) 
        
    #     ee_pos = ObsTerm(func=push_observations.ee_frame_pos_rel)    # End-effector position
    #     ee_quat = ObsTerm(func=push_observations.ee_frame_quat_rel)  # End-effector quaternion
        
    #     # Cube observations (world frame)
    #     cube_pos_w = ObsTerm(func=push_observations.cube_pos_w, params={"asset_cfg": SceneEntityCfg("cube")})
    #     cube_rot_w = ObsTerm(func=push_observations.cube_rot_w, params={"asset_cfg": SceneEntityCfg("cube")})
        
    #     # Target observations (world frame)
    #     # target_pos_w = ObsTerm(func=push_observations.target_pos_w, params={"command_name": "ee_pose"})
    #     # target_rot_w = ObsTerm(func=push_observations.target_quat_w, params={"command_name": "ee_pose"})

    #     # Target observations (robot base frame)
    #     target_pos = ObsTerm(func=push_observations.target_pos_rel, params={"command_name": "ee_pose"})
    #     target_rot = ObsTerm(func=push_observations.target_quat_rel, params={"command_name": "ee_pose"})
        
    #     # Cube observations (relative to goal frame) - frame-invariant
    #     cube_pos_goal = ObsTerm(
    #         func=push_observations.cube_in_target_frame,
    #         params={"command_name": "ee_pose", "asset_cfg": SceneEntityCfg("cube")}
    #     )
        
    #     # Privileged observations
    #     # time_left = ObsTerm(func=isaaclab_mdp.remaining_time_s)
                
    #     joint_vel = ObsTerm(func=isaaclab_mdp.joint_vel)
        
    #     ee_velocity = ObsTerm(func=push_observations.ee_velocity_rel)
            
    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = True
    #         self.history_length = 1

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    # critic: CriticCfg = CriticCfg()


@configclass
class CommandsCfg:
    """Command terms for the MDP.
    
    Note: The threshold parameter controls all related distances and visualizations.
    - min_distance = threshold + 0.01
    - goal_pose_visualizer radius = threshold
    - curr_pose_visualizer radius = 0.0 (always)
    """

    # Target position for the cube to reach (relative offsets from cube position)
    ee_pose = dex_cmd.ObjectRelativePoseCommandCfg(
        asset_name="robot",  # Reference frame (robot base)
        object_name="cube",  # The object to generate commands for
        resampling_time_range=(10e9, 10e9),  # Never resample during episode
        debug_vis=True,  # Enable visualization of target and current positions
        position_only=True,  # Only generate position commands (no orientation)
        make_quat_unique=False,
        min_distance=0.02,  # Will be updated in __post_init__ to threshold + 0.01
        ranges=dex_cmd.ObjectRelativePoseCommandCfg.Ranges(
            # These are OFFSETS from cube's current position (not absolute positions!)
            pos_x=(-0.10, 0.10),  # Target 15-35cm forward from cube
            pos_y=(-0.10, 0.10),   # Target ±20cm lateral from cube
            pos_z=(0.0, 0.0),    # Same height as cube (no vertical offset)
            roll=(0, 0),
            pitch=(0, 0),
            yaw=(0.0, 0),
        ),
        goal_pose_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "position_far": sim_utils.SphereCfg(
                    radius=0.01,  # Will be updated in __post_init__ to threshold
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red
                ),
                "position_near": sim_utils.SphereCfg(
                    radius=0.01,  # Will be updated in __post_init__ to threshold
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # Green
                ),
            },
        ),
        curr_pose_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/Command/body_pose",
            markers={
                "position_far": sim_utils.SphereCfg(
                    radius=0.0,  # Always 0.0
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),  # Orange
                ),
                "position_near": sim_utils.SphereCfg(
                    radius=0.0,  # Always 0.0
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0)),  # Blue
                ),
            },
        ),
    )

@configclass
class RewardsCfg:
    """Reward specifications for the push MDP."""

    # # SPARSE REWARD ONLY - Success when cube reaches target
    # reaching_goal = RwdTerm(
    #     func=push_mdp.object_reached_goal,
    #     params={
    #         "object_cfg": SceneEntityCfg("cube"),
    #         "goal_cfg": "ee_pose",
    #         "threshold": 0.01,  # Will be synchronized with commands.threshold in __post_init__
    #     },
    #     weight=1.0,  # Sparse reward: +1 for success
    # )
    
    # Action penalties for smoother control
    # action_magnitude = RwdTerm(func=push_mdp.action_l2_clamped, weight=-1e-6)
    
    # action_rate = RwdTerm(func=push_mdp.action_rate_l2_clamped, weight=-1e-6)
    
    # joint_vel = RwdTerm(
    #     func=push_mdp.joint_vel_l2_clamped,
    #     weight=-1e-6,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    # )
    
    # # Safety reward
    # abnormal_robot = RwdTerm(func=push_mdp.abnormal_robot_state, weight=-1.0)
    



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


@configclass
class PushEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for cube pushing tasks."""

    # Central threshold parameter - tune this to adjust all related parameters
    threshold: float = 0.10

    # Scene settings
    scene: PushSceneCfg = PushSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    # Unused managers
    commands = CommandsCfg()
    curriculum = None
    events = None  # Will be set by derived classes

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10  # 10Hz control
        self.episode_length_s = 5.0  # 10 second episodes
        
        # simulation settings
        self.sim.dt =  0.01  # 100Hz
        self.sim.render_interval = self.decimation
        
        # simulation settings
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        # self.sim.physx.gpu_max_rigid_contact_count = 2**23
        # self.sim.physx.gpu_max_rigid_patch_count = 2**23
        # self.sim.physx.gpu_collision_stack_size = 2**31
        
        # render settings
        # self.sim.render.enable_dlssg = True
        # self.sim.render.enable_ambient_occlusion = True
        # self.sim.render.enable_reflections = True
        # self.sim.render.enable_dl_denoiser = True
        
        
        # Synchronize threshold between commands and rewards
        # Use self.threshold as the source of truth
        threshold = self.threshold
        
        
        # Update min_distance to threshold + 0.01
        self.commands.ee_pose.min_distance = threshold + 0.01
        
        # Update goal pose visualizer radius to threshold
        self.commands.ee_pose.goal_pose_visualizer_cfg.markers["position_far"].radius = threshold
        self.commands.ee_pose.goal_pose_visualizer_cfg.markers["position_near"].radius = threshold
        
        # Keep current pose visualizer radius at 0.0
        self.commands.ee_pose.curr_pose_visualizer_cfg.markers["position_far"].radius = 0.0
        self.commands.ee_pose.curr_pose_visualizer_cfg.markers["position_near"].radius = 0.0



 
