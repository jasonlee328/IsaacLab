# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RwdTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from dataclasses import MISSING

from isaaclab_tasks.manager_based.manipulation.push.config.franka_robotiq_2f85.actions import (
    FrankaRobotiq2f85RelativeAction,
    FrankaRobotiq2f85CustomRelativeAction,
)
import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.push import mdp as push_mdp
from isaaclab_tasks.manager_based.manipulation.push.mdp import observations as push_observations  
from isaaclab_tasks.manager_based.manipulation.push.mdp import commands as push_commands
from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp import commands as dex_cmd

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import (
    FRANKA_ROBOTIQ_GRIPPER_CFG,
    FRANKA_ROBOTIQ_GRIPPER_CUSTOM_OMNI_CFG,
    FRANKA_ROBOTIQ_GRIPPER_CUSTOM_OMNI_PAT_CFG
)  # isort: skip
from . import actions



@configclass
class RlStateSceneCfg(InteractiveSceneCfg):

    robot: ArticulationCfg = FRANKA_ROBOTIQ_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/FrameTransformer", 
            markers={
                "frame": FRAME_MARKER_CFG.markers["frame"].replace(scale=(0.08, 0.08, 0.08)),
                "connecting_line": FRAME_MARKER_CFG.markers["connecting_line"]
            }
        ),
    )

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.4, 0.0, 0.0203],  # Closer to robot position before randomization
            rot=[1, 0, 0, 0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            semantic_tags=[("class", "cube")],
        ),
    )
    
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
    
    
    
@configclass
class BaseEventCfg:
    """Configuration for events (cloud variant with 13 joints)."""  
    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            # Robotiq gripper has 13 joints: 7 arm + 1 main drive + 2 inner finger + 2 inner knuckle + 1 outer knuckle
            "default_pose": [
                0.0, 0.8, 0.0, -1.1, 0.0, 2.1, 0.785,  # 7 arm joints
                0.0,  # finger_joint (main drive)
                0.0, 0.0,  # left_inner_finger_joint, right_inner_finger_joint
                0.0, 0.0,  # left_inner_finger_knuckle_joint, right_inner_finger_knuckle_joint
                0.0,  # right_outer_knuckle_joint
            ]
        },
    )

    randomize_cube_position = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.50, 0.60),  
                "y": (0.03, 0.07),  
                "z": (0.0203, 0.0203),  # Fixed height on table
                "yaw": (-0.5, 0.5)  # Small rotation randomization
            },
            "min_separation": 0.0,  # No separation needed for single cube
            "asset_cfgs": [SceneEntityCfg("cube")],
        },
    )
    
    
@configclass
class CustomEventCfg:
    """Configuration for events (custom pre-assembled USD with 13 joints)."""  
    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            # Custom Robotiq gripper has 13 joints total (see GitHub #1299)
            "default_pose": [
                0.0, 0.93, 0.0, -1.27, 0.0, 2.17, 0.0,  # 7 arm joints
                0.0, 0.0,  # right_outer_knuckle_joint, left_outer_knuckle_joint
                0.0, 0.0,  # right_inner_finger_joint, left_inner_finger_joint
                0.0, 0.0,  # RevoluteJoint, RevoluteJoint_0 (unnamed passive joints)
            ]
        #  "default_pose": [
        #         0.0, 0.50, 0.0, -2.03, 0.0, 2.6, 0.0,  # 7 arm joints
        #         0.0, 0.0,  # right_outer_knuckle_joint, left_outer_knuckle_joint
        #         0.0, 0.0,  # right_inner_finger_joint, left_inner_finger_joint
        #         0.0, 0.0,  # RevoluteJoint, RevoluteJoint_0 (unnamed passive joints)
        #     ]
        },
    )

    randomize_cube_position = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.50, 0.60),  
                "y": (0.03, 0.07),  
                "z": (0.0203, 0.0203),  # Fixed height on table
                "roll": (0.0, 0.0),  # No roll
                "pitch": (0.0, 0.0),  # No pitch
                "yaw": (-0.5, 0.5)  # Small rotation randomization
            },
            "min_separation": 0.0,  # No separation needed for single cube
            "asset_cfgs": [SceneEntityCfg("cube")],
        },
    )

    
@configclass
class CommandsCfg:
    """Command terms for the MDP.
    
    Uses 2D polar sampling for target generation:
    - min_radius: Minimum distance from cube to target
    - max_radius: Maximum distance from cube to target  
    - Samples uniformly in radius and angle (full 360°)
    """

    # Target position for the cube to reach (using polar sampling)
    ee_pose = push_commands.ObjectRelativePoseCommandCfg(
        asset_name="robot",  # Reference frame (robot base)
        object_name="cube",  # The object to generate commands for
        resampling_time_range=(10e9, 10e9),  # Never resample during episode
        debug_vis=True,  # Enable visualization of target and current positions
        position_only=True,  # Only generate position commands (no orientation)
        make_quat_unique=False,
        min_radius=0.05,  # Minimum push distance (e.g., cube radius)
        max_radius=0.10,  # Maximum push distance
        ranges=push_commands.ObjectRelativePoseCommandCfg.Ranges(
            # Orientation ranges (not used since position_only=True)
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        )
    )


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
        # gripper_pos = ObsTerm(func=stack_mdp.gripper_pos) 
        
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

    policy: PolicyCfg = PolicyCfg()
    
    
@configclass
class RewardsCfg:

    reaching_goal = RwdTerm(
        func=push_mdp.object_reached_goal,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "goal_cfg": "ee_pose",
            "threshold": 0.03,  # Will be synchronized with commands.threshold in __post_init__
        },
        weight=1.0,  
    )
    
    
@configclass
class TerminationsCfg:
    """Termination terms for the push MDP."""

    time_out = DoneTerm(func=isaaclab_mdp.time_out, time_out=True)

    cube_falling = DoneTerm(
        func=isaaclab_mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube")}
    )


@configclass
class FrankaRobotiq2f85RLStateCfg(ManagerBasedRLEnvCfg):
    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: FrankaRobotiq2f85RelativeAction = FrankaRobotiq2f85RelativeAction()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: BaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(1.8, 1.2, 1.2),lookat=(0.5, -0.2, 0.0),  origin_type="env", env_index=0, asset_name="robot")
    

    def __post_init__(self):
        self.decimation = 10  # 10Hz control
        self.episode_length_s = 5.0  # 10 second episodes
        self.sim.dt =  0.01 
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23


@configclass
class FrankaRobotiq2f85CustomOmniRelTrainCfg(FrankaRobotiq2f85RLStateCfg):
    """Configuration for Franka cube pushing task using Omniverse-style Robotiq gripper (per-joint tuning)."""
    
    def __post_init__(self):

        super().__post_init__()
        # Use custom configurations for Omniverse-style pre-assembled USD
        self.events = CustomEventCfg()  # 13-joint structure
        self.actions = FrankaRobotiq2f85CustomRelativeAction()  # outer_knuckle_joint actions
        self.scene.robot = FRANKA_ROBOTIQ_GRIPPER_CUSTOM_OMNI_PAT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)  # Reset to origin (table is at 0.5, 0, 0)

        
        self.scene.robot.spawn.activate_contact_sensors = True  # Enable contact sensing
        self.scene.robot.init_state.joint_pos = {
            "panda_joint1": 0.0,      # base rotation
            "panda_joint2": 0.8,     # shoulder forward
            "panda_joint3": 0.0,      # elbow rotation
            "panda_joint4": -1.1,     # elbow bend
            "panda_joint5": 0.0,      # wrist rotation
            "panda_joint6": 2.1,      # wrist bend
            "panda_joint7": 0.785,      # flange rotation (45 degrees)
            ".*_outer_knuckle_joint": 0.0,   # Main actuated joints - OPEN position
            ".*_inner_finger_joint": 0.0,    # Passive joints
        }
        
        # Robotiq gripper configuration (Omniverse-style uses outer_knuckle_joints)
        self.gripper_joint_names = ["left_outer_knuckle_joint", "right_outer_knuckle_joint"]
        self.gripper_open_val = 0.0  # Robotiq opens at 0.0
        self.gripper_threshold = 0.8  # Threshold for gripper state
        
        # Update FrameTransformer to use panda_link7 instead of panda_hand
        self.scene.ee_frame.target_frames = [
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_link7",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.2418], 
                ),
            ),
        ]
        
        
        
        
        
        
############# NUDGE TASK #############

   
@configclass
class NudgeObservationsCfg:
    """Custom observation specifications for the reorientation task.
    
    This includes yaw angle and orientation delta observations that are more
    suitable for learning rotation tasks compared to full quaternions.
    """
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group - optimized for reorientation."""
        
        # joint_pos = ObsTerm(func=isaaclab_mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=isaaclab_mdp.joint_vel_rel)
        joint_pos = ObsTerm(func=push_observations.arm_joint_pos_rel)
        joint_vel = ObsTerm(func=push_observations.arm_joint_vel_rel)
        gripper_pos = ObsTerm(func=push_observations.gripper_state_binary)
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
        
        # Distractor observations (current state)
        distractor_positions = ObsTerm(
            func=push_observations.distractor_positions_rel,
            params={
                "distractor_1_cfg": SceneEntityCfg("distractor_1")
            }
        )
        
        distractor_orientations = ObsTerm(
            func=push_observations.distractor_yaw_current,
            params={
                "distractor_1_cfg": SceneEntityCfg("distractor_1")
            }
        )
        
        # Distractor initial poses (target state to maintain)
        distractor_initial_positions = ObsTerm(
            func=push_observations.distractor_initial_positions_rel,
            params={
                "distractor_1_cfg": SceneEntityCfg("distractor_1")
            }
        )
        
        distractor_initial_orientations = ObsTerm(
            func=push_observations.distractor_yaw_initial,
            params={
                "distractor_1_cfg": SceneEntityCfg("distractor_1")
            }
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()
    


@configclass
class FrankaRobotiq2f85CustomOmniNudgeEnvCfg(FrankaRobotiq2f85CustomOmniRelTrainCfg):
    """Configuration for reorientation task with Franka + Robotiq gripper with distractors."""
    
    def __post_init__(self):

        super().__post_init__()
        

        
        ##### EVENTS #####
        cube_x_range = (0.625, 0.625) 
        cube_y_range = (0.0, 0.0)
        distractor_distance = 0.0470
        
        ##### COMMANDS #####
        min_radius = 0.05
        max_radius = 0.05
        yaw_range = (-1.57, 1.57)  
    
        
        ##### REWARDS #####
        threshold = 0.01  
        orientation_threshold = 0.0173  
        distractor_distance_threshold = 0.02
        distractor_orientation_threshold = 0.0173
        
        
        self.scene.distractor_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Distractor1",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.5, 0.1, 0.0203],  # Will be positioned by event
                rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",  # Red for distractors
                scale=(1.0, 1.0, 1.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        self.observations = NudgeObservationsCfg()
        self.events.randomize_cube_position.params["pose_range"]["x"] = cube_x_range
        self.events.randomize_cube_position.params["pose_range"]["y"] = cube_y_range
        self.nudge_distractor_config = {
            "distractor_1_cfg": SceneEntityCfg("distractor_1"),
            "distractor_2_cfg": None,
            "command_name": "ee_pose",
            "distractor_distance": distractor_distance,  
        }
        

        self.commands.ee_pose.ranges.yaw = yaw_range
        self.commands.ee_pose.position_only = False  
        self.commands.ee_pose.success_threshold = threshold
        self.commands.ee_pose.min_radius = min_radius
        self.commands.ee_pose.max_radius = max_radius
        self.rewards.reaching_goal = None  
        self.rewards.distance_orientation_goal_distractors = RwdTerm(
            func=push_mdp.distance_orientation_goal_with_distractors,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "goal_cfg": "ee_pose",
                "distractor_cfgs": [
                    SceneEntityCfg("distractor_1"),
                ],
                "distance_threshold": threshold,
                "orientation_threshold": orientation_threshold,
                "distractor_distance_threshold": distractor_distance_threshold, 
                "distractor_orientation_threshold": distractor_orientation_threshold, 
            },
            weight=1.0, 
        )
        

############# PUSH TASK #############


  
@configclass
class PushObservationsCfg:
    """Custom observation specifications for the reorientation task.
    
    This includes yaw angle and orientation delta observations that are more
    suitable for learning rotation tasks compared to full quaternions.
    """
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group - optimized for reorientation."""
        
        joint_pos = ObsTerm(func=isaaclab_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=isaaclab_mdp.joint_vel_rel)
        
        ee_pos = ObsTerm(func=push_observations.ee_frame_pos_rel)
        ee_quat = ObsTerm(func=push_observations.ee_frame_quat_rel)
        
        # Cube observations
        cube_pos = ObsTerm(func=push_observations.cube_pos_rel, params={"asset_cfg": SceneEntityCfg("cube")})
        cube_yaw = ObsTerm(func=push_observations.cube_yaw_angle, params={"asset_cfg": SceneEntityCfg("cube")})
        
        # Target observations
        target_pos = ObsTerm(func=push_observations.target_pos_rel, params={"command_name": "ee_pose"})
        target_yaw = ObsTerm(func=push_observations.target_yaw_angle, params={"command_name": "ee_pose"})
        
        # Cube position relative to goal (frame-invariant)
        cube_pos_goal = ObsTerm(
            func=push_observations.cube_in_target_frame,
            params={"command_name": "ee_pose", "asset_cfg": SceneEntityCfg("cube")}
        )
        
        # Distractor observations (current state)
        distractor_positions = ObsTerm(
            func=push_observations.distractor_positions_rel,
            params={
                "distractor_1_cfg": SceneEntityCfg("distractor_1")
            }
        )
        
        distractor_orientations = ObsTerm(
            func=push_observations.distractor_orientations_current,
            params={
                "distractor_1_cfg": SceneEntityCfg("distractor_1")
            }
        )
        
        # Distractor initial poses (target state to maintain)
        distractor_initial_positions = ObsTerm(
            func=push_observations.distractor_initial_positions_rel,
            params={
                "distractor_1_cfg": SceneEntityCfg("distractor_1")
            }
        )
        
        distractor_initial_orientations = ObsTerm(
            func=push_observations.distractor_initial_orientations,
            params={
                "distractor_1_cfg": SceneEntityCfg("distractor_1")
            }
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()
    


@configclass
class FrankaRobotiq2f85CustomOmniPushEnvCfg(FrankaRobotiq2f85CustomOmniRelTrainCfg):
    """Configuration for reorientation task with Franka + Robotiq gripper with distractors."""
    
    def __post_init__(self):

        super().__post_init__()
        

        
        ##### EVENTS #####
        cube_x_range = (0.725, 0.725) 
        cube_y_range = (0.0, 0.0)
        distractor_min_distance = 0.10
        distractor_max_distance = 0.20
        
        ##### COMMANDS #####
        min_radius = 0.25
        max_radius = 0.25
        yaw_range = (-1.57, 1.57)  
    
        
        ##### REWARDS #####
        threshold = 0.05  
        orientation_threshold = 3.14  
        distractor_distance_threshold = 0.02
        distractor_orientation_threshold = 0.1
        
        
        self.scene.distractor_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Distractor1",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.5, 0.1, 0.0203],  # Will be positioned by event
                rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",  # Red for distractors
                scale=(1.0, 1.0, 1.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        self.observations = PushObservationsCfg()
        self.events.randomize_cube_position.params["pose_range"]["x"] = cube_x_range
        self.events.randomize_cube_position.params["pose_range"]["y"] = cube_y_range
   
        self.push_distractor_config = {
            "distractor_1_cfg": SceneEntityCfg("distractor_1"),
            "distractor_2_cfg": None,
            "cube_cfg": SceneEntityCfg("cube"),
            "command_name": "ee_pose",
            "min_distance_from_cube": distractor_min_distance,
            "max_distance_from_cube": distractor_max_distance, 
        }
        

        self.commands.ee_pose.ranges.yaw = yaw_range
        self.commands.ee_pose.position_only = False  
        self.commands.ee_pose.success_threshold = threshold
        self.commands.ee_pose.min_radius = min_radius
        self.commands.ee_pose.max_radius = max_radius
        self.rewards.reaching_goal = None  
        self.rewards.distance_orientation_goal_distractors = RwdTerm(
            func=push_mdp.distance_orientation_goal_with_distractors,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "goal_cfg": "ee_pose",
                "distractor_cfgs": [
                    SceneEntityCfg("distractor_1"),
                ],
                "distance_threshold": threshold,
                "orientation_threshold": orientation_threshold,
                "distractor_distance_threshold": distractor_distance_threshold, 
                "distractor_orientation_threshold": distractor_orientation_threshold, 
            },
            weight=1.0, 
        )






############# FLIP TASK #############




@configclass
class FlipObservationsCfg:
    """Custom observation specifications for the reorientation task.
    
    This includes full orientation (roll, pitch, yaw) observations that are
    suitable for learning flip/reorientation tasks where all rotation axes matter.
    """
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group - optimized for reorientation."""
        
        joint_pos = ObsTerm(func=isaaclab_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=isaaclab_mdp.joint_vel_rel)
        
        ee_pos = ObsTerm(func=push_observations.ee_frame_pos_rel)
        ee_quat = ObsTerm(func=push_observations.ee_frame_quat_rel)
        
        # Cube observations - full orientation (roll, pitch, yaw)
        cube_pos = ObsTerm(func=push_observations.cube_pos_rel, params={"asset_cfg": SceneEntityCfg("cube")})
        cube_orientation = ObsTerm(func=push_observations.cube_orientation_euler, params={"asset_cfg": SceneEntityCfg("cube")})
        
        # Target observations - full orientation (roll, pitch, yaw)
        target_pos = ObsTerm(func=push_observations.target_pos_rel, params={"command_name": "ee_pose"})
        target_orientation = ObsTerm(func=push_observations.target_orientation_euler, params={"command_name": "ee_pose"})
        
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
        
        # Distractor observations (current state)
        distractor_positions = ObsTerm(
            func=push_observations.distractor_positions_rel,
            params={
                "distractor_1_cfg": SceneEntityCfg("distractor_1")
            }
        )
        
        # distractor_orientations = ObsTerm(
        #     func=push_observations.distractor_orientations_current,
        #     params={
        #         "distractor_1_cfg": SceneEntityCfg("distractor_1")
        #     }
        # )
        
        # Distractor initial poses (target state to maintain)
        distractor_initial_positions = ObsTerm(
            func=push_observations.distractor_initial_positions_rel,
            params={
                "distractor_1_cfg": SceneEntityCfg("distractor_1")
            }
        )
        
        # distractor_initial_orientations = ObsTerm(
        #     func=push_observations.distractor_initial_orientations,
        #     params={
        #         "distractor_1_cfg": SceneEntityCfg("distractor_1")
        #     }
        # )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()
    


@configclass
class FrankaRobotiq2f85CustomOmniFlipEnvCfg(FrankaRobotiq2f85CustomOmniRelTrainCfg):
    """Configuration for reorientation task with Franka + Robotiq gripper with distractors."""
    
    def __post_init__(self):

        super().__post_init__()
        

        
        ##### EVENTS #####
        cube_x_range = (0.625, 0.625) 
        cube_y_range = (0.0, 0.0)
        cube_yaw_range = (0.0, 0.0)
        distractor_distance = 0.0470
        
        ##### COMMANDS #####
        min_radius = 0.00
        max_radius = 0.00
        # yaw_range = (-1.57, 1.57)  
    
        
        ##### REWARDS #####
        threshold = 0.05  
        orientation_threshold = 0.0173  
        distractor_distance_threshold = 0.05
        distractor_orientation_threshold = 0.1
        
        
        self.scene.distractor_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Distractor1",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.5, 0.1, 0.0203],  # Will be positioned by event
                rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",  # Red for distractors
                scale=(1.0, 1.0, 1.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        self.observations = FlipObservationsCfg()
        self.events.randomize_cube_position.params["pose_range"]["x"] = cube_x_range
        self.events.randomize_cube_position.params["pose_range"]["y"] = cube_y_range
        self.events.randomize_cube_position.params["pose_range"]["yaw"] = cube_yaw_range
        self.nudge_distractor_config = {
            "distractor_1_cfg": SceneEntityCfg("distractor_1"),
            "distractor_2_cfg": None,
            "command_name": "ee_pose",
            "distractor_distance": distractor_distance,  
        }
        


        self.commands.ee_pose.position_only = False  
        self.commands.ee_pose.success_threshold = threshold
        self.commands.ee_pose.min_radius = min_radius
        self.commands.ee_pose.max_radius = max_radius
        # Discrete flip options: 4 ways to flip the cube by 90 degrees
        self.commands.ee_pose.discrete_orientation_options = [
            (1.5708, 0.0, 0.0),   # Roll right 90° (left face to top)
            (-1.5708, 0.0, 0.0),  # Roll left 90° (right face to top)
            (0.0, 1.5708, 0.0),   # Pitch forward 90° (back face to top)
            (0.0, -1.5708, 0.0),  # Pitch backward 90° (front face to top)
        ]
        self.rewards.reaching_goal = None  
        self.rewards.distance_orientation_goal_distractors = RwdTerm(
            func=push_mdp.distance_orientation_goal_with_distractors,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "goal_cfg": "ee_pose",
                "distractor_cfgs": [
                    SceneEntityCfg("distractor_1"),
                ],
                "distance_threshold": threshold,
                "orientation_threshold": orientation_threshold,
                "distractor_distance_threshold": distractor_distance_threshold, 
                "distractor_orientation_threshold": distractor_orientation_threshold, 
            },
            weight=1.0, 
        )
        



















class DistractorResetEnv(ManagerBasedRLEnv):
    """Custom environment that positions distractors after command reset and tracks their initial poses."""
    
    def __init__(self, cfg, render_mode=None, **kwargs):
        """Initialize environment and setup distractor tracking."""
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize dictionary to store initial distractor poses (in world coordinates)
        # Format: {distractor_name: (initial_pos_w, initial_quat_w)}
        self.distractor_initial_poses_w = {}
    
    def _reset_idx(self, env_ids: Sequence[int]):
        """Override reset to position distractors AFTER command manager resets and store their initial poses."""
        super()._reset_idx(env_ids)
        
        # Position distractors based on task type
        distractor_config = None
        
        if hasattr(self.cfg, 'nudge_distractor_config'):
            distractor_config = self.cfg.nudge_distractor_config
            push_mdp.position_distractors_cardinal_to_target(
                self, 
                env_ids,
                **distractor_config
            )
        elif hasattr(self.cfg, 'push_distractor_config'):
            distractor_config = self.cfg.push_distractor_config
            push_mdp.position_distractors_between_target(
                self, 
                env_ids,
                **distractor_config
            )
        
        # Store initial poses (supports any number of distractors)
        if distractor_config is not None:
            # Dynamically find all distractor configs (distractor_1_cfg, distractor_2_cfg, distractor_3_cfg, etc.)
            distractor_cfgs = []
            i = 1
            while True:
                key = f"distractor_{i}_cfg"
                if key in distractor_config:
                    cfg = distractor_config[key]
                    if cfg is not None:  # Only add if not None
                        distractor_cfgs.append(cfg)
                    i += 1
                else:
                    break  # No more distractors
            
            # Store initial poses for all found distractors
            for distractor_cfg in distractor_cfgs:
                distractor_name = distractor_cfg.name
                
                if distractor_name not in self.scene.keys():
                    continue
                
                distractor_asset = self.scene[distractor_name]
                
                if distractor_name not in self.distractor_initial_poses_w:
                    self.distractor_initial_poses_w[distractor_name] = (
                        distractor_asset.data.root_pos_w[:, :3].clone(),
                        distractor_asset.data.root_quat_w.clone(),
                    )
                else:
                    self.distractor_initial_poses_w[distractor_name][0][env_ids] = \
                        distractor_asset.data.root_pos_w[env_ids, :3].clone()
                    self.distractor_initial_poses_w[distractor_name][1][env_ids] = \
                        distractor_asset.data.root_quat_w[env_ids].clone()