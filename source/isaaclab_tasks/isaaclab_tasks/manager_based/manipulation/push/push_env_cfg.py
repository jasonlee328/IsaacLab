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
from . import mdp


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
        # joint_vel = ObsTerm(func=isaaclab_mdp.joint_vel_rel)
        actions = ObsTerm(func=isaaclab_mdp.last_action)
        
        # Cube observations (world frame)
        cube_pos = ObsTerm(func=isaaclab_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube")})
        cube_rot = ObsTerm(func=isaaclab_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube")})
        
        # Target observations (world frame)
        # target_pos = ObsTerm(func=isaaclab_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("target")})
        target_pos = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        # End-effector observations
        # eef_pos = ObsTerm(func=stack_mdp.ee_frame_pos)
        # eef_quat = ObsTerm(func=stack_mdp.ee_frame_quat)
        # gripper_pos = ObsTerm(func=stack_mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # Target position for the cube to reach
    ee_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",  # Reference frame (robot base)
        object_name="cube",  # The object to generate commands for
        success_vis_asset_name="cube",  # Asset to visualize success
        resampling_time_range=(10e9, 10e9),
        debug_vis=True,  # Enable visualization of target and current positions
        position_only=True,  # Only generate position commands (no orientation)
        make_quat_unique=False,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.6),  # Target x position range in robot base frame
            pos_y=(-0.325, 0.325),  # Target y position range
            pos_z=(0.0203, 0.0203),  # Target z position (table height)
            roll=(0, 0),
            pitch=(0, 0),
            yaw=(0.0, 0),
        ),
        # Goal position visualization (red when far, green when near)
        goal_pose_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "position_far": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red
                ),
                "position_near": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # Green
                ),
            },
        ),
        # Current cube position visualization (red when far, green when near)
        curr_pose_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/Command/body_pose",
            markers={
                "position_far": sim_utils.SphereCfg(
                    radius=0.025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),  # Orange
                ),
                "position_near": sim_utils.SphereCfg(
                    radius=0.025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 1.0)),  # Cyan
                ),
            },
        ),
        # Success marker visualization
        success_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/SuccessMarkers",
            markers={
                "success": sim_utils.SphereCfg(
                    radius=0.02,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                )
            },
        ),
    )

@configclass
class RewardsCfg:
    """Reward specifications for the push MDP."""

    # SPARSE REWARD ONLY - Success when cube reaches target
    reaching_goal = RwdTerm(
        func=push_mdp.object_reached_goal,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "goal_cfg": "ee_pose",
            "threshold": 0.1,  
        },
        weight=1.0,  # Sparse reward: +1 for success
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
    
    # Unused managers
    commands = CommandsCfg()
    curriculum = None
    events = None  # Will be set by derived classes

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # 25Hz control
        self.episode_length_s = 10.0  # 10 second episodes
        
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation
        
        # physics settings
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
