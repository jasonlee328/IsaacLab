# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.push.push_env_cfg import PushEnvCfg
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

##
# Environment configuration
##

@configclass
class PushEventCfg:
    """Configuration for push task events."""
    
    # Initialize robot pose with small randomization
    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )
    
    # Randomize robot joint states
    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,  # Fixed 2cm std for joint randomization
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    # Randomize cube spawn position within 3cm radius
    randomize_cube_position = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.47, 0.53),  # 3cm radius around center position (0.5)
                "y": (-0.03, 0.03),  # 3cm radius 
                "z": (0.0203, 0.0203),  # Fixed height on table
                "yaw": (-0.5, 0.5)  # Small rotation randomization
            },
            "min_separation": 0.0,  # No separation needed for single cube
            "asset_cfgs": [SceneEntityCfg("cube")],
        },
    )
    
    # # Randomize target position - will be configured in __post_init__
    # randomize_target_position = EventTerm(
    #     func=franka_stack_events.randomize_object_pose,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": (-1, 1),  # Default range, will be updated
    #             "y": (-1, 1),  # Default range, will be updated
    #             "z": (0.0203, 0.0203),  # Fixed height on table
    #             "yaw": (0.0, 0.0)  # No rotation for target
    #         },
    #         "min_separation": 0.0,
    #         "asset_cfgs": [SceneEntityCfg("target")],
    #     },
    # )


@configclass
class FrankaPushCubeEnvCfg(PushEnvCfg):
    """Configuration for Franka cube pushing task with sparse rewards."""
    
    # Task-specific parameters
    target_spawn_radius: float = 0.50  # Default 15cm radius around cube for target spawn
    target_goal_radius: float = 0.05   # Default 5cm radius for goal region
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # Set push-specific events
        self.events = PushEventCfg()
        
        # Update target spawn range based on parameters
        # Assuming cube spawns around (0.5, 0.0)
        # self.events.randomize_target_position.params["pose_range"]["x"] = (
        #     0.5 - self.target_spawn_radius, 
        #     0.5 + self.target_spawn_radius
        # )
        # self.events.randomize_target_position.params["pose_range"]["y"] = (
        #     0.0 - self.target_spawn_radius,
        #     0.0 + self.target_spawn_radius
        # )
        
        # Update sparse reward threshold with our goal radius
        self.rewards.reaching_goal.params["threshold"] = self.target_goal_radius
        
        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        
        
        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        
        # Gripper utilities
        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005
        
        # Rigid body properties
        cube_properties = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )
        
        # Single cube configuration
        self.scene.cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.5, 0.0, 0.0203],  # Center position before randomization
                rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube")],
            ),
        )
        
        # Target region configuration (visual indicator)
        target_properties = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            disable_gravity=True,  # Target shouldn't fall
        )
        
        # # Create a flat cylinder as target region indicator
        # self.scene.target = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Target",
        #     init_state=RigidObjectCfg.InitialStateCfg(
        #         pos=[0.5, 0.2, 0.0203],  # Default position before randomization
        #         rot=[1, 0, 0, 0]
        #     ),
        #     spawn=sim_utils.CylinderCfg(
        #         radius=self.target_goal_radius,
        #         height=0.01,  # Very flat cylinder
        #         visual_material=sim_utils.PreviewSurfaceCfg(
        #             diffuse_color=(1.0, 0.0, 0.0),  # Red color
        #             opacity=1.0,  # Fully opaque
        #         ),
        #         rigid_props=target_properties,
        #         semantic_tags=[("class", "target")],
        #     ),
        # )
        
        
        # Frame transformer for end-effector tracking
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
