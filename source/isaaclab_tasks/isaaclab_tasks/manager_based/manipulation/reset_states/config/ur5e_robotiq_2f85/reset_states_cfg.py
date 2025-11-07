# Copyright (c) 2024-2025, The Octi Lab Project Developers. (https://github.com/zoctipus/OctiLab/blob/main/CONTRIBUTORS.md).
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

from __future__ import annotations

import numpy as np
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import OCTILAB_CLOUD_ASSETS_DIR
from isaaclab_assets.robots.ur5e_robotiq_gripper import EXPLICIT_UR5E_ROBOTIQ_2F85
from isaaclab.utils import (
    get_octilab_assets_path,
    get_octilab_grasp_datasets_path,
    get_octilab_partial_assembly_datasets_path,
    get_octilab_reset_state_datasets_path,
)

from isaaclab_tasks.manager_based.manipulation.reset_states.config.ur5e_robotiq_2f85.actions import (
    Ur5eRobotiq2f85RelativeOSCAction,
)

from ... import mdp as task_mdp


@configclass
class ResetStatesSceneCfg(InteractiveSceneCfg):
    """Scene configuration for reset states environment."""

    robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

    insertive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{OCTILAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            # assume very light
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # receptive_object: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/ReceptiveObject",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{OCTILAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd",
    #         scale=(1, 1, 1),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=4,
    #             solver_velocity_iteration_count=0,
    #             disable_gravity=False,
    #             # receptive object does not move
    #             kinematic_enabled=True,
    #         ),
    #         # since kinematic_enabled=True, mass does not matter
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    # )


    receptive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{OCTILAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                # receptive object does not move
                kinematic_enabled=False,
            ),
            # since kinematic_enabled=True, mass does not matter
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Environment
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, -0.881), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{OCTILAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention/pat_vention.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    ur5_metal_support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/UR5MetalSupport",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0, -0.013), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{OCTILAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention2/Ur5MetalSupport/ur5plate.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.868)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=10000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class ResetStatesBaseEventCfg:
    """Configuration for randomization."""

    # startup: low friction to avoid slip
    reset_robot_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.3),
            "dynamic_friction_range": (0.2, 0.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("robot"),
            "make_consistent": True,
        },
    )

    insertive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.3),
            "dynamic_friction_range": (0.2, 0.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "make_consistent": True,
        },
    )

    receptive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.3),
            "dynamic_friction_range": (0.2, 0.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "make_consistent": True,
        },
    )

    # visualize object orientations
    visualize_object_orientations = EventTerm(
        func=task_mdp.object_orientation_visualization_event,
        mode="interval",
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
        interval_range_s=(0.0, 0.0),  # Update every step
    )

    # reset

    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})

    reset_robot_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.01, 0.01),
                "y": (-0.059, -0.019),
                "z": (-0.01, 0.01),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfgs": {"robot": SceneEntityCfg("robot"), "ur5_metal_support": SceneEntityCfg("ur5_metal_support")},
        },
    )

    reset_receptive_object_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.3, 0.55),
                "y": (-0.1, 0.3),
                "z": (0.1, 0.1),
                "roll": (0, 0),
                "pitch": (0, 0),
                "yaw": (-np.pi/12, np.pi/12),
            },
            "velocity_range": {},
            "asset_cfgs": {"receptive_object": SceneEntityCfg("receptive_object")},
            "offset_asset_cfg": SceneEntityCfg("ur5_metal_support"),
            "use_bottom_offset": True,
        },
    )


@configclass
class ObjectAnywhereEEAnywhereEventCfg(ResetStatesBaseEventCfg):
    reset_insertive_object_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.3, 0.55),
                "y": (-0.1, 0.3),
                "z": (0.2, 0.2),
                "roll": (-np.pi, np.pi),
                "pitch": (-np.pi, np.pi),
                "yaw": (-np.pi, np.pi),
            },
            "velocity_range": {},
            "asset_cfgs": {"insertive_object": SceneEntityCfg("insertive_object")},
            "offset_asset_cfg": SceneEntityCfg("ur5_metal_support"),
            "use_bottom_offset": True,
        },
    )

    reset_end_effector_pose = EventTerm(
        func=task_mdp.reset_end_effector_round_fixed_asset,
        mode="reset",
        params={
            "fixed_asset_cfg": SceneEntityCfg("robot"),
            "fixed_asset_offset": None,
            "pose_range_b": {
                "x": (0.3, 0.7),
                "y": (-0.4, 0.4),
                "z": (0.0, 0.5),
                "roll": (0.0, 0.0),
                "pitch": (np.pi / 4, 3 * np.pi / 4),
                "yaw": (np.pi / 2, 3 * np.pi / 2),
            },
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
            ),
        },
    )


@configclass
class ObjectRestingEEGraspedEventCfg(ResetStatesBaseEventCfg):
    reset_insertive_object_pose_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "base_paths": ["reset_state_datasets/ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
        },
    )

    #Move the end effector to be near the receptive object
    reset_end_effector_pose_from_grasp_dataset = EventTerm(
        func=task_mdp.reset_end_effector_from_grasp_dataset,
        mode="reset",
        params={
            "base_path": str(get_octilab_grasp_datasets_path()),
            "fixed_asset_cfg": SceneEntityCfg("insertive_object"),
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
            ),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["finger_joint", ".*right.*", ".*left.*"]),
            "pose_range_b": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (-0.02, 0.02),
                "roll": (-np.pi / 16, np.pi / 16),
                "pitch": (-np.pi / 16, np.pi / 16),
                "yaw": (-np.pi / 16, np.pi / 16),
            },
        },
    )

@configclass
class ObjectRestingEERoundInsertiveEventCfg(ResetStatesBaseEventCfg):
    """Reset insertive object from reset states dataset, then position end effector around it."""
    
    # First, reset the insertive object from the reset states dataset (same as ObjectRestingEEGraspedEventCfg)
    reset_insertive_object_pose_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "base_paths": [str(get_octilab_reset_state_datasets_path() / "ObjectAnywhereEEAnywhere")],
            "probs": [1.0],
        },
    )

    # Then, position the end effector around the insertive object (instead of using grasp dataset)
    reset_end_effector_pose_around_insertive = EventTerm(
        func=task_mdp.reset_end_effector_round_fixed_asset,
        mode="reset",
        params={
            "fixed_asset_cfg": SceneEntityCfg("insertive_object"),  # Target the insertive object
            "fixed_asset_offset": None,  # No additional offset
            "pose_range_b": {
                "x": (-0.1, 0.1),      # Distance range from insertive object
                "y": (-0.1, 0.1),     # Lateral range
                "z": (0.0, 0.1),      # Height range
                "roll": (0.0, 0.0),   # No roll variation
                "pitch": (np.pi / 6, 5 * np.pi / 6),  # Pitch range (30° to 150°)
                "yaw": (0.0, 2 * np.pi),              # Full yaw range (0° to 360°)
            },
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
            ),
        },
    )


@configclass
class ObjectAnywhereEEGraspedEventCfg(ResetStatesBaseEventCfg):
    #first reset the insertive object to be anywhere

    reset_insertive_object_pose_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "base_paths": ["reset_state_datasets/ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
        },
    )


    reset_insertive_object_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.3, 0.55),
                "y": (-0.1, 0.3),
                "z": (0.0, 0.3),
                "roll": (-np.pi, np.pi),
                "pitch": (-np.pi, np.pi),
                "yaw": (-np.pi, np.pi),
            },
            "velocity_range": {},
            "asset_cfgs": {"insertive_object": SceneEntityCfg("insertive_object")},
            "offset_asset_cfg": SceneEntityCfg("ur5_metal_support"),
            "use_bottom_offset": True,
        },
    )

    # Then reset the end effector at the relative grasp pose
    reset_end_effector_pose_from_grasp_dataset = EventTerm(
        func=task_mdp.reset_end_effector_from_grasp_dataset,
        mode="reset",
        params={
            "base_path": str(get_octilab_grasp_datasets_path()),
            "fixed_asset_cfg": SceneEntityCfg("insertive_object"),
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
            ),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["finger_joint", ".*right.*", ".*left.*"]),
            "pose_range_b": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )



@configclass
class ObjectStackedOnReceptiveEventCfg(ResetStatesBaseEventCfg):
    """Configuration for resetting the insertive object stacked on top of the receptive object."""
    
    # First, reset the entire scene from the reset states dataset
    reset_scene_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "base_paths": [str(get_octilab_reset_state_datasets_path() / "ObjectAnywhereEEAnywhere")],
            "probs": [1.0],
        },
    )
    
    # Then, stack the insertive object on top of the receptive object
    stack_insertive_object_on_receptive = EventTerm(
        func=task_mdp.reset_insertive_object_stacked_on_receptive,
        mode="reset",
        params={
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "stack_height_offset": 0.041,  # 10cm height offset
        },
    )


@configclass
class ObjectNearReceptiveEEGraspedEventCfg(ResetStatesBaseEventCfg):
    """Configuration for resetting insertive object near receptive object and grasping it."""

    reset_insertive_object_pose_near_receptive = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "base_paths": ["reset_state_datasets/ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
        },
    )
    
    # Step 1: Reset insertive object near receptive object (position and orientation relative)
    reset_insertive_object_near_receptive = EventTerm(
        func=task_mdp.reset_insertive_object_relative_to_receptive,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.02, 0.02),      # Box size: 10cm x 10cm x 5cm
                "y": (-0.02, 0.02), 
                "z": (0.04, 0.06),
                "roll": (-np.pi/8, np.pi/8),    # Small orientation variation
                "pitch": (-np.pi/8, np.pi/8),
                "yaw": (-np.pi/8, np.pi/8),
            },
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
            "use_bottom_offset": False,  # Use bottom offset of insertive object
        },
    )
    
    # Step 2: Reset end effector to grasp the insertive object
    reset_end_effector_pose_from_grasp_dataset = EventTerm(
        func=task_mdp.reset_end_effector_from_grasp_dataset,
        mode="reset",
        params={
            "base_path": str(get_octilab_grasp_datasets_path()),
            "fixed_asset_cfg": SceneEntityCfg("insertive_object"),
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
            ),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["finger_joint", ".*right.*", ".*left.*"]),
            "pose_range_b": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )




@configclass
class ObjectPartiallyAssembledEEAnywhereEventCfg(ResetStatesBaseEventCfg):
    reset_insertive_object_pose_from_partial_assembly_dataset = EventTerm(
        func=task_mdp.reset_insertive_object_from_partial_assembly_dataset,
        mode="reset",
        params={
            "base_path": str(get_octilab_partial_assembly_datasets_path()),
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "pose_range_b": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_end_effector_pose = EventTerm(
        func=task_mdp.reset_end_effector_round_fixed_asset,
        mode="reset",
        params={
            "fixed_asset_cfg": SceneEntityCfg("robot"),
            "fixed_asset_offset": None,
            "pose_range_b": {
                "x": (0.3, 0.7),
                "y": (-0.4, 0.4),
                "z": (0.5, 0.5),
                "roll": (0.0, 0.0),
                "pitch": (np.pi / 4, 3 * np.pi / 4),
                "yaw": (np.pi / 2, 3 * np.pi / 2),
            },
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
            ),
        },
    )


@configclass
class ObjectPartiallyAssembledEEGraspedEventCfg(ResetStatesBaseEventCfg):
    reset_insertive_object_pose_from_partial_assembly_dataset = EventTerm(
        func=task_mdp.reset_insertive_object_from_partial_assembly_dataset,
        mode="reset",
        params={
            "base_path": str(get_octilab_partial_assembly_datasets_path()),
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "pose_range_b": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_end_effector_pose_from_grasp_dataset = EventTerm(
        func=task_mdp.reset_end_effector_from_grasp_dataset,
        mode="reset",
        params={
            "base_path": str(get_octilab_grasp_datasets_path()),
            "fixed_asset_cfg": SceneEntityCfg("insertive_object"),
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
            ),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["finger_joint", ".*right.*", ".*left.*"]),
            "pose_range_b": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
                "roll": (-np.pi / 32, np.pi / 32),
                "pitch": (-np.pi / 32, np.pi / 32),
                "yaw": (-np.pi / 32, np.pi / 32),
            },
        },
    )


@configclass
class ResetStatesTerminationCfg:
    """Configuration for reset states termination conditions."""

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)

    success = DoneTerm(
        func=task_mdp.check_reset_state_success,
        params={
            "object_cfgs": [SceneEntityCfg("insertive_object"), SceneEntityCfg("receptive_object")],
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_body_name": "robotiq_base_link",
            "collision_analyzer_cfgs": [
                task_mdp.CollisionAnalyzerCfg(
                    num_points=1024,
                    max_dist=0.5,
                    min_dist=-0.0005,
                    asset_cfg=SceneEntityCfg("robot"),
                    obstacle_cfgs=[SceneEntityCfg("insertive_object")],
                ),
                task_mdp.CollisionAnalyzerCfg(
                    num_points=1024,
                    max_dist=0.5,
                    min_dist=0.0,
                    asset_cfg=SceneEntityCfg("robot"),
                    obstacle_cfgs=[SceneEntityCfg("receptive_object")],
                ),
                task_mdp.CollisionAnalyzerCfg(
                    num_points=1024,
                    max_dist=0.5,
                    min_dist=-0.0005,
                    asset_cfg=SceneEntityCfg("insertive_object"),
                    obstacle_cfgs=[SceneEntityCfg("receptive_object")],
                ),
            ],
            "max_robot_pos_deviation": 0.05,
            "max_object_pos_deviation": MISSING,
            "pos_z_threshold": -0.01,
            "consecutive_stability_steps": 5,
        },
        time_out=True,
    )


@configclass
class ResetStatesObservationsCfg:
    """Configuration for reset states observations."""

    pass


@configclass
class ResetStatesRewardsCfg:
    """Configuration for reset states rewards."""

    pass


def make_insertive_object(usd_path: str):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


    # def make_insertive_object(usd_path: str):
    # return RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/InsertiveObject",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=usd_path,
    #         scale=(1, 1, 1),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=4,
    #             solver_velocity_iteration_count=0,
    #             disable_gravity=False,
    #             kinematic_enabled=True,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    # )



def make_receptive_object(usd_path: str):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


# variants = {
#     "scene.insertive_object": {
#         "fbleg": make_insertive_object(f"{OCTILAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareLeg/square_leg.usd"),
#         "fbdrawerbottom": make_insertive_object(
#             f"{OCTILAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBottom/drawer_bottom.usd"
#         ),
#         "peg": make_insertive_object(f"{OCTILAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd"),
#     },
#     "scene.receptive_object": {
#         "fbtabletop": make_receptive_object(
#             f"{OCTILAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareTableTop/square_table_top.usd"
#         ),
#         "fbdrawerbox": make_receptive_object(
#             f"{OCTILAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBox/drawer_box.usd"
#         ),
#         "peghole": make_receptive_object(f"{OCTILAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd"),
#     },
# }


variants = {
    "scene.insertive_object": {
        "cube": make_insertive_object(str(get_octilab_assets_path() / "props" / "insertive_cube" / "insertive_cube.usd")),
        "peg": make_insertive_object(f"{OCTILAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd")
    },
    "scene.receptive_object": {
        "cube": make_receptive_object(str(get_octilab_assets_path() / "props" / "receptive_cube" / "receptive_cube.usd")),
        "peg": make_receptive_object(f"{OCTILAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd")
    },
}




@configclass
class UR5eRobotiq2f85ResetStatesCfg(ManagerBasedRLEnvCfg):
    """Configuration for reset states environment with UR5e Robotiq 2F85 gripper."""

    scene: ResetStatesSceneCfg = ResetStatesSceneCfg(num_envs=1, env_spacing=1.5)
    events: ResetStatesBaseEventCfg = MISSING
    terminations: ResetStatesTerminationCfg = ResetStatesTerminationCfg()
    observations: ResetStatesObservationsCfg = ResetStatesObservationsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: ResetStatesRewardsCfg = ResetStatesRewardsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

    def __post_init__(self):
        self.decimation = 12
        self.episode_length_s = 2.0
        # simulation settings
        self.sim.dt = 1 / 120.0

        # Contact and solver settings
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005

        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**31

        # Render settings
        self.sim.render.enable_dlssg = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True


@configclass
class ObjectAnywhereEEAnywhereResetStatesCfg(UR5eRobotiq2f85ResetStatesCfg):
    events: ObjectAnywhereEEAnywhereEventCfg = ObjectAnywhereEEAnywhereEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = np.inf


@configclass
class ObjectRestingEEGraspedResetStatesCfg(UR5eRobotiq2f85ResetStatesCfg):
    events: ObjectRestingEEGraspedEventCfg = ObjectRestingEEGraspedEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = 0.01


@configclass
class ObjectAnywhereEEGraspedResetStatesCfg(UR5eRobotiq2f85ResetStatesCfg):
    events: ObjectAnywhereEEGraspedEventCfg = ObjectAnywhereEEGraspedEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = 0.05


@configclass
class ObjectPartiallyAssembledEEAnywhereResetStatesCfg(UR5eRobotiq2f85ResetStatesCfg):
    events: ObjectPartiallyAssembledEEAnywhereEventCfg = ObjectPartiallyAssembledEEAnywhereEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = 0.005


@configclass
class ObjectPartiallyAssembledEEGraspedResetStatesCfg(UR5eRobotiq2f85ResetStatesCfg):
    events: ObjectPartiallyAssembledEEGraspedEventCfg = ObjectPartiallyAssembledEEGraspedEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = 0.005


@configclass
class ObjectRestingEERoundInsertiveResetStatesCfg(UR5eRobotiq2f85ResetStatesCfg):
    events: ObjectRestingEERoundInsertiveEventCfg = ObjectRestingEERoundInsertiveEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = 0.01


@configclass
class ObjectStackedOnReceptiveResetStatesCfg(UR5eRobotiq2f85ResetStatesCfg):
    events: ObjectStackedOnReceptiveEventCfg = ObjectStackedOnReceptiveEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = 0.01


@configclass
class ObjectNearReceptiveEEGraspedResetStatesCfg(UR5eRobotiq2f85ResetStatesCfg):
    events: ObjectNearReceptiveEEGraspedEventCfg = ObjectNearReceptiveEEGraspedEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = 0.025
