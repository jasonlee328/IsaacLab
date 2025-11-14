# Copyright (c) 2024-2025, The Octi Lab Project Developers. (https://github.com/zoctipus/OctiLab/blob/main/CONTRIBUTORS.md).
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

from __future__ import annotations

from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    ClippedBinaryJointPositionActionCfg,
    DefaultJointPositionStaticActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_ROBOTIQ_GRIPPER_CUSTOM_OMNI_PAT_CFG
from isaaclab_tasks.manager_based.manipulation.push.config.franka_robotiq_2f85.actions import (
    ROBOTIQ_CUSTOM_GRIPPER_BINARY_ACTIONS,
    ROBOTIQ_CUSTOM_COMPLIANT_JOINTS,
)

from isaaclab_tasks.manager_based.manipulation.reset_states.mdp.utils import read_metadata_from_usd_directory

from ...mdp.actions.actions_cfg import PreprocessedOperationalSpaceControllerActionCfg

FRANKA_ROBOTIQ_2F85_RELATIVE_OSC = PreprocessedOperationalSpaceControllerActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    # Use robotiq_arg2f_base_link to match UR5e pattern and align with observations/rewards
    # This ensures OSC controls the same frame that observations/rewards reference (with gripper_offset applied for TCP)
    body_name="panda_link7",
    body_offset=PreprocessedOperationalSpaceControllerActionCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)  # No offset: control gripper base link directly (matches UR5e pattern)
    ),
    action_root_offset=PreprocessedOperationalSpaceControllerActionCfg.OffsetCfg(
        # Robot base coordinate frame transformation (not end-effector frame)
        pos=read_metadata_from_usd_directory(FRANKA_ROBOTIQ_GRIPPER_CUSTOM_OMNI_PAT_CFG.spawn.usd_path).get("offset", {}).get("pos", [0.0, 0.0, 0.0]),
        rot=read_metadata_from_usd_directory(FRANKA_ROBOTIQ_GRIPPER_CUSTOM_OMNI_PAT_CFG.spawn.usd_path).get("offset", {}).get("quat", [1.0, 0.0, 0.0, 0.0]),
    ),
    # Increased action scaling for better exploration (1.5x position, 1.5x rotation)
    # This allows larger movements per action, helping the policy explore more effectively
    scale_xyz_axisangle=(0.04, 0.04, 0.04, 0.04, 0.04, 0.2),
    input_clip=(-1.0, 1.0),
    controller_cfg=OperationalSpaceControllerCfg(
        target_types=["pose_rel"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=False,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_stiffness_task=(200.0, 200.0, 200.0, 5.0, 5.0, 5.0),
        motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        nullspace_control="none",
    ),
    position_scale=1.0,
    orientation_scale=1.0,
    stiffness_scale=1.0,
    damping_ratio_scale=1.0,
)


@configclass
class FrankaRobotiq2f85RelativeOSCAction:
    arm = FRANKA_ROBOTIQ_2F85_RELATIVE_OSC
    gripper = ROBOTIQ_CUSTOM_GRIPPER_BINARY_ACTIONS  # Custom USD uses outer_knuckle_joint
    compliant_joints = ROBOTIQ_CUSTOM_COMPLIANT_JOINTS  # Custom USD uses inner_finger_joint


# Relative Joint Position Action for RL training
FRANKA_ROBOTIQ_2F85_RELATIVE_JOINT_POSITION = RelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    scale=0.02,
    use_zero_offset=True,
)


@configclass
class FrankaRobotiq2f85RelativeJointPositionAction:
    """Relative joint position action for Franka Robotiq 2F85 (for RL training)."""
    arm = FRANKA_ROBOTIQ_2F85_RELATIVE_JOINT_POSITION
    gripper = ROBOTIQ_CUSTOM_GRIPPER_BINARY_ACTIONS
    compliant_joints = ROBOTIQ_CUSTOM_COMPLIANT_JOINTS


# Custom gripper binary actions with clipping (matching UR5e pattern)
ROBOTIQ_CUSTOM_GRIPPER_BINARY_ACTIONS_CLIPPED = ClippedBinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=[".*_outer_knuckle_joint"],  # Pre-assembled USD uses outer_knuckle_joints
    open_command_expr={".*_outer_knuckle_joint": 0.0},  # Open position
    close_command_expr={".*_outer_knuckle_joint": 0.8},  # Closed position
    input_clip=(-1.0, 1.0),  # Match UR5e gripper action clipping
)


@configclass
class FrankaRobotiq2f85CustomRelativeAction:
    """Configurable relative joint position action for Franka Robotiq 2F85 with custom pre-assembled USD.
    
    This action uses relative joint position control for the arm and binary position control for the gripper.
    panda_joint6 (wrist bend, equivalent to UR5e wrist_3_joint) is scaled by 0.2, matching UR5e pattern.
    
    Args:
        arm_scale: Scale factor for relative joint position actions (default: 0.02 for most joints, 0.2 for panda_joint6).
            This controls how much the joints move per unit of action input.
    """
    # Default arm action with per-joint scaling (panda_joint6 scaled by 0.2, matching UR5e wrist_3_joint)
    arm: RelativeJointPositionActionCfg = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale={
            "(?!panda_joint6).*": 0.02,  # All joints except panda_joint6: 0.02
            "panda_joint6": 0.2,  # panda_joint6 (wrist bend, equivalent to UR5e wrist_3_joint): 0.2
        },
        use_zero_offset=True,
    )
    # Gripper binary action with clipping (matching UR5e pattern)
    gripper: ClippedBinaryJointPositionActionCfg = ROBOTIQ_CUSTOM_GRIPPER_BINARY_ACTIONS_CLIPPED
    # Compliant joints (passive inner finger joints)
    compliant_joints: DefaultJointPositionStaticActionCfg = ROBOTIQ_CUSTOM_COMPLIANT_JOINTS
