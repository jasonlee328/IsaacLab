# Copyright (c) 2024-2025, The Octi Lab Project Developers. (https://github.com/zoctipus/OctiLab/blob/main/CONTRIBUTORS.md).
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

from __future__ import annotations


from isaaclab.utils import configclass

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, DefaultJointPositionStaticActionCfg, RelativeJointPositionActionCfg



ROBOTIQ_GRIPPER_BINARY_ACTIONS = BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["finger_joint"],
    open_command_expr={"finger_joint": 0.0},
    close_command_expr={"finger_joint": 0.785398},
)

ROBOTIQ_COMPLIANT_JOINTS = DefaultJointPositionStaticActionCfg(
    asset_name="robot", joint_names=["left_inner_finger_joint", "right_inner_finger_joint"]
)
    
FRANKA_ROBOTIQ_2F85_RELATIVE_ACTION = RelativeJointPositionActionCfg(
    asset_name="robot", joint_names=["panda_joint.*"], scale=0.02
) 
    
@configclass
class FrankaRobotiq2f85RelativeAction:
    arm = FRANKA_ROBOTIQ_2F85_RELATIVE_ACTION
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
    compliant_joints = ROBOTIQ_COMPLIANT_JOINTS


# Custom actions for pre-assembled Robotiq USD (uses outer_knuckle_joints)
ROBOTIQ_CUSTOM_GRIPPER_BINARY_ACTIONS = BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=[".*_outer_knuckle_joint"],  # Pre-assembled USD uses outer_knuckle_joints
    open_command_expr={".*_outer_knuckle_joint": 0.0},  # Open position
    close_command_expr={".*_outer_knuckle_joint": 0.8},  # Closed position
)

ROBOTIQ_CUSTOM_COMPLIANT_JOINTS = DefaultJointPositionStaticActionCfg(
    asset_name="robot", joint_names=[".*_inner_finger_joint"]  # Passive joints in custom USD
)


@configclass
class FrankaRobotiq2f85CustomRelativeAction:
    """Actions for custom pre-assembled Robotiq USD (GitHub #1299)."""
    arm = FRANKA_ROBOTIQ_2F85_RELATIVE_ACTION
    gripper = ROBOTIQ_CUSTOM_GRIPPER_BINARY_ACTIONS
    compliant_joints = ROBOTIQ_CUSTOM_COMPLIANT_JOINTS 