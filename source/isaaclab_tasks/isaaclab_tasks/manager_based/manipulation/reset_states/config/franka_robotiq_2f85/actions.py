# Copyright (c) 2024-2025, The Octi Lab Project Developers. (https://github.com/zoctipus/OctiLab/blob/main/CONTRIBUTORS.md).
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

from __future__ import annotations

from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
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
    body_name="robotiq_arg2f_base_link",
    body_offset=PreprocessedOperationalSpaceControllerActionCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)  # No offset: control gripper base link directly (matches UR5e pattern)
    ),
    action_root_offset=PreprocessedOperationalSpaceControllerActionCfg.OffsetCfg(
        # Robot base coordinate frame transformation (not end-effector frame)
        pos=read_metadata_from_usd_directory(FRANKA_ROBOTIQ_GRIPPER_CUSTOM_OMNI_PAT_CFG.spawn.usd_path).get("offset", {}).get("pos", [0.0, 0.0, 0.0]),
        rot=read_metadata_from_usd_directory(FRANKA_ROBOTIQ_GRIPPER_CUSTOM_OMNI_PAT_CFG.spawn.usd_path).get("offset", {}).get("quat", [1.0, 0.0, 0.0, 0.0]),
    ),
    scale_xyz_axisangle=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
    input_clip=(-1.0, 1.0),
    controller_cfg=OperationalSpaceControllerCfg(
        target_types=["pose_rel"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=False,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_stiffness_task=(200.0, 200.0, 200.0, 5.0, 5.0, 5.0),
        motion_damping_ratio_task=(3.0, 3.0, 3.0, 1.0, 1.0, 1.0),
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
