# Copyright (c) 2024-2025, The Octi Lab Project Developers. (https://github.com/zoctipus/OctiLab/blob/main/CONTRIBUTORS.md).
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

from __future__ import annotations

from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.ur5e_robotiq_gripper import EXPLICIT_UR5E_ROBOTIQ_2F85
from isaaclab_assets.robots.ur5e_robotiq_gripper.actions import ROBOTIQ_COMPLIANT_JOINTS, ROBOTIQ_GRIPPER_BINARY_ACTIONS

from isaaclab_tasks.manager_based.manipulation.reset_states.mdp.utils import read_metadata_from_usd_directory

from ...mdp.actions.actions_cfg import PreprocessedOperationalSpaceControllerActionCfg

UR5E_ROBOTIQ_2F85_RELATIVE_OSC = PreprocessedOperationalSpaceControllerActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name="robotiq_base_link",
    body_offset=PreprocessedOperationalSpaceControllerActionCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
    ),
    action_root_offset=PreprocessedOperationalSpaceControllerActionCfg.OffsetCfg(
        pos=read_metadata_from_usd_directory(EXPLICIT_UR5E_ROBOTIQ_2F85.spawn.usd_path).get("offset").get("pos"),
        rot=read_metadata_from_usd_directory(EXPLICIT_UR5E_ROBOTIQ_2F85.spawn.usd_path).get("offset").get("quat"),
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
class Ur5eRobotiq2f85RelativeOSCAction:
    arm = UR5E_ROBOTIQ_2F85_RELATIVE_OSC
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
    compliant_joints = ROBOTIQ_COMPLIANT_JOINTS
