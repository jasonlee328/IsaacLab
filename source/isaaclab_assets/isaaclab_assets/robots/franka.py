# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control
* :obj:`FRANKA_ROBOTIQ_GRIPPER_CFG`: Franka robot with Robotiq_2f_85 gripper

Reference: https://github.com/frankaemika/franka_ros
"""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  # UR5 style - disable gravity
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=36, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
    ),
    # actuators={
    #     "panda_shoulder": ImplicitActuatorCfg(
    #         joint_names_expr=["panda_joint[1-4]"],
    #         effort_limit_sim=87.0,
    #         velocity_limit_sim=0.175,
    #         stiffness=5000.0,
    #         damping=1000.0,
    #     ),
    #     "panda_forearm": ImplicitActuatorCfg(
    #         joint_names_expr=["panda_joint[5-7]"],
    #         effort_limit_sim=12.0,
    #         velocity_limit_sim=0.175,
    #         stiffness=5000.0,
    #         damping=1000.0,
    #     ),
    #     "panda_hand": ImplicitActuatorCfg(
    #         joint_names_expr=["panda_finger_joint.*"],
    #         effort_limit_sim=200.0,
    #         velocity_limit_sim=0.04,
    #         stiffness=2e3,
    #         damping=1e2,
    #     ),
    # },
    actuators={
        # UR5-style individual joint tuning for Franka
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-7]"],
            stiffness={
                "panda_joint1": 4.63,   # Base rotation - similar to shoulder_pan
                "panda_joint2": 5.41,   # Shoulder lift - similar to shoulder_lift  
                "panda_joint3": 8.06,   # Elbow rotation - similar to elbow
                "panda_joint4": 7.28,   # Elbow bend - similar to wrist_1
                "panda_joint5": 8.04,   # Wrist rotation - similar to wrist_2
                "panda_joint6": 7.18,   # Wrist bend - similar to wrist_3
                "panda_joint7": 4.63,   # Flange rotation - similar to shoulder_pan
            },
            damping={
                "panda_joint1": 8.84,   # Base rotation
                "panda_joint2": 6.47,   # Shoulder lift
                "panda_joint3": 9.46,   # Elbow rotation
                "panda_joint4": 2.80,   # Elbow bend
                "panda_joint5": 2.41,   # Wrist rotation
                "panda_joint6": 1.90,   # Wrist bend
                "panda_joint7": 8.84,   # Flange rotation
            },
            velocity_limit_sim=3.14,  # UR5 velocity limit
            effort_limit_sim={
                "panda_joint1": 87.0,   # Franka's actual effort limit for joint 1
                "panda_joint2": 87.0,   # Franka's actual effort limit for joint 2
                "panda_joint3": 87.0,   # Franka's actual effort limit for joint 3
                "panda_joint4": 87.0,   # Franka's actual effort limit for joint 4
                "panda_joint5": 12.0,   # Franka's actual effort limit for joint 5
                "panda_joint6": 12.0,   # Franka's actual effort limit for joint 6
                "panda_joint7": 12.0,   # Franka's actual effort limit for joint 7
            },
            armature=0.01,  # UR5 armature value
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.04,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)




# # """Configuration of Franka Emika Panda robot."""


FRANKA_PANDA_HIGH_PD_CFG = FRANKA_PANDA_CFG.copy()
# # FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
# # FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 50000.0
# # FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 1000.0
# # FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 50000.0
# # FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 1000.0
# # """Configuration of Franka Emika Panda robot with stiffer PD control.

# # This configuration is useful for task-space control using differential IK.
# # """


FRANKA_ROBOTIQ_GRIPPER_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"


FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.variants = {"Gripper": "Robotiq_2F_85"}
FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "finger_joint": 0.0,
    ".*_inner_finger_joint": 0.0,
    ".*_inner_finger_knuckle_joint": 0.0,
    ".*_outer_.*_joint": 0.0,
}
FRANKA_ROBOTIQ_GRIPPER_CFG.init_state.pos = (-0.85, 0, 0.76)
FRANKA_ROBOTIQ_GRIPPER_CFG.actuators = {
    "panda_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit_sim=5200.0,
        velocity_limit_sim=2.175,
        stiffness=1100.0,
        damping=80.0,
    ),
    "panda_forearm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        effort_limit_sim=720.0,
        velocity_limit_sim=2.61,
        stiffness=1000.0,
        damping=80.0,
    ),
    "gripper_drive": ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],  # "right_outer_knuckle_joint" is its mimic joint
        effort_limit_sim=1650,
        velocity_limit_sim=10.0,
        stiffness=17,
        damping=0.02,
    ),
    # enable the gripper to grasp in a parallel manner
    "gripper_finger": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_joint"],
        effort_limit_sim=50,
        velocity_limit_sim=10.0,
        stiffness=0.2,
        damping=0.001,
    ),
    # set PD to zero for passive joints in close-loop gripper
    "gripper_passive": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
        effort_limit_sim=1.0,
        velocity_limit_sim=10.0,
        stiffness=0.0,
        damping=0.0,
    ),
}


"""Configuration of Franka Emika Panda robot with Robotiq_2f_85 gripper."""


##
# Configuration - Franka Panda with Robotiq 2F-85 Gripper (Custom Pre-Assembled USD)
##

# NOTE: This configuration uses a local pre-assembled Franka + Robotiq 2F-85 USD file.
# The cloud variant {"Gripper": "Robotiq_2F_85"} is broken due to missing configuration files.
# See GitHub issue: https://github.com/isaac-sim/IsaacLab/issues/1299
# This uses a pre-assembled USD with proper joint structure based on the community solution.

# FRANKA_ROBOTIQ_GRIPPER_CUSTOM_CFG = FRANKA_PANDA_CFG.copy()
# FRANKA_ROBOTIQ_GRIPPER_CUSTOM_CFG.spawn.usd_path = "/home/jason/IsaacLab/Franka/Collected_franka_robotiq/franka_robotiq.usd"
# FRANKA_ROBOTIQ_GRIPPER_CUSTOM_CFG.spawn.variants = None  # Pre-assembled file, no variants needed
# FRANKA_ROBOTIQ_GRIPPER_CUSTOM_CFG.spawn.rigid_props.disable_gravity = True
# FRANKA_ROBOTIQ_GRIPPER_CUSTOM_CFG.init_state.joint_pos = {
#     "panda_joint1": 0.0,
#     "panda_joint2": -0.569,
#     "panda_joint3": 0.0,
#     "panda_joint4": -2.810,
#     "panda_joint5": 0.0,
#     "panda_joint6": 3.037,
#     "panda_joint7": 0.741,
#     # Robotiq 2F-85 gripper joints (pre-assembled USD structure from GitHub #1299)
#     # The local USD has: outer_knuckle_joints (main actuated) + inner_finger_joints (passive)
#     ".*_outer_knuckle_joint": 0.0,      # Main actuated joints - open position
#     ".*_inner_finger_joint": 0.0,       # Passive joints
# }
# FRANKA_ROBOTIQ_GRIPPER_CUSTOM_CFG.init_state.pos = (-0.85, 0, 0.76)
# FRANKA_ROBOTIQ_GRIPPER_CUSTOM_CFG.actuators = {
#     "panda_shoulder": ImplicitActuatorCfg(
#         joint_names_expr=["panda_joint[1-4]"],
#         effort_limit_sim=5200.0,
#         velocity_limit_sim=2.175,
#         stiffness=1100.0,
#         damping=80.0,
#     ),
#     "panda_forearm": ImplicitActuatorCfg(
#         joint_names_expr=["panda_joint[5-7]"],
#         effort_limit_sim=720.0,
#         velocity_limit_sim=2.61,
#         stiffness=1000.0,
#         damping=80.0,
#     ),
#     # Robotiq 2F-85 gripper - based on GitHub issue #1299 solution
#     # The pre-assembled USD uses outer_knuckle_joints as the main actuated joints
#     "robotiq_gripper": ImplicitActuatorCfg(
#         joint_names_expr=[".*_outer_knuckle_joint"],  # Main actuated joints (left + right)
#         effort_limit_sim=200.0,
#         velocity_limit_sim=0.2,
#         stiffness=2e3,
#         damping=1e2,
#     ),
# }


"""
Configuration of Franka Emika Panda robot with Robotiq 2F-85 gripper using local pre-assembled USD.

Joint Structure (9 named joints):
- 7 arm joints: panda_joint1 to panda_joint7
- 2 main gripper joints: left_outer_knuckle_joint, right_outer_knuckle_joint (actuated)
- 2 passive finger joints: left_inner_finger_joint, right_inner_finger_joint

Gripper Control:
- Open: outer_knuckle_joint = 0.0
- Closed: outer_knuckle_joint = 0.8
"""


##
# Configuration - Franka Panda with Robotiq 2F-85 Gripper (Omniverse Style - Robot Agnostic)
##

# Base articulation configuration (no actuators yet - robot agnostic)
FRANKA_ROBOTIQ_BASE_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="./nvidia_assets/Franka/Collected_franka_robotiq/franka_robotiq.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=36,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.85, 0, 0.76),
        rot=(1, 0, 0, 0),
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            ".*_outer_knuckle_joint": 0.0,  # Open position
            ".*_inner_finger_joint": 0.0,   # Passive joints
        },
    ),
    soft_joint_pos_limit_factor=1.0,
)

# Complete configuration with actuators (robot-specific tuning)
FRANKA_ROBOTIQ_GRIPPER_CUSTOM_OMNI_CFG = FRANKA_ROBOTIQ_BASE_ARTICULATION.copy()
FRANKA_ROBOTIQ_GRIPPER_CUSTOM_OMNI_CFG.actuators = {
    # Arm actuators with per-joint tuning (robot-specific parameters)
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint.*"],  # All 7 arm joints
        stiffness={
            "panda_joint1": 1100.0,   # Base rotation - tuned for Franka dynamics
            "panda_joint2": 1100.0,   # Shoulder lift
            "panda_joint3": 1100.0,   # Elbow rotation
            "panda_joint4": 1100.0,   # Elbow bend
            "panda_joint5": 1000.0,   # Wrist rotation
            "panda_joint6": 1000.0,   # Wrist bend
            "panda_joint7": 1100.0,   # Flange rotation
        },
        damping={
            "panda_joint1": 80.0,
            "panda_joint2": 80.0,
            "panda_joint3": 80.0,
            "panda_joint4": 80.0,
            "panda_joint5": 80.0,
            "panda_joint6": 80.0,
            "panda_joint7": 80.0,
        },
        # velocity_limit_sim=3.14,  # rad/s
        velocity_limit_sim={
            "panda_joint1": 2.175,
            "panda_joint2": 2.175,
            "panda_joint3": 2.175,
            "panda_joint4": 2.175,
            "panda_joint5": 2.61,
            "panda_joint6": 2.61,
            "panda_joint7": 2.61,
        },
        effort_limit_sim={
            "panda_joint1": 5200.0,   # Franka's actual effort limits
            "panda_joint2": 5200.0,
            "panda_joint3": 5200.0,
            "panda_joint4": 5200.0,
            "panda_joint5": 720.0,
            "panda_joint6": 720.0,
            "panda_joint7": 720.0,
        },
        armature=0.01,
    ),
    # Robotiq gripper actuators (pre-assembled USD uses outer_knuckle_joints)
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=[".*_outer_knuckle_joint"],  # Main actuated joints
        effort_limit_sim=200.0,
        velocity_limit_sim=0.2,
        stiffness=2e3,
        damping=1e2,
    ),
    # Passive inner finger joints (compliant, minimal control)
    "inner_finger": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_joint"],
        effort_limit_sim=50.0,
        velocity_limit_sim=10.0,
        stiffness=0.2,
        damping=0.001,
    ),
}


"""
Configuration of Franka Emika Panda robot with Robotiq 2F-85 gripper (Omniverse/UR5E style).

This configuration follows the robot-agnostic pattern:
1. Base articulation (spawn, init_state) - robot-independent structure
2. Actuators (stiffness, damping, limits) - robot-specific tuning

Key Features:
- Per-joint stiffness and damping tuning
- Realistic effort limits based on Franka specifications
- Separate actuator groups: arm, gripper, inner_finger
- Uses local pre-assembled USD (GitHub #1299)

Joint Structure (13 total joints):
- 7 arm joints: panda_joint1 to panda_joint7
- 2 main gripper joints: left_outer_knuckle_joint, right_outer_knuckle_joint (actuated)
- 2 inner finger joints: left_inner_finger_joint, right_inner_finger_joint (passive/compliant)
- 2 unnamed joints: RevoluteJoint, RevoluteJoint_0 (passive)

Gripper Control:
- Open: outer_knuckle_joint = 0.0
- Closed: outer_knuckle_joint = 0.8
"""

FRANKA_ROBOTIQ_GRIPPER_CUSTOM_OMNI_PAT_CFG = FRANKA_ROBOTIQ_BASE_ARTICULATION.copy()
FRANKA_ROBOTIQ_GRIPPER_CUSTOM_OMNI_PAT_CFG.actuators = {
    # Arm actuators with per-joint tuning (robot-specific parameters)
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint.*"],  # All 7 arm joints
        stiffness={
            "panda_joint1": 4.63,   # Base rotation - adapted from shoulder_pan
            "panda_joint2": 5.41,   # Shoulder lift - adapted from shoulder_lift
            "panda_joint3": 8.06,   # Elbow rotation - adapted from elbow
            "panda_joint4": 7.28,   # Elbow bend - adapted from wrist_1
            "panda_joint5": 8.04,   # Wrist rotation - adapted from wrist_2
            "panda_joint6": 7.18,   # Wrist bend - adapted from wrist_3
            "panda_joint7": 4.63,   # Flange rotation - adapted from shoulder_pan
        },
        damping={
            "panda_joint1": 8.84,
            "panda_joint2": 6.47,
            "panda_joint3": 9.46,
            "panda_joint4": 2.80,
            "panda_joint5": 2.41,
            "panda_joint6": 1.90,
            "panda_joint7": 8.84,
        },
        velocity_limit_sim=3.14,  # rad/s (same as UR5E)
        effort_limit_sim={
            "panda_joint1": 87.0,   # Franka's effort limits
            "panda_joint2": 87.0,
            "panda_joint3": 87.0,
            "panda_joint4": 87.0,
            "panda_joint5": 12.0,
            "panda_joint6": 12.0,
            "panda_joint7": 12.0,
        },
    ), 
    # Robotiq gripper actuators (pre-assembled USD uses outer_knuckle_joints)
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=[".*_outer_knuckle_joint"],  # Main actuated joints
        effort_limit_sim=200.0,
        velocity_limit_sim=0.2,
        stiffness=2e3,
        damping=1e2,
    ),
    # Passive inner finger joints (compliant, minimal control)
    "inner_finger": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_joint"],
        effort_limit_sim=50.0,
        velocity_limit_sim=10.0,
        stiffness=0.2,
        damping=0.001,
    ),
}
