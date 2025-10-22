# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import pose_commands as dex_cmd
from . import relative_pose_commands as rel_cmd

ALIGN_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.1, 0.1, 0.1),
        ),
        "position_far": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "position_near": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    }
)


@configclass
class ObjectUniformPoseCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = dex_cmd.ObjectUniformPoseCommand

    asset_name: str = MISSING
    """Name of the coordinate referencing asset in the environment for which the commands are generated respect to."""

    object_name: str = MISSING
    """Name of the object in the environment for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""

        roll: tuple[float, float] = MISSING
        """Range for the roll angle (in rad)."""

        pitch: tuple[float, float] = MISSING
        """Range for the pitch angle (in rad)."""

        yaw: tuple[float, float] = MISSING
        """Range for the yaw angle (in rad)."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

    position_only: bool = True
    """Command goal position only. Command includes goal quat if False"""

    # Pose Markers
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = ALIGN_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    curr_pose_visualizer_cfg: VisualizationMarkersCfg = ALIGN_MARKER_CFG.replace(prim_path="/Visuals/Command/body_pose")
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    success_vis_asset_name: str = MISSING
    """Name of the asset in the environment for which the success color are indicated."""

    # success markers
    success_visualizer_cfg = VisualizationMarkersCfg(prim_path="/Visuals/SuccessMarkers", markers={})
    """The configuration for the success visualization marker. User needs to add the markers"""


@configclass
class ObjectRelativePoseCommandCfg(CommandTermCfg):
    """Configuration for relative pose command generator.
    
    This configuration generates target poses relative to the object's current position,
    which is useful for tasks like pushing where you want to avoid the object spawning
    on top of the target.
    """

    class_type: type = rel_cmd.ObjectRelativePoseCommand

    asset_name: str = "robot"
    """Name of the coordinate referencing asset in the environment for which the commands are generated respect to."""

    object_name: str = MISSING
    """Name of the object in the environment for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    @configclass
    class Ranges:
        """Relative distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the relative x position offset (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the relative y position offset (in m)."""

        pos_z: tuple[float, float] = MISSING
        """Range for the relative z position offset (in m)."""

        roll: tuple[float, float] = MISSING
        """Range for the roll angle (in rad)."""

        pitch: tuple[float, float] = MISSING
        """Range for the pitch angle (in rad)."""

        yaw: tuple[float, float] = MISSING
        """Range for the yaw angle (in rad)."""

    ranges: Ranges = MISSING
    """Ranges for the relative commands."""

    position_only: bool = True
    """Command goal position only. Command includes goal quat if False"""

    min_distance: float = 0.0
    """Minimum distance between object and target (in m). 
    
    Set this to your success threshold to ensure the object never spawns already in success.
    For example, if success threshold is 0.10m, set min_distance to 0.10m (or slightly higher).
    """

    # Pose Markers
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = ALIGN_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_pose_relative"
    )
    """The configuration for the goal pose visualization marker. Defaults to ALIGN_MARKER_CFG."""

    curr_pose_visualizer_cfg: VisualizationMarkersCfg = ALIGN_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose_relative"
    )
    """The configuration for the current pose visualization marker. Defaults to ALIGN_MARKER_CFG."""
