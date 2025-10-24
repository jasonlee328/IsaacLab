# Copyright (c) 2024-2025, The ISAAC Lab Project Developers.
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from . import commands


@configclass
class BlocksPoseCommandCfg(CommandTermCfg):
    """Configuration for blocks pose command generator with relative targets and exclusion zones."""

    class_type: type = commands.BlocksPoseCommand

    # Asset references
    asset_name: str = MISSING
    """Name of the coordinate referencing asset (robot) in the environment."""

    object_name: str = MISSING
    """Name of the object in the environment for which commands are generated."""

    # Command behavior
    position_only: bool = True
    """If True, only position commands are generated. If False, both position and orientation."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique by ensuring positive real part."""

    relative_orientation: bool = False
    """If True, orientation commands are relative to object's current orientation. 
    If False, orientation commands are absolute in robot base frame."""

    # Success thresholds
    success_position_threshold: float = 0.05
    """Position error threshold for considering the target reached (in meters)."""

    success_orientation_threshold: float = 0.2
    """Orientation error threshold for considering alignment achieved (in radians).
    This is the sum of absolute roll and pitch errors."""

    @configclass
    class SliceCounts:
        """Optional discrete slicing for each axis."""
        
        pos_x: int | None = None
        """Number of discrete slices for x position. None means continuous sampling."""
        
        pos_y: int | None = None
        """Number of discrete slices for y position. None means continuous sampling."""
        
        pos_z: int | None = None
        """Number of discrete slices for z position. None means continuous sampling."""
        
        roll: int | None = None
        """Number of discrete slices for roll angle. None means continuous sampling."""
        
        pitch: int | None = None
        """Number of discrete slices for pitch angle. None means continuous sampling."""
        
        yaw: int | None = None
        """Number of discrete slices for yaw angle. None means continuous sampling."""

    @configclass
    class Ranges:
        """Relative distribution ranges for pose commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for relative x position offset (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for relative y position offset (in m)."""

        pos_z: tuple[float, float] = MISSING
        """Range for relative z position offset (in m)."""

        roll: tuple[float, float] = (0.0, 0.0)
        """Range for roll angle (in rad). Only used if position_only=False."""

        pitch: tuple[float, float] = (0.0, 0.0)
        """Range for pitch angle (in rad). Only used if position_only=False."""

        yaw: tuple[float, float] = (0.0, 0.0)
        """Range for yaw angle (in rad). Only used if position_only=False."""

    @configclass
    class ExclusionRanges:
        """Exclusion zones within the sampling ranges.
        
        These define "forbidden zones" where targets cannot be placed.
        For example, if pos_x range is (-0.2, 0.2) and exclusion is (-0.01, 0.01),
        then x can be sampled from [-0.2, -0.01] OR [0.01, 0.2], but not [-0.01, 0.01].
        """

        pos_x: tuple[float, float] = (0.0, 0.0)
        """Exclusion zone for x position (in m)."""

        pos_y: tuple[float, float] = (0.0, 0.0)
        """Exclusion zone for y position (in m)."""

        pos_z: tuple[float, float] = (0.0, 0.0)
        """Exclusion zone for z position (in m)."""

        roll: tuple[float, float] = (0.0, 0.0)
        """Exclusion zone for roll angle (in rad)."""

        pitch: tuple[float, float] = (0.0, 0.0)
        """Exclusion zone for pitch angle (in rad)."""

        yaw: tuple[float, float] = (0.0, 0.0)
        """Exclusion zone for yaw angle (in rad)."""

    ranges: Ranges = MISSING
    """Ranges for the relative commands."""

    exclusion_ranges: ExclusionRanges = ExclusionRanges()
    """Exclusion zones within the ranges. Defaults to no exclusion."""
    
    slice_counts: SliceCounts = SliceCounts()
    """Optional per-axis discrete slicing. Defaults to continuous sampling for all axes."""