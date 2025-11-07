# Copyright (c) 2024-2025, The Octi Lab Project Developers. (https://github.com/zoctipus/OctiLab/blob/main/CONTRIBUTORS.md).
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

"""Utility functions for path handling in IsaacLab."""

import os
from pathlib import Path


def get_isaaclab_root() -> Path:
    """Get the IsaacLab root directory.

    This function finds the IsaacLab root directory by looking for the IsaacLab files
    or by traversing up from the current file location.

    Returns:
        Path: The absolute path to the IsaacLab root directory.

    Raises:
        RuntimeError: If the IsaacLab root directory cannot be found.
    """
    # First, try to find IsaacLab root in the current working directory or parent directories
    current_path = Path.cwd()
    for parent in [current_path] + list(current_path.parents):
        isaaclab_sh = parent / ".isaaclab"
        if isaaclab_sh.exists():
            return parent
    
    # If not found, try to find it relative to this file's location
    # This file is in source/isaaclab/isaaclab/utils/paths.py
    # So we need to go up 4 levels to get to the root
    current_file = Path(__file__).resolve()
    isaaclab_root = current_file.parents[4]  # Go up 4 levels: utils -> isaaclab -> isaaclab -> source -> root
    
    # Last resort: try environment variable
    isaaclab_root_env = os.environ.get("ISAACLAB_ROOT")
    if isaaclab_root_env:
        return Path(isaaclab_root_env)
    
    return isaaclab_root


def get_isaaclab_assets_path() -> Path:
    """Get the path to the IsaacLab assets directory.

    Returns:
        Path: The absolute path to the IsaacLab assets directory.
    """
    return get_isaaclab_root() / "source" / "isaaclab_assets" / "isaaclab_assets"


def get_isaaclab_data_path() -> Path:
    """Get the path to the IsaacLab data directory.

    Returns:
        Path: The absolute path to the IsaacLab data directory.
    """
    return get_isaaclab_root() / "data_storage"


def get_isaaclab_grasp_datasets_path() -> Path:
    """Get the path to the grasp datasets directory.
    
    Returns:
        Path: The absolute path to the grasp datasets directory.
    """
    return get_isaaclab_root() / "grasp_datasets"


def get_isaaclab_reset_state_datasets_path() -> Path:
    """Get the path to the reset state datasets directory.
    
    Returns:
        Path: The absolute path to the reset state datasets directory.
    """
    return get_isaaclab_root() / "reset_state_datasets"


def get_isaaclab_partial_assembly_datasets_path() -> Path:
    """Get the path to the partial assembly datasets directory.
    
    Returns:
        Path: The absolute path to the partial assembly datasets directory.
    """
    return get_isaaclab_root() / "partial_assembly_datasets"


# Temporary compatibility aliases
def get_octilab_assets_path() -> Path:
    """Deprecated: Use get_isaaclab_assets_path instead."""
    return get_isaaclab_assets_path()


def get_octilab_data_path() -> Path:
    """Deprecated: Use get_isaaclab_data_path instead."""
    return get_isaaclab_data_path()


def get_octilab_grasp_datasets_path() -> Path:
    """Deprecated: Use get_isaaclab_grasp_datasets_path instead."""
    return get_isaaclab_grasp_datasets_path()


def get_octilab_reset_state_datasets_path() -> Path:
    """Deprecated: Use get_isaaclab_reset_state_datasets_path instead."""
    return get_isaaclab_reset_state_datasets_path()


def get_octilab_partial_assembly_datasets_path() -> Path:
    """Deprecated: Use get_isaaclab_partial_assembly_datasets_path instead."""
    return get_isaaclab_partial_assembly_datasets_path()
