import hashlib
import io
import logging
import numpy as np
import os
import random
import tempfile
import torch
import trimesh
import yaml
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from functools import lru_cache

import isaaclab.utils.math as math_utils
import isaacsim.core.utils.torch as torch_utils
import omni
import warp as wp
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.warp import convert_to_warp_mesh

def read_metadata_from_usd_directory(usd_path: str) -> dict:
    """Read metadata from metadata.yaml in the same directory as the USD file."""
    # Get the directory containing the USD file
    usd_dir = os.path.dirname(usd_path)

    # Look for metadata.yaml in the same directory
    metadata_path = os.path.join(usd_dir, "metadata.yaml")
    rank = int(os.getenv("RANK", "0"))
    download_dir = os.path.join(tempfile.gettempdir(), f"rank_{rank}")
    with open(retrieve_file_path(metadata_path, download_dir=download_dir)) as f:
        metadata_file = yaml.safe_load(f)

    return metadata_file