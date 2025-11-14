#!/usr/bin/env python3
"""Check red_block.usd dimensions using USD tools."""

# Initialize Isaac Sim first
import isaaclab.app
import argparse

parser = argparse.ArgumentParser(description="Check red block dimensions")
parser.add_argument("--headless", action="store_true", default=True, help="Run headless")
args_cli = parser.parse_args()

app_launcher = isaaclab.app.AppLauncher(args_cli)
simulation_app = app_launcher.app
from pxr import UsdGeom, Usd, Gf
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import numpy as np

# Path to red block
red_block_path = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"

print(f"Checking USD file: {red_block_path}")
print("="*70)

# Open the USD stage
stage = Usd.Stage.Open(red_block_path)

if not stage:
    print("ERROR: Could not open USD stage!")
    exit(1)

# Get the root prim
root_prim = stage.GetDefaultPrim()
if not root_prim:
    # Try getting the first valid prim
    for prim in stage.Traverse():
        if prim.IsValid():
            root_prim = prim
            break

print(f"Root prim: {root_prim.GetPath()}")

# Create bounding box cache
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])

# Compute bounding box
bbox = bbox_cache.ComputeWorldBound(root_prim)
range_bbox = bbox.GetRange()
unscaled_size = np.asarray(range_bbox.GetSize())

print(f"\nUnscaled dimensions:")
print(f"  X: {unscaled_size[0]:.6f} m ({unscaled_size[0]*100:.2f} cm)")
print(f"  Y: {unscaled_size[1]:.6f} m ({unscaled_size[1]*100:.2f} cm)")
print(f"  Z: {unscaled_size[2]:.6f} m ({unscaled_size[2]*100:.2f} cm)")

# Get scale from transform
xformable = UsdGeom.Xformable(root_prim)
wt = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
scale_x = wt.GetColumn(0).GetLength()
scale_y = wt.GetColumn(1).GetLength()
scale_z = wt.GetColumn(2).GetLength()
scale = np.array([scale_x, scale_y, scale_z])

print(f"\nScale factors: {scale}")

# Final scaled size
final_size = unscaled_size * scale

print(f"\nFinal scaled dimensions:")
print(f"  X: {final_size[0]:.6f} m ({final_size[0]*100:.2f} cm)")
print(f"  Y: {final_size[1]:.6f} m ({final_size[1]*100:.2f} cm)")
print(f"  Z: {final_size[2]:.6f} m ({final_size[2]*100:.2f} cm)")

print("\n" + "="*70)
print(f"RED BLOCK SIZE: {final_size[0]*100:.2f} cm × {final_size[1]*100:.2f} cm × {final_size[2]*100:.2f} cm")
print("="*70)

# Also print min/max corners
min_corner = range_bbox.GetMin()
max_corner = range_bbox.GetMax()
print(f"\nBounding box corners:")
print(f"  Min: ({min_corner[0]:.6f}, {min_corner[1]:.6f}, {min_corner[2]:.6f})")
print(f"  Max: ({max_corner[0]:.6f}, {max_corner[1]:.6f}, {max_corner[2]:.6f})")

