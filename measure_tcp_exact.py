#!/usr/bin/env python3
"""
Measure exact TCP distance from USD file.
Run with: ./isaaclab.sh -p measure_tcp_exact.py --headless
"""

import argparse
from isaaclab.app import AppLauncher

# Create argument parser
parser = argparse.ArgumentParser(description="Measure exact TCP offset from Franka Robotiq USD file.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
from pxr import Usd, UsdGeom
import omni.usd

print("\n" + "="*80)
print("EXACT TCP MEASUREMENT FROM USD FILE")
print("="*80)

# Get the current stage
context = omni.usd.get_context()
stage = context.get_stage()

# Load the Franka+Robotiq USD
usd_path = "/home/jason/IsaacLab/nvidia_assets/Franka/Collected_franka_robotiq/franka_robotiq.usd"
print(f"\nLoading: {usd_path}")

# Create prim and reference the USD
robot_prim = stage.DefinePrim("/World/robot", "Xform")
robot_prim.GetReferences().AddReference(usd_path)

# Force stage update
context.get_stage()

print("\n" + "="*80)
print("FINDING ALL PRIMS IN ROBOT")
print("="*80)

# Find all prims
all_prims = []
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if "/World/robot" in path:
        all_prims.append(path)

# Filter for relevant prims
print("\nRobot links found:")
links = [p for p in all_prims if "link" in p.lower()]
for link in sorted(links):
    print(f"  {link}")

print("\nGripper/Robotiq prims found:")
gripper_prims = [p for p in all_prims if any(k in p.lower() for k in ["robotiq", "gripper", "finger", "tcp", "tool"])]
for prim in sorted(gripper_prims):
    print(f"  {prim}")

# Get panda_link7 transform
link7_path = None
for prim_path in all_prims:
    if "panda_link7" in prim_path.lower():
        link7_path = prim_path
        break

if not link7_path:
    print("\nERROR: panda_link7 not found!")
    simulation_app.close()
    exit(1)

print(f"\n✓ Found panda_link7: {link7_path}")

# Get the transform
link7_prim = stage.GetPrimAtPath(link7_path)
xformable = UsdGeom.Xformable(link7_prim)

# Get world transform
link7_xform = xformable.ComputeLocalToWorldTransform(0)
link7_pos = np.array(link7_xform.ExtractTranslation())

print(f"  Position: {link7_pos}")

print("\n" + "="*80)
print("MEASURING DISTANCES FROM panda_link7")
print("="*80)

distances = []
for prim_path in sorted(all_prims):
    if any(k in prim_path.lower() for k in ["robotiq", "finger", "gripper", "tcp", "tool"]):
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            xformable = UsdGeom.Xformable(prim)
            try:
                xform = xformable.ComputeLocalToWorldTransform(0)
                pos = np.array(xform.ExtractTranslation())
                dist = np.linalg.norm(pos - link7_pos)
                distances.append((prim_path, dist, pos))
                print(f"{prim_path:70s} | {dist*1000:7.2f}mm")
            except:
                pass

print("\n" + "="*80)
print("TOP 10 FURTHEST POINTS (LIKELY TCP CANDIDATES)")
print("="*80)

sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)
for i, (path, dist, pos) in enumerate(sorted_distances[:10]):
    print(f"{i+1:2d}. {path:65s} | {dist*1000:7.2f}mm")

if sorted_distances:
    max_dist = sorted_distances[0][1]
    print(f"\n{'='*80}")
    print(f"🎯 RECOMMENDED TCP OFFSET")
    print(f"{'='*80}")
    print(f"   Distance from panda_link7 to TCP: {max_dist*1000:.2f}mm")
    print(f"   ")
    print(f"   Use this in your config:")
    print(f"   offset=OffsetCfg(")
    print(f"       pos=[0.0, 0.0, {-max_dist:.4f}],  # {max_dist*1000:.1f}mm (NEGATIVE because z-axis points backward)")
    print(f"   )")
    print(f"{'='*80}\n")

simulation_app.close()

