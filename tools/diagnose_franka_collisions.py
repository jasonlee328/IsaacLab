#!/usr/bin/env python3
"""Diagnostic script to check if Franka collision meshes are accessible in the stage."""

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
from pxr import Usd, UsdPhysics, UsdGeom

# Create stage
from isaaclab.sim import SimulationContext
sim = SimulationContext()

# Spawn Franka robot
from isaaclab_assets.robots.franka import FRANKA_ROBOTIQ_GRIPPER_CUSTOM_OMNI_PAT_CFG
robot_cfg = FRANKA_ROBOTIQ_GRIPPER_CUSTOM_OMNI_PAT_CFG.copy()
robot_cfg.prim_path = "/World/Robot"

# Spawn the robot
robot_cfg.spawn.func("/World/Robot", robot_cfg.spawn)

# Play simulation to ensure everything is loaded
sim.reset()

# Get the stage
stage = sim_utils.get_current_stage()

# Check ALL body links
body_link_names = [
    "panda_link0", "panda_link1", "panda_link2", "panda_link3",
    "panda_link4", "panda_link5", "panda_link6", "panda_link7",
    "panda_hand", "panda_leftfinger", "panda_rightfinger",
    "left_outer_knuckle", "right_outer_knuckle",
    "left_inner_finger", "right_inner_finger"
]

total_collision_meshes = 0
links_with_collisions = []

for body_name in body_link_names:
    link_prim_path = f"/World/Robot/{body_name}"
    link_prim = stage.GetPrimAtPath(link_prim_path)
    
    # Skip if prim doesn't exist
    if not link_prim.IsValid():
        continue
    
    # Try get_all_matching_child_prims (the function used by collision analyzer)
    collision_prims = sim_utils.get_all_matching_child_prims(
        link_prim_path,
        predicate=lambda p: p.GetTypeName() in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone")
        and p.HasAPI(UsdPhysics.CollisionAPI),
    )
    
    num_collisions = len(collision_prims)
    total_collision_meshes += num_collisions
    
    if num_collisions > 0:
        links_with_collisions.append((body_name, num_collisions))
        print(f"{body_name}: {num_collisions} collision mesh(es)")

print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"Total body links checked: {len(body_link_names)}")
print(f"Links with collision meshes: {len(links_with_collisions)}")
print(f"Total collision meshes found: {total_collision_meshes}")
print(f"\nLinks with collisions:")
for link_name, count in links_with_collisions:
    print(f"  - {link_name}: {count} mesh(es)")
print(f"\n{'='*80}\n")

# Close
sim.stop()
app.close()

