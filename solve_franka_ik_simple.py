#!/usr/bin/env python3
"""Solve IK for Franka arm using TracIK."""

import argparse
import numpy as np

import isaaclab.app
parser = argparse.ArgumentParser(description="Solve Franka IK")
parser.add_argument("--headless", action="store_true", default=True, help="Run headless")
args_cli = parser.parse_args()

app_launcher = isaaclab.app.AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import Isaac Sim modules
try:
    from trac_ik_python.trac_ik import IK
    TRAC_IK_AVAILABLE = True
except ImportError:
    TRAC_IK_AVAILABLE = False
    print("WARNING: trac_ik_python not available, will use alternative method")

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_ROBOTIQ_GRIPPER_CFG

@configclass
class IKSceneCfg(InteractiveSceneCfg):
    """Configuration for IK solving scene."""
    robot = FRANKA_ROBOTIQ_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def solve_ik_trac(target_pos, target_quat, q_init):
    """Solve IK using TracIK."""
    # Get robot URDF - assuming Franka Panda
    base_link = "panda_link0"
    tip_link = "panda_hand"
    
    # You might need to provide the URDF path
    # For now, let's use the standard Franka URDF
    import os
    urdf_path = None
    
    # Try to find URDF
    possible_paths = [
        "/home/jason/IsaacLab/source/isaaclab_assets/data/robots/franka/franka.urdf",
        "/home/jason/.local/share/ov/pkg/isaac-sim-*/exts/omni.isaac.franka/data/franka_description/robots/franka_panda.urdf",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            urdf_path = path
            break
    
    if urdf_path is None:
        print("ERROR: Could not find Franka URDF file")
        return None
    
    print(f"Using URDF: {urdf_path}")
    
    # Create IK solver
    ik_solver = IK(base_link, tip_link, urdf_string=open(urdf_path).read())
    
    # Solve IK
    # TracIK expects: qinit, x, y, z, rx, ry, rz, rw
    # Where rx, ry, rz, rw is the quaternion (x, y, z, w format)
    solution = ik_solver.get_ik(
        q_init,
        target_pos[0], target_pos[1], target_pos[2],
        target_quat[1], target_quat[2], target_quat[3], target_quat[0]  # Convert w,x,y,z to x,y,z,w
    )
    
    return solution


def solve_ik_iterative(scene, robot, target_pos, target_quat, q_init, max_iterations=100):
    """Solve IK iteratively using Jacobian."""
    import torch
    
    sim = SimulationContext.instance()
    
    # Reset simulation first to initialize everything
    sim.reset()
    
    # Now set initial joint positions
    full_joint_pos = torch.tensor([q_init], device=sim.device, dtype=torch.float32)
    robot.write_joint_state_to_sim(full_joint_pos, torch.zeros_like(full_joint_pos))
    
    # Step once to apply
    scene.update(sim.get_physics_dt())
    sim.step()
    
    # Find end-effector frame
    ee_frame_name = "panda_hand"
    body_names = robot.data.body_names
    
    if ee_frame_name not in body_names:
        for name in ["panda_hand", "panda_leftfinger", "panda_link8"]:
            if name in body_names:
                ee_frame_name = name
                break
    
    ee_frame_idx = body_names.index(ee_frame_name)
    print(f"Using EE frame: {ee_frame_name} (index: {ee_frame_idx})")
    
    # Convert target to torch
    target_pos_torch = torch.tensor([target_pos], device=sim.device, dtype=torch.float32)
    target_quat_torch = torch.tensor([target_quat], device=sim.device, dtype=torch.float32)
    
    # Iterative IK with better parameters
    tolerance = 0.002  # 2mm
    step_size = 1.0  # Larger step size
    
    for i in range(max_iterations):
        # Update scene
        scene.update(sim.get_physics_dt())
        
        # Get current EE pose
        ee_pos = robot.data.body_pos_w[:, ee_frame_idx, :]
        ee_quat = robot.data.body_quat_w[:, ee_frame_idx, :]
        
        # Compute position error
        pos_error = target_pos_torch - ee_pos
        error_norm = torch.norm(pos_error).item()
        
        if i % 10 == 0:
            print(f"Iteration {i}: error = {error_norm:.6f} m")
        
        if error_norm < tolerance:
            print(f"Converged in {i} iterations!")
            return robot.data.joint_pos[0, :7].cpu().numpy()
        
        # Get Jacobian (only for first 7 joints)
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_frame_idx, :, :7]
        
        # Use only position part of Jacobian (first 3 rows)
        jac_pos = jacobian[:, :3, :]  # (1, 3, 7)
        
        # Use damped pseudo-inverse method (more stable)
        # J# = J^T * (J*J^T + lambda^2*I)^-1
        jac_T = jac_pos.transpose(1, 2)  # (1, 7, 3)
        jjt = torch.bmm(jac_pos, jac_T)  # (1, 3, 3)
        
        # Adaptive damping based on error
        lambda_val = 0.01 + 0.1 * min(error_norm, 1.0)
        damping = lambda_val ** 2 * torch.eye(3, device=sim.device).unsqueeze(0)
        jjt_damped = jjt + damping
        
        # Compute damped pseudo-inverse
        try:
            jjt_inv = torch.linalg.inv(jjt_damped)
            j_pinv = torch.bmm(jac_T, jjt_inv)  # (1, 7, 3)
            
            # Compute delta_q
            delta_q = torch.bmm(j_pinv, pos_error.unsqueeze(2)).squeeze(2)  # (1, 7)
            
            # Limit step size
            delta_norm = torch.norm(delta_q).item()
            if delta_norm > 0.5:  # Limit to 0.5 rad per step
                delta_q = delta_q * (0.5 / delta_norm)
            
            # Update joint positions (only first 7)
            current_q = robot.data.joint_pos.clone()
            current_q[:, :7] += delta_q * step_size
            
            # Clamp to joint limits (approximate Franka limits)
            joint_limits_lower = torch.tensor([[-2.9, -1.76, -2.9, -3.07, -2.9, -0.02, -2.9]], device=sim.device)
            joint_limits_upper = torch.tensor([[2.9, 1.76, 2.9, -0.07, 2.9, 3.75, 2.9]], device=sim.device)
            current_q[:, :7] = torch.clamp(current_q[:, :7], joint_limits_lower, joint_limits_upper)
            
            # Write to simulation
            robot.write_joint_state_to_sim(current_q, torch.zeros_like(current_q))
            sim.step()
            
        except Exception as e:
            print(f"Warning: Matrix inversion failed at iteration {i}: {e}")
            break
    
    print(f"Did not converge after {max_iterations} iterations")
    return robot.data.joint_pos[0, :7].cpu().numpy()


def main():
    """Solve IK for target position."""
    
    # Target position and orientation
    target_pos = np.array([0.625, 0.0, 0.55])
    target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
    
    print("="*70)
    print(f"Target Position: {target_pos}")
    print(f"Target Orientation (quat w,x,y,z): {target_quat}")
    print("="*70)
    
    # Initial joint configuration (start from home pose)
    q_init = [
        0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785,  # 7 arm joints (home pose)
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # gripper joints
    ]

    solution = None
    
    # Try TracIK first
    if TRAC_IK_AVAILABLE:
        print("\nAttempting IK solution using TracIK...")
        solution = solve_ik_trac(target_pos, target_quat, q_init[:7])
        
        if solution is not None:
            print("\n" + "="*70)
            print("SOLUTION (TracIK):")
            print("="*70)
            for i, angle in enumerate(solution):
                print(f"  Joint {i}: {angle:.4f} rad ({np.degrees(angle):.2f}°)")
            
            print(f"\nPython array format:")
            print(f"[{', '.join([f'{x:.4f}' for x in solution])}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]")
            print("="*70)
    
    # Fallback to iterative IK if TracIK failed or unavailable
    if solution is None:
        print("\nAttempting IK solution using iterative Jacobian method...")
        
        # Create simulation
        sim_cfg = sim_utils.SimulationCfg(dt=0.01)
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
        
        # Add ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
        
        # Create scene
        scene_cfg = IKSceneCfg(num_envs=1, env_spacing=2.0)
        scene = InteractiveScene(scene_cfg)
        robot = scene["robot"]
        
        solution = solve_ik_iterative(scene, robot, target_pos, target_quat, q_init, max_iterations=500)
        
        print("\n" + "="*70)
        print("SOLUTION (Iterative):")
        print("="*70)
        for i, angle in enumerate(solution):
            print(f"  Joint {i}: {angle:.4f} rad ({np.degrees(angle):.2f}°)")
        
        print(f"\nPython array format:")
        print(f"[{', '.join([f'{x:.4f}' for x in solution])}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]")
        print("="*70)
    
    simulation_app.close()


if __name__ == "__main__":
    main()

