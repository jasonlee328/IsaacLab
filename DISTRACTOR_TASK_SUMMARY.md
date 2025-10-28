# Push Task with Distractors - Implementation Summary

## Overview
The push task with distractors extends the basic push task by adding two red distractor cubes that the robot must avoid while pushing the blue target cube to its goal position.

## Key Features

### 1. Object Spawning
- **Target Cube (Blue)**: Spawns randomly at x=[0.45, 0.65], y=[-0.15, 0.15]
- **Distractor Cubes (Red)**: Two red cubes spawn in the same range
- **Minimum Separation**: All cubes maintain at least 15cm separation from each other
- **Randomization**: All cube positions are randomized together using `randomize_all_cubes` event

### 2. Target Position Generation
- Uses custom `PushObjectDistractorAwareCommand` class
- Ensures target position maintains:
  - Minimum distance from target cube (accounting for success sphere)
  - At least 15cm from all distractor cubes
- Target position range: x=[0.45, 0.65], y=[-0.15, 0.15]

### 3. Termination Conditions
- **Distractor Movement**: Episode fails if any distractor moves > 3cm from initial position
- **Time Out**: Standard episode timeout
- **Cube Falling**: If target cube falls off table

### 4. Observations
The policy receives:
- Robot joint positions and velocities
- End-effector position and orientation
- Target cube position and orientation
- Target goal position
- Distractor cube positions (6D vector)
- Distractor cube orientations (8D quaternions)

### 5. Rewards
- **Sparse Reward**: +1.0 when target cube reaches goal position (within 2cm)
- **Optional Penalty**: -0.5 * distance penalty for getting too close to distractors

## Implementation Details

### Event Configuration
```python
# All cubes randomized together with separation
self.events.randomize_all_cubes = EventTerm(
    func=franka_stack_events.randomize_object_pose,
    params={
        "pose_range": {...},
        "min_separation": 0.15,  # 15cm minimum
        "asset_cfgs": [cube, distractor_1, distractor_2]
    }
)

# Store initial distractor positions for movement detection
self.events.store_distractor_positions = EventTerm(
    func=push_mdp.store_distractor_initial_positions,
    params={...}
)
```

### Command Generation
The `PushObjectDistractorAwareCommand` class:
1. Samples random target positions
2. Checks distance from target cube (with success sphere)
3. Checks distance from all distractors
4. Resamples if constraints are violated
5. Uses fallback positions if no valid position found

### Termination Function
The `distractor_moved` function:
- Tracks initial positions stored during reset
- Computes displacement from initial positions
- Returns True if any distractor moved > threshold

## Usage

Run the environment:
```bash
python train.py --task Blocks-Robotiq2f85-CustomOmni-Push-Distractor
```

Test the environment:
```bash
python test_distractor_env.py --num_envs 2
```

## Task Objective
The robot must:
1. Push the blue target cube to the randomly generated goal position
2. Avoid touching or disturbing the red distractor cubes
3. Complete the task before timeout

The task tests the robot's ability to:
- Navigate around obstacles
- Perform precise manipulation
- Plan paths that avoid collisions
- Execute controlled pushing motions
