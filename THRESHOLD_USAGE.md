# Threshold Parameter Usage

The push task now has a centralized `threshold` parameter that automatically controls all related distances and visualizations.

## How It Works

The `threshold` parameter is defined in `CommandsCfg` and automatically sets:

- **Reward threshold**: `rewards.reaching_goal.params.threshold = threshold`
- **Min distance**: `commands.ee_pose.min_distance = threshold + 0.01`
- **Goal visualizer radius**: `commands.ee_pose.goal_pose_visualizer_cfg.markers.*.radius = threshold`
- **Current pose visualizer radius**: `commands.ee_pose.curr_pose_visualizer_cfg.markers.*.radius = 0.0` (always)

## Hydra Usage

### Basic Threshold Tuning
```bash
# Easy task (larger success area)
env.threshold=0.15

# Hard task (smaller success area)  
env.threshold=0.05

# Very precise task
env.threshold=0.03
```

### Complete Example Commands

#### Easy Task (15cm threshold)
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Push-Cube-Franka-Easy-v0 \
  --headless \
  --logger wandb \
  --log_project_name Isaac-RL \
  --num_envs 2048 \
  --run_name easy-push-15cm \
  env.threshold=0.15
```

#### Hard Task (5cm threshold)
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Push-Cube-Franka-Easy-v0 \
  --headless \
  --logger wandb \
  --log_project_name Isaac-RL \
  --num_envs 2048 \
  --run_name hard-push-5cm \
  env.threshold=0.05
```

#### Combined with Other Parameters
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Push-Cube-Franka-Easy-v0 \
  --headless \
  --logger wandb \
  --log_project_name Isaac-RL \
  --num_envs 2048 \
  --run_name custom-push \
  env.threshold=0.08 \
  env.events.randomize_cube_position.params.pose_range.x=[0.4,0.6] \
  env.commands.ee_pose.ranges.pos_x=[-0.1,0.1]
```

## What Gets Updated Automatically

When you change `env.threshold=X`:

1. **Success criteria**: Cube must be within X meters of target
2. **Min target distance**: Targets spawn at least X+0.01 meters from cube
3. **Visualization**: Goal spheres have radius X (red/green spheres)
4. **Current pose visualization**: Always radius 0.0 (orange/blue spheres)

## Benefits

- **Single parameter**: Tune difficulty with one value
- **Consistent**: All related parameters stay synchronized
- **Visual feedback**: Visualization automatically matches success criteria
- **Easy experimentation**: Simple Hydra override for hyperparameter tuning
