# Isaac Lab Push Task - Hyperparameter Tuning Reference

This guide shows you exactly how to tune parameters for your push task experiments.

## Quick Start

Your base command is:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Push-Cube-Franka-Easy-v0 \
  --headless \
  --logger wandb \
  --log_project_name isaac-lab-push \
  --num_envs 8192
```

Add parameter overrides at the end to tune hyperparameters.

---

## 1. Cube Position Parameters

**Where it's defined:** `push_joint_pos_env_cfg.py` → `PushEventCfg.randomize_cube_position`

| Parameter | Hydra Override | Default | Description |
|-----------|---------------|---------|-------------|
| X position | `env.events.randomize_cube_position.params.pose_range.x=[min,max]` | `[0.3, 0.3]` | Cube X coordinate |
| Y position | `env.events.randomize_cube_position.params.pose_range.y=[min,max]` | `[0.0, 0.0]` | Cube Y coordinate |
| Z position | `env.events.randomize_cube_position.params.pose_range.z=[min,max]` | `[0.0203, 0.0203]` | Cube Z (height) |
| Rotation | `env.events.randomize_cube_position.params.pose_range.yaw=[min,max]` | `[-0.5, 0.5]` | Cube rotation |

**Examples:**
```bash
# Fixed position (no randomization)
env.events.randomize_cube_position.params.pose_range.x=[0.4,0.4]
env.events.randomize_cube_position.params.pose_range.y=[0.0,0.0]

# Random in a range
env.events.randomize_cube_position.params.pose_range.x=[0.3,0.5]
env.events.randomize_cube_position.params.pose_range.y=[-0.1,0.1]
```

---

## 2. Target Position (Goal) Parameters

**Where it's defined:** `push_env_cfg.py` → `CommandsCfg.ee_pose.ranges`

| Parameter | Hydra Override | Default | Description |
|-----------|---------------|---------|-------------|
| Target X range | `env.commands.ee_pose.ranges.pos_x=[min,max]` | `[0.2, 0.4]` | Where targets spawn (X) |
| Target Y range | `env.commands.ee_pose.ranges.pos_y=[min,max]` | `[-0.1, 0.1]` | Where targets spawn (Y) |
| Target Z | `env.commands.ee_pose.ranges.pos_z=[min,max]` | `[0.0203, 0.0203]` | Target height |
| Goal threshold | `env.rewards.reaching_goal.params.threshold` | `0.03` | Distance to consider "reached" |

**Examples:**
```bash
# Closer targets
env.commands.ee_pose.ranges.pos_x=[0.15,0.35]
env.commands.ee_pose.ranges.pos_y=[-0.05,0.05]

# Easier goal (larger threshold)
env.rewards.reaching_goal.params.threshold=0.08

# Harder goal (smaller threshold)
env.rewards.reaching_goal.params.threshold=0.02
```

---

## 3. Robot Starting Position

**Where it's defined:** `push_joint_pos_env_cfg.py` → `PushEventCfg.init_franka_arm_pose`

| Parameter | Hydra Override | Default | Description |
|-----------|---------------|---------|-------------|
| Initial joint pose | `env.events.init_franka_arm_pose.params.default_pose=[j1,j2,j3,j4,j5,j6,j7,g1,g2]` | See below | 7 joint + 2 gripper values |
| Joint randomization | `env.events.randomize_franka_joint_state.params.std` | `0.0` | Gaussian noise on joints |

**Default joint configuration:** `[0.0, -0.1, 0.0, -2.4, 0.0, 2.8, 0.1, 0.04, 0.04]`

**Examples:**
```bash
# Different starting pose (use quotes!)
'env.events.init_franka_arm_pose.params.default_pose=[0.0,-0.2,0.0,-2.2,0.0,2.5,0.0,0.04,0.04]'

# Add randomization to starting pose
env.events.randomize_franka_joint_state.params.std=0.1
```

---

## 4. PPO Algorithm Parameters

**Where it's defined:** `rsl_rl_ppo_cfg.py` → `FrankaPushCubePPORunnerCfg.algorithm`

| Parameter | Hydra Override | Default | Description |
|-----------|---------------|---------|-------------|
| Learning rate | `agent.algorithm.learning_rate` | `3e-4` | Adam learning rate |
| Discount (γ) | `agent.algorithm.gamma` | `0.99` | Reward discount factor |
| GAE lambda (λ) | `agent.algorithm.lam` | `0.95` | GAE parameter |
| Clip coefficient | `agent.algorithm.clip_param` | `0.2` | PPO clip range |
| Entropy coefficient | `agent.algorithm.entropy_coef` | `0.0` | Entropy bonus weight |
| Value loss coef | `agent.algorithm.value_loss_coef` | `0.5` | Critic loss weight |
| Learning epochs | `agent.algorithm.num_learning_epochs` | `4` | Updates per batch |
| Mini-batches | `agent.algorithm.num_mini_batches` | `256` | Number of mini-batches |
| Max grad norm | `agent.algorithm.max_grad_norm` | `0.5` | Gradient clipping |

**Examples:**
```bash
# Lower learning rate
agent.algorithm.learning_rate=1e-4

# Higher discount (longer horizon)
agent.algorithm.gamma=0.995
agent.algorithm.lam=0.98

# Add exploration bonus
agent.algorithm.entropy_coef=0.01

# More training per batch
agent.algorithm.num_learning_epochs=8
```

---

## 5. Network Architecture

**Where it's defined:** `rsl_rl_ppo_cfg.py` → `FrankaPushCubePPORunnerCfg.policy`

| Parameter | Hydra Override | Default | Description |
|-----------|---------------|---------|-------------|
| Actor network | `agent.policy.actor_hidden_dims=[sizes]` | `[256,256,256]` | Hidden layer sizes |
| Critic network | `agent.policy.critic_hidden_dims=[sizes]` | `[256,256,256]` | Hidden layer sizes |
| Initial noise | `agent.policy.init_noise_std` | `0.6` | Action noise at start |
| Activation | `agent.policy.activation` | `tanh` | Activation function |

**Examples:**
```bash
# Larger network
agent.policy.actor_hidden_dims=[512,512,512]
agent.policy.critic_hidden_dims=[512,512,512]

# Smaller network
agent.policy.actor_hidden_dims=[128,128]
agent.policy.critic_hidden_dims=[128,128]

# More exploration
agent.policy.init_noise_std=0.8
```

---

## 6. Training Configuration

**Where it's defined:** `rsl_rl_ppo_cfg.py` → `FrankaPushCubePPORunnerCfg`

| Parameter | Hydra Override | Default | Description |
|-----------|---------------|---------|-------------|
| Steps per env | `agent.num_steps_per_env` | `100` | Rollout length |
| Max iterations | `--max_iterations` | `20000` | Total training iterations |
| Save interval | `agent.save_interval` | `20` | Checkpoint frequency |
| Log interval | `agent.log_interval` | `10` | Logging frequency |
| Run name | `agent.run_name` | (auto) | Experiment identifier |

**Examples:**
```bash
# Longer rollouts
agent.num_steps_per_env=200

# Shorter training
--max_iterations 10000

# Name your run
agent.run_name=my_experiment_v1
```

---

## 7. Environment Settings

**Where it's defined:** `push_env_cfg.py` → `PushEnvCfg`

| Parameter | Hydra Override | Default | Description |
|-----------|---------------|---------|-------------|
| Control frequency | `env.decimation` | `4` | Control decimation |
| Episode length | `env.episode_length_s` | `10.0` | Max episode duration (seconds) |
| Sim timestep | `env.sim.dt` | `0.01` | Physics timestep |
| Num envs | `--num_envs` | `4096` | Parallel environments |

**Examples:**
```bash
# More envs for faster training
--num_envs 16384

# Shorter episodes
env.episode_length_s=5.0

# Higher control frequency
env.decimation=2
```

---

## Complete Example Runs

### Experiment 1: Baseline
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Push-Cube-Franka-Easy-v0 \
  --headless --logger wandb --log_project_name isaac-lab-push \
  --num_envs 8192 \
  agent.run_name=baseline
```

### Experiment 2: Easy Setup (Close cube, large goal threshold)
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Push-Cube-Franka-Easy-v0 \
  --headless --logger wandb --log_project_name isaac-lab-push \
  --num_envs 8192 \
  env.events.randomize_cube_position.params.pose_range.x=[0.3,0.3] \
  env.events.randomize_cube_position.params.pose_range.y=[0.0,0.0] \
  env.commands.ee_pose.ranges.pos_x=[0.2,0.35] \
  env.rewards.reaching_goal.params.threshold=0.08 \
  agent.run_name=easy_setup
```

### Experiment 3: Aggressive Learning
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Push-Cube-Franka-Easy-v0 \
  --headless --logger wandb --log_project_name isaac-lab-push \
  --num_envs 8192 \
  agent.algorithm.learning_rate=5e-4 \
  agent.algorithm.num_learning_epochs=8 \
  agent.algorithm.entropy_coef=0.02 \
  agent.policy.init_noise_std=0.8 \
  agent.run_name=aggressive_learning
```

### Experiment 4: Full Custom
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Push-Cube-Franka-Easy-v0 \
  --headless --logger wandb --log_project_name isaac-lab-push \
  --num_envs 8192 \
  --max_iterations 15000 \
  --seed 42 \
  agent.algorithm.learning_rate=2e-4 \
  agent.algorithm.gamma=0.98 \
  agent.policy.actor_hidden_dims=[512,512,256] \
  agent.num_steps_per_env=150 \
  env.events.randomize_cube_position.params.pose_range.x=[0.35,0.4] \
  env.commands.ee_pose.ranges.pos_x=[0.25,0.45] \
  env.rewards.reaching_goal.params.threshold=0.04 \
  agent.run_name=custom_full
```

---

## Tips for Clean Experimentation

### 1. **Always name your runs**
```bash
agent.run_name=descriptive_experiment_name
```
This makes it clear what you tuned in WandB.

### 2. **Track what you changed**
Use descriptive run names that indicate what you tuned:
```bash
agent.run_name=lr_1e-4_gamma95_close_cube
```

### 3. **Use seeds for reproducibility**
```bash
--seed 42
```

### 4. **Quick testing**
Before long runs, test with:
```bash
--num_envs 512 --max_iterations 100
```

### 5. **Check logged configs**
After training, your exact config is saved in `logs/rsl_rl/{experiment_name}/{timestamp}/params/`:
- `env.yaml` - Full environment config
- `agent.yaml` - Full agent config

This ensures you always know exactly what parameters were used!

---

## How Hydra Overrides Work

The pattern is: `config.path.to.parameter=value`

For example:
- `push_env_cfg.py` has `rewards.reaching_goal` → override with `env.rewards.reaching_goal.params.threshold=0.05`
- `rsl_rl_ppo_cfg.py` has `algorithm.learning_rate` → override with `agent.algorithm.learning_rate=1e-4`

The prefix `env.` or `agent.` tells Hydra which config to modify.

