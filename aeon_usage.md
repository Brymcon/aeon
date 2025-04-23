# Aeon Usage Guide

Aeon is a high-performance implementation of Proximal Policy Optimization (PPO) with advanced features for efficient training on modern hardware. This guide explains how to leverage the key improvements while minimizing code changes.

## Quick Start

```python
import gym
from aeon import AdvancedPPO

# Create environment
env = gym.make("HalfCheetah-v4")

# Configure PPO with advanced options
config = {
    # Basic PPO hyperparameters
    'clip_range': 0.2,
    'entropy_coef': 0.01,
    'gamma': 0.99,
    'batch_size': 2048,
    
    # GPU configuration
    'gpu_config': {
        'cuda_device': 0,  # or [0,1] for multi-GPU
        'memory_fraction': 0.9,
        'mixed_precision': True
    },
    
    # Checkpoint settings
    'checkpoint_dir': './checkpoints',
    'checkpoint_frequency': 20,
    
    # Optimization enhancements
    'use_lr_scheduler': True
}

# Create and train agent
agent = AdvancedPPO(env, config)
results = agent.train(total_timesteps=1_000_000)

print(f"Training complete! Final reward: {results['final_avg_reward']}")
print(f"Checkpoint saved to: {results['checkpoint_path']}")
```

## Key Features

### 1. GPU Memory Management

Efficiently utilize GPU memory with automatic memory fraction allocation:

```python
'gpu_config': {
    'cuda_device': 0,  # Single GPU
    'memory_fraction': 0.9  # Use 90% of available memory
}
```

For multi-GPU training:

```python
'gpu_config': {
    'cuda_device': [0, 1, 2, 3],  # Specify GPUs to use
    'memory_fraction': 0.8
}
```

### 2. Mixed Precision Training

Enable mixed precision (FP16) for 2-3x speedup:

```python
'gpu_config': {
    'mixed_precision': True  # Use FP16 where possible
}
```

### 3. Checkpointing

Automatically save and load training progress:

```python
# Save settings
'checkpoint_dir': './checkpoints',  
'checkpoint_frequency': 20,  # Save every 20 batches

# Load from checkpoint
checkpoint_path = 'checkpoints/checkpoint_500000.pt'
agent.load_checkpoint(checkpoint_path)
```

### 4. Discrete Action Spaces

Support for various action spaces:

```python
# For continuous environments (default)
'action_type': 'continuous'

# For discrete environments
'action_type': 'discrete'

# For hybrid environments
'action_type': 'mixed',
'discrete_dims': [3, 4],  # Dimensions of discrete actions
'continuous_dim': 2       # Dimension of continuous action part
```

### 5. Advanced Learning Rate Scheduling

Adaptive learning rate scheduling based on performance:

```python
'use_lr_scheduler': True  # Automatically adjust learning rate
```

## Migrating from Standard PPO

If you're transitioning from standard PPO implementations:

1. Replace imports:
   ```python
   from aeon import AdvancedPPO  # Instead of standard PPO
   ```

2. Update configuration:
   ```python
   # Old way
   ppo = PPO(env, lr=3e-4, gamma=0.99)
   
   # New way
   config = {
       'lr': 3e-4,
       'gamma': 0.99,
       'gpu_config': {'mixed_precision': True}
   }
   ppo = AdvancedPPO(env, config)
   ```

3. Enable checkpointing:
   ```python
   results = ppo.train(total_timesteps=1000000)
   # Checkpoints are saved automatically
   ```

## Performance Tips

1. **Memory optimization**: For large environments, reduce `batch_size` and enable mixed precision

2. **Multi-GPU training**: For faster training on complex tasks:
   ```python
   'gpu_config': {'cuda_device': [0, 1]}
   ```

3. **Checkpointing frequency**: Adjust based on task stability:
   ```python
   # For stable tasks
   'checkpoint_frequency': 50
   
   # For unstable tasks
   'checkpoint_frequency': 10
   ```

4. **Action space**: Select the appropriate type for your environment:
   ```python
   # For MuJoCo/continuous control
   'action_type': 'continuous'
   
   # For Atari/discrete control
   'action_type': 'discrete'
   ```

## Advanced Usage

### Custom Environment Integration

For environments with complex observation/action spaces:

```python
class CustomEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-10, 10, shape=(64,))
        self.action_space = gym.spaces.MultiDiscrete([3, 4, 5])
        
    # Implement step, reset, etc.

env = CustomEnv()
config = {
    'action_type': 'discrete',
    # Other settings
}
agent = AdvancedPPO(env, config)
```

### Resuming Training

To continue training from a checkpoint:

```python
agent = AdvancedPPO(env, config)
timesteps_completed = agent.load_checkpoint("checkpoints/checkpoint_500000.pt")
remaining_steps = 1000000 - timesteps_completed
results = agent.train(total_timesteps=remaining_steps)
```

### Analyzing Training Progress

Access full training statistics:

```python
results = agent.train(total_timesteps=1000000)
stats = results['training_stats']

# Plot learning curves
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(stats['rewards'])
plt.title('Training Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig('training_curve.png')
```

## Comparison with AeonMini

Aeon provides full features but may require more resources. Consider using AeonMini when:

1. You're working with limited GPU memory (< 16GB)
2. You need maximum throughput for simpler environments
3. You don't need all advanced features (checkpointing, multi-GPU) 
