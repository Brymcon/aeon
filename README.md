# Aeon AI - Advanced PPO Implementation

## Overview

Aeon AI is a sophisticated implementation of Proximal Policy Optimization (PPO) for reinforcement learning tasks. This implementation features several advanced techniques to enhance stability, sample efficiency, and overall performance compared to standard PPO algorithms. 

## Features

- **Adaptive Actor-Critic Architecture**
  - Shared trunk network with layer normalization
  - Adaptive gating mechanism
  - Value function ensemble for uncertainty estimation
  - Orthogonal initialization with learned variance
  
- **Advanced PPO Techniques**
  - Value function clipping
  - Adaptive KL penalty
  - Entropy annealing
  - Prioritized experience replay
  
- **Optimization Enhancements**
  - Lookahead optimizer
  - Gradient clipping
  - Dynamic batch sizing
  - Evolutionary hyperparameter optimization
  
- **Stability Improvements**
  - Adaptive clip range adjustment
  - State and reward normalization
  - Generalized Advantage Estimation (GAE)
  - KL-based early stopping
  
- **Automated Learning**
  - Curriculum learning capabilities
  - Dynamic hyperparameter adjustment
  - Intrinsic motivation through curiosity
  - Expert demonstration learning

## Architecture

### AdaptiveActorCritic

The neural network architecture used in this implementation includes:

1. **Shared Trunk**: Multiple linear layers with layer normalization and activation functions
2. **Adaptive Gating**: A sigmoid-based gating mechanism to control information flow
3. **Actor Head**: Policy network that outputs mean and standard deviation for a Gaussian policy
4. **Critic Ensemble**: Multiple value function heads to estimate uncertainty

### AdvancedPPO

The main algorithm class that implements:

1. **Initialization**: Setup of networks, optimizers, and normalization statistics
2. **Update**: Advanced PPO update with various clipping and regularization techniques
3. **Training Loop**: Episodic training with vectorized environments
4. **Adaptive Methods**: Normalization, advantage estimation, and hyperparameter adjustment

## Usage

```python
import gym
from manning_ai import AdvancedPPO

# Create environment
env = gym.make('HalfCheetah-v4')

# Configure PPO
config = {
    'clip_range': 0.2,
    'entropy_coef': 0.01,
    'kl_target': 0.01,
    'adaptive_lr': True,
    'gamma': 0.999,
    'gae_lambda': 0.98,
    'batch_size': 512,
    'epochs': 15,
    'lr': 3e-4,
    'wd': 1e-6
}

# Create agent
ppo = AdvancedPPO(env, config)

# Train
results = ppo.train(total_timesteps=1_000_000)

# Print results
print(f"Training complete with {results['episodes']} episodes")
print(f"Best reward: {results['best_reward']}")
print(f"Final average reward: {results['final_avg_reward']}")
```

### Hyperparameter Auto-Tuning

Manning AI includes an evolutionary optimization approach for finding optimal hyperparameters:

```python
from aeon_ai import AutoTuner

# Define tuning configuration
auto_tune_config = {
    'population_size': 20,
    'mutation_rate': 0.3,
    'metric': '100_episode_return',
    'parameters': {
        'clip_range': (0.1, 0.3),
        'entropy_coef': (0.001, 0.1),
        'gae_lambda': (0.9, 0.99)
    },
    'eval_timesteps': 100000
}

# Create AutoTuner
tuner = AutoTuner(base_config=config, 
                  tuning_config=auto_tune_config,
                  env_factory=lambda: gym.make('HalfCheetah-v4'))

# Run optimization for 10 generations
best_config = tuner.run(generations=10)

# Create optimized agent
optimized_ppo = AdvancedPPO(env, best_config)
```

### Using Intrinsic Motivation

Enhancing exploration with curiosity-driven learning:

```python
from aeon_ai import AdvancedPPO, CuriosityModule

# Create environment and agent
env = gym.make('SparseRewardEnvironment-v0')
ppo = AdvancedPPO(env, config)

# Create curiosity module
curiosity = CuriosityModule(
    obs_dim=env.observation_space.shape[0],
    act_dim=env.action_space.shape[0]
)

# During training loop:
state, _ = env.reset()
done = False

while not done:
    # Get action from policy
    action = ppo.get_action(state)
    
    # Step environment
    next_state, reward, done, _, _ = env.step(action)
    
    # Get intrinsic reward
    intrinsic_reward = curiosity.get_intrinsic_reward(
        torch.FloatTensor(state),
        torch.FloatTensor(action),
        torch.FloatTensor(next_state)
    )
    
    # Combine rewards
    combined_reward = reward + intrinsic_reward.item()
    
    # Store in buffer with combined reward
    ppo.store(state, action, combined_reward, next_state, done)
    
    # Move to next state
    state = next_state
```

### Expert Demonstrations

Accelerating learning with expert demonstrations:

```python
from aeon_ai import AdvancedPPO, DemonstrationBuffer

# Create environment and agent
env = gym.make('ComplexTask-v0')
ppo = AdvancedPPO(env, config)

# Create demonstration buffer
demo_buffer = DemonstrationBuffer(capacity=10000)

# Load expert demonstrations
demo_buffer.load_demonstrations_from_file('expert_demos.pt')

# In your training loop, mix agent experiences with demonstrations
for update in range(updates):
    # Sample batch with demonstrations
    batch = demo_buffer.sample(
        batch_size=512,
        agent_buffer=ppo.buffer,
        demo_ratio=0.3  # 30% demonstrations, 70% agent experience
    )
    
    # Update policy using mixed batch
    ppo.update_from_batch(batch)
```

## Key Components

### Lookahead Optimizer

An implementation of the Lookahead optimizer that maintains slow and fast weights to improve optimization stability.

```python
optimizer = optim.AdamW(policy.parameters(), lr=3e-4)
lookahead = Lookahead(optimizer, k=5, alpha=0.5)
```

### AdaptiveClipper

A component that adjusts the PPO clip range dynamically based on the KL divergence between old and new policies.

```python
clipper = AdaptiveClipper(init_value=0.2, kl_target=0.01, rate=1.5)
```

### Generalized Advantage Estimation

Implementation of GAE for more accurate and lower variance advantage estimation:

```python
advantages = compute_gae(rewards, values, dones, next_value)
```

### Prioritized Experience Replay

A dataset implementation that allows prioritized sampling of experiences:

```python
dataset = ExperienceDataset(obs, acts, advantages, returns, values, log_probs)
```

## Implementation Details

### State Normalization

States are normalized using running statistics to improve training stability:

```python
norm_state = adaptive_normalize(state, obs_mean, obs_var)
```

### Value Function Clipping

The value function is clipped similar to the policy to prevent large updates:

```python
value_clipped = old_values + torch.clamp(value - old_values, -clip_range, clip_range)
```

### Dynamic Batch Sizing

Batch sizes are adjusted based on episode length variance:

```python
if std_episode_length < 0.1 * mean_episode_length:
    batch_size = min(4096, batch_size * 1.5)
```

### Curriculum Learning

Environment difficulty is adjusted based on agent performance:

```python
if recent_success_rate > success_rate_threshold:
    difficulty_level += 1
```

## Performance Considerations

- Use GPU acceleration for larger networks
- Adjust batch sizes based on available memory
- Consider environment vectorization for faster data collection
- Monitor KL divergence to ensure stable updates
- Balance intrinsic and extrinsic rewards for optimal exploration
-Layer-wise Adaptive Learning Rates - Implementing a custom optimizer that adjusts learning rates per layer based on     gradient statistics.
-Advanced Parallelization - Adding true vectorized environment support with shared memory for significantly faster data collection.
-Multi-GPU Training - Distributing the PPO training across multiple GPUs for very large models.
-Model Distillation - Implementing a technique to distill the trained policy into a smaller, faster model for deployment.

## References

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
- Zhang, M., & Boutilier, C. (2020). ADAPT: Action-Dependent Adaptive Policy Trees. arXiv preprint arXiv:2009.14279.
- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. arXiv preprint arXiv:1801.01290.
- Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). Curiosity-driven exploration by self-supervised prediction. arXiv preprint arXiv:1705.05363.
- Vecerik, M., Hester, T., Scholz, J., Wang, F., Pietquin, O., Piot, B., ... & Mnih, V. (2017). Leveraging demonstrations for deep reinforcement learning on robotics problems with sparse rewards. arXiv preprint arXiv:1707.08817.

## License

This implementation is provided under the MIT License. 
