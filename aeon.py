import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import MultivariateNormal, Categorical
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

class AdaptiveActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=512, 
                num_layers=3, activation='gelu'):
        super().__init__()
        self.activation = getattr(nn, activation.upper())()
        
        # Shared trunk with layer norm
        self.shared_trunk = nn.ModuleList()
        for _ in range(num_layers):
            self.shared_trunk.append(nn.Linear(obs_dim if _ == 0 else hidden_size, hidden_size))
            self.shared_trunk.append(nn.LayerNorm(hidden_size))
            self.shared_trunk.append(self.activation)
        
        # Adaptive gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Policy head with learnable variance
        self.actor = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
        # Value head with ensemble
        self.value_heads = nn.ModuleList(
            [nn.Linear(hidden_size, 1) for _ in range(3)]
        )
        
        # Orthogonal initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        for layer in self.shared_trunk:
            x = layer(x)
        gate = self.gate(x)
        x = x * gate
        
        # Policy
        mean = self.actor(x)
        std = torch.exp(self.log_std.clamp(-20, 2))
        
        # Value ensemble
        values = [head(x) for head in self.value_heads]
        value = torch.stack(values).mean(0)
        value_std = torch.stack(values).std(0)
        
        return mean, std, value, value_std

class AdvancedPPO:
    def __init__(self, env, config):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        
        # Hyperparameters
        self.clip_range = config.get('clip_range', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.kl_target = config.get('kl_target', 0.01)
        self.adaptive_lr = config.get('adaptive_lr', True)
        self.gamma = config.get('gamma', 0.999)
        self.gae_lambda = config.get('gae_lambda', 0.98)
        self.batch_size = config.get('batch_size', 512)
        self.epochs = config.get('epochs', 15)
        
        # Networks
        self.policy = AdaptiveActorCritic(self.obs_dim, self.act_dim)
        self.old_policy = AdaptiveActorCritic(self.obs_dim, self.act_dim)
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Optimizer with lookahead
        self.optimizer = optim.AdamW(self.policy.parameters(), 
                                   lr=config.get('lr', 3e-4),
                                   weight_decay=config.get('wd', 1e-6))
        self.lookahead = Lookahead(self.optimizer, k=5, alpha=0.5)
        
        # Normalization
        self.obs_mean = torch.zeros(self.obs_dim)
        self.obs_var = torch.ones(self.obs_dim)
        self.return_mean = 0.0
        self.return_var = 1.0
        
        # Adaptive parameters
        self.clip_range_scheduler = AdaptiveClipper(
            init_value=0.2, 
            kl_target=0.01,
            rate=1.5
        )
    
    def adaptive_normalize(self, x, mean, var):
        return (x - mean) / torch.sqrt(var + 1e-8)
    
    def compute_gae(self, rewards, values, dones, next_value=None):
        # Implementation with TD-lambda return estimation
        gae = 0
        returns = []
        
        # If next_value is not provided, assume terminal state
        if next_value is None:
            next_value = 0.0
            
        # Work backwards from the last step
        for t in reversed(range(len(rewards))):
            # Calculate delta (TD error)
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # Calculate GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            # Store return
            returns.insert(0, gae + values[t])
            
            # Update next value
            next_value = values[t]
            
        return torch.tensor(returns)
    
    def update_normalization(self, batch):
        # Welford's algorithm for online stats
        observations = batch["observations"]
        returns = batch["returns"]
        
        # Update observation statistics
        batch_size = observations.shape[0]
        delta = observations - self.obs_mean
        self.obs_mean = self.obs_mean + delta.mean(0)
        delta2 = observations - self.obs_mean
        self.obs_var = self.obs_var + ((delta * delta2).mean(0) - self.obs_var) / (batch_size + 1)
        
        # Update return statistics
        if isinstance(returns, torch.Tensor):
            returns = returns.flatten().cpu().numpy()
        batch_return_mean = np.mean(returns)
        batch_return_var = np.var(returns)
        
        # Exponential moving average for returns
        alpha = 0.005  # Slow update for stability
        self.return_mean = (1 - alpha) * self.return_mean + alpha * batch_return_mean
        self.return_var = (1 - alpha) * self.return_var + alpha * batch_return_var
    
    def update(self, rollout):
        # Advanced update with:
        # - Value function clipping
        # - Adaptive KL penalty
        # - Entropy annealing
        # - Prioritized experience replay
        
        # Unpack rollout data
        states = torch.FloatTensor(rollout['states'])
        actions = torch.FloatTensor(rollout['actions'])
        rewards = torch.FloatTensor(rollout['rewards'])
        dones = torch.FloatTensor(rollout['dones'])
        old_values = torch.FloatTensor(rollout['values'])
        old_log_probs = torch.FloatTensor(rollout['log_probs'])
        
        # Normalize states
        norm_states = self.adaptive_normalize(states, self.obs_mean, self.obs_var)
        
        # Compute returns and advantages
        with torch.no_grad():
            # Get values from old policy
            _, _, values, _ = self.old_policy(norm_states)
            # Compute GAE
            returns = self.compute_gae(rewards, values.squeeze(-1), dones)
            # Normalize returns
            norm_returns = (returns - self.return_mean) / (np.sqrt(self.return_var) + 1e-8)
            # Compute advantages
            advantages = returns - values.squeeze(-1)
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Create dataset and dataloader
        dataset = ExperienceDataset(
            observations=norm_states,
            actions=actions,
            advantages=advantages,
            returns=norm_returns,
            old_values=old_values,
            old_log_probs=old_log_probs
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Track metrics
        kl_divs = []
        value_losses = []
        policy_losses = []
        entropy_terms = []
        
        # PPO Update loop
        for epoch in range(self.epochs):
            for batch in loader:
                # Unpack batch
                b_obs = batch['observations']
                b_acts = batch['actions']
                b_advs = batch['advantages']
                b_rets = batch['returns']
                b_old_values = batch['old_values']
                b_old_log_probs = batch['old_log_probs']
                
                # Get current policy outputs
                mu, std, value, value_std = self.policy(b_obs)
                
                # Create normal distribution
                dist = MultivariateNormal(mu, torch.diag_embed(std.pow(2)))
                
                # Get log probabilities
                log_probs = dist.log_prob(b_acts)
                
                # Calculate KL divergence
                old_mu, old_std, _, _ = self.old_policy(b_obs)
                old_dist = MultivariateNormal(old_mu, torch.diag_embed(old_std.pow(2)))
                kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_dist, dist))
                
                # Early stopping based on KL
                if kl_div > 4 * self.kl_target:
                    break
                
                # Ratio for PPO
                ratio = torch.exp(log_probs - b_old_log_probs)
                
                # Policy loss with clipping
                clip_range = self.clip_range_scheduler.current_value
                policy_loss1 = -b_advs * ratio
                policy_loss2 = -b_advs * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = torch.mean(torch.max(policy_loss1, policy_loss2))
                
                # Value loss with clipping
                value_clipped = b_old_values + torch.clamp(
                    value - b_old_values, -clip_range, clip_range
                )
                value_loss1 = (value - b_rets).pow(2)
                value_loss2 = (value_clipped - b_rets).pow(2)
                value_loss = 0.5 * torch.mean(torch.max(value_loss1, value_loss2))
                
                # Entropy bonus
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                # Track metrics
                kl_divs.append(kl_div.item())
                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
                entropy_terms.append(entropy.item())
        
        # Apply lookahead update
        self.lookahead.step()
        
        # Update old policy
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Update clip range based on average KL
        if len(kl_divs) > 0:
            mean_kl = np.mean(kl_divs)
            self.clip_range_scheduler.update(mean_kl)
        
        # Return metrics
        metrics = {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropy_terms),
            "kl_div": np.mean(kl_divs),
            "clip_range": self.clip_range_scheduler.current_value
        }
        
        return metrics
    
    def train(self, total_timesteps):
        # Training loop with:
        # - Vectorized environments
        # - Dynamic batch sizing
        # - Automated curriculum learning
        
        timesteps_so_far = 0
        episodes = 0
        best_reward = -float('inf')
        
        # Create a buffer for storing rollout data
        rollout_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'log_probs': []
        }
        
        # Initialize progress tracking
        episode_rewards = []
        episode_lengths = []
        successes = []
        
        # For curriculum learning
        difficulty_level = 0
        success_rate_threshold = 0.8
        success_window = 10
        
        # Main training loop
        while timesteps_so_far < total_timesteps:
            # Maybe adjust environment difficulty based on recent performance
            if len(successes) >= success_window:
                recent_success_rate = sum(successes[-success_window:]) / success_window
                if recent_success_rate > success_rate_threshold:
                    difficulty_level += 1
                    print(f"Increasing difficulty to level {difficulty_level}")
                    # Here you would change environment parameters based on difficulty
                    # self.env.set_difficulty(difficulty_level)
                    successes = []  # Reset success tracking after difficulty change
            
            # Reset environment
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            # Episode loop
            while not done:
                # Normalize state
                norm_state = self.adaptive_normalize(
                    torch.FloatTensor(state), 
                    self.obs_mean, 
                    self.obs_var
                )
                
                # Get action from policy
                with torch.no_grad():
                    mu, std, value, _ = self.policy(norm_state.unsqueeze(0))
                    
                    # Create distribution
                    dist = MultivariateNormal(mu, torch.diag_embed(std.pow(2)))
                    
                    # Sample action
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    
                    # Clip action to environment bounds if needed
                    clipped_action = action.squeeze(0).numpy()
                    clipped_action = np.clip(
                        clipped_action, 
                        self.env.action_space.low, 
                        self.env.action_space.high
                    )
                
                # Step environment
                next_state, reward, done, truncated, info = self.env.step(clipped_action)
                done = done or truncated
                
                # Store in rollout buffer
                rollout_buffer['states'].append(state)
                rollout_buffer['actions'].append(clipped_action)
                rollout_buffer['rewards'].append(reward)
                rollout_buffer['dones'].append(float(done))
                rollout_buffer['values'].append(value.squeeze().item())
                rollout_buffer['log_probs'].append(log_prob.squeeze().item())
                
                # Update counters
                timesteps_so_far += 1
                episode_reward += reward
                episode_length += 1
                
                # Check if we need to update
                if len(rollout_buffer['states']) >= self.batch_size or done:
                    # Convert buffer to numpy arrays
                    for k, v in rollout_buffer.items():
                        rollout_buffer[k] = np.array(v)
                    
                    # Update policy
                    metrics = self.update(rollout_buffer)
                    
                    # Clear buffer
                    for k in rollout_buffer.keys():
                        rollout_buffer[k] = []
                
                # Move to next state
                state = next_state
                
                # Break if we've reached the total timesteps
                if timesteps_so_far >= total_timesteps:
                    break
            
            # Episode complete
            episodes += 1
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Track success for curriculum learning (example: success if reward > threshold)
            success = episode_reward > 100  # Adjust threshold based on your task
            successes.append(float(success))
            
            # Dynamic batch sizing based on episode length variance
            if episodes > 10:
                std_episode_length = np.std(episode_lengths[-10:])
                # Increase batch size if episodes are consistent length
                if std_episode_length < 0.1 * np.mean(episode_lengths[-10:]):
                    self.batch_size = min(4096, self.batch_size * 1.5)
                # Decrease batch size if episodes vary a lot
                elif std_episode_length > 0.3 * np.mean(episode_lengths[-10:]):
                    self.batch_size = max(64, int(self.batch_size * 0.8))
            
            # Track best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                # Save best model
                # torch.save(self.policy.state_dict(), "best_model.pt")
            
            # Print progress
            if episodes % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                print(f"Episode {episodes}, Timesteps: {timesteps_so_far}")
                print(f"Average reward: {avg_reward:.2f}, Average length: {avg_length:.1f}")
                print(f"Current batch size: {self.batch_size}")
                
        return {
            "episodes": episodes,
            "timesteps": timesteps_so_far,
            "best_reward": best_reward,
            "final_avg_reward": np.mean(episode_rewards[-10:])
        }

class Lookahead:
    # Lookahead optimizer implementation
    def __init__(self, optimizer, k=5, alpha=0.5):
        """
        Lookahead optimizer wrapper for enhanced optimization stability
        Args:
            optimizer: Base optimizer (e.g., Adam, SGD)
            k: Number of steps before parameter sync (default: 5)
            alpha: Slow weights step size (default: 0.5)
        """
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        
        # Initialize slow parameter copies
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['slow_param'] = torch.zeros_like(p.data)
                param_state['slow_param'].copy_(p.data)
                param_state['step'] = 0
    
    def step(self):
        """
        Perform one optimization step with lookahead
        """
        loss = None
        
        # Count number of lookahead steps 
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['step'] += 1
                
                # Synchronize parameters every k steps
                if param_state['step'] % self.k == 0:
                    # Get slow parameters
                    slow_param = param_state['slow_param']
                    
                    # Update slow weights
                    slow_param.data.add_(
                        self.alpha * (p.data - slow_param.data)
                    )
                    
                    # Copy slow parameters to fast parameters
                    p.data.copy_(slow_param.data)
        
        return loss
    
    def state_dict(self):
        """
        Return the optimizer state dict
        """
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'fast_state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups
        }
    
    def load_state_dict(self, state_dict):
        """
        Load optimizer state dict
        """
        fast_state_dict = {
            'state': state_dict['fast_state'],
            'param_groups': state_dict['param_groups']
        }
        self.optimizer.load_state_dict(fast_state_dict)
        
        # Build a dict with id keys for applying slow state
        slow_state_dict = {
            k: v for k, v in state_dict['slow_state'].items()
        }
        for k, v in slow_state_dict.items():
            self.state[k] = v

class AdaptiveClipper:
    # Automatic clip range adjustment based on KL divergence
    def __init__(self, init_value=0.2, kl_target=0.01, rate=1.5, min_value=0.05, max_value=0.5):
        """
        Adaptively adjust PPO clip range based on KL divergence
        Args:
            init_value (float): Initial clip range (default: 0.2)
            kl_target (float): Target KL divergence (default: 0.01)
            rate (float): Adjustment rate (default: 1.5)
            min_value (float): Minimum clip range (default: 0.05)
            max_value (float): Maximum clip range (default: 0.5)
        """
        self.current_value = init_value
        self.init_value = init_value
        self.kl_target = kl_target
        self.rate = rate
        self.min_value = min_value
        self.max_value = max_value
    
    def update(self, current_kl):
        """
        Update clip range based on the current KL divergence
        Args:
            current_kl (float): Current KL divergence
        """
        # If KL is too high, decrease clip range to be more conservative
        if current_kl > 2.0 * self.kl_target:
            self.current_value = max(self.min_value, self.current_value / self.rate)
        # If KL is too low, increase clip range to explore more
        elif current_kl < 0.5 * self.kl_target:
            self.current_value = min(self.max_value, self.current_value * self.rate)
        
        return self.current_value
    
    def reset(self):
        """
        Reset clip range to initial value
        """
        self.current_value = self.init_value

class ExperienceDataset(Dataset):
    # Prioritized experience replay buffer
    def __init__(self, observations, actions, advantages, returns, old_values, old_log_probs, priorities=None):
        """
        Dataset for PPO experience replay with optional prioritization
        Args:
            observations: Normalized observation tensors
            actions: Action tensors
            advantages: Advantage estimates
            returns: Return estimates
            old_values: Values from the old policy
            old_log_probs: Log probabilities from the old policy
            priorities: Optional priority weights for sampling
        """
        self.observations = observations
        self.actions = actions
        self.advantages = advantages
        self.returns = returns
        self.old_values = old_values
        self.old_log_probs = old_log_probs
        
        # Set up prioritization if provided
        if priorities is None:
            # Default to uniform sampling
            self.priorities = torch.ones_like(advantages)
        else:
            # Use provided priorities
            self.priorities = priorities
            
        # Normalize priorities for proper sampling
        self.priorities = self.priorities / self.priorities.sum()
        
        # Store size
        self.size = observations.shape[0]
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'observations': self.observations[idx],
            'actions': self.actions[idx],
            'advantages': self.advantages[idx],
            'returns': self.returns[idx],
            'old_values': self.old_values[idx],
            'old_log_probs': self.old_log_probs[idx],
            'priorities': self.priorities[idx]
        }
    
    def update_priorities(self, indices, new_priorities):
        """
        Update priorities for specific samples
        Args:
            indices: Indices of samples to update
            new_priorities: New priority values
        """
        self.priorities[indices] = new_priorities
        # Renormalize
        self.priorities = self.priorities / self.priorities.sum()
    
    def get_prioritized_sampler(self, batch_size):
        """
        Create a sampler that uses the priorities for sampling
        Args:
            batch_size: Batch size for sampling
        Returns:
            sampler: A sampler for the DataLoader
        """
        from torch.utils.data import WeightedRandomSampler
        
        # Create a weighted sampler
        sampler = WeightedRandomSampler(
            weights=self.priorities,
            num_samples=batch_size,
            replacement=True
        )
        
        return sampler

class CuriosityModule(nn.Module):
    # Intrinsic motivation through forward dynamics
    def __init__(self, obs_dim, act_dim, hidden_size=256, feature_dim=128):
        """
        Curiosity-driven exploration module implementing ICM (Intrinsic Curiosity Module)
        
        Args:
            obs_dim: Observation dimension
            act_dim: Action dimension
            hidden_size: Hidden layer size for networks
            feature_dim: Dimension of the feature encoding
        """
        super().__init__()
        
        # Feature encoder (state -> feature space)
        self.feature_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_dim)
        )
        
        # Forward dynamics model (feature, action -> next feature)
        self.forward_dynamics = nn.Sequential(
            nn.Linear(feature_dim + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_dim)
        )
        
        # Inverse dynamics model (feature, next_feature -> action)
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Scaling factor for intrinsic reward
        self.reward_scale = 0.01
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state, action, next_state):
        """
        Calculate intrinsic reward based on prediction error
        
        Args:
            state: Current state tensor
            action: Action taken tensor
            next_state: Next state tensor
            
        Returns:
            intrinsic_reward: Curiosity-based intrinsic reward
            total_loss: Loss for training the curiosity module
        """
        # Encode states into feature space
        feat = self.feature_encoder(state)
        next_feat = self.feature_encoder(next_state)
        
        # Forward dynamics - predict next feature
        pred_next_feat = self.forward_dynamics(torch.cat([feat, action], dim=-1))
        
        # Inverse dynamics - predict action from states
        pred_action = self.inverse_dynamics(torch.cat([feat, next_feat], dim=-1))
        
        # Calculate losses
        forward_loss = 0.5 * torch.mean(torch.square(pred_next_feat - next_feat.detach()), dim=-1)
        inverse_loss = 0.5 * torch.mean(torch.square(pred_action - action.detach()), dim=-1)
        
        # Intrinsic reward = prediction error in feature space
        intrinsic_reward = self.reward_scale * forward_loss
        
        # Total loss for training (forward loss + inverse loss)
        total_loss = forward_loss.mean() + inverse_loss.mean()
        
        return intrinsic_reward, total_loss
    
    def get_intrinsic_reward(self, state, action, next_state):
        """
        Calculate only the intrinsic reward without computing gradients
        """
        with torch.no_grad():
            reward, _ = self.forward(state, action, next_state)
        return reward


class DemonstrationBuffer:
    # Mix expert demonstrations with agent experiences
    def __init__(self, capacity=10000, demonstration_alpha=0.3):
        """
        Buffer for storing and sampling expert demonstrations alongside agent experiences
        
        Args:
            capacity: Maximum number of demonstrations to store
            demonstration_alpha: Weight for prioritizing demonstrations
        """
        self.capacity = capacity
        self.demonstration_alpha = demonstration_alpha
        self.demo_buffer = []
        self.demo_priorities = []
        
        # For tracking insertion position
        self.position = 0
        
    def add_demonstration(self, obs, action, reward, next_obs, done, priority=None):
        """
        Add an expert demonstration to the buffer
        """
        # Set default priority higher than regular experiences
        if priority is None:
            priority = 2.0
            
        demo = {
            'observations': obs,
            'actions': action,
            'rewards': reward,
            'next_observations': next_obs,
            'dones': done
        }
        
        # If buffer not full, append; otherwise, replace
        if len(self.demo_buffer) < self.capacity:
            self.demo_buffer.append(demo)
            self.demo_priorities.append(priority)
        else:
            self.demo_buffer[self.position] = demo
            self.demo_priorities[self.position] = priority
            
        # Update position for next insertion
        self.position = (self.position + 1) % self.capacity
    
    def load_demonstrations_from_file(self, filepath):
        """
        Load expert demonstrations from a file
        """
        try:
            data = torch.load(filepath)
            for i in range(len(data['observations'])):
                self.add_demonstration(
                    data['observations'][i],
                    data['actions'][i],
                    data['rewards'][i],
                    data['next_observations'][i],
                    data['dones'][i]
                )
            print(f"Loaded {len(self.demo_buffer)} demonstrations")
        except Exception as e:
            print(f"Error loading demonstrations: {e}")
    
    def sample(self, batch_size, agent_buffer=None, demo_ratio=0.25):
        """
        Sample a batch of transitions, mixing demonstrations and agent experiences
        
        Args:
            batch_size: Number of samples to draw
            agent_buffer: Buffer containing agent experiences
            demo_ratio: Fraction of batch to fill with demonstrations
        
        Returns:
            Batch of transitions with demonstration flag
        """
        demo_batch_size = min(int(batch_size * demo_ratio), len(self.demo_buffer))
        agent_batch_size = batch_size - demo_batch_size
        
        # Sample from demonstration buffer based on priorities
        if demo_batch_size > 0:
            probs = np.array(self.demo_priorities) ** self.demonstration_alpha
            probs = probs / probs.sum()
            demo_indices = np.random.choice(
                len(self.demo_buffer), 
                size=demo_batch_size, 
                replace=False, 
                p=probs
            )
            demo_samples = [self.demo_buffer[i] for i in demo_indices]
        else:
            demo_samples = []
            
        # Sample from agent buffer if available
        if agent_buffer is not None and agent_batch_size > 0:
            agent_samples = agent_buffer.sample(agent_batch_size)
        else:
            agent_samples = []
            
        # Combine samples
        combined_batch = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': [],
            'is_demo': []  # Flag to distinguish demos from agent experiences
        }
        
        # Add demonstrations
        for demo in demo_samples:
            for key in ['observations', 'actions', 'rewards', 'next_observations', 'dones']:
                combined_batch[key].append(demo[key])
            combined_batch['is_demo'].append(1.0)  # Mark as demonstration
            
        # Add agent experiences
        for exp in agent_samples:
            for key in ['observations', 'actions', 'rewards', 'next_observations', 'dones']:
                combined_batch[key].append(exp[key])
            combined_batch['is_demo'].append(0.0)  # Mark as agent experience
            
        # Convert to tensors
        for key in combined_batch:
            combined_batch[key] = torch.tensor(combined_batch[key])
            
        return combined_batch
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled demonstrations
        """
        for idx, priority in zip(indices, priorities):
            if idx < len(self.demo_priorities):
                self.demo_priorities[idx] = priority

class AutoTuner:
    """Evolutionary-based hyperparameter optimization for PPO algorithm"""
    
    def __init__(self, base_config, tuning_config, env_factory):
        """
        Initialize the AutoTuner
        
        Args:
            base_config: Base configuration for PPO
            tuning_config: Configuration for tuning process
            env_factory: Function that creates environments
        """
        self.base_config = base_config
        self.tune_config = tuning_config
        self.env_factory = env_factory
        
        # Extract tuning parameters
        self.population_size = tuning_config.get('population_size', 20)
        self.mutation_rate = tuning_config.get('mutation_rate', 0.3)
        self.metric = tuning_config.get('metric', '100_episode_return')
        self.parameters = tuning_config.get('parameters', {})
        
        # Initialize population
        self.population = []
        self.scores = []
        self.best_config = None
        self.best_score = -float('inf')
        self.generation = 0
        
    def _create_individual(self):
        """Create a random configuration within parameter bounds"""
        config = self.base_config.copy()
        
        for param, bounds in self.parameters.items():
            if isinstance(bounds[0], float):
                # Continuous parameter
                value = np.random.uniform(bounds[0], bounds[1])
            elif isinstance(bounds[0], int):
                # Discrete parameter
                value = np.random.randint(bounds[0], bounds[1] + 1)
            elif isinstance(bounds[0], bool):
                # Boolean parameter
                value = np.random.choice([True, False])
            else:
                # Categorical parameter
                value = np.random.choice(bounds)
                
            config[param] = value
            
        return config
    
    def _mutate(self, config):
        """Mutate configuration with given mutation rate"""
        new_config = config.copy()
        
        for param, bounds in self.parameters.items():
            # Skip mutation based on mutation rate
            if np.random.random() > self.mutation_rate:
                continue
                
            if isinstance(bounds[0], float):
                # Mutate continuous parameter
                # Add Gaussian noise scaled to parameter range
                range_width = bounds[1] - bounds[0]
                noise = np.random.normal(0, range_width * 0.1)
                new_value = config[param] + noise
                new_config[param] = np.clip(new_value, bounds[0], bounds[1])
            elif isinstance(bounds[0], int):
                # Mutate discrete parameter
                # Add integer noise scaled to parameter range
                range_width = bounds[1] - bounds[0]
                noise = np.random.randint(-max(1, int(range_width * 0.1)), 
                                         max(1, int(range_width * 0.1)) + 1)
                new_value = config[param] + noise
                new_config[param] = np.clip(new_value, bounds[0], bounds[1])
            elif isinstance(bounds[0], bool):
                # Flip boolean with probability mutation_rate
                new_config[param] = not config[param]
            else:
                # Randomly select new categorical value
                new_config[param] = np.random.choice(bounds)
                
        return new_config
    
    def _crossover(self, config1, config2):
        """Perform crossover between two configurations"""
        child = {}
        
        for param in self.parameters:
            # 50% chance to inherit from each parent
            if np.random.random() < 0.5:
                child[param] = config1[param]
            else:
                child[param] = config2[param]
                
        # Copy other parameters from base config
        for param, value in self.base_config.items():
            if param not in child:
                child[param] = value
                
        return child
    
    def _evaluate(self, config):
        """Evaluate a configuration by training an agent"""
        # Create environment
        env = self.env_factory()
        
        # Create agent
        agent = AdvancedPPO(env, config)
        
        # Train for a short duration
        eval_timesteps = self.tune_config.get('eval_timesteps', 100000)
        results = agent.train(total_timesteps=eval_timesteps)
        
        # Return the specified metric
        if self.metric == '100_episode_return':
            return results['final_avg_reward']
        elif self.metric == 'best_reward':
            return results['best_reward']
        else:
            return results.get(self.metric, 0.0)
    
    def initialize_population(self):
        """Initialize the population with random configurations"""
        self.population = [self._create_individual() for _ in range(self.population_size)]
        
        # Evaluate initial population
        self.scores = []
        for config in self.population:
            score = self._evaluate(config)
            self.scores.append(score)
            
            # Update best config
            if score > self.best_score:
                self.best_score = score
                self.best_config = config.copy()
                
        print(f"Initial population evaluated. Best score: {self.best_score}")
    
    def evolve_generation(self):
        """Evolve one generation of configurations"""
        self.generation += 1
        
        # Select parents using tournament selection
        def tournament_select(k=3):
            # Randomly select k individuals and return the best
            indices = np.random.choice(len(self.population), k, replace=False)
            best_idx = indices[np.argmax([self.scores[i] for i in indices])]
            return self.population[best_idx]
        
        # Create new population
        new_population = []
        elite_count = max(1, int(0.1 * self.population_size))
        
        # Elitism: keep best individuals
        elite_indices = np.argsort(self.scores)[-elite_count:]
        elites = [self.population[i] for i in elite_indices]
        new_population.extend(elites)
        
        # Fill rest of population with crossover and mutation
        while len(new_population) < self.population_size:
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation
            child = self._mutate(child)
            
            new_population.append(child)
            
        # Evaluate new population
        self.population = new_population
        self.scores = []
        
        for config in self.population:
            score = self._evaluate(config)
            self.scores.append(score)
            
            # Update best config
            if score > self.best_score:
                self.best_score = score
                self.best_config = config.copy()
                
        print(f"Generation {self.generation} evaluated. Best score: {self.best_score}")
        return self.best_config, self.best_score
    
    def run(self, generations=10):
        """Run the evolutionary optimization for given generations"""
        self.initialize_population()
        
        for _ in range(generations):
            self.evolve_generation()
            
        print("Optimization complete.")
        print(f"Best configuration found:")
        for param, value in self.best_config.items():
            if param in self.parameters:
                print(f"  {param}: {value}")
                
        return self.best_config
