import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.cuda.amp import GradScaler, autocast  # Mixed precision

class MemoryEfficientActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=512):
        super().__init__()
        # Shared trunk (saves memory)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        # Separate heads
        self.actor_mean = nn.Linear(hidden_size, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))  # Learned variance
        self.critic = nn.Linear(hidden_size, 1)

        # Orthogonal init (better for RL)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01 if m == self.critic else 1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.trunk(x)
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd.clamp(-20, 2))
        value = self.critic(x)
        return mean, std, value

class AdaptiveClipper:
    """Dynamically adjusts PPO's clip range based on KL divergence."""
    def __init__(self, init_value=0.2, target_kl=0.01, increase_factor=1.1, decrease_factor=0.9):
        self.value = init_value
        self.target_kl = target_kl
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.min_value = 0.05
        self.max_value = 0.5
    
    def update(self, kl):
        """Update clip range based on KL divergence."""
        if kl > 2.0 * self.target_kl:
            self.value = max(self.min_value, self.value * self.decrease_factor)
        elif kl < 0.5 * self.target_kl:
            self.value = min(self.max_value, self.value * self.increase_factor)
        return self.value

class PPOTrainer:
    def __init__(self, env, batch_size=4096, micro_batch=512, gamma=0.99, gae_lambda=0.95, entropy_coef=0.01):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = MemoryEfficientActorCritic(self.obs_dim, self.act_dim).to(self.device)
        self.old_policy = MemoryEfficientActorCritic(self.obs_dim, self.act_dim).to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        # Optimizer with gradient accumulation
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=3e-4, eps=1e-5)
        self.scaler = GradScaler()  # For mixed precision

        # Adaptive batching (prevents OOM)
        self.batch_size = batch_size
        self.micro_batch = micro_batch  # Processed in chunks
        self.accumulation_steps = 4  # Accumulate gradients for 4 steps
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.adaptive_clipper = AdaptiveClipper(init_value=0.2, target_kl=0.01)
        self.epochs = 5
        
        # Stats
        self.stats = {"policy_loss": [], "value_loss": [], "clip_frac": [], "kl_div": [], "entropy": []}

    def compute_gae(self, rewards, values, dones):
        """Memory-efficient GAE computation with in-place operations."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_value = values[t+1] if t < len(values)-1 else 0
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        return advantages

    def collect_rollout(self, num_steps):
        """Collect experience batch using vectorized environment."""
        obs = torch.zeros((num_steps + 1, self.obs_dim), dtype=torch.float32, device=self.device)
        actions = torch.zeros((num_steps, self.act_dim), dtype=torch.float32, device=self.device)
        log_probs = torch.zeros(num_steps, dtype=torch.float32, device=self.device)
        rewards = torch.zeros(num_steps, dtype=torch.float32, device=self.device)
        dones = torch.zeros(num_steps, dtype=torch.float32, device=self.device)
        values = torch.zeros(num_steps + 1, dtype=torch.float32, device=self.device)
        
        state, _ = self.env.reset()
        obs[0] = torch.as_tensor(state, device=self.device)
        
        for step in range(num_steps):
            with torch.no_grad():
                mean, std, value = self.policy(obs[step].unsqueeze(0))
                dist = MultivariateNormal(mean, torch.diag_embed(std))
                action = dist.sample()
                log_prob = dist.log_prob(action)
                values[step] = value.squeeze()
            
            action_np = action.cpu().numpy().squeeze()
            next_state, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            obs[step + 1] = torch.as_tensor(next_state, device=self.device)
            actions[step] = action.squeeze()
            log_probs[step] = log_prob.squeeze()
            rewards[step] = torch.as_tensor(reward, device=self.device)
            dones[step] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
            
            if done:
                state, _ = self.env.reset()
                obs[step + 1] = torch.as_tensor(state, device=self.device)
        
        with torch.no_grad():
            _, _, final_value = self.policy(obs[-1].unsqueeze(0))
            values[-1] = final_value.squeeze()
        
        return obs[:-1], actions, log_probs, rewards, dones, values

    def update(self, rollout_data):
        """Update policy using PPO with micro-batching and adaptive clipping."""
        obs, actions, old_log_probs, rewards, dones, values = rollout_data
        
        advantages = self.compute_gae(rewards, values, dones)
        returns = advantages + values[:-1]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_losses, value_losses, clip_fracs, kls, entropies = [], [], [], [], []
        
        for epoch in range(self.epochs):
            indices = torch.randperm(self.batch_size)
            
            step_count = 0
            for i in range(0, self.batch_size, self.micro_batch):
                mb_idx = indices[i:i+self.micro_batch]
                mb_obs, mb_actions = obs[mb_idx], actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                
                with autocast():
                    mean, std, values = self.policy(mb_obs)
                    dist = MultivariateNormal(mean, torch.diag_embed(std))
                    log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()
                    
                    with torch.no_grad():
                        old_mean, old_std, _ = self.old_policy(mb_obs)
                        old_dist = MultivariateNormal(old_mean, torch.diag_embed(old_std))
                        kl_div = torch.distributions.kl.kl_divergence(old_dist, dist).mean()
                    
                    clip_range = self.adaptive_clipper.update(kl_div.item())
                    
                    ratio = (log_probs - mb_old_log_probs).exp()
                    clip_frac = (ratio.abs() > (1 + clip_range)).float().mean()
                    
                    obj1 = mb_advantages * ratio
                    obj2 = mb_advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                    policy_loss = -torch.min(obj1, obj2).mean()
                    
                    old_values = values[mb_idx]
                    value_pred_clipped = old_values + torch.clamp(
                        values - old_values, -clip_range, clip_range)
                    value_loss1 = (values - mb_returns).pow(2)
                    value_loss2 = (value_pred_clipped - mb_returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                    
                    total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
                
                total_loss = total_loss / self.accumulation_steps
                
                self.scaler.scale(total_loss).backward()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                clip_fracs.append(clip_frac.item())
                kls.append(kl_div.item())
                entropies.append(entropy.item())
                
                step_count += 1
                if step_count % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
        
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        stats = {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "clip_frac": np.mean(clip_fracs),
            "kl_div": np.mean(kls),
            "entropy": np.mean(entropies),
            "clip_range": self.adaptive_clipper.value
        }
        
        for k, v in stats.items():
            if k in self.stats:
                self.stats[k].append(v)
        
        return stats
        
    def train(self, total_timesteps):
        """Train policy for specified number of timesteps."""
        timesteps_so_far = 0
        results = []
        
        while timesteps_so_far < total_timesteps:
            batch_size = min(self.batch_size, total_timesteps - timesteps_so_far)
            rollout = self.collect_rollout(batch_size)
            
            stats = self.update(rollout)
            results.append(stats)
            
            timesteps_so_far += batch_size
            
            if len(results) % 10 == 0:
                print(f"Timesteps: {timesteps_so_far}/{total_timesteps}")
                print(f"  Policy Loss: {stats['policy_loss']:.4f}")
                print(f"  Value Loss: {stats['value_loss']:.4f}")
                print(f"  KL Div: {stats['kl_div']:.4f}")
                print(f"  Clip Range: {stats['clip_range']:.3f}")
                print(f"  Clip Fraction: {stats['clip_frac']:.3f}")
                
                if stats['clip_frac'] > 0.3:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.9
                    print("  Reducing learning rate due to high clip fraction")
        
        return self.stats
