#!/usr/bin/env python
"""
Aeon Agent: Command-line tool for training and evaluating PPO agents using the Aeon framework
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime

# Import aeon PPO implementation
from aeon import AdvancedPPO, AeonUtils

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Aeon RL agent for training and evaluation")
    
    # Environment settings
    parser.add_argument("--env", type=str, default="HalfCheetah-v4", 
                        help="Gym environment ID")
    parser.add_argument("--action-type", type=str, choices=["continuous", "discrete", "mixed"],
                        default="continuous", help="Action space type")
    
    # Training settings
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total timesteps for training")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    
    # PPO hyperparameters
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="PPO batch size")
    parser.add_argument("--micro-batch", type=int, default=512,
                        help="Micro batch size for gradient accumulation")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda parameter")
    parser.add_argument("--clip-range", type=float, default=0.2,
                        help="PPO clip range")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--value-coef", type=float, default=0.5,
                        help="Value loss coefficient")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of PPO epochs per update")
    parser.add_argument("--use-lr-scheduler", action="store_true",
                        help="Use learning rate scheduler")
    
    # GPU and performance options
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use multiple GPUs if available")
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--memory-fraction", type=float, default=0.9,
                        help="Fraction of GPU memory to use")
    
    # Saving and loading
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name (default: env_id-timestamp)")
    parser.add_argument("--save-dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--checkpoint-freq", type=int, default=20,
                        help="Checkpoint frequency (in batches)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run evaluation only")
    parser.add_argument("--render", action="store_true",
                        help="Render environment during evaluation")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Number of episodes for evaluation")
    
    # Advanced options
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file (overrides command-line args)")
    
    return parser.parse_args()

def load_config_from_file(config_file):
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)

def create_env(env_id, seed=0):
    """Create and wrap a gym environment."""
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env

def build_ppo_config(args):
    """Build PPO configuration dictionary from arguments."""
    # Get GPU device
    if args.multi_gpu:
        cuda_device = list(range(torch.cuda.device_count()))
    else:
        cuda_device = int(args.device.split(':')[-1]) if 'cuda' in args.device else 0
    
    # Create configuration
    config = {
        # Basic PPO parameters
        'clip_range': args.clip_range,
        'entropy_coef': args.entropy_coef,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        
        # Action space configuration
        'action_type': args.action_type,
        
        # GPU configuration
        'gpu_config': {
            'cuda_device': cuda_device,
            'memory_fraction': args.memory_fraction,
            'mixed_precision': args.mixed_precision
        },
        
        # Checkpointing
        'checkpoint_dir': os.path.join(args.save_dir, args.exp_name),
        'checkpoint_frequency': args.checkpoint_freq,
        
        # Optimization enhancements
        'use_lr_scheduler': args.use_lr_scheduler,
    }
    
    return config

def train(env, config, args):
    """Train the agent."""
    print(f"Starting training on {args.env} with {args.timesteps} timesteps")
    print(f"Configuration: {config}")
    
    # Create agent
    start_time = time.time()
    agent = AdvancedPPO(env, config)
    
    # Resume from checkpoint if specified
    start_timestep = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_timestep = agent.load_checkpoint(args.resume)
        print(f"Loaded checkpoint at timestep {start_timestep}")
    
    # Train agent
    results = agent.train(total_timesteps=args.timesteps)
    
    # Print and save results
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    print(f"Final average reward: {results['final_avg_reward']:.2f}")
    print(f"Best reward: {results['best_reward']:.2f}")
    print(f"Episodes: {results['episodes']}")
    print(f"Final checkpoint: {results['checkpoint_path']}")
    
    # Save training curve
    save_training_curve(results['training_stats'], args)
    
    return agent, results

def evaluate(agent, env, num_episodes=10, render=False):
    """Evaluate the agent's performance."""
    rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Get action from policy
            action = agent.get_action(state)
            
            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update stats
            episode_reward += reward
            episode_length += 1
            
            # Render if requested
            if render:
                env.render()
            
            # Update state
            state = next_state
        
        # Store results
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode+1}/{num_episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # Print summary
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"Evaluation results over {num_episodes} episodes:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean episode length: {mean_length:.1f}")
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'rewards': rewards,
        'episode_lengths': episode_lengths
    }

def save_training_curve(stats, args):
    """Save training curves as plots."""
    save_path = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Plot rewards
    if 'rewards' in stats and len(stats['rewards']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(stats['rewards'])
        plt.title(f'Training Rewards ({args.env})')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(os.path.join(save_path, 'rewards.png'))
    
    # Plot losses if available
    if 'policy_losses' in stats and len(stats['policy_losses']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(stats['policy_losses'], label='Policy Loss')
        if 'value_losses' in stats:
            plt.plot(stats['value_losses'], label='Value Loss')
        plt.title(f'Training Losses ({args.env})')
        plt.xlabel('Update')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'losses.png'))
    
    # Plot KL divergence if available
    if 'kl_divs' in stats and len(stats['kl_divs']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(stats['kl_divs'])
        plt.title(f'KL Divergence ({args.env})')
        plt.xlabel('Update')
        plt.ylabel('KL Divergence')
        plt.savefig(os.path.join(save_path, 'kl_div.png'))
    
    # Save all stats as JSON
    with open(os.path.join(save_path, 'training_stats.json'), 'w') as f:
        # Convert numpy arrays and tensors to lists for JSON serialization
        json_stats = {}
        for k, v in stats.items():
            if isinstance(v, list) and len(v) > 0:
                if isinstance(v[0], (np.ndarray, torch.Tensor)):
                    json_stats[k] = [float(x) for x in v]
                else:
                    json_stats[k] = v
            elif isinstance(v, (np.ndarray, torch.Tensor)):
                json_stats[k] = [float(x) for x in v]
            else:
                json_stats[k] = v
        
        json.dump(json_stats, f, indent=2)

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load config from file if specified
    if args.config:
        config_dict = load_config_from_file(args.config)
        # Update args with config values
        for k, v in config_dict.items():
            if hasattr(args, k):
                setattr(args, k, v)
    
    # Set experiment name
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"{args.env}-{timestamp}"
    
    # Create experiment directory
    os.makedirs(os.path.join(args.save_dir, args.exp_name), exist_ok=True)
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    env = create_env(args.env, args.seed)
    
    # Build PPO configuration
    config = build_ppo_config(args)
    
    # Save configuration
    with open(os.path.join(args.save_dir, args.exp_name, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Train or load agent
    if args.eval_only and args.resume:
        # Load agent for evaluation
        agent = AdvancedPPO(env, config)
        agent.load_checkpoint(args.resume)
        print(f"Loaded agent from {args.resume} for evaluation")
    else:
        # Train agent
        agent, results = train(env, config, args)
    
    # Evaluate agent
    if args.eval_only or args.render:
        eval_env = create_env(args.env, args.seed + 100)  # Different seed for eval
        eval_results = evaluate(agent, eval_env, args.eval_episodes, args.render)
        
        # Save evaluation results
        eval_path = os.path.join(args.save_dir, args.exp_name, "eval_results.json")
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)
    
    print(f"All results saved to {os.path.join(args.save_dir, args.exp_name)}")

if __name__ == "__main__":
    main() 
