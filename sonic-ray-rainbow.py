import ray
import retro
import gymnasium as gym
import numpy as np
import torch
import argparse
import os
from datetime import datetime
import json
from ray.rllib.agents.dqn import RainbowTrainer

# Import key classes from the original implementation
from typing import List, Dict, Any

TRAINING_LEVELS = [
    "GreenHillZone.Act1",
    "SpringYardZone.Act1", 
    "StarLightZone.Act1"
]

class CurriculumManager:
    def __init__(self, levels=TRAINING_LEVELS):
        self.levels = levels
        self.current_level_idx = 0
        self.level_completion_threshold = 0.5
        self.completion_history = {level: [] for level in levels}
        
    def should_advance_level(self, success_rate):
        return success_rate >= self.level_completion_threshold
    
    def get_current_level(self):
        return self.levels[self.current_level_idx]
    
    def advance_level(self):
        if self.current_level_idx < len(self.levels) - 1:
            self.current_level_idx += 1
            return True
        return False
    
    def update_completion(self, level, completed):
        self.completion_history[level].append(completed)
        self.completion_history[level] = self.completion_history[level][-100:]
    
    def get_success_rate(self, level):
        history = self.completion_history[level]
        return sum(history) / len(history) if history else 0.0

class SonicMetrics:
    def __init__(self):
        self.reset_episode_metrics()
        self.total_episodes = 0
        self.total_deaths = 0
        self.level_completions = {level: 0 for level in TRAINING_LEVELS}
        self.best_times = {level: float('inf') for level in TRAINING_LEVELS}
        self.ring_records = {level: 0 for level in TRAINING_LEVELS}
        self.episode_rewards = []
        self.completion_times = []
        
    def reset_episode_metrics(self):
        self.steps = 0
        self.rings_collected = 0
        self.max_rings = 0
        self.enemies_defeated = 0
        self.current_reward = 0
        self.episode_start_time = datetime.now()

    def on_episode_end(self, level):
        self.total_episodes += 1
        self.episode_rewards.append(self.current_reward)
        episode_time = (datetime.now() - self.episode_start_time).total_seconds()
        self.completion_times.append(episode_time)
        
        if level in self.best_times:
            self.best_times[level] = min(self.best_times[level], episode_time)

    def get_training_stats(self):
        completion_rate = sum(self.level_completions.values()) / max(1, self.total_episodes)
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        avg_completion_time = np.mean(self.completion_times) if self.completion_times else 0
        
        return {
            'total_episodes': self.total_episodes,
            'total_deaths': self.total_deaths,
            'completion_rate': completion_rate,
            'average_reward': avg_reward,
            'average_completion_time': avg_completion_time,
            'best_times': self.best_times,
            'level_completions': self.level_completions
        }
class MetricsCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_history = []
        self.current_metrics = None

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get info from the most recent step
            if hasattr(self.locals, 'infos') and len(self.locals['infos']) > 0:
                info = self.locals['infos'][0]  # Get info from first environment
                if 'episode_metrics' in info:
                    metrics = info['episode_metrics']
                    self.metrics_history.append(metrics)
                    
                    # Log to tensorboard
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            self.logger.record(f"metrics/{key}", value)
                    
                    # Save metrics to file
                    metrics_file = os.path.join(self.log_dir, "training_metrics.json")
                    with open(metrics_file, 'w') as f:
                        json.dump(self.metrics_history, f, indent=4)
        
        return True

class RainbowSonicWrapper(gym.Wrapper):
    def __init__(self, env, initial_level=TRAINING_LEVELS[0]):
        super().__init__(env)
        self.metrics = SonicMetrics()
        self.current_level = initial_level
        self.reset_metrics()
        
    def reset(self, **kwargs):
        kwargs['state'] = kwargs.get('state', self.current_level)
        obs, info = self.env.reset(**kwargs)
        self.reset_metrics()
        self.metrics.reset_episode_metrics()
        return obs, info

    def reset_metrics(self):
        self.previous_info = None
        self.prev_x = 0
        self.max_x = 0
        self.standing_still_counter = 0
        self.prev_score = 0
        self.last_checkpoint_x = 0
        self.checkpoint_rewards = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_info = self.unwrapped.data.lookup_all()
        
        # Custom reward calculation (similar to your original implementation)
        custom_reward = 0
        
        # 1. Progress and exploration rewards
        current_x = current_info.get('x', 0)
        x_progress = current_x - self.prev_x
        
        # Basic progress reward
        if x_progress > 0:
            custom_reward += x_progress * 0.1
            
            # Checkpoint rewards
            if current_x - self.last_checkpoint_x >= 100:
                custom_reward += 50
                self.last_checkpoint_x = current_x
            
            # New maximum progress bonus
            if current_x > self.max_x:
                custom_reward += (current_x - self.max_x) * 0.2
                self.max_x = current_x

        # Penalize standing still
        if abs(x_progress) < 1:
            self.standing_still_counter += 1
            if self.standing_still_counter > 60:
                custom_reward -= 0.1
        else:
            self.standing_still_counter = 0

        # 2. Ring management
        if self.previous_info is not None:
            prev_rings = self.previous_info.get('rings', 0)
            current_rings = current_info.get('rings', 0)
            rings_delta = current_rings - prev_rings
            
            if rings_delta > 0:
                custom_reward += rings_delta * 1.0
                self.metrics.rings_collected += rings_delta
                self.metrics.max_rings = max(self.metrics.max_rings, current_rings)
                
                # Extra reward for collecting rings while moving
                if x_progress > 0:
                    custom_reward += rings_delta * 0.5
            
            elif rings_delta < 0:
                if current_rings == 0:
                    custom_reward -= 25  # Severe penalty for losing all rings
                else:
                    custom_reward -= abs(rings_delta) * 5

        # 3. Score and enemies
        current_score = current_info.get('score', 0)
        score_delta = current_score - self.prev_score
        if score_delta > 0:
            custom_reward += score_delta * 0.2
            self.metrics.enemies_defeated += 1

        # 4. Life management
        if self.previous_info is not None:
            life_diff = current_info.get('lives', 0) - self.previous_info.get('lives', 0)
            if life_diff < 0:
                custom_reward -= 100
                terminated = True

        # 5. Level completion
        if current_info.get('level_end_bonus', 0) > 0:
            custom_reward += 1000
            self.metrics.level_completions[self.current_level] += 1
            
            episode_time = (datetime.now() - self.metrics.episode_start_time).total_seconds()
            if episode_time < self.metrics.best_times[self.current_level]:
                self.metrics.best_times[self.current_level] = episode_time
                custom_reward += 500
            
            terminated = True

        # Update states
        self.prev_x = current_x
        self.prev_score = current_score
        self.previous_info = current_info
        self.metrics.steps += 1

        # Clip final reward
        total_reward = np.clip(custom_reward, -10.0, 10.0)
        self.metrics.current_reward += total_reward

        # Prepare info dictionary
        info['episode_metrics'] = {
            'steps': self.metrics.steps,
            'rings_collected': self.metrics.rings_collected,
            'max_rings': self.metrics.max_rings,
            'enemies_defeated': self.metrics.enemies_defeated,
            'current_reward': self.metrics.current_reward,
            'total_deaths': self.metrics.total_deaths,
        }
        
        if terminated or truncated:
            self.metrics.on_episode_end(self.current_level)
            info['episode_metrics'].update(self.metrics.get_training_stats())

        return obs, total_reward, terminated, truncated, info

class RainbowCurriculumTrainer:
    def __init__(self, args):
        # Initialize Ray
        ray.init(ignore_reinit_error=True)
        
        # Curriculum Manager
        self.curriculum_manager = CurriculumManager()
        
        # Configuration for Rainbow Trainer
        self.config = {
            "env": RainbowSonicWrapper,
            "env_config": {
                "game": args.game,
                "state": self.curriculum_manager.get_current_level()
            },
            "framework": "torch",
            "rainbow_config": {
                "num_atoms": 51,  # Distributional RL parameter
                "v_min": -10.0,   # Minimum value
                "v_max": 10.0,    # Maximum value
            },
            "double_q": True,
            "dueling": True,
            "prioritized_replay": True,
            "n_step": 3,  # Multi-step learning
            "lr": 1e-4,
            "buffer_size": 100000,
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.02,
                "epsilon_timesteps": 100000
            },
            "num_workers": args.num_envs,
            "num_envs_per_worker": 1
        }
        
        # Create Trainer
        self.trainer = RainbowTrainer(config=self.config)

    def train(self, total_timesteps):
        # Training loop with curriculum management
        for iteration in range(total_timesteps // 10000):
            # Train for a fixed number of steps
            result = self.trainer.train()
            
            # Extract current level metrics
            current_level = self.curriculum_manager.get_current_level()
            
            # Check for level advancement
            success_rate = result.get('episode_reward_mean', 0)
            self.curriculum_manager.update_completion(current_level, success_rate > 0)
            
            if self.curriculum_manager.should_advance_level(success_rate):
                if self.curriculum_manager.advance_level():
                    new_level = self.curriculum_manager.get_current_level()
                    print(f"\nAdvancing to new level: {new_level}")
                    
                    # Update environment configuration
                    self.trainer.workers.foreach_worker(
                        lambda worker: setattr(worker.env, 'current_level', new_level)
                    )
            
            # Logging
            print(f"Iteration {iteration}: Level={current_level}, Reward={success_rate}")
        
        # Save final model
        self.trainer.save("./models/rainbow_sonic_final")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="SonicTheHedgehog-Genesis")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--num_envs", type=int, default=4)
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize and train Rainbow agent
    rainbow_trainer = RainbowCurriculumTrainer(args)
    rainbow_trainer.train(args.timesteps)

if __name__ == "__main__":
    main()