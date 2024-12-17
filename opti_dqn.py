import argparse
import retro
from datetime import datetime
import torch
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import optuna
from optuna.integration import PyTorchLightningPruningCallback
TRAINING_LEVELS = [
    "GreenHillZone.Act1",
    "SpringYardZone.Act1",
    "StarLightZone.Act1"
]
class SonicMetrics:
    """Clase para seguimiento de métricas durante el entrenamiento"""
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


class SonicRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.metrics = SonicMetrics()
        self.reset_metrics()
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reset_metrics()
        self.metrics.reset_episode_metrics()
        self.current_level = kwargs.get('state', TRAINING_LEVELS[0])
        self.metrics.current_level = self.current_level
        return obs, info

    def reset_metrics(self):
        self.previous_info = None
        self.prev_x = 0
        self.max_x = 0
        self.standing_still_counter = 0
        self.prev_score = 0
        self.current_level = None
        self.last_checkpoint_x = 0
        self.checkpoint_rewards = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_info = self.unwrapped.data.lookup_all()
        
        # Initialize custom reward
        custom_reward = 0
        
        # 1. Progress and exploration rewards
        current_x = current_info.get('x', 0)
        x_progress = current_x - self.prev_x
        
        # Basic progress reward
        if x_progress > 0:
            custom_reward += x_progress * 0.1
            
            # Checkpoint rewards (every 100 units)
            if current_x - self.last_checkpoint_x >= 100:
                custom_reward += 50  # Significant reward for reaching checkpoints
                self.last_checkpoint_x = current_x
            
            # New maximum progress bonus
            if current_x > self.max_x:
                custom_reward += (current_x - self.max_x) * 0.2
                self.max_x = current_x

        """        # Penalize standing still
        if abs(x_progress) < 1:
            self.standing_still_counter += 1
            if self.standing_still_counter > 60:
                custom_reward -= 0.1
        else:
            self.standing_still_counter = 0"""

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
            custom_reward += 10000
            self.metrics.level_completions[self.current_level] += 1
            
            episode_time = (datetime.now() - self.metrics.episode_start_time).total_seconds()
            if episode_time < self.metrics.best_times[self.current_level]:
                self.metrics.best_times[self.current_level] = episode_time
                custom_reward += 500  # Extra reward for beating best time
            
            terminated = True

        # Update states
        self.prev_x = current_x
        self.prev_score = current_score
        self.previous_info = current_info
        self.metrics.steps += 1

        # Clip final reward
        total_reward = np.clip(custom_reward, -10.0, 10.0)
        self.metrics.current_reward += total_reward

        # Update info dictionary with metrics
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
    

    
class MultiBinaryToDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # Generate all possible actions for the MultiBinary space
        self.discrete_actions = np.array(np.meshgrid(*[[0, 1]] * env.action_space.n)).T.reshape(-1, env.action_space.n)
        self.action_space = gym.spaces.Discrete(len(self.discrete_actions))

    def action(self, action):
        # Map discrete action to MultiBinary action
        return self.discrete_actions[action]
class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        super().__init__(env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        total_reward = 0
        for i in range(self.n):
            if self.curac is None or (i == 0 and self.rng.rand() > self.stickprob):
                self.curac = ac
            ob, reward, terminated, truncated, info = self.env.step(self.curac)
            total_reward += reward
            if terminated or truncated:
                break
        return ob, total_reward, terminated, truncated, info

def make_retro(game, state=None):
    env = retro.make(game, state or retro.State.DEFAULT, render_mode="rgb_array")
    env = Monitor(env)
    return StochasticFrameSkip(env, n=4, stickprob=0.25)

env = make_retro('SonicTheHedgehog-Genesis')
env = SonicRewardWrapper(env)
env = MultiBinaryToDiscreteWrapper(env)
def objective(trial):
    # Hyperparameters for DQN
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-2)
    gamma = trial.suggest_float("gamma", 0.8, 0.9997)
    batch_size = trial.suggest_int("batch_size", 32, 64)
    buffer_size = 15000
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)
    tau = trial.suggest_float("tau", 0.9, 1.0)
    target_update_interval = trial.suggest_int("target_update_interval", 1000, 10000)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        tau=tau,
        target_update_interval=target_update_interval,
        verbose=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.learn(total_timesteps=25000, callback=PyTorchLightningPruningCallback, progress_bar=True)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    return -mean_reward  # Minimize negative reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=20)
    args = parser.parse_args()

    #setup pruner
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    # Set up Optuna study
    study = optuna.create_study(direction="minimize", pruner = pruner)
    study.optimize(objective, n_trials=args.n_trials)
    print("Best hyperparameters found:", study.best_params)

if __name__ == "__main__":
    main()
