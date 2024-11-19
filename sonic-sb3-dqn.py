import argparse
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from datetime import datetime
import retro
import os
import json
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns


TRAINING_LEVELS = [
    "GreenHillZone.Act1",
    "SpringYardZone.Act1",
    "StarLightZone.Act1"
]
class CurriculumManager:
    def __init__(self, levels=TRAINING_LEVELS):
        self.levels = levels
        self.current_level_idx = 0
        self.level_completion_threshold = 0.7  # 70% success rate needed to advance
        self.completion_history = {level: [] for level in levels}
        
    def should_advance_level(self, success_rate):
        if success_rate >= self.level_completion_threshold:
            return True
        return False
    
    def get_current_level(self):
        return self.levels[self.current_level_idx]
    
    def advance_level(self):
        if self.current_level_idx < len(self.levels) - 1:
            self.current_level_idx += 1
            return True
        return False
    
    def update_completion(self, level, completed):
        self.completion_history[level].append(completed)
        # Keep only last 100 attempts
        self.completion_history[level] = self.completion_history[level][-100:]
    
    def get_success_rate(self, level):
        history = self.completion_history[level]
        if not history:
            return 0.0
        return sum(history) / len(history)

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

        # Update info dictionary
        info.update({
            'episode_metrics': {
                'steps': self.metrics.steps,
                'rings_collected': self.metrics.rings_collected,
                'max_rings': self.metrics.max_rings,
                'enemies_defeated': self.metrics.enemies_defeated,
                'distance': self.max_x,
                'current_reward': self.metrics.current_reward,
                'total_deaths': self.metrics.total_deaths,
                'checkpoint_rewards': self.checkpoint_rewards
            }
        })

        return obs, total_reward, terminated, truncated, info
class CurriculumCallback(BaseCallback):
    def __init__(self, curriculum_manager, eval_env, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.eval_env = eval_env
        self.check_freq = check_freq
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            current_level = self.curriculum_manager.get_current_level()
            success_rate = self.curriculum_manager.get_success_rate(current_level)
            
            if self.curriculum_manager.should_advance_level(success_rate):
                if self.curriculum_manager.advance_level():
                    new_level = self.curriculum_manager.get_current_level()
                    print(f"\nAdvancing to new level: {new_level}")
                    # Update environment with new level
                    self.eval_env.reset()
                    
        return True
class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, render_freq=10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.render_freq = render_freq
        self.best_mean_reward = -np.inf

    def _on_step(self):
        # Render the environment every `render_freq` timesteps during evaluation
        if self.n_calls % self.render_freq == 0:
            self.eval_env.render()
        return super()._on_step()
    

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
    def __init__(self, env, n=4, stickprob=0.25, exploration_prob=0.1):
        super().__init__(env)
        self.n = n
        self.stickprob = stickprob
        self.exploration_prob = exploration_prob
        self.curac = None
        self.rng = np.random.RandomState()

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        # Occasional random action for exploration
        if self.rng.rand() < self.exploration_prob:
            ac = self.env.action_space.sample()
        
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            if self.curac is None:
                self.curac = ac
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            elif i == 1:
                self.curac = ac
            
            ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            
            if terminated or truncated:
                break
        
        return ob, totrew, terminated, truncated, info
class MetricsCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get metrics from environment
            env = self.training_env.envs[0]
            if hasattr(env, 'metrics'):
                metrics = env.metrics.get_training_stats()
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

class StabilizedDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Ensure we have the optimizer
        if not hasattr(self, "optimizer"):
            self.optimizer = self.policy.optimizer
            
        losses = []
        for _ in range(gradient_steps):
            # Check if we need to sample new data
            if not self._can_sample(batch_size):
                return
            
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            with torch.no_grad():
                # Calculate next Q-values and target Q-values
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Get the maximum Q-value along the action dimension
                max_next_q_values, _ = next_q_values.max(dim=1, keepdim=True)
                # Calculate target Q-values with discounted future rewards
                target_q_values = replay_data.rewards.reshape(-1, 1) + \
                    (1 - replay_data.dones.reshape(-1, 1)) * self.gamma * max_next_q_values

            # Get current Q-values and select the ones for the actions taken
            current_q_values = self.q_net(replay_data.observations)
            actions_column = replay_data.actions.reshape(-1, 1)
            current_q_values = torch.gather(current_q_values, dim=1, index=actions_column)

            # Ensure shapes match
            assert current_q_values.shape == target_q_values.shape, \
                f"Shape mismatch: current_q_values {current_q_values.shape} vs target_q_values {target_q_values.shape}"
            
            # Calculate loss with properly shaped tensors
            loss = torch.nn.functional.smooth_l1_loss(current_q_values, target_q_values)
            
            # Clip loss to prevent explosion
            loss = torch.clamp(loss, -100, 100)
            
            losses.append(loss.item())
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            
            self.optimizer.step()

        if len(losses) > 0:
            self.logger.record("train/loss", np.mean(losses))
            
    def _can_sample(self, batch_size):
        """Check if enough samples are available in the replay buffer"""
        return bool(
            self.replay_buffer is not None and
            self.replay_buffer.size() > batch_size and
            self.replay_buffer.size() >= self.learning_starts
        )


def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs, render_mode="rgb_array")
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)   
    return env

def wrap_deepmind_retro(env):
    """Configure environment for retro games with DQN-style preprocessing"""
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="SonicTheHedgehog-Genesis")
    parser.add_argument("--state", default="GreenHillZone.Act1")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--save_interval", type=int, default=50000, help="Timesteps interval to save video")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint to resume training")
    return parser.parse_args()

def get_unique_video_folder(base_path):
    """Generate a unique folder for each run based on the timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_path, f"videos_{timestamp}")

def main():

    args = parse_args()
    def make_env(game, state, scenario, rank):
        def _init():
            env = make_retro(game=game, state=state, scenario=scenario)
            env = SonicRewardWrapper(env) 
            env = MultiBinaryToDiscreteWrapper(env)
            env = wrap_deepmind_retro(env)
            video_folder = get_unique_video_folder("./videos")
            env = RecordVideo(
                env,
                video_folder=video_folder,
                name_prefix="training_run",
                episode_trigger=lambda episode_id: episode_id % args.save_interval == 0
            )
            env = RecordEpisodeStatistics(env)
            env = Monitor(env, f"./logs/train_{rank}")
            return env
        return _init
    
    # Create output directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Initialize curriculum manager
    curriculum_manager = CurriculumManager()
    
    # Create environments
    env = SubprocVecEnv([make_env(args.game, curriculum_manager.get_current_level(), args.scenario, i) 
                         for i in range(args.num_envs)])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    
    # Create evaluation environment
    eval_env = make_env(args.game, curriculum_manager.get_current_level(), args.scenario, 0)()
    eval_env = VecFrameStack(VecTransposeImage(SubprocVecEnv([lambda: eval_env])), n_stack=4)
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // args.num_envs,
        save_path="./models/",
        name_prefix="sonic_dqn"
    )

    metrics_callback = MetricsCallback(
        check_freq=1000,
        log_dir="./logs/metrics/"
    )
    
    eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path="./models/best_model",
        log_path="./logs/eval/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True
    )
    curriculum_callback = CurriculumCallback(
        curriculum_manager,
        eval_env,
        check_freq=10000
    )
    if args.checkpoint is not None:
        model = StabilizedDQN.load(args.checkpoint, env=env)
    else:
        model = StabilizedDQN(
            policy="CnnPolicy",
            env=env,
            learning_rate=1e-6,
            buffer_size=30000,
            learning_starts=1000,
            batch_size=32,
            tau=0.2,
            gamma=0.99,
            train_freq=8,
            gradient_steps=1,
            target_update_interval=500,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02,
            max_grad_norm=0.1,
            verbose=1,
            tensorboard_log="./logs/tensorboard/",
            device="cuda" if torch.cuda.is_available() else "cpu",
            policy_kwargs=dict(
                features_extractor_kwargs=dict(features_dim=128),
                net_arch=[64, 32],
                normalize_images=True
            )
        )
        # Train the model
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=False,
            callback=[checkpoint_callback, eval_callback, metrics_callback, curriculum_callback],
            log_interval=50,
            progress_bar = True
        )

    # Save the final model
    model.save("./models/sonic_dqn_final")
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
