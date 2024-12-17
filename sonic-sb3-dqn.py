import argparse
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from datetime import datetime
import retro
import os   
from gymnasium.spaces import Box
import json
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=516):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # Adjusted input channels to 4
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate the flattened size for the fully connected layer
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))
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

class RightMovementBiasWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.rng = np.random.RandomState()
    
    def step(self, action):
        # 50% chance of forcing right movement

        if self.rng.rand() < 0.5:
            action = 2  # Right action

        # right jump action
        elif self.rng.rand() < 0.1:
            action = 4
        return self.env.step(action)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
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

        # Update metrics without modifying rewards
        current_x = current_info.get('x', 0)
        self.metrics.steps += 1

        if self.previous_info is not None:
            prev_rings = self.previous_info.get('rings', 0)
            current_rings = current_info.get('rings', 0)
            rings_delta = current_rings - prev_rings
            if rings_delta > 0:
                self.metrics.rings_collected += rings_delta
                self.metrics.max_rings = max(self.metrics.max_rings, current_rings)

        if current_info.get('level_end_bonus', 0) > 0:
            self.metrics.level_completions[self.current_level] += 1
            episode_time = (datetime.now() - self.metrics.episode_start_time).total_seconds()
            if episode_time < self.metrics.best_times[self.current_level]:
                self.metrics.best_times[self.current_level] = episode_time
            terminated = True

        # Update info dictionary with metrics
        info['episode_metrics'] = {
            'steps': self.metrics.steps,
            'rings_collected': self.metrics.rings_collected,
            'max_rings': self.metrics.max_rings,
            'enemies_defeated': self.metrics.enemies_defeated,
            'current_reward': self.metrics.current_reward + reward,  # Add default reward for tracking
            'total_deaths': self.metrics.total_deaths,
        }
        
        if terminated or truncated:
            self.metrics.on_episode_end(self.current_level)
            info['episode_metrics'].update(self.metrics.get_training_stats())

        # Return the original reward
        return obs, reward, terminated, truncated, info
"""
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

        return obs, total_reward, terminated, truncated, info"""
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
    

class SonicActionWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        Wrapper to create a discrete action space for Sonic with 8 meaningful actions.
        
        Args:
            env (gym.Env): The original Sonic environment
        """
        super().__init__(env)
        
        # Define the action mapping
        self.action_space = gym.spaces.Discrete(8)
        
        # Action switch mapping (based on button press combinations)
        self.action_switch = {
            # No Operation
            0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Left
            1: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            # Right
            2: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            # Left, Down
            3: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            # Right, Down
            4: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            # Down
            5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            # Down, B
            6: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            # B
            7: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
    
    def step(self, action):
        """
        Convert the discrete action to the corresponding multi-binary action.
        
        Args:
            action (int): Discrete action index (0-7)
        
        Returns:
            tuple: Observation, reward, done, info after taking the action
        """
        # Convert discrete action to multi-binary action
        multi_binary_action = self.action_switch[action]
        
        # Take a step in the environment with the converted action
        return self.env.step(multi_binary_action)
    
    def reset(self, **kwargs):
        """
        Reset the environment.
        
        Returns:
            observation: The initial observation after resetting
        """
        return self.env.reset(**kwargs)
class ImprovedFrameSkip(gym.Wrapper):
    def __init__(self, env, n=4, stickprob=0.1, exploration_prob=0.05):
        """
        A more controlled frame skipping wrapper for Sonic.
        
        Args:
            env: The environment to wrap
            n: Number of frames to skip
            stickprob: Probability of repeating the previous action
            exploration_prob: Probability of inserting a random action
        """
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
        terminated = False
        truncated = False
        totrew = 0
        totinfo = {}

        for i in range(self.n):
            # Maintain previous action with low probability
            if self.curac is None or self.rng.rand() > self.stickprob:
                self.curac = ac

            # Occasional very brief random exploration
            if i == 0 and self.rng.rand() < self.exploration_prob:
                # Briefly inject a small random variation, favoring forward momentum
                if self.rng.rand() < 0.7:  # 70% chance to keep forward direction
                    self.curac = np.array(self.curac)
                    # Slightly modify to maintain general direction
                    self.curac[7] = 1   # Ensure right movement
                else:
                    self.curac = self.env.action_space.sample()

            ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            totinfo = info  # Keep the last info
            
            if terminated or truncated:
                break

        return ob, totrew, terminated, truncated, totinfo
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

def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs, render_mode="rgb_array")
    env = ImprovedFrameSkip(env, n=4, stickprob=0.1, exploration_prob=0.05)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)   
    return env

def wrap_deepmind_retro(env):
    """Configure environment for retro games with DQN-style preprocessing"""
    env = WarpFrame(env)
    return env

class VideoRecordingCallback(BaseCallback):
    def __init__(self, save_interval=50000, video_folder=None, verbose=1):
        super().__init__(verbose)
        self.save_interval = save_interval
        self.video_folder = video_folder or get_unique_video_folder("./videos")
        
    def _on_step(self) -> bool:
        # Record video every save_interval steps
        if self.n_calls % self.save_interval == 0:
            if hasattr(self.training_env, 'env') and hasattr(self.training_env.env, 'video_recorder'):
                try:
                    self.training_env.env.video_recorder.capture_frame()
                except Exception as e:
                    print(f"Video recording error: {e}")
        
        return True
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="SonicTheHedgehog-Genesis")
    parser.add_argument("--state", default="GreenHillZone.Act1")
    parser.add_argument("--scenario", default="contest")
    parser.add_argument("--save_interval", type=int, default=50000, help="Timesteps interval to save video")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint to resume training")
    return parser.parse_args()

def get_unique_video_folder(base_path):
    """Generate a unique folder for each run based on the timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_path, f"videos_{timestamp}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    def make_env(game, state, rank):
        def _init():
            # Create the environment
            env = make_retro(game=game, state=state)
            env = SonicActionWrapper(env)
            #env = RightMovementBiasWrapper(env)
            #env = SonicRewardWrapper(env)
            env.current_level = state 
            env = wrap_deepmind_retro(env)
            
            # Set up video recording
            video_folder = get_unique_video_folder("./videos")
            env = RecordVideo(
                env,
                video_folder=video_folder,
                name_prefix="sonic_training",
                
                episode_trigger=lambda episode_id: episode_id % 50 == 0,  # Record every 50 episode for debugging
                video_length=0  # Record full episodes
            )

            # Record episode statistics
            env = RecordEpisodeStatistics(env)
            env = Monitor(env, f"./logs/train_{rank}")
            return env
        return _init
    
    # Create output directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Initialize curriculum manager
    #curriculum_manager = CurriculumManager()
    
    # Create training environments
    env = SubprocVecEnv([make_env(args.game, args.state, i) 
                         for i in range(args.num_envs)])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    
    # Create evaluation environment
    eval_env = SubprocVecEnv([make_env(args.game, args.state, 999)])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // args.num_envs,
        save_path="./models/",
        name_prefix="sonic_dqn"
    )

    video_callback = VideoRecordingCallback(
        save_interval=args.save_interval, 
        video_folder=get_unique_video_folder("./videos")
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
    """    
    curriculum_callback = CurriculumCallback(
        curriculum_manager,
        eval_env,
        check_freq=10000
    )"""

    tensorboard_log_dir = os.path.join(
    "./logs/tensorboard/", 
    datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    # Load or create the model
    if args.checkpoint is not None:
        print(f"Cargando el modelo desde {args.checkpoint}...")
        model = DQN.load(args.checkpoint, env=env)
    else:
        print("Creando un nuevo modelo...")
        print(env.observation_space)
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=516),
            net_arch=[]  # No additional layers, architecture is fully defined in the CustomCNN
        )
        model = DQN(
            policy="CnnPolicy",
            env=env,
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            tau=0.9,
            gamma=0.95,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=10000,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            max_grad_norm=0.4,
            verbose=1,
            tensorboard_log=tensorboard_log_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
            policy_kwargs=policy_kwargs
        )
    
    # Train the model
    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=False,
        callback=[checkpoint_callback, eval_callback, metrics_callback],
        log_interval=50,
        progress_bar=True
    )

    # Save the final model
    model.save("./models/sonic_dqn_final")
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
