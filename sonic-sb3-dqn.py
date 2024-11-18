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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import retro
import os
class SonicRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_info = {}
        self.reset_metrics()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reset_metrics()
        return obs, info

    def reset_metrics(self):
        self.farthest_distance = 0
        self.previous_info = {}
        self.prev_x = self.unwrapped.data.lookup_all().get('x', 0)
        self.max_x = self.prev_x

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_info = self.unwrapped.data.lookup_all()
        prev_info = self.previous_info or {}
        
        custom_reward = 0


        # 1. Reward for advancement
        current_x = current_info.get('x', 0)
        if current_x - self.farthest_distance > 100:
            self.farthest_distance = current_x
            advance_reward = self.farthest_distance * 0.01
            custom_reward += advance_reward

        # 2. Reward for killing enemies
        if prev_info:
            score_diff = current_info.get('score', 0) - prev_info.get('score', 0)
            enemy_kill_reward = max(0, score_diff) * 10
            custom_reward += enemy_kill_reward

        # 3. Penalty for losing lives
        if prev_info:
            life_diff = current_info.get('lives', 0) - prev_info.get('lives', 0)
            if life_diff < 0:
                custom_reward += -50
                self.farthest_distance = 0  # Reset farthest distance on death

        # 4. Penalty for losing rings
        if prev_info:
            ring_diff = prev_info.get('rings', 0) - current_info.get('rings', 0)
            ring_loss_penalty = -5 * ring_diff if ring_diff > 0 else 0
            custom_reward += ring_loss_penalty

        # 5. Reward for collecting rings
        if prev_info:
            ring_gain = max(0, current_info.get('rings', 0) - prev_info.get('rings', 0))
            ring_reward = ring_gain * 0.9
            custom_reward += ring_reward

        # 6. Bonus for completing the level
        level_complete_bonus = 10000 if current_info.get('level_end_bonus', 0) > 0 else 0
        custom_reward += level_complete_bonus

        # 7. Exploration bonus
        x_progress = max(0, current_x - self.prev_x)
        progress_reward = np.sqrt(x_progress) * 0.25
        custom_reward += progress_reward
        if x_progress > 0:
            custom_reward += 0.005

        # Combine custom reward with clipped environment reward
        total_reward = custom_reward + np.clip(reward * 0.05, -1.0, 1.0)

        # Prevent extreme rewards
        total_reward = np.clip(total_reward, -2.0, 2.0)

        # Update previous state
        self.prev_x = current_x
        self.previous_info = current_info

        return obs, total_reward, terminated, truncated, info

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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="SonicTheHedgehog-Genesis")
    parser.add_argument("--state", default="GreenHillZone.Act1")
    parser.add_argument("--scenario", default="contest")
    parser.add_argument("--save_interval", type=int, default=100, help="Episodes interval to save video")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint to resume training")
    args = parser.parse_args()


    def get_unique_video_folder(base_path):
        """Generate a unique folder for each run based on the timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(base_path, f"videos_{timestamp}")

    def make_env(game, state, scenario, rank):
        def _init():
            env = make_retro(game=game, state=state, scenario=scenario)
            env = SonicRewardWrapper(env) 
            env = MultiBinaryToDiscreteWrapper(env)
            env = wrap_deepmind_retro(env)
            video_folder = get_unique_video_folder("./videos")
            env = RecordVideo(
                env,
                video_folder="videos1",
                name_prefix="training_run",
                episode_trigger=lambda x: x % 10 == 0
            )
            env = RecordEpisodeStatistics(env)
            env = Monitor(env, f"./logs/train_{rank}")
            return env
        return _init
    
    # Create output directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Create vectorized environment
    env = SubprocVecEnv([make_env(args.game, args.state, args.scenario, i) for i in range(args.num_envs)])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    # Create evaluation environment
    eval_env = SubprocVecEnv([make_env(args.game, args.state, args.scenario, 0)])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // args.num_envs,
        save_path="./models/",
        name_prefix="sonic_dqn"
    )
    
    eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path="./models/best_model",
        log_path="./logs/eval/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True
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
            callback=[checkpoint_callback, eval_callback],
            log_interval=50,
            progress_bar = True
        )

    # Save the final model
    model.save("./models/sonic_dqn_final")
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
