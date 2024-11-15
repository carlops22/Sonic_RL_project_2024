import argparse
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import retro
import os

class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, render_freq=10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.render_freq = render_freq

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
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
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
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info

class RewardScaler(gym.RewardWrapper):
    """Scale rewards to be in a reasonable range for DQN"""
    def reward(self, reward):
        return reward * 0.01

def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs, render_mode=None)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def wrap_deepmind_retro(env):
    """Configure environment for retro games with DQN-style preprocessing"""
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = RewardScaler(env)
    return env

def make_env(game, state, scenario, rank):
    def _init():
        env = make_retro(game=game, state=state, scenario=scenario)
        env = MultiBinaryToDiscreteWrapper(env)
        env = wrap_deepmind_retro(env)
        env = Monitor(env, f"./logs/train_{rank}")
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="SonicTheHedgehog-Genesis")
    parser.add_argument("--state", default="GreenHillZone.Act1")
    parser.add_argument("--scenario", default="contest")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint to resume training")
    args = parser.parse_args()

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
        model = DQN.load(args.checkpoint, env=env)
    else:
    # Initialize DQN model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DQN(
            policy="CnnPolicy",
            env=env,
            learning_rate=2.5e-4,
            buffer_size=60000,
            learning_starts=10000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            max_grad_norm=10,
            verbose=1,
            tensorboard_log="./logs/tensorboard/"
        )
        model.to(device)
        # Train the model
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, eval_callback],
            log_interval=100
        )

    # Save the final model
    model.save("./models/sonic_dqn_final")
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
