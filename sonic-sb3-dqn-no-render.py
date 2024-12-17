import argparse
import gymnasium as gym
import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import retro
import os

# ... [previous wrapper classes remain the same] ...

def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(
        game, 
        state, 
        **kwargs,
        render_mode=None  # Disable rendering
    )
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def make_env(game, state, scenario, rank):
    def _init():
        env = make_retro(game=game, state=state, scenario=scenario)
        env = wrap_deepmind_retro(env)
        env = Monitor(env, f"./logs/train_{rank}", allow_early_resets=True)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="SonicTheHedgehog-Genesis")
    parser.add_argument("--state", default="GreenHillZone.Act1")
    parser.add_argument("--scenario", default="contest")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    args = parser.parse_args()

    # Create output directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)
    os.makedirs("logs/eval", exist_ok=True)
    
    # Create vectorized environment with multiple parallel environments
    env = DummyVecEnv([make_env(args.game, args.state, args.scenario, i) for i in range(args.num_envs)])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    # Create evaluation environment (single environment is enough for eval)
    eval_env = DummyVecEnv([make_env(args.game, args.state, args.scenario, 0)])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // args.num_envs,  # Adjust frequency based on number of envs
        save_path="./models/",
        name_prefix="sonic_dqn"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model",
        log_path="./logs/eval/",
        eval_freq=10000 // args.num_envs,  # Adjust frequency based on number of envs
        n_eval_episodes=5,
        deterministic=True
    )

    # Initialize DQN model with optimized parameters
    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=2.5e-4,
        buffer_size=100000,
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
        tensorboard_log="./logs/tensorboard/",
        device="cuda"  # Use GPU if available
    )

    # Train the model
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10
    )

    # Save final model
    model.save("./models/sonic_dqn_final")

if __name__ == "__main__":
    main()
