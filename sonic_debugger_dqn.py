import argparse
import os
import pathlib
import gymnasium as gym
import retro
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import WarpFrame
from gymnasium.wrappers import RecordVideo

class CompatibleActionWrapper(gym.Wrapper):
    """
    Wrapper to make the environment compatible with the trained model's action space
    while still forcing right movement
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Restore the original action space
        self.action_space = gym.spaces.Discrete(8)
        
        # Sonic action space (typically 12 actions)
        # Example right movement action that might be similar to training
        self.right_actions = [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Basic right movement
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # Right + jump
        ]
    
    def step(self, action):
        # Ignore input action, always use a right-movement action
        # This ensures we're always moving right while maintaining action space compatibility
        return self.env.step(self.right_actions[0])
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

def make_debug_env(game="SonicTheHedgehog-Genesis", state="GreenHillZone.Act1"):
    """Create environment for debugging with compatible action space"""
    os.makedirs("./debug_videos", exist_ok=True)
    
    # Create environment with original action space
    env = retro.make(game, state, render_mode="rgb_array")
    
    # Add compatibility wrapper before other wrappers
    env = CompatibleActionWrapper(env)
    env = WarpFrame(env)
    
    # Add video recording
    env = RecordVideo(
        env, 
        video_folder="./debug_videos", 
        name_prefix="sonic_right_debug",
        episode_trigger=lambda episode_id: True
    )
    
    return env

def robust_model_load(checkpoint_path):
    """
    Robust model loading that handles various path scenarios
    """
    checkpoint_path = pathlib.Path(checkpoint_path)
    
    possible_paths = [
        checkpoint_path,
        checkpoint_path.with_suffix('.zip'),
        checkpoint_path.parent / (checkpoint_path.name + '.zip'),
        checkpoint_path.parent / checkpoint_path.stem / (checkpoint_path.stem + '.zip')
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"Loading model from: {path}")
            return path
    
    raise FileNotFoundError(f"Could not find model checkpoint at any of these paths:\n" + 
                             "\n".join(str(p) for p in possible_paths))

def load_and_debug_model(checkpoint_path):
    """Load model and run debugging session with action space compatibility"""
    # Robustly find the model file
    model_path = robust_model_load(checkpoint_path)
    
    # Create environment using DummyVecEnv
    def env_creator():
        return make_debug_env()
    
    vec_env = DummyVecEnv([env_creator])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecTransposeImage(vec_env)
    
    # Load model with robust path
    model = DQN.load(model_path, env=vec_env)
    
    # Debugging session
    obs = vec_env.reset()
    total_reward = 0
    steps = 0
    max_steps = 1000  # Prevent infinite running
    
    try:
        while steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            
            total_reward += reward[0]
            steps += 1
            
            print(f"Step: {steps}, Reward: {reward[0]}, Action: {action[0]}")
            
            if done[0]:
                print(f"Episode finished. Total Reward: {total_reward}, Total Steps: {steps}")
                obs = vec_env.reset()
                total_reward = 0
                steps = 0
    
    except KeyboardInterrupt:
        print("Debugging session interrupted.") 
    finally:
        vec_env.close()

def main():
    parser = argparse.ArgumentParser(description="Sonic Right Movement Debugger")
    parser.add_argument("checkpoint", help="Path to the model checkpoint")
    args = parser.parse_args()
    
    load_and_debug_model(args.checkpoint)

if __name__ == "__main__":
    main()