import argparse
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import retro
import os

class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        super
        self.__init__(env)
        self.n = n
        self.stickprob = stickprob

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
            ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info
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

def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, render_mode="rgb_array", **kwargs)

    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def wrap_deepmind_retro(env):
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="SonicTheHedgehog-Genesis")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default="contest")
    parser.add_argument("--save_interval", type=int, default=100, help="Episodes interval to save video")
    parser.add_argument("--model_path", type=str, default="dqn.zip", help="Path to save/load the model")
    args = parser.parse_args()

    def make_env():
        env = make_retro(game=args.game, state=args.state, scenario=args.scenario)
        env = SonicActionWrapper(env)
        env = wrap_deepmind_retro(env)
        env = RecordVideo(
            env,
            video_folder="videos1",
            name_prefix="training_run",
            episode_trigger=lambda episode_id: episode_id % args.save_interval == 0
        )
        env = RecordEpisodeStatistics(env)
        return env

    venv = VecTransposeImage(DummyVecEnv([make_env]))

    # Cargar el modelo si existe, sino, crear uno nuevo
    if os.path.exists(args.model_path):
        print(f"Cargando el modelo desde {args.model_path}...")
        model = DQN.load(args.model_path, env=venv)
    else:
        print("Creando un nuevo modelo...")
        model = DQN(
            policy="CnnPolicy",
            env=venv,
            learning_rate=1e-4,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            verbose=1,
        )

    # Entrenamiento con intervalos de guardado
    total_timesteps = 200_000
    model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,  # Continúa desde donde se quedó
            log_interval=50,
            progress_bar=True
        )
    # Guardar el modelo en el intervalo especificado
    model.save(args.model_path)
        
    print("Entrenamiento completo.")

if __name__ == "__main__":
    main()
