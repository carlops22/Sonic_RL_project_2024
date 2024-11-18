import argparse
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
import retro
from stable_baselines3.common.vec_env import DummyVecEnv
import os


class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        super().__init__(env)
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
                ob, rew, terminated, truncated, info = self.env.step(self.curac, want_render=False)
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, render_mode="rgb_array", **kwargs)  # Cambiado a render_mode="rgb_array"
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind_retro(env):
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Airstriker-Genesis")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--save_interval", type=int, default=100, help="Episodes interval to save video")
    parser.add_argument("--model_path", type=str, default="ppo.zip", help="Path to save/load the model")
    args = parser.parse_args()

    def make_env():
        env = make_retro(game=args.game, state=args.state, scenario=args.scenario)
        env = wrap_deepmind_retro(env)
        env = RecordVideo(
            env,
            video_folder="videos1",
            name_prefix="training_run",
            episode_trigger=lambda episode_id: episode_id % args.save_interval == 0
        )
        env = RecordEpisodeStatistics(env)
        return env

    venv = VecTransposeImage(VecFrameStack(DummyVecEnv([make_env]), n_stack=4))

    # Cargar el modelo si existe, sino, crear uno nuevo
    if os.path.exists(args.model_path):
        print(f"Cargando el modelo desde {args.model_path}...")
        model = PPO.load(args.model_path, env=venv)
    else:
        print("Creando un nuevo modelo...")
        model = PPO(
            policy="CnnPolicy",
            env=venv,
            learning_rate=lambda f: f * 0.01,#2.5e-4,
            n_steps=128,
            batch_size=32,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
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
