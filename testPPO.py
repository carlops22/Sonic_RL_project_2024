import argparse
import retro
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import WarpFrame
import os


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
        terminated, truncated, totrew = False, False, 0
        for i in range(self.n):
            if self.curac is None or (i == 0 and self.rng.rand() > self.stickprob):
                self.curac = ac
            ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs, render_mode='rgb_array')
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="SonicTheHedgehog-Genesis")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--model", type=str, default="star.zip", help="Path to load the model")
    args = parser.parse_args()

    # Cargar el modelo si existe, sino mostrar un mensaje
    if os.path.exists(args.model):
        print(f"Cargando el modelo desde {args.model}...")
        model = PPO.load(args.model)
    else:
        print("No se encontró el archivo del modelo. Por favor, entrena y guarda el modelo en 'ppo.zip'.")
        return

    # Crear entorno de prueba y aplicar los wrappers necesarios
    env = make_retro(game=args.game, state=args.state)
    env = WarpFrame(env)  # Redimensiona las imágenes a (84, 84, 3)
    env = DummyVecEnv([lambda: env])  # Convertir a entorno vectorizado
    env = VecFrameStack(env, n_stack=4)  # Apilar 4 frames

    print(f"Nivel: {args.state}")
    max_steps = 4000
    episodios_totales = 100
    for episode in range(episodios_totales):
        obs = env.reset()
        #print(f'Ejecutando episodio {episode + 1}...')

        maxDist = 0
        time_out = True
        for i in range(max_steps):
            action, _ = model.predict(obs, deterministic=False)  # Modo de predicción determinístico
            obs, reward, done, info = env.step(action)
            
            if info[0]['x'] > maxDist: maxDist = info[0]['x']
            # Revisar si se ha ganado el nivel (detiene el episodio)
            if info[0]['level_end_bonus']:
                print(f"Episodio {episode + 1}: VICTORIA alcanzada! - Tiempo = ({(266 * i) / 4000} segundos)")
                time_out = False
                break
            if info[0]['lives'] < 1:
                print(f"Episodio {episode + 1}: Muerte")
                time_out = False
                break
        
        if time_out: print(f'Episodio {episode +1} tiempo máximo excedido')
        print(f'distancia maxima = {maxDist}')

    env.close()


if __name__ == "__main__":
    main()
