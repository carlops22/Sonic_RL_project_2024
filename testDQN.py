import argparse
import retro
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.atari_wrappers import WarpFrame
from gymnasium.wrappers.time_limit import TimeLimit
import os

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

class ImprovedFrameSkip(gym.Wrapper):
    def __init__(self, env, n=4, stickprob=0.1, exploration_prob=0.05):
        super().__init__(env)
        self.n = n
        self.stickprob = stickprob
        self.exploration_prob = exploration_prob
        self.curac = [0] * 12  # Ensure it's a 12-element list (array)
        self.rng = np.random.RandomState()

    def reset(self, **kwargs):
        self.curac = [0] * 12  # Reset the action state
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        totinfo = {}

        for i in range(self.n):
            if self.curac is None or self.rng.rand() > self.stickprob:
                self.curac = [0] * 12  # Reset action state if needed
                self.curac[ac] = 1  # Convert action to a multi-binary vector
            
            if i == 0 and self.rng.rand() < self.exploration_prob:
                # Occasional random exploration logic
                if self.rng.rand() < 0.7:
                    self.curac[7] = 1  # Example of right movement
                
            ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            totinfo = info
            
            if terminated or truncated:
                break

        return ob, totrew, terminated, truncated, totinfo
def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs, render_mode="rgb_array")
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)   
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="SonicTheHedgehog-Genesis")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--model", type=str, default="star.zip", help="Path to load the model")
    parser.add_argument("--scenario", type=str, default="contest")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
    args = parser.parse_args()

    # Cargar el modelo si existe, sino mostrar un mensaje
    if os.path.exists(args.model):
        print(f"Cargando el modelo desde {args.model}...")
        model = DQN.load(args.model)
    else:
        print("No se encontró el archivo del modelo. Por favor, entrena y guarda el modelo en 'ppo.zip'.")
        return
    def make_env(game, state):
        def _init():
            # Create the environment
            env = make_retro(game=game, state=state)
            env = SonicActionWrapper(env)
            #env = RightMovementBiasWrapper(env)
            #env = SonicRewardWrapper(env)
            env.current_level = state 
            env = WarpFrame(env)
            
            return env
        return _init
    
    env = SubprocVecEnv([make_env(args.game, args.state)])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
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
