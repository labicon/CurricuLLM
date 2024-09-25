import gymnasium as gym

from stable_baselines3.common.utils import set_random_seed

import Curriculum
import importlib

def make_env(env_id: str, rank: int, task = None, seed: int = 0, render_mode: str = None):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        importlib.reload(Curriculum)
        if render_mode is None:
            env = gym.make(env_id)
        else:
            env = gym.make(env_id, render_mode=render_mode)
            
        if task is not None:
            env.set_task(task)
        # check_env(env) # check the environment
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init