import gymnasium as gym

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env

import Curriculum

def make_env(env_id: str, rank: int, task = None, seed: int = 0, render_mode: str = None):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
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

def get_task_env(env_id: str):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    env = gym.make(env_id)
    try:
        task_list = env.unwrapped.task_list
    except:
        task_list = None
    
    try:
        task_threshold = env.unwrapped.task_threshold
    except:
        task_threshold = None
    return task_list, task_threshold

def get_env_maximum_step(env_id: str):
    """
    Get maximum step of the environment.
    """
    env = gym.make(env_id)
    max_step = env.spec.max_episode_steps

    return max_step