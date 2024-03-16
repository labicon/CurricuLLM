import os

import gymnasium as gym
from typing import Any, Callable, Dict, Optional, Type, Union

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env

import Curriculum
from evaluation.evalcallback_feedback import CurriculumEvalCallback

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

def single_task_training(model, env_id, task, logger_path, eval_callback: CurriculumEvalCallback, num_cpu=4, total_timesteps=200_000):
    # Create the vectorized environment
    training_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])

    eval_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])

    # change eval env and training env
    eval_callback.change_environment(eval_env, task)
    model.set_env(training_env)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(logger_path + "/final_model")