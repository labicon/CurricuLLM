import gymnasium as gym
import os

from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure

from evaluation.evalcallback import CurriculumEvalCallback

from utils.envs_utils import *


if __name__ == "__main__":
    env_id = "Curriculum/Hopper-v5"
    num_cpu = 4

    task = None

    max_step = get_env_maximum_step(env_id)

    # Create the logger
    logger_path = "./logs/" + env_id + "_ppo_1/"
    new_logger = configure(logger_path, ["stdout", "csv"])

    # Create the vectorized environment
    training_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])

    eval_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])

    # Create the callback: check every 10000 steps
    eval_callback = CurriculumEvalCallback(eval_env,
                                            log_path=logger_path, best_model_save_path=logger_path, eval_freq=4_000,
                                            deterministic=True, render=False, warn=False, 
                                            task=task)

    model = PPO("MlpPolicy", training_env, verbose=1)
    model.set_logger(new_logger)

    task_finished = model.learn(total_timesteps=600_000, callback=eval_callback)

    model.save(logger_path + "ppo_final")
