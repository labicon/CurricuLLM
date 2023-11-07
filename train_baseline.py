import gymnasium as gym
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure

import Curriculum

from policy.curriculum_ppo import CurriculumPPO

from evaluation.evaluation import curriculum_evaluate_policy
from evaluation.evalcallback import CurriculumEvalCallback


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
            env = gym.make(env_id, task=task)
        else:
            env = gym.make(env_id, task=task, render_mode=render_mode)
        check_env(env) # check the environment
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

if __name__ == "__main__":
    env_id = "Curriculum/HalfCheetah-v4"
    num_cpu = 4

    # task_list, task_threshold = get_task_env(env_id)
    task = None
    task_list = None

    max_step = get_env_maximum_step(env_id)
    # task_threshold = {task: task_threshold[task] * max_step for task in task_threshold}
    task_threshold = None

    # Create the logger
    logger_path = "./logs/" + env_id + "_baseline/"
    new_logger = configure(logger_path, ["stdout", "csv"])

    # Create the vectorized environment
    training_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])

    eval_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])

    # Create the callback: check every 10000 steps
    eval_callback = CurriculumEvalCallback(eval_env,
                                            log_path=logger_path, eval_freq=5_000,
                                            deterministic=True, render=False, warn=False, 
                                            task=task, task_list=task_list, task_threshold=task_threshold)

    model = CurriculumPPO("MlpPolicy", 
                              training_env, 
                              task_list=task_list, 
                              task_threshold=task_threshold, 
                              verbose=1)
    model.set_logger(new_logger)

    for i in range(30):
        print(f"Training iteration: {i}")
        if task is not None:
            print(f"Task: {task}")
            print(f"Task Threshold: {task_threshold[task]}")
        else:
            print("Task: Main")

        task_finished = model.learn(total_timesteps=500_000, callback=eval_callback)

        if task_finished:
            print("Task complete! Saving model...")
            model.save(f"./logs/{env_id}/ppo_{task}")

            # # Visualize the trained policy
            # obs = eval_env.reset()
            # for _ in range(1000):
            #     action, _states = model.predict(obs)
            #     obs, rewards, dones, info = eval_env.step(action)
            #     eval_env.render()

            try:
                task = task_list[task_list.index(task) + 1]
            except:
                print("All tasks complete!")
                task = None

            del training_env, eval_env
            training_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])
            eval_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])
            eval_callback.eval_env = eval_env
            eval_callback.change_current_task(task)
            model.set_env(training_env)
        else:
            print("Task not complete! Continuing training...")
