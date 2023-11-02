import gymnasium as gym
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure

from envs.curriculum_halfcheetah import HalfCheetahEnv

from policy.curriculum_ppo import CurriculumPPO

from utils.evaluation import curriculum_evaluate_policy
from utils.evalcallback import CurriculumEvalCallback


def make_env(env_id: str, rank: int, task = None, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, task=task)
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

    task_list, task_threshold = get_task_env(env_id)
    task = task_list[0]

    max_step = get_env_maximum_step(env_id)

    # Create the logger
    logger_path = "./logs/" + env_id + "/"
    new_logger = configure(logger_path, ["stdout", "csv", "tensorboard"])

    # Create the vectorized environment
    training_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])

    eval_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])

    # Create the callback: check every 10000 steps
    eval_callback = CurriculumEvalCallback(eval_env,
                                            log_path=logger_path, eval_freq=5_000,
                                            deterministic=True, render=False, warn=False, task=task)

    model = CurriculumPPO("MlpPolicy", 
                              training_env, 
                              task_list=task_list, 
                              task_threshold=task_threshold, 
                              verbose=1)
    model.set_logger(new_logger)

    for i in range(30):
        print(f"Task: {task}")
        print(f"Task Threshold: {task_threshold[task]}")

        model.learn(total_timesteps=20_000, callback=eval_callback)

        print("Evaluating the trained policy")
        mean_reward_main, std_reward_main, mean_reward_task, std_reward_task = \
            curriculum_evaluate_policy(model, 
                                       eval_env, 
                                       n_eval_episodes=4, 
                                       deterministic=False, render=False, warn=False)

        print(f"mean_reward_main: {mean_reward_main}, std_reward_main: {std_reward_main}")
        print(f"mean_reward_task: {mean_reward_task}, std_reward_task: {std_reward_task}")

        model.save(f"./logs/{env_id}/ppo_{task}/model_{i}")

        if mean_reward_task > task_threshold[task] * max_step:
            print("Task complete!")
            try:
                task = task_list[task_list.index(task) + 1]
            except:
                print("All tasks complete!")
                task = None
            training_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])
            eval_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])
            eval_callback.task = task
            model.set_env(training_env)
