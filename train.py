import gymnasium as gym
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure

from policy.curriculum_ppo import CurriculumPPO

from evaluation.evalcallback import CurriculumEvalCallback

from utils.envs_utils import *


if __name__ == "__main__":
    env_id = "Curriculum/HalfCheetah-v4"
    num_cpu = 4

    task_list, task_threshold = get_task_env(env_id)
    task = task_list[0]

    max_step = get_env_maximum_step(env_id)
    task_threshold = {task: task_threshold[task] * max_step for task in task_threshold}

    # Create the logger
    logger_path = "./logs/" + env_id + "_2/"
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
