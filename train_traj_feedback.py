import gymnasium as gym
import os
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure

from evaluation.evalcallback_feedback import CurriculumEvalCallback

from utils.envs_utils import *

from gpt.curriculum_api_traj import generate_curriculum, feedback

def single_task_training(model, env_id, task, logger_path, eval_callback: CurriculumEvalCallback, num_cpu=4, total_timesteps=200_000):
    # Create the vectorized environment
    training_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])

    eval_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])

    # change eval env and training env
    eval_callback.change_environment(eval_env, task)
    model.set_env(training_env)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(logger_path + f"/ppo_{task}")

def load_training_log(logger_path, task):
    training_log = np.load(logger_path + "/evaluations.npz", allow_pickle=True)

    reward_task = training_log["results_task"].mean(axis=1)
    task_list = training_log["task"]
    reward_dict = training_log["results_dict"]

    task_indicies = np.where(task_list == task)[0]
    reward_task = reward_task[task_indicies]
    reward_dict = reward_dict[task_indicies]

    averaged_dicts = []

    for row in reward_dict:
        sum_dict = {}
        for col in row:
            for key in col:
                sum_dict[key] = sum_dict.get(key, 0) + col[key]

        avg_dict = {key: value/len(row) for key, value in sum_dict.items()}
        averaged_dicts.append(avg_dict)

    reward_df = pd.DataFrame(averaged_dicts)

    return reward_task, reward_df

if __name__ == "__main__":
    env_id = "Curriculum/Hopper-v5"
    generate_curriculum("hopper")
    task_list = []
    while True:
        task = input("Enter the task list, enter end if there is no more task: ")
        if task == "end":
            break
        else:
            task_list.append(task)

    task = task_list[0]

    # Create the logger
    logger_path = "./logs/" + env_id + "_4/"
    new_logger = configure(logger_path, ["stdout", "csv"])

    # Create the vectorized environment
    num_cpu = 4
    training_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])

    eval_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])

    # Create the callback: check every 10000 steps
    eval_callback = CurriculumEvalCallback(eval_env,
                                            log_path=logger_path, best_model_save_path=logger_path, eval_freq=4_000,
                                            deterministic=True, render=False, warn=False, 
                                            task=task)

    model = PPO("MlpPolicy", 
                    training_env, 
                    verbose=1)
    # model = PPO.load(logger_path + "ppo_velocity_management")
    previous_task = task
    for i in range(10):     
        single_task_training(model, env_id, task, logger_path, eval_callback, num_cpu=num_cpu, total_timesteps=400_000)

        reward_task, reward_df = load_training_log(logger_path, task)

        print("Task reward: ", np.around(reward_task, decimals=4))
        for column in reward_df.columns:
            print(column, np.around(reward_df[column].to_numpy(), decimals=2))

        # get trajectory
        obs = eval_env.reset()
        obs_trajectory = [obs[0]]
        action_trajectory = []
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            action_trajectory.append(action[0])
            obs, rewards, dones, info = eval_env.step(action)
            obs_trajectory.append(obs[0])

        # (length, dimension) array
        obs_trajectory = np.array(obs_trajectory)
        action_trajectory = np.array(action_trajectory)

        feedback("hopper", reward_task, reward_df, task, obs_trajectory, action_trajectory)

        user_input = input("1: move to next task, 2: continue training, 3: exit and revise reward function")

        if user_input == "1":
            current_task_idx = task_list.index(task)
            previous_task = task
            try:
                task = task_list[current_task_idx + 1]
                print("Moving to next task: ", task)
            except:
                print("No more tasks")
                break
        elif user_input == "2":
            continue
        elif user_input == "3":
            break
        else:
            print("Invalid input")
            break


