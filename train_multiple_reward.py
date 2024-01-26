import gymnasium as gym
import os
import numpy as np
import pandas as pd

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure

from evaluation.evalcallback_feedback import CurriculumEvalCallback

from utils.envs_utils import *

from gpt.curriculum_api_multiple import generate_curriculum, generate_reward, feedback

def single_task_training(model, env_id, task, logger_path, eval_callback: CurriculumEvalCallback, num_cpu=4, total_timesteps=200_000):
    # Create the vectorized environment
    training_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])

    eval_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])

    # change eval env and training env
    eval_callback.change_environment(eval_env, task)
    model.set_env(training_env)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(logger_path + "/final_model")

def analyze_trajectory_ant(obs_trajectory, goal_trajectory):
    # obs_trajectory: list of observations
    # Get list of torso_coord, torso_orientation, torso_velocity, torso_angular_velocity, goal_pos, gosl_distance
    torso_coord = []
    torso_orientation = []
    torso_velocity = []
    torso_angular_velocity = []
    goal_pos = []
    goal_distance = []

    for obs, goal in zip(obs_trajectory, goal_trajectory):
        torso_coord.append(obs[0:3])
        torso_orientation.append(obs[3:7])
        torso_velocity.append(obs[15:18])
        torso_angular_velocity.append(obs[18:21])
        goal_pos.append(goal)
        goal_distance.append(np.linalg.norm(obs[0:2] - goal))

    # change to np array
    torso_coord = np.array(torso_coord)
    torso_orientation = np.array(torso_orientation)
    torso_velocity = np.array(torso_velocity)
    torso_angular_velocity = np.array(torso_angular_velocity)
    goal_pos = np.array(goal_pos)
    goal_distance = np.array(goal_distance)

    # Calculate mean and std of each variable
    statistics = {}
    statistics["torso_coord_mean"] = np.mean(torso_coord, axis=0)
    statistics["torso_coord_std"] = np.std(torso_coord, axis=0)
    statistics["torso_orientation_mean"] = np.mean(torso_orientation, axis=0)
    statistics["torso_orientation_std"] = np.std(torso_orientation, axis=0)
    statistics["torso_velocity_mean"] = np.mean(torso_velocity, axis=0)
    statistics["torso_velocity_std"] = np.std(torso_velocity, axis=0)
    statistics["torso_angular_velocity_mean"] = np.mean(torso_angular_velocity, axis=0)
    statistics["torso_angular_velocity_std"] = np.std(torso_angular_velocity, axis=0)
    statistics["goal_pos_mean"] = np.mean(goal_pos, axis=0)
    statistics["goal_pos_std"] = np.std(goal_pos, axis=0)
    statistics["goal_distance_mean"] = np.mean(goal_distance, axis=0)
    statistics["goal_distance_std"] = np.std(goal_distance, axis=0)

    return statistics

if __name__ == "__main__":
    env_name = "AntMaze_UMaze"
    # generate_curriculum(env_name)
    task_list = ['basic_locomotion', 'orientation_control', 'goal_orientation', 'navigation_turning', 'original_task']
    # while True:
    #     task = input("Enter the task list, enter end if there is no more task: ")
    #     if task == "end":
    #         break
    #     else:
    #         task_list.append(task)

    curriculum_length = len(task_list)
    task_idx = input("Enter the task idx to start training: ")
    task_idx = int(task_idx)
    task = task_list[task_idx]

    if task_idx != 0:
        previous_task = task_list[task_idx - 1]
        best_reward_sample = input("Enter the sample number of the best reward function: ")
        previous_model_path = f"./logs/{env_name}_SAC/{previous_task}/sample_{best_reward_sample}/final_model.zip"

    for sample_num in range(5):
        # Create a number of environments and train

        env_id = f"Curriculum/{env_name}-v{sample_num}"
        # Create the logger
        logger_path = f"./logs/{env_name}_SAC/{task}/sample_{sample_num}"
        new_logger = configure(logger_path, ["stdout", "csv"])

        # Create the vectorized environment
        num_cpu = 4
        training_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])

        eval_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])

        # Create the callback
        eval_callback = CurriculumEvalCallback(eval_env, 
                                            log_path=logger_path, best_model_save_path=logger_path, 
                                            eval_freq=1000, 
                                            deterministic=True, render=False, warn=False,
                                            task=task)
        
        if task_idx == 0:
            # model = PPO("MultiInputPolicy",
            #             training_env,
            #             verbose=1)
            model = SAC("MultiInputPolicy",
                        training_env,
                        verbose=1)
        else:
            # model = PPO.load(previous_model_path)
            model = SAC.load(previous_model_path)

        single_task_training(model, env_id, task, logger_path, eval_callback, num_cpu=num_cpu, total_timesteps=400_000)

    # Evaluate the trained model
    statistics = []
    for sample_num in range(5):
        env_id = f"Curriculum/{env_name}-v{sample_num}"
        num_cpu = 4
        eval_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])
        model = SAC.load(f"./logs/{env_name}_SAC/{task}/sample_{sample_num}/final_model")
        
        # Get trajectory
        obs = eval_env.reset()
        obs_trajectory = [obs['observation'][0]]   
        action_trajectory = []
        goal_trajectory = [obs['desired_goal'][0]]
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            action_trajectory.append(action[0])
            obs, rewards, dones, info = eval_env.step(action)
            obs_trajectory.append(obs['observation'][0])
            goal_trajectory.append(obs['desired_goal'][0])

        # Analyze trajectory statistics
        single_trajectory_statistics = analyze_trajectory_ant(obs_trajectory, goal_trajectory)
        statistics.append(single_trajectory_statistics)
    
    # Ask LLM to choose the best trajectory and reward function
    feedback(env_name, task, statistics)

    user_input = input("Type the idx of reward function to choose: ")
    print("Train next task with reward function: ", user_input)



