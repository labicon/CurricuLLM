import gymnasium as gym
import os
import numpy as np
import pandas as pd

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback

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

def analyze_trajectory_fetch(obs_trajectory, goal_trajectory):
    # obs_trajectory: list of observations
    # Get list of end effector position, block position, relative block linear velocity, end effector velocity, goal_pos, gosl_distance

    end_effector_pos = []
    block_pos = []
    block_velocity = []
    end_effector_velocity = []
    goal_pos = []
    goal_distance = []

    for obs, goal in zip(obs_trajectory, goal_trajectory):
        end_effector_pos.append(obs[0:3])
        block_pos.append(obs[3:6])
        block_velocity.append(obs[15:18])
        end_effector_velocity.append(obs[20:23])
        goal_pos.append(goal)
        goal_distance.append(np.linalg.norm(obs[3:6] - goal))

    # change to np array
    end_effector_pos = np.array(end_effector_pos)
    block_pos = np.array(block_pos)
    block_velocity = np.array(block_velocity)
    end_effector_velocity = np.array(end_effector_velocity)
    goal_pos = np.array(goal_pos)
    goal_distance = np.array(goal_distance)

    # Calculate mean and std of each variable
    statistics = {}
    statistics["end_effector_pos_mean"] = np.mean(end_effector_pos, axis=0)
    statistics["end_effector_pos_std"] = np.std(end_effector_pos, axis=0)
    statistics["block_pos_mean"] = np.mean(block_pos, axis=0)
    statistics["block_pos_std"] = np.std(block_pos, axis=0)
    statistics["block_relative_velocity_mean"] = np.mean(block_velocity, axis=0)
    statistics["block_relative_velocity_std"] = np.std(block_velocity, axis=0)
    statistics["end_effector_velocity_mean"] = np.mean(end_effector_velocity, axis=0)
    statistics["end_effector_velocity_std"] = np.std(end_effector_velocity, axis=0)
    statistics["goal_pos_mean"] = np.mean(goal_pos, axis=0)
    statistics["goal_pos_std"] = np.std(goal_pos, axis=0)
    statistics["goal_distance_mean"] = np.mean(goal_distance, axis=0)
    statistics["goal_distance_std"] = np.std(goal_distance, axis=0)

    return statistics

if __name__ == "__main__":
    env_name = "Fetch_Push"
    # generate_curriculum(env_name)
    # task_list = ["reduced_distance_to_goal", "move_block_to_target_position"] #"align_end_effector_with_block", "match_end_effector_velocity_with_block", 
    # # while True:
    # #     task = input("Enter the task list, enter end if there is no more task: ")
    # #     if task == "end":
    # #         break
    # #     else:
    # #         task_list.append(task)

    # curriculum_length = len(task_list)
    # task_idx = input("Enter the task idx to start training: ")
    # task_idx = int(task_idx)
    # task = task_list[task_idx]
    task = None

    # if task_idx != 0:
    #     previous_task = task_list[task_idx - 1]
    #     best_reward_sample = input("Enter the sample number of the best reward function: ")
    #     previous_model_path = f"./logs/{env_name}_SAC/{previous_task}/sample_{best_reward_sample}/final_model.zip"

    for sample_num in range(1):
        # Create the environment
        # env_id = f"Curriculum/{env_name}-v{sample_num}"
        env_id = "FetchPushDense-v2"
        # Create the logger
        logger_path = f"./logs/{env_name}_SAC/{task}/sample_{sample_num}_no_curriculum"
        new_logger = configure(logger_path, ["stdout", "csv"])

        # Create the vectorized environment
        num_cpu = 4
        training_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])

        eval_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])

        # Create the callback
        # eval_callback = CurriculumEvalCallback(eval_env, 
                                            # log_path=logger_path, best_model_save_path=logger_path, 
                                            # eval_freq=1000, 
                                            # deterministic=True, render=False, warn=False,
                                            # task=task)
        eval_callback = EvalCallback(eval_env,
                                    best_model_save_path=logger_path,
                                    log_path=logger_path,
                                    eval_freq=1000,
                                    deterministic=True,
                                    render=False,
                                    warn=False)

        # if task_idx == 0:
        #     # model = PPO("MultiInputPolicy",
        #                 # training_env,
        #                 # verbose=1)
        #     model = SAC("MultiInputPolicy",
        #                 training_env,
        #                 verbose=1)
        # else:
        #     # model = PPO.load(previous_model_path)
        #     model = SAC.load(previous_model_path)

        model = SAC("MultiInputPolicy",
                    training_env,
                    verbose=1)

        # single_task_training(model, env_id, task, logger_path, eval_callback, num_cpu=num_cpu, total_timesteps=1_200_000)
        model.learn(total_timesteps=1_200_000, callback=eval_callback)
        model.save(logger_path + "/final_model")

        del model, training_env, eval_env, eval_callback

    # # Evaluate the trained model
    # statistics = []
    # for sample_num in range(5):
    #     env_id = f"Curriculum/{env_name}-v{sample_num}"
    #     num_cpu = 4
    #     eval_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])
    #     model = SAC.load(f"./logs/{env_name}_SAC/{task}/sample_{sample_num}/final_model")
    #     # model = PPO.load(f"./logs/{env_name}_PPO/{task}/sample_{sample_num}/final_model")
        
    #     # Get trajectory
    #     obs = eval_env.reset()
    #     obs_trajectory = [obs['observation'][0]]   
    #     action_trajectory = []
    #     goal_trajectory = [obs['desired_goal'][0]]
    #     for i in range(1000):
    #         action, _states = model.predict(obs, deterministic=True)
    #         action_trajectory.append(action[0])
    #         obs, rewards, dones, info = eval_env.step(action)
    #         obs_trajectory.append(obs['observation'][0])
    #         goal_trajectory.append(obs['desired_goal'][0])

    #     # Analyze trajectory statistics
    #     single_trajectory_statistics = analyze_trajectory_fetch(obs_trajectory, goal_trajectory)
    #     statistics.append(single_trajectory_statistics)

    #     # print(single_trajectory_statistics["torso_velocity_mean"])
    
    # # Ask LLM to choose the best trajectory and reward function
    # feedback(env_name, task, statistics)

    # user_input = input("Type the idx of reward function to choose: ")
    # print("Train next task with reward function: ", user_input)



