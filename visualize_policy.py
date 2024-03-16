import gymnasium as gym
import os
import imageio
import numpy as np
import time

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.logger import configure

import Curriculum

from utils.train_utils import *

import sys

if __name__ == "__main__":
    env_id = "Curriculum/AntMaze_UMaze-v0"
    num_cpu = 4

    if len(sys.argv) > 1:
        task = sys.argv[1]
        sample_num = sys.argv[2]
        log_dir = "./logs/AntMaze_UMaze_SAC_empty/" + task + "/sample_" + sample_num
    else:
        log_dir = "./logs/Fetch_Push_SAC/reduced_distance_to_goal/sample_0"
        task = None

    model = SAC.load(log_dir + "/final_model.zip")
    # model = PPO.load(log_dir + "/final_model.zip")

    # test_env = SubprocVecEnv([make_env(env_id, i, task=task, render_mode="rgb_array") for i in range(num_cpu)])

    # # Visualize the policy
    # obs = test_env.reset()
    # test_env = VecVideoRecorder(test_env, video_folder=log_dir,
    #                    record_video_trigger=lambda x: x == 0, video_length=1000,
    #                    name_prefix="Curriculum")
    # test_env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, rewards, dones, info = test_env.step(action)
    
    # test_env.close()

    test_env = SubprocVecEnv([make_env(env_id, i, task=task, render_mode="human") for i in range(num_cpu)])

    obs = test_env.reset()
    test_env.render("human")
    # time.sleep(5)
    # input("Press Enter to continue...")
    # images = [img]
    for i in range(1000):
        # action = np.random.uniform(-1, 1, size=(num_cpu, 8))
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = test_env.step(action)
        test_env.render("human")
        time.sleep(0.05)
        # images.append(img)
        # print(img)
        # print(rewards)
        # print(dones)
        # print(info)

    # print(images)
    # imageio.mimsave(log_dir + "/rollout.gif", [np.array(img) for i, img in enumerate(images)], fps=20)