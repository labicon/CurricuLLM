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
    env_id = "FetchPush-v2"
    num_cpu = 16

    if len(sys.argv) > 1:
        task = sys.argv[1]
        sample_num = sys.argv[2]
        log_dir = "./logs/AntMaze_UMaze_SAC_empty/" + task + "/sample_" + sample_num
    else:
        log_dir = "./logs/Fetch/Push_6/[Push to Goal]/sample_1"
        task = None

    test_env = SubprocVecEnv([make_env(env_id, i, render_mode="rgb_array") for i in range(num_cpu)])

    model = SAC.load(log_dir + "/final_model.zip", env=test_env)
    # model = PPO.load(log_dir + "/final_model.zip")

    # Visualize the policy
    obs = test_env.reset()
    test_env = VecVideoRecorder(test_env, video_folder=log_dir,
                        record_video_trigger=lambda x: x == 0, video_length=500,
                        name_prefix="Curriculum")
    test_env.reset()
    for i in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = test_env.step(action)
    
    test_env.close()

    # test_env = SubprocVecEnv([make_env(env_id, i, render_mode="human") for i in range(num_cpu)])

    # obs = test_env.reset()
    # test_env.render("human")
    # # time.sleep(5)
    # # input("Press Enter to continue...")
    # # images = [img]
    # for i in range(50):
    #     # action = np.random.uniform(-1, 1, size=(num_cpu, 8))
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, rewards, dones, info = test_env.step(action)
    #     test_env.render("human")
    #     time.sleep(0.05)
    #     # images.append(img)
    #     # print(img)
    #     # print(rewards)
    #     # print(dones)
    #     # print(info)

    # print(images)
    # imageio.mimsave(log_dir + "/rollout.gif", [np.array(img) for i, img in enumerate(images)], fps=20)