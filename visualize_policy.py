import gymnasium as gym
import os

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.logger import configure

import Curriculum

from utils.envs_utils import *

import sys

if __name__ == "__main__":
    env_id = "Curriculum/AntMaze_UMaze-v0"
    num_cpu = 4

    test_env = SubprocVecEnv([make_env(env_id, i, render_mode="rgb_array") for i in range(num_cpu)])

    log_dir = "./logs/AntMaze_UMaze/basic_locomotion/sample_4"
    if len(sys.argv) > 1:
        task = sys.argv[1]
        sample_num = sys.argv[2]
        log_dir = "./logs/AntMaze_UMaze_SAC/" + task + "/sample_" + sample_num 
        
    model = SAC.load(log_dir + "/final_model.zip")

    # Visualize the policy
    obs = test_env.reset()
    # print(obs)
    test_env = VecVideoRecorder(test_env, video_folder=log_dir,
                       record_video_trigger=lambda x: x == 0, video_length=1000,
                       name_prefix="Curriculum-AntMaze-UMaze")
    test_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = test_env.step(action)
    
    test_env.close()