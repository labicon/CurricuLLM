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

if __name__ == "__main__":
    env_id = "Curriculum/Hopper-v5"
    num_cpu = 4

    test_env = SubprocVecEnv([make_env(env_id, i, render_mode="rgb_array") for i in range(num_cpu)])

    log_dir = "./logs/Curriculum/Hopper-v5_4/"

    model = PPO.load(log_dir + "ppo_hopping_forward")

    # Visualize the policy
    obs = test_env.reset()
    print(obs)
    test_env = VecVideoRecorder(test_env, video_folder=log_dir,
                       record_video_trigger=lambda x: x == 0, video_length=1000,
                       name_prefix="Curriculum-Hopper-v5")
    test_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = test_env.step(action)
    
    test_env.close()