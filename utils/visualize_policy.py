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

from utils.envs_utils import *

if __name__ == "__main__":
    env_id = "Curriculum/HalfCheetah-v4"
    num_cpu = 4

    logger_path = "./logs/" + env_id + "_2/"

    test_env = SubprocVecEnv([make_env(env_id, i, render_mode="human") for i in range(num_cpu)])

    model = CurriculumPPO.load("./logs/Curriculum/HalfCheetah-v4_2/ppo_0")

    # Visualize the policy
    obs = test_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = test_env.step(action)
        test_env.render()