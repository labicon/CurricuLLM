import gymnasium as gym
import os
import numpy as np
import pandas as pd

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure

from evaluation.evalcallback_feedback import CurriculumEvalCallback

from utils.train_utils import *
from train_fetch import Curriculum_Module

if __name__ == "__main__":
    env_name = "Fetch_Push"
    env_path = "./environments/Curriculum/envs/Fetch_Push.py"
    logger_path = "./logs/Fetch/Push/"

    # Generate curriculum module
    curriculum_module = Curriculum_Module(env_name, env_path, logger_path)

    # Generate curriculum and return list of dictionaries with task details
    curriculum_module.generate_curriculum()
    
    # Train curriculum
    curriculum_module.train_curriculum()