import gymnasium as gym
import os
import numpy as np
import pandas as pd

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure

from evaluation.evalcallback_feedback import CurriculumEvalCallback
from utils.train_utils import *
from gpt.curriculum_api import CurriculumAPI

class Curriculum_Module:
    def __init__(self, env_name):
        self.env_name = env_name
        self.prompt_path = "./gpt/prompt/"
        self.gpt_api = CurriculumAPI(self.env_name, self.prompt_path)
        
    def generate_curriculum(self):
        # Generate curriculum and return list of dictionaries with task details
        self.curriculum_info = self.gpt_api.generate_curriculum()
        self.curriculum_length = len(self.curriculum_info)

    def generate_env_with_task(self, task, previous_reward_code=None, env_version=0):
        # Created environment with task and save as version = env_version
        # First, generate reward code from given task info
        reward_code = None
        max_attempt = 5
        attempt = 0
        while reward_code is None and attempt < max_attempt:
            reward_code = self.gpt_api.generate_rewards(task)
            attempt += 1
            if reward_code is None:
                print("Failed to generate reward code. Retrying...")

        # If previous reward code is given, new reward is sum of previous and current reward
        if previous_reward_code is not None:
            reward_code = self.gpt_api.add_rewards(previous_reward_code, reward_code)

