import gymnasium as gym
import gc
import torch

from utils.train_utils import *
from train_fetch import Curriculum_Module, Reward_Addition_Module

if __name__ == "__main__":
    env_name = "Fetch_Push"
    env_path = "./environments/Curriculum/envs/Fetch_Push.py"
    logger_path = "./logs/Fetch/Push_6/"

    # Curriculum experiments
    curriculum_module = Curriculum_Module(env_name, env_path, logger_path)
    curriculum_module.generate_curriculum()
    curriculum_module.train_curriculum()

    del curriculum_module
    gc.collect()
    torch.cuda.empty_cache()

    # Reward addition experiments
    reward_addition_module = Reward_Addition_Module(env_name, env_path, logger_path)
    reward_addition_module.train_with_reward_addition()

    del reward_addition_module
    gc.collect()
    torch.cuda.empty_cache()
