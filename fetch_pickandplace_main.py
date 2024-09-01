import gymnasium as gym
import gc
import torch

from utils.train_utils import *
from train_fetch_pickandplace import Curriculum_Module, HER_Module

if __name__ == "__main__":
    seed = 14
    
    env_name = "FetchPickAndPlace" # "FetchPush"
    env_path = "./environments/Curriculum/envs/FetchPickAndPlace_source.py" # FetchPush.py
    logger_path = f"./logs/Fetch/PickAndPlace_updated_{seed}/" # f"./logs/Fetch/Push_{seed}/"

    # Curriculum experiments
    curriculum_module = Curriculum_Module(env_name, env_path, logger_path, seed)
    curriculum_module.generate_curriculum()
    curriculum_module.train_curriculum()

    del curriculum_module
    gc.collect()
    torch.cuda.empty_cache()

    # # HER experiments
    # her_module = HER_Module(env_name, env_path, logger_path, seed)
    # her_module.train_with_her()

    # del her_module
    # gc.collect()
    # torch.cuda.empty_cache()
