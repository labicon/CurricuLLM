import gymnasium as gym
import gc
import torch

from utils.train_utils import *
from train_ant import Curriculum_Module, HER_Module

if __name__ == "__main__":
    seed = 10
    
    env_name = "AntMaze_UMaze"
    env_path = "./environments/Curriculum/envs/AntMaze_UMaze_source.py"
    logger_path = f"./logs/AntMaze_UMaze_updated_{seed}/"

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
