import gymnasium as gym
import gc
import torch

from utils.train_utils import *
from train_ant import Curriculum_Module, Reward_Addition_Module, HER_Module

if __name__ == "__main__":
    seed = 1
    
    env_name = "AntMaze_UMaze"
    env_path = "./environments/Curriculum/envs/AntMaze_UMaze.py"
    logger_path = f"./logs/AntMaze_UMaze_{seed}/"

    # # Curriculum experiments
    # curriculum_module = Curriculum_Module(env_name, env_path, logger_path)
    # curriculum_module.generate_curriculum()
    # curriculum_module.train_curriculum(seed=seed)

    # del curriculum_module
    # gc.collect()
    # torch.cuda.empty_cache()

    # Reward addition experiments
    reward_addition_module = Reward_Addition_Module(env_name, env_path, logger_path)
    reward_addition_module.train_with_reward_addition(seed=seed)

    del reward_addition_module
    gc.collect()
    torch.cuda.empty_cache()

    # # HER experiments
    # her_module = HER_Module(env_name, env_path, logger_path)
    # her_module.train_with_her(seed=seed)

    # del her_module
    # gc.collect()
    # torch.cuda.empty_cache()
