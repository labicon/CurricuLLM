import gymnasium as gym
import gc
import torch

from utils.train_utils import *
from train_adroit_relocate import Curriculum_Module, Reward_Addition_Module, SAC_Module

if __name__ == "__main__":
    seed = 13
    
    env_name = "AdroitHandRelocate"
    env_path = "./environments/Curriculum/envs/AdroitHandRelocate.py"
    logger_path = f"./logs/Adroit_Relocate_{seed}/"

    # Curriculum experiments
    curriculum_module = Curriculum_Module(env_name, env_path, logger_path, seed)
    curriculum_module.generate_curriculum()
    curriculum_module.train_curriculum()

    del curriculum_module
    gc.collect()
    torch.cuda.empty_cache()

    # # Reward addition experiments
    # reward_addition_module = Reward_Addition_Module(env_name, env_path, logger_path, seed)
    # reward_addition_module.train_with_reward_addition()

    # del reward_addition_module
    # gc.collect()
    # torch.cuda.empty_cache()

    # # SAC Experiments
    # sac_module = SAC_Module(env_name, env_path, logger_path)
    # sac_module.train_sac(seed=seed)

    # del sac_module
    # gc.collect()
    # torch.cuda.empty_cache()
