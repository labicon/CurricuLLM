import gymnasium as gym
import gc
import torch
import argparse

from utils.train_utils import *
from train_ant import Curriculum_Module, HER_Module, SAC_Module, Scratch_Module

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recieve seed number")
    parser.add_argument("--seed", type=int, help="Random seed", default=0)
    args = parser.parse_args()
    seed = args.seed
    
    env_name = "AntMaze_UMaze"
    env_path = "./environments/Curriculum/envs/AntMaze_UMaze_source.py"
    logger_path = f"./logs/AntMaze_UMaze_updated_{seed}/"

    # Curriculum experiments
    curriculum_module = Curriculum_Module(env_name, env_path, logger_path, seed)
    curriculum_module.generate_curriculum()
    curriculum_module.train_curriculum()
    # curriculum_module.resume_curriculum(resume_idx=1, resume_sample_idx=0, resume_from_training=True)

    del curriculum_module
    gc.collect()
    torch.cuda.empty_cache()

    # # HER experiments
    # her_module = HER_Module(env_name, env_path, logger_path, seed)
    # her_module.train_with_her()

    # del her_module
    # gc.collect()
    # torch.cuda.empty_cache()

    # # SAC experiments
    sac_module = SAC_Module(env_name, env_path, logger_path, seed)
    sac_module.train_with_sac()

    del sac_module
    gc.collect()
    torch.cuda.empty_cache()

    # # Scratch experiments
    scratch_module = Scratch_Module(env_name, env_path, logger_path, seed)
    scratch_module.train_curriculum()
