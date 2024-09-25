import argparse
import yaml

from utils.train_utils import *
from train.Curriculum_Module import Curriculum_Module
from train.HER_Module import HER_Module
from train.SAC_Module import SAC_Module
from train.Zeroshot_Module import Zeroshot_Module

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recieve task name, experiment name, and seed number")
    parser.add_argument("--task", type=str, help="Task name", default="AntMaze_UMaze")
    parser.add_argument("--exp", type=str, help="Experiment name", default="curriculum")
    parser.add_argument("--seed", type=int, help="Random seed", default=0)
    parser.add_argument("--logdir", type=str, help="Log directory", default="./logs")
    args = parser.parse_args()

    task = args.task
    env_name = task
    exp = args.exp
    seed = args.seed
    logger_path = f"{args.logdir}/{task}/{exp}_{seed}/"

    with open(f"./configs/{task}.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    module_map = {
        "curriculum": Curriculum_Module,
        "her": HER_Module,
        "sac": SAC_Module,
        "zeroshot": Zeroshot_Module
    }

    env_path = cfg['env_path']

    module = module_map[exp](env_name, env_path, logger_path, cfg, seed)
    module.train()