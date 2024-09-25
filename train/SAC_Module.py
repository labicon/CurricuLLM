import gc
import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from evaluation.evalcallback_success import SuccessEvalCallback as EvalCallback
from utils.train_utils import *
from traj_feedback import analyze_trajectory_ant, analyze_trajectory_fetch

traj_analysis_function_map = {
    "AntMaze_UMaze": analyze_trajectory_ant,
    "FetchSlide": analyze_trajectory_fetch,
    "FetchPush": analyze_trajectory_fetch,
    "FetchPickAndPlace": analyze_trajectory_fetch,
    "FetchReach": analyze_trajectory_fetch,
}

training_algorithm_map = {
    "PPO": PPO,
    "SAC": SAC,
}

class SAC_Module:
    def __init__(self, env_name, env_path, logger_path, cfg, seed=0):
        self.env_name = env_name
        self.env_path = env_path
        self.logger_path = logger_path
        self.cfg = cfg['SacCfg']
        self.seed = seed

    def train(self):
        # Create the environment
        env_id = self.cfg['env_id']

        # Create the vectorized environment
        training_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.cfg['num_envs'])])
        eval_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.cfg['num_envs'])])

        # Create the callback
        eval_callback = EvalCallback(eval_env, 
                                    log_path=self.logger_path + "sac/", 
                                    best_model_save_path=self.logger_path + "sac/", 
                                    eval_freq=1000, 
                                    deterministic=True, render=False, warn=False)
        
        model = SAC(self.cfg['policy_network'],
                    training_env,
                    verbose=1,
                    )
        
        model.learn(total_timesteps=self.cfg['training_timesteps'], callback=eval_callback)
        model.save(self.logger_path + "sac/final_model.zip")

        del model, training_env, eval_env, eval_callback
        gc.collect()
        torch.cuda.empty_cache()