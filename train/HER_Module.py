import gc
import torch

from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
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

class HER_Module:
    def __init__(self, env_name, env_path, logger_path, cfg, seed=0):
        self.env_name = env_name
        self.env_path = env_path
        self.logger_path = logger_path
        self.cfg = cfg['HerCfg']
        self.seed = seed
        self.training_algorithm = training_algorithm_map[self.cfg['training_alg']]
        self.traj_analysis_function = traj_analysis_function_map[self.env_name]
        

    def train(self):
        goal_selection_strategy = GoalSelectionStrategy.FUTURE

        # Create the environment
        env_id = self.cfg['env_id']

        # Create the vectorized environment
        training_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.cfg['num_envs'])])
        eval_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.cfg['num_envs'])])

        # Create the callback
        eval_callback = EvalCallback(eval_env, 
                                    log_path=self.logger_path + "her/", 
                                    best_model_save_path=self.logger_path + "her/", 
                                    eval_freq=self.cfg['eval_freq'], 
                                    deterministic=True, render=False, warn=False)
        
        model = self.training_algorithm(self.cfg['policy_network'],
                                        training_env,
                                        learning_starts=self.cfg['num_envs'] * 1000,
                                        replay_buffer_class=HerReplayBuffer,
                                        # Parameters for HER
                                        replay_buffer_kwargs=dict(
                                            n_sampled_goal=4,
                                            goal_selection_strategy=goal_selection_strategy,
                                        ),
                                        verbose=1)
        
        model.learn(total_timesteps=self.cfg['training_timesteps'], callback=eval_callback)
        model.save(self.logger_path + "her/final_model.zip")

        del model, training_env, eval_env, eval_callback
        gc.collect()
        torch.cuda.empty_cache() # Free up unused memory