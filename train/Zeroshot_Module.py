import gc
import torch
import os

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

from evaluation.evalcallback_feedback import CurriculumEvalCallback
from utils.train_utils import *
from gpt.curriculum_api_chain_ant import CurriculumAPI_Ant
from gpt.curriculum_api_chain_fetch import CurriculumAPI_Fetch
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

api_map = {
    "Curriculum/AntMaze_UMaze": CurriculumAPI_Ant,
    "Curriculum/FetchSlide": CurriculumAPI_Fetch,
    "Curriculum/FetchPush": CurriculumAPI_Fetch,
}

class Zeroshot_Module:

    def __init__(self, env_name, env_path, logger_path, cfg, seed=0):
        self.env_name = env_name
        self.env_path = env_path
        self.prompt_path = "./gpt/prompt/"
        self.gpt_api = api_map[cfg['CurriculumCfg']['env_id']](self.env_name, self.prompt_path, logger_path)
        self.logger_path = logger_path
        self.cfg = cfg['ZeroshotCfg']
        self.seed = seed
        self.training_algorithm = training_algorithm_map[self.cfg['training_alg']]
        
    def train(self):
        self.load_task_info()

        for sample_num in range(self.cfg['num_samples']):
            task = self.curriculum_info[0]
            try:
                self.train_single(0, task, sample_num)
            except Exception as e:
                print(f"Error in training task {task['Name']} sample {sample_num}")
                print(e)
                # Save the error message
                os.makedirs(self.logger_path + f"{task['Name']}/sample_{sample_num}/", exist_ok=True)
                with open(self.logger_path + f"{task['Name']}/sample_{sample_num}/training_error.txt", "w") as file:
                    file.write(str(e))
                continue

    def train_single(self, curriculum_idx, task, sample_num):
        # Create the environment
        env_id = self.cfg['env_id']

        # Update env code
        reward_code = self.gpt_api.update_env_code(self.env_path, curriculum_idx, 
                                                    previous_reward_code=[], 
                                                    version_number=sample_num)

        # Create the vectorized environment
        training_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.cfg['num_envs'])])
        eval_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.cfg['num_envs'])])

        # Create the callback
        eval_callback = CurriculumEvalCallback(eval_env, 
                                            log_path=self.logger_path + f"{task['Name']}/sample_{sample_num}", 
                                            best_model_save_path=self.logger_path + f"{task['Name']}/sample_{sample_num}", 
                                            eval_freq=self.cfg['eval_freq'], 
                                            deterministic=True, render=False, warn=False)
        
        model = self.training_algorithm(self.cfg['policy_network'],
                                        training_env,
                                        verbose=1,
                                        )

        model.learn(total_timesteps=self.cfg['training_timesteps'], callback=eval_callback)
        model.save(self.logger_path + f"{task['Name']}/sample_{sample_num}/final_model.zip")

        del model, training_env, eval_env, eval_callback
        gc.collect()
        torch.cuda.empty_cache()  # Free up unused memory

    def load_task_info(self):
        # Load curriculum
        with open(self.prompt_path + self.env_name + "/original_task_info.md", "r") as file:
            task_txt = file.read()

        # Function to extract details from each task section
        def extract_task_details(task_section):

            details = {}
            lines = task_section.split("\n")
            for line in lines:
                if line.startswith("Task"):
                    details["Task"] = line.split(" ")[1]
                elif line.startswith("Name:"):
                    details["Name"] = line.split(": ")[1]
                elif line.startswith("Description:"):
                    details["Description"] = line.split(": ")[1]
                elif line.startswith("Reason:"):
                    details["Reason"] = ": ".join(line.split(": ")[1:])
            return details

        # Extract details for all tasks
        tasks_details = [extract_task_details(task_txt)]
        self.curriculum_info = tasks_details
        self.gpt_api.tasks_details = tasks_details
        self.curriculum_length = len(self.curriculum_info)