import gc
import torch
import re
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

class Curriculum_Module:
    def __init__(self, env_name, env_path, logger_path, cfg, seed=0):
        self.env_name = env_name
        self.env_path = env_path
        self.prompt_path = "./gpt/prompt/"
        self.gpt_api = api_map[cfg['CurriculumCfg']['env_id']](self.env_name, self.prompt_path, logger_path)
        self.logger_path = logger_path
        self.best_reward_code_list = []
        self.best_model_idx_list = []
        self.current_reward_code_list = []
        self.cfg = cfg['CurriculumCfg']
        self.seed = seed
        self.stats_summary = []
        self.training_algorithm = training_algorithm_map[self.cfg["training_alg"]]
        self.traj_analysis_function = traj_analysis_function_map[self.env_name]
        
    def generate_curriculum(self):
        # Generate curriculum and return list of dictionaries with task details
        self.curriculum_info = self.gpt_api.generate_curriculum()
        self.curriculum_length = len(self.curriculum_info)

    def train(self):
        self.generate_curriculum()

        for curriculum_idx in range(self.curriculum_length):
            for sample_num in range(self.cfg["num_samples"]):
                task = self.curriculum_info[curriculum_idx]
                try:
                    self.train_single(curriculum_idx, task, sample_num)
                except Exception as e:
                    print(f"Error in training task {task['Name']} sample {sample_num}")
                    print(e)
                    # Save error message in log path
                    os.makedirs(self.logger_path + f"{task['Name']}/sample_{sample_num}/", exist_ok=True)
                    with open(self.logger_path + f"{task['Name']}/sample_{sample_num}/training_error.txt", "w") as file:
                        file.write(str(e))
                    self.stats_summary.append({"Error": "Error in evaluating task"})
                    continue
            
            # Asl LLM to choose the best model
            best_sample_idx = self.gpt_api.feedback(self.env_name, task, curriculum_idx, self.stats_summary)
            trial = 1
            while best_sample_idx is None:
                print("Statistics Analysis error. Try again.")
                best_sample_idx = self.gpt_api.feedback(self.env_name, task, curriculum_idx, self.stats_summary)
                trial += 1
                if trial == 5:
                    best_sample_idx = 0

            self.best_model_idx_list.append(best_sample_idx)
            # Update best reward code list
            self.best_reward_code_list.append(self.current_reward_code_list[best_sample_idx])

            # Save the best reward code list
            with open(self.logger_path + f"{task['Name']}/best_reward_code.txt", "w") as file:
                file.write(self.current_reward_code_list[best_sample_idx])
                
            self.current_reward_code_list = []
            self.stats_summary = []

    def train_single(self, curriculum_idx, task, sample_num):
        # Create the environment
        env_id = self.cfg["env_id"]
        eval_env_id = self.cfg["env_id"]

        # Update env code
        reward_code = self.gpt_api.update_env_code(self.env_path, curriculum_idx,
                                     previous_reward_code=self.best_reward_code_list, 
                                     version_number=sample_num)
        self.current_reward_code_list.append(reward_code)

        # Create the vectorized environment
        training_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.cfg["num_envs"])])
        eval_env = SubprocVecEnv([make_env(eval_env_id, i, seed=self.seed) for i in range(self.cfg["num_envs"])])

        # Create the callback
        eval_callback = CurriculumEvalCallback(eval_env, 
                                            log_path=self.logger_path + f"{task['Name']}/sample_{sample_num}", 
                                            best_model_save_path=self.logger_path + f"{task['Name']}/sample_{sample_num}", 
                                            eval_freq=self.cfg["eval_freq"], 
                                            deterministic=True, render=False, warn=False)
        
        if curriculum_idx == 0:
            model = self.training_algorithm(self.cfg["policy_network"],
                                            training_env,
                                            verbose=1,
                                            tensorboard_log=self.logger_path)
        else:
            previous_task = self.curriculum_info[curriculum_idx - 1]['Name']
            pre_tuned_model_path = self.logger_path + previous_task + f"/sample_{self.best_model_idx_list[-1]}/final_model"

            print("Loading model from " + pre_tuned_model_path)
            model = self.training_algorithm.load(pre_tuned_model_path)
            model.set_env(training_env)

        if curriculum_idx == self.curriculum_length - 1 or curriculum_idx == self.curriculum_length - 2:
            model.learn(total_timesteps=self.cfg['long_training_timesteps'], callback=eval_callback, tb_log_name=f"{task['Name']}_sample_{sample_num}")
        else:
            model.learn(total_timesteps=self.cfg['short_training_timesteps'], callback=eval_callback, tb_log_name=f"{task['Name']}_sample_{sample_num}")

        model.save(self.logger_path + f"{task['Name']}/sample_{sample_num}/final_model.zip")

        try:
            # Get trajectory
            obs = eval_env.reset()
            obs_trajectory = [obs['observation'][0]]
            goal_trajectory = [obs['desired_goal'][0]]
            for _ in range(7000):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, _ = eval_env.step(action)
                obs_trajectory.append(obs['observation'][0])
                goal_trajectory.append(obs['desired_goal'][0])

            self.stats_summary.append(self.traj_analysis_function(obs_trajectory, goal_trajectory))
        except Exception as e:
            print(f"Error in evaluating task {task['Name']} sample {sample_num}")
            print(e)
            # Save error message in log path
            with open(self.logger_path + f"{task['Name']}/sample_{sample_num}/evaluation_error.txt", "w") as file:
                file.write(str(e))
            self.stats_summary.append({"Error": "Error in evaluating task"})

        del model, training_env, eval_env, eval_callback
        gc.collect()
        torch.cuda.empty_cache()  # Free up unused memory

    def load_curriculum(self):
        # Load curriculum
        with open(self.logger_path + "curriculum.md", "r") as file:
            curriculum_txt = file.read()

        # Split the string into individual task sections
        task_sections = re.split(r"\n\n(?=Task)", curriculum_txt)

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
        tasks_details = [extract_task_details(section) for section in task_sections]
        self.curriculum_info = tasks_details
        self.gpt_api.tasks_details = tasks_details
        self.curriculum_length = len(self.curriculum_info)

    def load_rewards(self, resume_idx):
        # Load rewards
        for task in self.curriculum_info[:resume_idx]:
            with open(self.logger_path + f"{task['Name']}/best_reward_code.txt", "r") as file:
                reward_code = file.read()
                self.best_reward_code_list.append(reward_code)

            # Load best model indices
            with open(self.logger_path + f"{task['Name']}.md", "r") as file:
                decision_txt = file.read()

            decision = decision_txt.splitlines()[0]
            print("For task " + task["Name"] + ", GPT decided " + decision)
            numbers = re.findall(r"\d+", decision)
            if numbers:
                self.best_model_idx_list.append(int(numbers[0]))
            else:
                print("No number found in the decision.")
                self.best_model_idx_list.append(0)

    def load_current_rewards(self, resume_idx):
        for sample in range(self.num_samples):
            with open(
                self.logger_path + f"{self.curriculum_info[resume_idx]['Name']}/sample_{sample}/reward_code.md", "r"
            ) as file:
                response = file.read()
                if "Error" in response:
                    self.current_reward_code_list.append(None)
                else:
                    self.current_reward_code_list.append(response)

    def resume_curriculum(self, resume_idx, resume_sample_idx=0, resume_from_training=True):
        print(f"Resuming curriculum at task {resume_idx}")
        # Load curriculum and rewards
        self.load_curriculum()
        self.load_rewards(resume_idx)

        prev_task = self.curriculum_info[resume_idx - 1]
        print(f"Resuming from task {prev_task['Name']}")
        for idx, task in enumerate(self.curriculum_info[resume_idx:], start=resume_idx):
            if resume_from_training:
                print(f"Training task {task['Name']}")
                start_idx = resume_sample_idx
                for sample_num in range(start_idx, self.num_samples):
                    try:
                        self.train_single(idx, task, sample_num)
                    except Exception as e:
                        print(f"Error in training task {task['Name']} sample {sample_num}")
                        print(e)
                        # Save error message in log path
                        with open(self.logger_path + f"{task['Name']}/sample_{sample_num}/training_error.txt", "w") as file:
                            file.write(str(e))
                        self.stats_summary.append({"Error": "Error in evaluating task"})
                        continue
                start_idx = 0
            else:
                self.load_current_rewards(resume_idx)
                print(f"Loaded current rewards for task {task['Name']}")

            # Asl LLM to choose the best model
            best_sample_idx = self.gpt_api.feedback(self.env_name, task, idx, self.stats_summary)
            trial = 1
            while best_sample_idx is None:
                print("Statistics Analysis error. Try again.")
                best_sample_idx = self.gpt_api.feedback(self.env_name, task, idx, self.stats_summary)
                trial += 1
                if trial == 5:
                    best_sample_idx = 0

            self.best_model_idx_list.append(best_sample_idx)
            # Update best reward code list
            self.best_reward_code_list.append(self.current_reward_code_list[best_sample_idx])

            # Save the best reward code list
            with open(self.logger_path + f"{task['Name']}/best_reward_code.txt", "w") as file:
                file.write(self.current_reward_code_list[best_sample_idx])
                
            self.current_reward_code_list = []
            self.stats_summary = []

            resume_from_training = True


