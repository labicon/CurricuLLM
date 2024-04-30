import numpy as np
import gc
import torch
import re

from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from evaluation.evalcallback_feedback import CurriculumEvalCallback
from evaluation.evalcallback_success import SuccessEvalCallback as EvalCallback
from utils.train_utils import *
from gpt.curriculum_api import CurriculumAPI
from gpt.utils import file_to_string

class Curriculum_Module:
    def __init__(self, env_name, env_path, logger_path, seed=0):
        self.env_name = env_name
        self.env_path = env_path
        self.prompt_path = "./gpt/prompt/"
        self.gpt_api = CurriculumAPI(self.env_name, self.prompt_path, logger_path)
        self.logger_path = logger_path
        self.best_reward_code_list = []
        self.best_model_idx_list = []
        self.current_reward_code_list = []
        self.num_cpu = 16
        self.num_samples = 3
        self.seed = seed
        
    def generate_curriculum(self):
        # Generate curriculum and return list of dictionaries with task details
        self.curriculum_info = self.gpt_api.generate_curriculum()
        self.curriculum_length = len(self.curriculum_info)

    def train_curriculum(self):
        for curriculum_idx in range(self.curriculum_length):
            for sample_num in range(self.num_samples):
                task = self.curriculum_info[curriculum_idx]
                try:
                    self.train_single(curriculum_idx, task, sample_num)
                except Exception as e:
                    print(f"Error in training task {task['Name']} sample {sample_num}")
                    print(e)
                    continue
            # Evaluate the trained models
            statistics = []
            for sample_num in range(self.num_samples):
                try:
                    env_id = f"Curriculum/{self.env_name}-v{sample_num}"
                    eval_env = SubprocVecEnv([make_env(env_id, i) for i in range(self.num_cpu)])
                    model_path = self.logger_path + f"{task['Name']}/sample_{sample_num}/final_model.zip"
                    model = PPO.load(model_path)
                    
                    # Get trajectory
                    obs = eval_env.reset()
                    obs_trajectory = [obs['observation'][0]]
                    goal_trajectory = [obs['desired_goal'][0]]
                    for _ in range(1400):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, _, _, _ = eval_env.step(action)
                        obs_trajectory.append(obs['observation'][0])
                        goal_trajectory.append(obs['desired_goal'][0])

                    statistics.append(analyze_trajectory_ant(obs_trajectory, goal_trajectory))
                except Exception as e:
                    print(f"Error in evaluating task {task['Name']} sample {sample_num}")
                    print(e)
                    statistics.append({"Error": "Error in evaluating task"})
                    continue
            
            # Asl LLM to choose the best model
            best_sample_idx = self.gpt_api.feedback(self.env_name, task, curriculum_idx, statistics)
            trial = 1
            while best_sample_idx is None:
                print("Statistics Analysis error. Try again.")
                best_sample_idx = self.gpt_api.feedback(self.env_name, task, curriculum_idx, statistics)
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


    def train_single(self, curriculum_idx, task, sample_num):
        # Create the environment
        env_id = f"Curriculum/{self.env_name}-v{sample_num}"

        # Update env code
        reward_code = self.gpt_api.update_env_code(self.env_path, curriculum_idx,
                                     previous_reward_code=self.best_reward_code_list, 
                                     version_number=sample_num)
        self.current_reward_code_list.append(reward_code)

        # Create the vectorized environment
        training_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.num_cpu)])
        eval_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.num_cpu)])

        # Create the callback
        eval_callback = CurriculumEvalCallback(eval_env, 
                                            log_path=self.logger_path + f"{task['Name']}/sample_{sample_num}", 
                                            best_model_save_path=self.logger_path + f"{task['Name']}/sample_{sample_num}", 
                                            eval_freq=1000, 
                                            deterministic=True, render=False, warn=False)
        
        if curriculum_idx == 0:
            model = PPO("MultiInputPolicy",
                        training_env,
                        verbose=1)
        else:
            previous_task = self.curriculum_info[curriculum_idx - 1]['Name']
            pre_tuned_model_path = self.logger_path + previous_task + f"/sample_{self.best_model_idx_list[-1]}/final_model"
            model = PPO.load(pre_tuned_model_path)
            model.set_env(training_env)

        if curriculum_idx == self.curriculum_length - 1:
            model.learn(total_timesteps=10_000_000, callback=eval_callback)
        else:
            model.learn(total_timesteps=1_000_000, callback=eval_callback)
        model.save(self.logger_path + f"{task['Name']}/sample_{sample_num}/final_model.zip")

        del model, training_env, eval_env, eval_callback
        gc.collect()
        torch.cuda.empty_cache()  # Free up unused memory

def analyze_trajectory_ant(obs_trajectory, goal_trajectory):
    # obs_trajectory: list of observations
    # Get list of torso_coord, torso_orientation, torso_velocity, torso_angular_velocity, goal_pos, gosl_distance
    torso_coord = []
    torso_orientation = []
    torso_velocity = []
    torso_angular_velocity = []
    goal_pos = []
    goal_distance = []

    for obs, goal in zip(obs_trajectory, goal_trajectory):
        torso_coord.append(obs[0:3])
        torso_orientation.append(obs[3:7])
        torso_velocity.append(np.linalg.norm(obs[15:17]))
        torso_angular_velocity.append(np.linalg.norm(obs[18:21]))
        goal_pos.append(goal)
        goal_distance.append(np.linalg.norm(obs[0:2] - goal))

    # change to np array
    torso_coord = np.array(torso_coord)
    torso_orientation = np.array(torso_orientation)
    torso_velocity = np.array(torso_velocity)
    torso_angular_velocity = np.array(torso_angular_velocity)
    goal_pos = np.array(goal_pos)
    goal_distance = np.array(goal_distance)

    # Calculate mean and std of each variable
    statistics = {}
    statistics["torso_coord_mean"] = np.mean(torso_coord, axis=0)
    statistics["torso_coord_std"] = np.std(torso_coord, axis=0)
    statistics["torso_orientation_mean"] = np.mean(torso_orientation, axis=0)
    statistics["torso_orientation_std"] = np.std(torso_orientation, axis=0)
    statistics["torso_velocity_mean"] = np.mean(torso_velocity, axis=0)
    statistics["torso_velocity_std"] = np.std(torso_velocity, axis=0)
    statistics["torso_angular_velocity_mean"] = np.mean(torso_angular_velocity, axis=0)
    statistics["torso_angular_velocity_std"] = np.std(torso_angular_velocity, axis=0)
    statistics["goal_pos_mean"] = np.mean(goal_pos, axis=0)
    statistics["goal_pos_std"] = np.std(goal_pos, axis=0)
    statistics["goal_distance_mean"] = np.mean(goal_distance, axis=0)
    statistics["goal_distance_std"] = np.std(goal_distance, axis=0)

    return statistics


class Reward_Addition_Module:
    def __init__(self, env_name, env_path, logger_path, seed=0):
        self.env_name = env_name
        self.env_path = env_path
        self.logger_path = logger_path
        self.best_reward_code_list = []
        self.num_cpu = 16
        self.num_samples = 3
        self.seed = seed
        
    def extract_curriculum(self):
        # extract curriculum and return list of dictionaries with task details
        curriculum_txt = file_to_string(self.logger_path + "curriculum.md")
        # Split the string into individual task sections
        task_sections = re.split(r'\n\n(?=Task)', curriculum_txt)

        # Function to extract details from each task section
        def extract_task_details(task_section):

            details = {}
            lines = task_section.split('\n')
            for line in lines:
                if line.startswith('Task'):
                    details['Task'] = line.split(' ')[1]
                elif line.startswith('Name:'):
                    details['Name'] = line.split(': ')[1]
                elif line.startswith('Description:'):
                    details['Description'] = line.split(': ')[1]
                elif line.startswith('Reason:'):
                    details['Reason'] = ': '.join(line.split(': ')[1:])
            return details

        # Extract details for all tasks
        self.curriculum_info = [extract_task_details(section) for section in task_sections]
        self.curriculum_length = len(self.curriculum_info)

    def update_env_code(self):
        # Extract reward code from best_reward_code.txt
        for task in self.curriculum_info:
            best_reward_code = file_to_string(self.logger_path + f"{task['Name']}/best_reward_code.txt")
            self.best_reward_code_list.append(best_reward_code)

        reward_code_summary = self.add_rewards(self.best_reward_code_list)

        with open(self.env_path, 'r') as file:
            original_code = file.read()

        # Append the new code block to the original code
        # Indent the code block with 4 spaces to the beginning of each line
        reward_code_summary = '\n'.join('    ' + line for line in reward_code_summary.splitlines())
        new_code = original_code + '\n\n' + reward_code_summary

        # Save as a new file with specific version number
        new_file_path = self.env_path.replace('.py', f'_v0.py')
        with open(new_file_path, 'w') as file:
            file.write(new_code)

        print(f"Updated environment code saved to {new_file_path}")

    def add_rewards(self, reward_code_list: list):
        # Find the length of the previous code block
        n_previous_code = len(reward_code_list)-1
        for idx, code in enumerate(reward_code_list):
            reward_code_list[idx] = code.replace("compute_reward_curriculum(", f"compute_reward_{idx}(")

        reward_code = ""
        for code in reward_code_list:
            reward_code += code + "\n\n"
        reward_code += """# Function to loop through compute_reward_X functions and sum their outputs
def compute_reward_curriculum(self):
    total_reward = 0
    total_reward_dict = {}
    """ + f"n = {n_previous_code}" + """
    for i in range(n + 1):  # Including n, hence n + 1
        # Construct the function name based on i
        function_name = f'compute_reward_{i}'
        # Get the function by name and call it
        function = getattr(self, function_name, None)
        if function:
            # Call the function and add its return value to the total sum
            reward, reward_dict = function()
            total_reward += reward
            total_reward_dict.update(reward_dict)
        else:
            raise NameError(f"Function {function_name} not found.")
    return total_reward, total_reward_dict"""
        return reward_code

    def train_with_reward_addition(self):
        self.extract_curriculum()
        self.update_env_code()

        curriculum_idx = self.curriculum_length - 1

        # Create the environment
        env_id = f"Curriculum/{self.env_name}-v0"

        # Create the vectorized environment
        training_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.num_cpu)])
        eval_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.num_cpu)])

        # Create the callback
        eval_callback = CurriculumEvalCallback(eval_env, 
                                            log_path=self.logger_path + "reward_addition/", 
                                            best_model_save_path=self.logger_path + "reward_addition/", 
                                            eval_freq=1000, 
                                            deterministic=True, render=False, warn=False)
        
        model = SAC("MultiInputPolicy",
                    training_env,
                    verbose=1)
        model.learn(total_timesteps=12_000_000, callback=eval_callback)
        model.save(self.logger_path + f"reward_addition/final_model.zip")

        del model, training_env, eval_env, eval_callback
        gc.collect()
        torch.cuda.empty_cache()  # Free up unused memory

class HER_Module:
    def __init__(self, env_name, env_path, logger_path, seed=0):
        self.env_name = env_name
        self.env_path = env_path
        self.logger_path = logger_path
        self.num_cpu = 16
        self.seed = seed

    def train_with_her(self):
        goal_selection_strategy = GoalSelectionStrategy.FUTURE

        # Create the environment
        env_id = f"{self.env_name}-v4"

        # Create the vectorized environment
        training_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.num_cpu)])
        eval_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.num_cpu)])

        # Create the callback
        eval_callback = EvalCallback(eval_env, 
                                    log_path=self.logger_path + "her/", 
                                    best_model_save_path=self.logger_path + "her/", 
                                    eval_freq=1000, 
                                    deterministic=True, render=False, warn=False)
        
        model = SAC("MultiInputPolicy",
                    training_env,
                    learning_starts=self.num_cpu * 1000,
                    replay_buffer_class=HerReplayBuffer,
                    # Parameters for HER
                    replay_buffer_kwargs=dict(
                        n_sampled_goal=4,
                        goal_selection_strategy=goal_selection_strategy,
                    ),
                    verbose=1)
        
        model.learn(total_timesteps=12_000_000, callback=eval_callback)
        model.save(self.logger_path + "her/final_model.zip")

        del model, training_env, eval_env, eval_callback
        gc.collect()
        torch.cuda.empty_cache() # Free up unused memory