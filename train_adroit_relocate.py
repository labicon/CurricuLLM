import numpy as np
import gc
import torch
import re

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from evaluation.evalcallback_feedback import CurriculumEvalCallback
from utils.train_utils import *
from gpt.curriculum_api import CurriculumAPI
from gpt.utils import file_to_string

class Curriculum_Module:
    def __init__(self, env_name, env_path, logger_path):
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
        
    def generate_curriculum(self):
        # Generate curriculum and return list of dictionaries with task details
        self.curriculum_info = self.gpt_api.generate_curriculum()
        self.curriculum_length = len(self.curriculum_info)

    def train_curriculum(self, seed=0):
        self.seed = seed
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
                    model = SAC.load(model_path)
                    
                    # Get trajectory
                    obs = eval_env.reset()
                    obs_trajectory = [obs['observation'][0]]
                    goal_trajectory = [obs['desired_goal'][0]]
                    for _ in range(400):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, _, _, _ = eval_env.step(action)
                        obs_trajectory.append(obs['observation'][0])
                        goal_trajectory.append(obs['desired_goal'][0])

                    statistics.append(analyze_trajectory_adroit(obs_trajectory, goal_trajectory))
                except Exception as e:
                    print(f"Error in evaluating task {task['Name']} sample {sample_num}")
                    print(e)
                    statistics.append({"Error": "Error in evaluating task"})
                    continue
            
            # Asl LLM to choose the best model
            best_sample_idx = self.gpt_api.feedback(self.env_name, task, statistics)
            trial = 1
            while best_sample_idx is None:
                print("Statistics Analysis error. Try again.")
                best_sample_idx = self.gpt_api.feedback(self.env_name, task, statistics)
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
        
        reward_code = self.gpt_api.update_env_code(self.env_path, task, 
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
            model = SAC("MlpPolicy",
                        training_env,
                        verbose=1)
        else:
            previous_task = self.curriculum_info[curriculum_idx - 1]['Name']
            pre_tuned_model_path = self.logger_path + previous_task + f"/sample_{self.best_model_idx_list[-1]}/final_model"
            model = SAC.load(pre_tuned_model_path)
            model.set_env(training_env)

        if curriculum_idx == self.curriculum_length - 1:
            model.learn(total_timesteps=10_000_000, callback=eval_callback)
        else:
            model.learn(total_timesteps=1_000_000, callback=eval_callback)
        model.save(self.logger_path + f"{task['Name']}/sample_{sample_num}/final_model.zip")

        del model, training_env, eval_env, eval_callback
        gc.collect()
        torch.cuda.empty_cache()  # Free up unused memory

def analyze_trajectory_adroit(obs_trajectory, goal_trajectory):
    '''
    obs_trajectory: list of observations
    Get list of variables
    (1) arm_position: Translation position of the arm in xyz direction
    (2) arm_angular_position: Angular position of arm
    (3) wrist_angular_position: Angular position of writst
    (4) mcp_angular_position_forefinger: Angular position of MCP(metacarpophalangeal, knuckle) joint of the forefinger
    (5) pip_angular_position_forefinger: Angular position of PIP(proximal interphalangeal, middle) joint of the forefinger
    (6) dip_angular_position_forefinger: Angular position of DIP(Distal Interphalangeal, tip) joint of the forefinger
    (7) mcp_angular_position_middlefinger: Angular position of MCP(metacarpophalangeal, knuckle) joint of the middlefinger
    (8) pip_angular_position_middlefingler: Angular position of PIP(proximal interphalangeal, middle) joint of the middlefinger
    (9) dip_angular_position_middlefinger: Angular position of DIP(Distal Interphalangeal, tip) joint of the middlefinger
    (10) mcp_angular_position_ringfinger: Angular position of MCP(metacarpophalangeal, knuckle) joint of the ringfinger
    (11) pip_angular_position_ringfinger: Angular position of PIP(proximal interphalangeal, middle) joint of the ringfinger
    (12) dip_angular_position_ringfinger: Angular position of DIP(Distal Interphalangeal, tip) joint of the ringfinger
    (13) cmc_angular_position_littlefinger: Angular Position of the CMC (carpometacarpal) joint of the little finger
    (14) mcp_angular_position_littlefinger: Angular position of MCP(metacarpophalangeal, knuckle) joint of the littlefinger
    (15) pip_angular_position_littlefinger: Angular position of PIP(proximal interphalangeal, middle) joint of the littlefinger
    (16) dip_angular_position_littlefinger: Angular position of DIP(Distal Interphalangeal, tip) joint of the littlefinger
    (17) cmc_angular_thumb: Angular position of CMC (carpometacarpal) joint of the thumb
    (18) mcp_angular_position_thumb: Angular position of MCP(metacarpophalangeal, knuckle) joint of the thumb
    (19) ip_angular_thumb: Angular position of the IP (Interphalangeal) joint of the thumb
    (20) positional_difference_ball: xyz positional difference from the palm to the ball, ball_position - palm_position
    (21) positional_difference_target: xyz positional difference from the palm to the target, target_position - palm_position
    (22) positional_difference_from_ball_to_target: xyz positional difference from the ball to the target, target_position - ball_position
    '''

    arm_position = []
    arm_angular_position = []
    wrist_angular_position = []
    mcp_angular_position_forefinger = []
    pip_angular_position_forefinger = []
    dip_angular_position_forefinger = []
    mcp_angular_position_middlefinger = []
    pip_angular_position_middlefingler = []
    dip_angular_position_middlefinger = []
    mcp_angular_position_ringfinger = []
    pip_angular_position_ringfinger = []
    dip_angular_position_ringfinger = []
    cmc_angular_position_littlefinger = []
    mcp_angular_position_littlefinger = []
    pip_angular_position_littlefinger = []
    dip_angular_position_littlefinger = []
    cmc_angular_thumb = []
    mcp_angular_position_thumb = []
    ip_angular_thumb = []
    positional_difference_ball = []
    positional_difference_target = []
    positional_difference_from_ball_to_target = []

    for obs in obs_trajectory:
        arm_position.append(obs[0:3])
        arm_angular_position.append(obs[3:6])
        wrist_angular_position.append(obs[6:8])
        mcp_angular_position_forefinger.append(obs[8:10])
        pip_angular_position_forefinger.append(obs[10])
        dip_angular_position_forefinger.append(obs[11])
        mcp_angular_position_middlefinger.append(obs[12:14])
        pip_angular_position_middlefingler.append(obs[14])
        dip_angular_position_middlefinger.append(obs[15])
        mcp_angular_position_ringfinger.append(obs[16:18])
        pip_angular_position_ringfinger.append(obs[18])
        dip_angular_position_ringfinger.append(obs[19])
        cmc_angular_position_littlefinger.append(obs[20])
        mcp_angular_position_littlefinger.append(obs[21:23])
        pip_angular_position_littlefinger.append(obs[23])
        dip_angular_position_littlefinger.append(obs[24])
        cmc_angular_thumb.append(obs[25:27])
        mcp_angular_position_thumb.append(obs[27:29])
        ip_angular_thumb.append(obs[29])
        positional_difference_ball.append(obs[30:33])
        positional_difference_target.append(obs[33:36])
        positional_difference_from_ball_to_target.append(obs[36:39])
        

    # change to np array
    arm_position = np.array(arm_position)
    arm_angular_position = np.array(arm_angular_position)
    wrist_angular_position = np.array(wrist_angular_position)
    mcp_angular_position_forefinger = np.array(mcp_angular_position_forefinger)
    pip_angular_position_forefinger = np.array(pip_angular_position_forefinger)
    dip_angular_position_forefinger = np.array(dip_angular_position_forefinger)
    mcp_angular_position_middlefinger = np.array(mcp_angular_position_middlefinger)
    pip_angular_position_middlefingler = np.array(pip_angular_position_middlefingler)
    dip_angular_position_middlefinger = np.array(dip_angular_position_middlefinger)
    mcp_angular_position_ringfinger = np.array(mcp_angular_position_ringfinger)
    pip_angular_position_ringfinger = np.array(pip_angular_position_ringfinger)
    dip_angular_position_ringfinger = np.array(dip_angular_position_ringfinger)
    cmc_angular_position_littlefinger = np.array(cmc_angular_position_littlefinger)
    mcp_angular_position_littlefinger = np.array(mcp_angular_position_littlefinger)
    pip_angular_position_littlefinger = np.array(pip_angular_position_littlefinger)
    dip_angular_position_littlefinger = np.array(dip_angular_position_littlefinger)
    cmc_angular_thumb = np.array(cmc_angular_thumb)
    mcp_angular_position_thumb = np.array(mcp_angular_position_thumb)
    ip_angular_thumb = np.array(ip_angular_thumb)
    positional_difference_ball = np.array(positional_difference_ball)
    positional_difference_target = np.array(positional_difference_target)
    positional_difference_from_ball_to_target = np.array(positional_difference_from_ball_to_target)


    # Calculate mean and std of each variable
    statistics = {}
    statistics["arm_position_mean"] = np.mean(arm_position, axis=0)
    statistics["arm_position_std"] = np.std(arm_position, axis=0)
    statistics["arm_angular_position_mean"] = np.mean(arm_angular_position, axis=0)
    statistics["arm_angular_position_std"] = np.std(arm_angular_position, axis=0)
    statistics["wrist_angular_position_mean"] = np.mean(wrist_angular_position, axis=0)
    statistics["wrist_angular_position_std"] = np.std(wrist_angular_position, axis=0)
    statistics["mcp_angular_position_forefinger_mean"] = np.mean(mcp_angular_position_forefinger, axis=0)
    statistics["mcp_angular_position_forefinger_std"] = np.std(mcp_angular_position_forefinger, axis=0)
    statistics["pip_angular_position_forefinger_mean"] = np.mean(pip_angular_position_forefinger, axis=0)
    statistics["pip_angular_position_forefinger_std"] = np.std(pip_angular_position_forefinger, axis=0)
    statistics["dip_angular_position_forefinger_mean"] = np.mean(dip_angular_position_forefinger, axis=0)
    statistics["dip_angular_position_forefinger_std"] = np.std(dip_angular_position_forefinger, axis=0)
    statistics["mcp_angular_position_middlefinger_mean"] = np.mean(mcp_angular_position_middlefinger, axis=0)
    statistics["mcp_angular_position_middlefinger_std"] = np.std(mcp_angular_position_middlefinger, axis=0)
    statistics["pip_angular_position_middlefingler_mean"] = np.mean(pip_angular_position_middlefingler, axis=0)
    statistics["pip_angular_position_middlefingler_std"] = np.std(pip_angular_position_middlefingler, axis=0)
    statistics["dip_angular_position_middlefinger_mean"] = np.mean(dip_angular_position_middlefinger, axis=0)
    statistics["dip_angular_position_middlefinger_std"] = np.std(dip_angular_position_middlefinger, axis=0)
    statistics["mcp_angular_position_ringfinger_mean"] = np.mean(mcp_angular_position_ringfinger, axis=0)
    statistics["mcp_angular_position_ringfinger_std"] = np.std(mcp_angular_position_ringfinger, axis=0)
    statistics["pip_angular_position_ringfinger_mean"] = np.mean(pip_angular_position_ringfinger, axis=0)
    statistics["pip_angular_position_ringfinger_std"] = np.std(pip_angular_position_ringfinger, axis=0)
    statistics["dip_angular_position_ringfinger_mean"] = np.mean(dip_angular_position_ringfinger, axis=0)
    statistics["dip_angular_position_ringfinger_std"] = np.std(dip_angular_position_ringfinger, axis=0)
    statistics["cmc_angular_position_littlefinger_mean"] = np.mean(cmc_angular_position_littlefinger, axis=0)
    statistics["cmc_angular_position_littlefinger_std"] = np.std(cmc_angular_position_littlefinger, axis=0)
    statistics["mcp_angular_position_littlefinger_mean"] = np.mean(mcp_angular_position_littlefinger, axis=0)
    statistics["mcp_angular_position_littlefinger_std"] = np.std(mcp_angular_position_littlefinger, axis=0)
    statistics["pip_angular_position_littlefinger_mean"] = np.mean(pip_angular_position_littlefinger, axis=0)
    statistics["pip_angular_position_littlefinger_std"] = np.std(pip_angular_position_littlefinger, axis=0)
    statistics["dip_angular_position_littlefinger_mean"] = np.mean(dip_angular_position_littlefinger, axis=0)
    statistics["dip_angular_position_littlefinger_std"] = np.std(dip_angular_position_littlefinger, axis=0)
    statistics["cmc_angular_thumb_mean"] = np.mean(cmc_angular_thumb, axis=0)
    statistics["cmc_angular_thumb_std"] = np.std(cmc_angular_thumb, axis=0)
    statistics["mcp_angular_position_thumb_mean"] = np.mean(mcp_angular_position_thumb, axis=0)
    statistics["mcp_angular_position_thumb_std"] = np.std(mcp_angular_position_thumb, axis=0)
    statistics["ip_angular_thumb_mean"] = np.mean(ip_angular_thumb, axis=0)
    statistics["ip_angular_thumb_std"] = np.std(ip_angular_thumb, axis=0)
    statistics["positional_difference_ball_mean"] = np.mean(positional_difference_ball, axis=0)
    statistics["positional_difference_ball_std"] = np.std(positional_difference_ball, axis=0)
    statistics["positional_difference_target_mean"] = np.mean(positional_difference_target, axis=0)
    statistics["positional_difference_target_std"] = np.std(positional_difference_target, axis=0)
    statistics["positional_difference_from_ball_to_target_mean"] = np.mean(positional_difference_from_ball_to_target, axis=0)
    statistics["positional_difference_from_ball_to_target_std"] = np.std(positional_difference_from_ball_to_target, axis=0)

    return statistics


class Reward_Addition_Module:
    def __init__(self, env_name, env_path, logger_path):
        self.env_name = env_name
        self.env_path = env_path
        self.logger_path = logger_path
        self.best_reward_code_list = []
        self.num_cpu = 16
        self.num_samples = 3
        
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
        n_previous_code = len(reward_code_list) - 1
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

    def train_with_reward_addition(self, seed=0):
        self.seed = seed
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
        
        model = SAC("MlpPolicy",
                    training_env,
                    verbose=1)
        model.learn(total_timesteps=12_000_000, callback=eval_callback)
        model.save(self.logger_path + f"reward_addition/final_model.zip")

        del model, training_env, eval_env, eval_callback
        gc.collect()
        torch.cuda.empty_cache()  # Free up unused memory