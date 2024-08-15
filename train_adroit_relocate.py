import numpy as np
import gc
import torch
import re

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from evaluation.evalcallback_feedback import CurriculumEvalCallback
from evaluation.evalcallback_success import SuccessEvalCallback as EvalCallback
from utils.train_utils import *
from gpt.curriculum_api_chain import CurriculumAPI
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
        self.stats_summary = []
    
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
        env_id = f"Curriculum/{self.env_name}"

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
            model = SAC("MlpPolicy",
                        training_env,
                        verbose=1)
        else:
            previous_task = self.curriculum_info[curriculum_idx - 1]['Name']
            pre_tuned_model_path = self.logger_path + previous_task + f"/sample_{self.best_model_idx_list[-1]}/final_model"
            model = SAC.load(pre_tuned_model_path)
            model.set_env(training_env)

        if curriculum_idx == self.curriculum_length - 1 or curriculum_idx == self.curriculum_length - 2:
            model.learn(total_timesteps=50_000_000, callback=eval_callback)
        else:
            model.learn(total_timesteps=10_000_000, callback=eval_callback)
        model.save(self.logger_path + f"{task['Name']}/sample_{sample_num}/final_model.zip")

        del model, training_env, eval_env, eval_callback
        gc.collect()
        torch.cuda.empty_cache()  # Free up unused memory

        try:
            env_id = f"Curriculum/{self.env_name}"
            eval_env = SubprocVecEnv([make_env(env_id, i) for i in range(self.num_cpu)])
            model_path = self.logger_path + f"{task['Name']}/sample_{sample_num}/final_model.zip"
            model = SAC.load(model_path)
            
            # Get trajectory
            obs = eval_env.reset()
            obs_trajectory = [obs[0]]
            for _ in range(800):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, _ = eval_env.step(action)
                obs_trajectory.append(obs[0])

            self.stats_summary.append(analyze_trajectory_adroit(obs_trajectory))

            del eval_env, model
            gc.collect()
            torch.cuda.empty_cache()  # Free up unused memory
        except Exception as e:
            print(f"Error in evaluating task {task['Name']} sample {sample_num}")
            print(e)
            self.stats_summary.append({"Error": "Error in evaluating task"})

    def load_and_retrain(self, model_path, sample_num):
        env_id = f"Curriculum/{self.env_name}"

        # Create the vectorized environment
        training_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.num_cpu)])
        eval_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.num_cpu)])

        # Create the callback
        eval_callback = CurriculumEvalCallback(eval_env,
                                                log_path= model_path + "additional_training/",
                                                best_model_save_path= model_path + "additional_training/",
                                                eval_freq=1000,
                                                deterministic=True, render=False, warn=False)
        
        model = SAC.load(model_path + "/final_model.zip")
        model.set_env(training_env)

        model.learn(total_timesteps=10_000_000, callback=eval_callback)
        model.save(model_path + "additional_training/final_model.zip")

        del model, training_env, eval_env, eval_callback
        gc.collect()
        torch.cuda.empty_cache()  # Free up unused memory

def analyze_trajectory_adroit(obs_trajectory):
    '''
    obs_trajectory: list of observations
    Get list of variables
    (1) arm_position: Translation position of the arm in xyz direction
    (2) arm_angular_position: Angular position of arm
    (3) wrist_angular_position: Angular position of writst
    (4) angular_position_forefinger: Angular position of joint of the forefinger joints in the order of horizontal MCP(metacarpophalangeal, knuckle) joint, and vertical MCP(metacarpophalangeal, knuckle) joint, PIP(proximal interphalangeal, middle), DIP(Distal Interphalangeal, tip)
    (5) angular_position_middlefinger: Angular position of joint of the middlefinger joints in the order of horizontal MCP(metacarpophalangeal, knuckle) joint, and vertical MCP(metacarpophalangeal, knuckle) joint, PIP(proximal interphalangeal, middle), DIP(Distal Interphalangeal, tip)
    (6) angular_position_ringfinger: Angular position of joint of the ringfinger joints in the order of horizontal MCP(metacarpophalangeal, knuckle) joint, and vertical MCP(metacarpophalangeal, knuckle) joint, PIP(proximal interphalangeal, middle), DIP(Distal Interphalangeal, tip)
    (7) angular_position_littlefinger: Angular position of joint of the littlefinger joints in the order of CMC (carpometacarpal) joint, horizontal MCP(metacarpophalangeal, knuckle) joint, vertical MCP(metacarpophalangeal, knuckle) joint, PIP(proximal interphalangeal, middle), DIP(Distal Interphalangeal, tip)
    (8) angular_thumb: Angular position of the thumb joints in the order of Horizontal CMC (carpometacarpal) joint, Vertical CMC (carpometacarpal) joint, horizontal MCP(metacarpophalangeal, knuckle) joint, vertical MCP(metacarpophalangeal, knuckle) joint, IP (Interphalangeal) joint
    (9) positional_difference_ball: xyz positional difference from the palm to the ball, ball_position - palm_position
    (10) positional_difference_target: xyz positional difference from the palm to the target, target_position - palm_position
    (11) positional_difference_from_ball_to_target: xyz positional difference from the ball to the target, target_position - ball_position
    '''

    arm_position = []
    arm_angular_position = []
    wrist_angular_position = []
    angular_position_forefinger = []
    angular_position_middlefinger = []
    angular_position_ringfinger = []
    angular_position_littlefinger = []
    angular_thumb = []
    positional_difference_ball = []
    positional_difference_target = []
    positional_difference_from_ball_to_target = []

    for obs in obs_trajectory:
        arm_position.append(obs[0:3])
        arm_angular_position.append(obs[3:6])
        wrist_angular_position.append(obs[6:8])
        angular_position_forefinger.append(obs[8:12])
        angular_position_middlefinger.append(obs[12:16])
        angular_position_ringfinger.append(obs[16:20])
        angular_position_littlefinger.append(obs[20:25])
        angular_thumb.append(obs[25:30])
        positional_difference_ball.append(obs[30:33])
        positional_difference_target.append(obs[33:36])
        positional_difference_from_ball_to_target.append(obs[36:])
        

    # change to np array
    arm_position = np.array(arm_position)
    arm_angular_position = np.array(arm_angular_position)
    wrist_angular_position = np.array(wrist_angular_position)
    angular_position_forefinger = np.array(angular_position_forefinger)
    angular_position_middlefinger = np.array(angular_position_middlefinger)
    angular_position_ringfinger = np.array(angular_position_ringfinger)
    angular_position_littlefinger = np.array(angular_position_littlefinger)
    angular_thumb = np.array(angular_thumb)
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
    statistics["angular_position_forefinger_mean"] = np.mean(angular_position_forefinger, axis=0)
    statistics["angular_position_forefinger_std"] = np.std(angular_position_forefinger, axis=0)
    statistics["angular_position_middlefinger_mean"] = np.mean(angular_position_middlefinger, axis=0)
    statistics["angular_position_middlefinger_std"] = np.std(angular_position_middlefinger, axis=0)
    statistics["angular_position_ringfinger_mean"] = np.mean(angular_position_ringfinger, axis=0)
    statistics["angular_position_ringfinger_std"] = np.std(angular_position_ringfinger, axis=0)
    statistics["angular_position_littlefinger_mean"] = np.mean(angular_position_littlefinger, axis=0)
    statistics["angular_position_littlefinger_std"] = np.std(angular_position_littlefinger, axis=0)
    statistics["angular_thumb_mean"] = np.mean(angular_thumb, axis=0)
    statistics["angular_thumb_std"] = np.std(angular_thumb, axis=0)
    statistics["positional_difference_ball_mean"] = np.mean(positional_difference_ball, axis=0)
    statistics["positional_difference_ball_std"] = np.std(positional_difference_ball, axis=0)
    statistics["positional_difference_target_mean"] = np.mean(positional_difference_target, axis=0)
    statistics["positional_difference_target_std"] = np.std(positional_difference_target, axis=0)
    statistics["positional_difference_from_ball_to_target_mean"] = np.mean(positional_difference_from_ball_to_target, axis=0)
    statistics["positional_difference_from_ball_to_target_std"] = np.std(positional_difference_from_ball_to_target, axis=0)

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
        final_task = self.curriculum_info[-1]
        reward_code_summary = file_to_string(self.logger_path + f"{final_task['Name']}/best_reward_code.txt")

        with open(self.env_path, 'r') as file:
            original_code = file.read()

        # Append the new code block to the original code
        reward_code_summary = '\n'.join(line for line in reward_code_summary.splitlines())
        new_code = original_code + '\n\n' + reward_code_summary

        # Save as a new file with specific version number
        new_file_path = self.env_path.replace('.py', f'_v0.py')
        with open(new_file_path, 'w') as file:
            file.write(new_code)

        print(f"Updated environment code saved to {new_file_path}")

    def train_with_reward_addition(self):
        self.extract_curriculum()
        self.update_env_code()

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
        model.learn(total_timesteps=50_000_000, callback=eval_callback)
        model.save(self.logger_path + f"reward_addition/final_model.zip")

        del model, training_env, eval_env, eval_callback
        gc.collect()
        torch.cuda.empty_cache()  # Free up unused memory

    def load_and_retrain(self, model_path):
        self.extract_curriculum()
        self.update_env_code()

        env_id = f"Curriculum/{self.env_name}-v0"

        # Create the vectorized environment
        training_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.num_cpu)])
        eval_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.num_cpu)])

        # Create the callback
        eval_callback = CurriculumEvalCallback(eval_env,
                                                log_path= model_path + "additional_training/",
                                                best_model_save_path= model_path + "additional_training/",
                                                eval_freq=1000,
                                                deterministic=True, render=False, warn=False)
        
        model = SAC.load(model_path + "/final_model.zip")
        model.set_env(training_env)

        model.learn(total_timesteps=10_000_000, callback=eval_callback)
        model.save(model_path + "additional_training/final_model.zip")

        del model, training_env, eval_env, eval_callback
        gc.collect()
        torch.cuda.empty_cache()  # Free up unused memory

class SAC_Module:
    def __init__(self, env_name, env_path, logger_path, seed=0):
        self.env_name = env_name
        self.env_path = env_path
        self.logger_path = logger_path
        self.num_cpu = 16
        self.num_samples = 3
        self.seed = seed

    def train_sac(self):
        # Create the environment
        env_id = f"{self.env_name}-v1"

        # Create the vectorized environment
        training_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.num_cpu)])
        eval_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.num_cpu)])

        # Create the callback
        eval_callback = EvalCallback(eval_env, 
                                    log_path=self.logger_path + "sac/", 
                                    best_model_save_path=self.logger_path + "sac/", 
                                    eval_freq=1000, 
                                    deterministic=True, render=False, warn=False)
        
        model = SAC("MlpPolicy",
                    training_env,
                    verbose=1)
        model.learn(total_timesteps=12_000_000, callback=eval_callback)
        model.save(self.logger_path + "sac/final_model.zip")

        del model, training_env, eval_env, eval_callback
        gc.collect()
        torch.cuda.empty_cache()  # Free up unused memory