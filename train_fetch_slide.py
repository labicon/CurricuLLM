import numpy as np
import gc
import torch
import re

from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from evaluation.evalcallback_feedback import CurriculumEvalCallback
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
                    # Save the error message
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
            model = SAC("MultiInputPolicy",
                        training_env,
                        verbose=1)
        else:
            previous_task = self.curriculum_info[curriculum_idx - 1]['Name']
            pre_tuned_model_path = self.logger_path + previous_task + f"/sample_{self.best_model_idx_list[-1]}/final_model"
            model = SAC.load(pre_tuned_model_path)
            model.set_env(training_env)

        if curriculum_idx == self.curriculum_length - 1 or curriculum_idx == self.curriculum_length - 2:
            model.learn(total_timesteps=5_000_000, callback=eval_callback)
        else:
            model.learn(total_timesteps=500_000, callback=eval_callback)
        model.save(self.logger_path + f"{task['Name']}/sample_{sample_num}/final_model.zip")

        try:
            # Get trajectory
            obs = eval_env.reset()
            obs_trajectory = [obs['observation'][0]]
            goal_trajectory = [obs['desired_goal'][0]]
            for _ in range(500):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, _ = eval_env.step(action)
                obs_trajectory.append(obs['observation'][0])
                goal_trajectory.append(obs['desired_goal'][0])

            self.stats_summary.append(analyze_trajectory_fetch(obs_trajectory, goal_trajectory))
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

        task = self.curriculum_info[resume_idx]

        # Load trajectory statistics
        self.stats_summary = []
        for sample in range(resume_sample_idx):
            try:
                # Load final model and roll out trajectory
                model_path = self.logger_path + f"{task['Name']}/sample_{sample}/"
                env_id = f"Curriculum/{self.env_name}"

                # Create the vectorized environment
                eval_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.num_cpu)])
                
                model = SAC.load(model_path + "/final_model.zip")
                model.set_env(eval_env)
                # Get trajectory
                obs = eval_env.reset()
                obs_trajectory = [obs['observation'][0]]
                goal_trajectory = [obs['desired_goal'][0]]
                for _ in range(500):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, _, _ = eval_env.step(action)
                    obs_trajectory.append(obs['observation'][0])
                    goal_trajectory.append(obs['desired_goal'][0])

                self.stats_summary.append(analyze_trajectory_fetch(obs_trajectory, goal_trajectory))
            except Exception as e:
                print(f"Error in evaluating task {task['Name']} sample {sample_num}")
                print(e)
                # Save error message in log path
                with open(self.logger_path + f"{task['Name']}/sample_{sample_num}/evaluation_error.txt", "w") as file:
                    file.write(str(e))
                self.stats_summary.append({"Error": "Error in evaluating task"})

            del model, eval_env
            gc.collect()
            torch.cuda.empty_cache()  # Free up unused memory

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


def analyze_trajectory_fetch(obs_trajectory, goal_trajectory):
    # obs_trajectory: list of observations
    # Get list of end effector position, block position, relative block linear velocity, end effector velocity, goal_pos, gosl_distance

    end_effector_pos = []
    block_pos = []
    gripper_distance = []
    block_relative_velocity = []
    end_effector_velocity = []
    goal_pos = []
    goal_distance = []

    for obs, goal in zip(obs_trajectory, goal_trajectory):
        end_effector_pos.append(obs[0:3])
        block_pos.append(obs[3:6])
        gripper_distance.append(abs(obs[9] - obs[10]))
        block_relative_velocity.append(obs[15:18])
        end_effector_velocity.append(obs[20:23])
        goal_pos.append(goal)
        goal_distance.append(np.linalg.norm(obs[3:6] - goal))

    # change to np array
    end_effector_pos = np.array(end_effector_pos)
    block_pos = np.array(block_pos)
    gripper_distance = np.array(gripper_distance)
    block_velocity = np.array(block_relative_velocity) + np.array(end_effector_velocity)
    end_effector_velocity = np.array(end_effector_velocity)
    goal_pos = np.array(goal_pos)
    goal_distance = np.array(goal_distance)

    # Calculate mean and std of each variable
    statistics = {}
    statistics["end_effector_pos_mean"] = np.mean(end_effector_pos, axis=0).round(2)
    statistics["end_effector_pos_std"] = np.std(end_effector_pos, axis=0).round(2)
    statistics["block_pos_mean"] = np.mean(block_pos, axis=0).round(2)
    statistics["block_pos_std"] = np.std(block_pos, axis=0).round(2)
    statistics["gripper_distance_mean"] = np.mean(gripper_distance, axis=0).round(2)
    statistics["gripper_distance_std"] = np.std(gripper_distance, axis=0).round(2)
    statistics["block_velocity_mean"] = np.mean(block_velocity, axis=0)
    statistics["block_velocity_std"] = np.std(block_velocity, axis=0)
    statistics["end_effector_velocity_mean"] = np.mean(end_effector_velocity, axis=0)
    statistics["end_effector_velocity_std"] = np.std(end_effector_velocity, axis=0)
    statistics["goal_pos_mean"] = np.mean(goal_pos, axis=0).round(2)
    statistics["goal_pos_std"] = np.std(goal_pos, axis=0).round(2)
    statistics["goal_distance_mean"] = np.mean(goal_distance, axis=0).round(2)
    statistics["goal_distance_std"] = np.std(goal_distance, axis=0).round(2)

    return statistics


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
        env_id = f"{self.env_name}-v2"

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
                    learning_starts = self.num_cpu * 100,
                    verbose=1,
                    replay_buffer_class=HerReplayBuffer,
                    # Parameters for HER
                    replay_buffer_kwargs=dict(
                        n_sampled_goal=4,
                        goal_selection_strategy=goal_selection_strategy,
                    ))
        
        model.learn(total_timesteps=13_000_000, callback=eval_callback)
        model.save(self.logger_path + "her/final_model.zip")

        del model, training_env, eval_env, eval_callback
        gc.collect()
        torch.cuda.empty_cache()

class SAC_Module:
    def __init__(self, env_name, env_path, logger_path, seed=0):
        self.env_name = env_name
        self.env_path = env_path
        self.logger_path = logger_path
        self.num_cpu = 16
        self.seed = seed

    def train_with_sac(self):
        # Create the environment
        env_id = f"{self.env_name}Dense-v2"

        # Create the vectorized environment
        training_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.num_cpu)])
        eval_env = SubprocVecEnv([make_env(env_id, i, seed=self.seed) for i in range(self.num_cpu)])

        # Create the callback
        eval_callback = EvalCallback(eval_env, 
                                    log_path=self.logger_path + "sac/", 
                                    best_model_save_path=self.logger_path + "sac/", 
                                    eval_freq=1000, 
                                    deterministic=True, render=False, warn=False)
        
        model = SAC("MultiInputPolicy",
                    training_env,
                    verbose=1,
                    )
        
        model.learn(total_timesteps=13_000_000, callback=eval_callback)
        model.save(self.logger_path + "her/final_model.zip")

        del model, training_env, eval_env, eval_callback
        gc.collect()
        torch.cuda.empty_cache()


class Scratch_Module:

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
        
    def train_curriculum(self):
        self.load_task_info()

        for curriculum_idx in range(self.curriculum_length):
            for sample_num in range(self.num_samples):
                task = self.curriculum_info[curriculum_idx]
                try:
                    self.train_single(curriculum_idx, task, sample_num)
                except Exception as e:
                    print(f"Error in training task {task['Name']} sample {sample_num}")
                    print(e)
                    # Save the error message
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
        
        model = SAC("MultiInputPolicy",
                    training_env,
                    verbose=1)

        model.learn(total_timesteps=10_000_000, callback=eval_callback)
        model.save(self.logger_path + f"{task['Name']}/sample_{sample_num}/final_model.zip")

        try:
            # Get trajectory
            obs = eval_env.reset()
            obs_trajectory = [obs['observation'][0]]
            goal_trajectory = [obs['desired_goal'][0]]
            for _ in range(500):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, _ = eval_env.step(action)
                obs_trajectory.append(obs['observation'][0])
                goal_trajectory.append(obs['desired_goal'][0])

            self.stats_summary.append(analyze_trajectory_fetch(obs_trajectory, goal_trajectory))
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

    def load_task_info(self):
        # Load curriculum
        with open(self.prompt_path + self.env_name + "original_task_info.md", "r") as file:
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