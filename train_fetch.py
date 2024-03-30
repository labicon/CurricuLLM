import numpy as np

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from evaluation.evalcallback_feedback import CurriculumEvalCallback
from utils.train_utils import *
from gpt.curriculum_api import CurriculumAPI

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
        
    def generate_curriculum(self):
        # Generate curriculum and return list of dictionaries with task details
        self.curriculum_info = self.gpt_api.generate_curriculum()
        self.curriculum_length = len(self.curriculum_info)

    def train_curriculum(self):
        for curriculum_idx in range(self.curriculum_length):
            for sample_num in range(5):
                task = self.curriculum_info[curriculum_idx]
                try:
                    self.train_single(curriculum_idx, task, sample_num)
                except Exception as e:
                    print(f"Error in training task {task['Name']} sample {sample_num}")
                    print(e)
                    continue
            # Evaluate the trained models
            statistics = []
            for sample_num in range(5):
                try:
                    env_id = f"Curriculum/{self.env_name}-v{sample_num}"
                    eval_env = SubprocVecEnv([make_env(env_id, i) for i in range(4)])
                    model_path = self.logger_path + f"{task['Name']}/sample_{sample_num}/final_model.zip"
                    model = SAC.load(model_path)
                    
                    # Get trajectory
                    obs = eval_env.reset()
                    obs_trajectory = [obs['observation'][0]]
                    goal_trajectory = [obs['desired_goal'][0]]
                    for _ in range(1000):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, _, _, _ = eval_env.step(action)
                        obs_trajectory.append(obs['observation'][0])
                        goal_trajectory.append(obs['desired_goal'][0])

                    statistics.append(analyze_trajectory_fetch(obs_trajectory, goal_trajectory))
                except Exception as e:
                    print(f"Error in evaluating task {task['Name']} sample {sample_num}")
                    print(e)
                    statistics.append({"Error": "Error in evaluating task"})
                    continue
            
            # Asl LLM to choose the best model
            best_sample_idx = self.gpt_api.feedback(self.env_name, task, statistics)
            if best_sample_idx is None:
                print("Statistics Analysis error. Try again.")
                while best_sample_idx is None:
                    best_sample_idx = self.gpt_api.feedback(self.env_name, task, statistics)

            self.best_model_idx_list.append(best_sample_idx)
            # Update best reward code list
            self.best_reward_code_list.append(self.current_reward_code_list[best_sample_idx])
            self.current_reward_code_list = []

        # Save the best reward code list
        with open(self.logger_path + "best_reward_code_list.txt", "w") as file:
            for code in self.best_reward_code_list:
                file.write(code + "\n\n")


    def train_single(self, curriculum_idx, task, sample_num):
        # Create the environment
        env_id = f"Curriculum/{self.env_name}-v{sample_num}"

        # Update env code
        reward_code = self.gpt_api.update_env_code(self.env_path, task, 
                                     previous_reward_code=self.best_reward_code_list, 
                                     version_number=sample_num)
        self.current_reward_code_list.append(reward_code)

        # Create the vectorized environment
        num_cpu = 8
        training_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
        eval_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

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
            pre_tuned_model_path = self.logger_path + previous_task + f"/sample_{self.best_model_idx_list[-1]}/final_model.zip"
            model = SAC.load(pre_tuned_model_path)

        model.learn(total_timesteps=10_000, callback=eval_callback)
        model.save(self.logger_path + f"{task['Name']}/sample_{sample_num}/final_model.zip")

        del model, training_env, eval_env, eval_callback

def analyze_trajectory_fetch(obs_trajectory, goal_trajectory):
    # obs_trajectory: list of observations
    # Get list of end effector position, block position, relative block linear velocity, end effector velocity, goal_pos, gosl_distance

    end_effector_pos = []
    block_pos = []
    block_velocity = []
    end_effector_velocity = []
    goal_pos = []
    goal_distance = []

    for obs, goal in zip(obs_trajectory, goal_trajectory):
        end_effector_pos.append(obs[0:3])
        block_pos.append(obs[3:6])
        block_velocity.append(obs[15:18])
        end_effector_velocity.append(obs[20:23])
        goal_pos.append(goal)
        goal_distance.append(np.linalg.norm(obs[3:6] - goal))

    # change to np array
    end_effector_pos = np.array(end_effector_pos)
    block_pos = np.array(block_pos)
    block_velocity = np.array(block_velocity)
    end_effector_velocity = np.array(end_effector_velocity)
    goal_pos = np.array(goal_pos)
    goal_distance = np.array(goal_distance)

    # Calculate mean and std of each variable
    statistics = {}
    statistics["end_effector_pos_mean"] = np.mean(end_effector_pos, axis=0)
    statistics["end_effector_pos_std"] = np.std(end_effector_pos, axis=0)
    statistics["block_pos_mean"] = np.mean(block_pos, axis=0)
    statistics["block_pos_std"] = np.std(block_pos, axis=0)
    statistics["block_relative_velocity_mean"] = np.mean(block_velocity, axis=0)
    statistics["block_relative_velocity_std"] = np.std(block_velocity, axis=0)
    statistics["end_effector_velocity_mean"] = np.mean(end_effector_velocity, axis=0)
    statistics["end_effector_velocity_std"] = np.std(end_effector_velocity, axis=0)
    statistics["goal_pos_mean"] = np.mean(goal_pos, axis=0)
    statistics["goal_pos_std"] = np.std(goal_pos, axis=0)
    statistics["goal_distance_mean"] = np.mean(goal_distance, axis=0)
    statistics["goal_distance_std"] = np.std(goal_distance, axis=0)

    return statistics