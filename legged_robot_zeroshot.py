# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import subprocess
import sys
import pickle
import pathlib

from gpt.curriculum_api import CurriculumAPI


class ZeroshotModule:
    def __init__(self, seed=0, iteration=None):
        self.prompt_path = "./gpt/prompt/random_command/"
        self.log_path = "./logs/hum_run/"
        self.reward_function_path = "./orbit/berkeley_humanoid/tasks/locomotion/llm_curriculum/mdp/rewards_source.py"
        self.command_path = "./orbit/berkeley_humanoid/tasks/locomotion/llm_curriculum/curriculum_env_cfg.py"
        self.gpt_api = CurriculumAPI(self.prompt_path, self.log_path)
        self.best_reward_list = []
        self.best_model_idx_list = []
        self.current_reward_list = []
        self.num_reward_samples = 16
        self.seed = seed
        self.max_iterations = iteration
        self.stats_summary = []

    def train_curriculum(self):
        self.load_task_info()

        # Train curriculum
        prev_task = None
        for idx, task in enumerate(self.curriculum_info):
            print(f"Training task {task['Name']}")
            for sample_num in range(5, self.num_reward_samples):
                try:
                    self.train_single(task, prev_task, idx, sample_num)
                except Exception as e:
                    print(f"Error in training task {task['Name']} Sample {sample_num}: {e}")
                    continue

            # Evaluate
            _ = self.evaluate_best_model(task, idx)
            prev_task = task

            self.stats_summary = []

    def train_single(self, task, prev_task, curriculum_idx, sample_num):
        # Environment Task id
        task_id = "Velocity-Flat-Berkeley-Curriculum-Humanoid-v0"

        # Update env code
        reward_code = self.gpt_api.update_env_code(
            self.reward_function_path,
            self.command_path,
            curriculum_idx=curriculum_idx,
            previous_reward_code=self.best_reward_list,
            version_number=sample_num,
        )
        self.current_reward_list.append(reward_code)

        # Train
        if prev_task is None:
            run_args = [
                "python",
                "./scripts/rsl_rl/train.py",
                f"--task={task_id}",
                f"--run_name={task['Name']}_v{sample_num}",
                "--headless",
            ]
        else:
            run_args = [
                "python",
                "./scripts/rsl_rl/train.py",
                f"--task={task_id}",
                f"--run_name={task['Name']}_v{sample_num}",
                "--headless",
                "--resume=True",
                f"--load_run={prev_task['Name']}_v{self.best_model_idx_list[-1]}",
            ]

        rl_filepath = f"{self.log_path}{task['Name']}/sample_{sample_num}/"
        os.makedirs(rl_filepath, exist_ok=True)
        with open(rl_filepath + "response.txt", "w") as file:
            process = subprocess.Popen(run_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for c in iter(lambda: process.stdout.read(1), b""):
                sys.stdout.buffer.write(c)
                file.buffer.write(c)
            for c in iter(lambda: process.stderr.read(1), b""):
                sys.stderr.buffer.write(c)
                file.buffer.write(c)
            process.wait()

        try:
            state_log = self.observation_rollout(task, sample_num)
            self.stats_summary.append(state_log)
        except Exception as e:
            print(f"Error in evaluating task {task['Name']} Sample {sample_num}: {e}")
            self.stats_summary.append({"Error": "Error in evaluating task"})

    def evaluate_best_model(self, task, curriculum_idx):
        # Evaluate best model

        # Ask LLM to choose the best model
        best_model_idx = self.gpt_api.feedback(task, curriculum_idx, self.stats_summary)
        trial = 1
        while best_model_idx is None:
            print("No best model selected. Retrying...")
            best_model_idx = self.gpt_api.feedback(task, curriculum_idx, self.stats_summary)
            trial += 1
            if trial == 5:
                best_model_idx = 0
                while self.stats_summary[best_model_idx]["Error"]:
                    print(f"Error in sample {best_model_idx}. Skipping...")
                    best_model_idx += 1

        self.best_model_idx_list.append(best_model_idx)
        self.best_reward_list.append(self.current_reward_list[best_model_idx])

        with open(self.log_path + f"{task['Name']}/best_reward_code.txt", "w") as file:
            file.write(self.current_reward_list[best_model_idx])

        self.current_reward_list = []

        return best_model_idx

    def observation_rollout(self, task, sample_num):
        traj_filepath = (
            str(pathlib.Path(__file__).parent.resolve()) + f"/{self.log_path}{task['Name']}/sample_{sample_num}/"
        )
        os.makedirs(traj_filepath, exist_ok=True)
        with open(traj_filepath + "traj_rollout.txt", "w") as file:
            print(f"--load_run={task['Name']}_v{sample_num}")
            process = subprocess.Popen(
                [
                    "python",
                    "./scripts/rsl_rl/play_save.py",
                    f"--task=Velocity-Flat-Berkeley-Curriculum-Humanoid-Play-v0",
                    f"--traj_log_path={traj_filepath}",
                    "--headless",
                    f"--load_run={task['Name']}_v{sample_num}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for c in iter(lambda: process.stdout.read(1), b""):
                sys.stdout.buffer.write(c)
                file.buffer.write(c)
            for c in iter(lambda: process.stderr.read(1), b""):
                sys.stderr.buffer.write(c)
                file.buffer.write(c)

            process.wait()

        print("Loading trajectory logs from: ", traj_filepath)
        with open(os.path.join(traj_filepath, "states.pkl"), "rb") as f:
            state_dict = pickle.load(f)

        return state_dict

    def load_task_info(self):
        # Load curriculum
        with open(self.log_path + "original_task_info.md", "r") as file:
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

    def load_current_rewards(self, resume_idx):
        for sample in range(self.num_reward_samples):
            with open(
                self.log_path + f"{self.curriculum_info[resume_idx]['Name']}/sample_{sample}/reward_code.md", "r"
            ) as file:
                response = file.read()
                if "Error" in response:
                    self.current_reward_list.append(None)
                else:
                    self.current_reward_list.append(response)

    def resume_curriculum(self, resume_idx, resume_sample_idx=0, resume_from_training=True):
        print(f"Resuming curriculum at task {resume_idx}")
        # Load curriculum and rewards
        self.load_task_info()

        for idx, task in enumerate(self.curriculum_info[resume_idx:], start=resume_idx):
            if resume_from_training:
                print(f"Training task {task['Name']}")
                start_idx = resume_sample_idx
                for sample_num in range(start_idx, self.num_reward_samples):
                    try:
                        self.train_single(task, None, idx, sample_num)
                    except Exception as e:
                        print(f"Error in training task {task['Name']} Sample {sample_num}: {e}")
                        continue
                start_idx = 0
            else:
                self.load_current_rewards(resume_idx)
                print(f"Loaded current rewards for task {task['Name']}")

            # Evaluate
            best_model_idx = self.evaluate_best_model(task, idx)
            self.stats_summary = []
            resume_from_training = True


if __name__ == "__main__":
    curriculum_module = ZeroshotModule(iteration=5000)
    curriculum_module.train_curriculum()
    # curriculum_module.resume_curriculum(0, 14, resume_from_training=True)
