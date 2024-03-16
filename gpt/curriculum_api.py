from openai import OpenAI
import yaml
import os
import numpy as np
import pandas as pd
import re

from gpt.utils import *

GPT_MODEL = "gpt-4-turbo-preview" # gpt-4-1106-preview, gpt-4-0613, gpt-4-32k, gpt-3.5-turbo-1106

class CurriculumAPI:
    def __init__(self, env, prompt_path):
        self.env = env
        self.client = get_client()
        self.prompt_path = prompt_path

    def generate_curriculum(self):
        initial_system = file_to_string(self.prompt_path + self.env + '/curriculum_system.txt')
        initial_user = file_to_string(self.prompt_path + self.env + '/curriculum_user.txt')

        tasks_string = gpt_interaction(self.client, GPT_MODEL, initial_system, initial_user)

        with open(f'{self.current_directory}/' + self.env + '/curriculum_assistant.md', 'w') as file:
            file.write(tasks_string)

        # Split the string into individual task sections
        task_sections = re.split(r'\n\n(?=Task)', tasks_string)

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
        tasks_details = [extract_task_details(section) for section in task_sections]

        # Return list of dictionaries with task details
        return tasks_details

    def generate_rewards(self, task_detail):
        reward_system = file_to_string(self.prompt_path + '/' + self.env + '/reward_system.txt')
        reward_user = file_to_string(self.prompt_path + '/' + self.env + '/reward_user.txt')

        # Concatenate the task details into the user strings
        reward_user = reward_user.replace('<<Task_Name>>', task_detail['Name'])
        reward_user = reward_user.replace('<<Task_Description>>', task_detail['Description'])
        reward_user = reward_user.replace('<<Task_Reason>>', task_detail['Reason'])

        # Get reward function from GPT
        reward_answer = gpt_interaction(self.client, GPT_MODEL, reward_system, reward_user)

        code_block_pattern = re.compile(r"python\n(.*?)\n", re.DOTALL)
        match = code_block_pattern.search(reward_answer)

        if match:
            code_block = match.group(1)
            print("Extracted Code Block:\n", code_block)
            return code_block
        else:
            print("No code block found.")
        return None
	
    def update_env_code(self, env_code_path, task, previous_reward_code=None, version_number=0):
        # Created environment with task and save as version = env_version
        # First, generate reward code from given task info
        reward_code = None
        max_attempt = 5
        attempt = 0
        while reward_code is None and attempt < max_attempt:
            reward_code = self.generate_rewards(task)
            attempt += 1
            if reward_code is None:
                print("Failed to generate reward code. Retrying...")

        # If previous reward code is given, new reward is sum of previous and current reward
        if previous_reward_code is not None:
            reward_code = self.add_rewards(previous_reward_code, reward_code)

        with open(env_code_path, 'r') as file:
            original_code = file.read()

        # Append the new code block to the original code
        new_code = original_code + '\n\n' + code_block

        # Save as a new file with specific version number
        new_file_path = env_code_path.replace('.py', f'_v{version_number}.py')
        with open(new_file_path, 'w') as file:
            file.write(new_code)

        print(f"Updated environment code saved to {new_file_path}")

    def add_rewards(self, previous_reward_code, current_reward_code):
        # Combine the two reward functions
        pass