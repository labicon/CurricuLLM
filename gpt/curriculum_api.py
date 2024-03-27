from openai import OpenAI
import yaml
import os
import numpy as np
import pandas as pd
import re

from gpt.utils import *

GPT_MODEL = "gpt-4-turbo-preview" # gpt-4-1106-preview, gpt-4-0613, gpt-4-32k, gpt-3.5-turbo-1106

class CurriculumAPI:
    def __init__(self, env_name, prompt_path, log_path):
        self.env = env_name
        self.client = get_client()
        self.prompt_path = prompt_path
        self.log_path = log_path

    def generate_curriculum(self):
        initial_system = file_to_string(self.prompt_path + self.env + '/curriculum_system.txt')
        initial_user = file_to_string(self.prompt_path + self.env + '/curriculum_user.txt')

        tasks_string = gpt_interaction(self.client, GPT_MODEL, initial_system, initial_user)

        # Ensure the directory exists and write the curriculum to a file
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path + 'curriculum.md', 'w') as file:
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
        reward_system = file_to_string(self.prompt_path + self.env + '/reward_system.txt')
        reward_user = file_to_string(self.prompt_path + self.env + '/reward_user.txt')

        # Concatenate the task details into the user strings
        reward_user = reward_user.replace('<<Task_Name>>', task_detail['Name'])
        reward_user = reward_user.replace('<<Task_Description>>', task_detail['Description'])
        reward_user = reward_user.replace('<<Task_Reason>>', task_detail['Reason'])

        # Get reward function from GPT
        reward_answer = gpt_interaction(self.client, GPT_MODEL, reward_system, reward_user)

        pattern = r"`python\n(.*?)\n`"
        match = re.search(pattern, reward_answer, re.DOTALL)

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

        # Save the reward code
        task_name = task['Name']
        save_string_to_file(self.log_path + f'{task_name}_reward_code_v{version_number}.md', reward_code)
        # If previous reward code is given, new reward is sum of previous and current reward
        if previous_reward_code is not None:
            reward_code_summary = self.add_rewards(previous_reward_code, reward_code)
        else:
            reward_code_summary = reward_code

        with open(env_code_path, 'r') as file:
            original_code = file.read()

        # Append the new code block to the original code
        # Indent the code block with 4 spaces to the beginning of each line
        reward_code_summary = '\n'.join('    ' + line for line in reward_code_summary.splitlines())
        new_code = original_code + '\n\n' + reward_code_summary

        # Save as a new file with specific version number
        new_file_path = env_code_path.replace('.py', f'_v{version_number}.py')
        with open(new_file_path, 'w') as file:
            file.write(new_code)

        print(f"Updated environment code saved to {new_file_path}")

        return reward_code

    def add_rewards(self, reward_code_list: list, current_reward_code):
        # Find the length of the previous code block
        n_previous_code = len(reward_code_list)
        for idx, code in enumerate(reward_code_list):
            reward_code_list[idx] = code.replace("compute_reward_curriculum(", f"compute_reward_{idx}(")

        new_code = current_reward_code.replace("compute_reward_curriculum(", f"compute_reward_{n_previous_code}(")
        new_code_list = reward_code_list + [new_code]

        reward_code = ""
        for code in new_code_list:
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
            print(f"Function {function_name} not found.")
    return total_reward, total_reward_dict"""
        return reward_code

    def feedback(self, env_name, task, statistics):
        feedback_system = file_to_string(self.prompt_path + env_name + '/feedback_system.txt')
        feedback_user = file_to_string(self.prompt_path + env_name + '/feedback_user.txt')

        # Concatenate the task details into the user strings
        feedback_user = feedback_user.replace('<<Task_Name>>', task['Name'])
        feedback_user = feedback_user.replace('<<Task_Description>>', task['Description'])
        feedback_user = feedback_user.replace('<<Task_Reason>>', task['Reason'])

        # Statistics to string
        feedback_statistics = ""
        for agent in range(len(statistics)):
            feedback_statistics += f"Agent {agent}:\n"
            for key, value in statistics[agent].items():
                feedback_statistics += f"{key}: {value}\n"
            feedback_statistics += "\n"

        feedback_user = feedback_user + "\n" + feedback_statistics

        gpt_answer = gpt_interaction(self.client, GPT_MODEL, feedback_system, feedback_user)

        decision = gpt_answer.split('\n')[0]
        print("For task " + task['Name'] + ", GPT decided " + decision) 
        numbers = re.findall(r'\d+', decision)
        if numbers:
            return int(numbers[0])
        else:
            print("No number found in the decision.")
            return None
