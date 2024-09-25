import os
import re

from gpt.utils import get_client, gpt_interaction, file_to_string, save_string_to_file

GPT_MODEL = "gpt-4-turbo-preview"  # gpt-4-1106-preview, gpt-4-0613, gpt-4-32k, gpt-3.5-turbo-1106


class CurriculumAPI:
    def __init__(self, prompt_path, log_path):
        self.client = get_client()
        self.prompt_path = prompt_path
        self.log_path = log_path

    def generate_curriculum(self):
        initial_system = file_to_string(self.prompt_path + "/curriculum_system.txt")
        initial_user = file_to_string(self.prompt_path + "/curriculum_user.txt")

        tasks_string = gpt_interaction(self.client, GPT_MODEL, initial_system, initial_user)

        # Ensure the directory exists and write the curriculum to a file
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path + "curriculum.md", "w") as file:
            file.write(tasks_string)

        # Split the string into individual task sections
        task_sections = re.split(r"\n\n(?=Task)", tasks_string)

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
                    details["Description"] = ": ".join(line.split(": ")[1:])
                elif line.startswith("Reason:"):
                    details["Reason"] = ": ".join(line.split(": ")[1:])
            return details

        # Extract details for all tasks
        tasks_details = [extract_task_details(section) for section in task_sections]
        self.tasks_details = tasks_details

        # Return list of dictionaries with task details
        return tasks_details

    def generate_rewards(self, curriculum_idx, reward_code_history):
        task_detail = self.tasks_details[curriculum_idx]

        reward_system = file_to_string(self.prompt_path + "/reward_system.txt")
        reward_user = file_to_string(self.prompt_path + "/reward_user.txt")

        # Concatenate the task details into the user strings
        reward_user = reward_user.replace("<<Task_Name>>", task_detail["Name"])
        reward_user = reward_user.replace("<<Task_Description>>", task_detail["Description"])
        reward_user = reward_user.replace("<<Task_Reason>>", task_detail["Reason"])

        # Add previous task and reward information
        if curriculum_idx > 0:
            i = curriculum_idx - 1
            task_history_details = self.tasks_details[i]
            reward_code = reward_code_history[i]
            reward_history = file_to_string(self.prompt_path + "/reward_history.txt")
            reward_history = reward_history.replace("<<Task_Name>>", task_history_details["Name"])
            reward_history = reward_history.replace("<<Task_Description>>", task_history_details["Description"])
            reward_history = reward_history.replace("<<Task_Reason>>", task_history_details["Reason"])
            reward_history = reward_history.replace("<<Task_Code>>", reward_code)

            reward_user = reward_user + "\n" + reward_history

        # Get reward function from GPT
        reward_answer = gpt_interaction(self.client, GPT_MODEL, reward_system, reward_user)

        pattern = r"`python\n(.*?)\n`"
        reward_match = re.search(pattern, reward_answer, re.DOTALL)

        pattern = r"`command\n(.*?)\n`"
        command_match = re.search(pattern, reward_answer, re.DOTALL)

        if reward_match and command_match:
            reward_code_block = reward_match.group(1)
            command_code_block = command_match.group(1)
            # print("Extracted Code Block:\n", code_block)
            return reward_code_block, command_code_block
        else:
            print("No code block found.")
        return None, None

    def update_env_code(self, reward_code_path, command_code_path, curriculum_idx, previous_reward_code=None, version_number=0):
        # Created environment with task and save as version = env_version
        # First, generate reward code from given task info
        reward_code, command_code = None, None
        max_attempt = 5
        attempt = 0
        while reward_code is None and attempt < max_attempt:
            reward_code, command_code = self.generate_rewards(curriculum_idx, previous_reward_code)
            attempt += 1
            if reward_code is None:
                print("Failed to generate reward code. Retrying...")

        # Save the reward code
        task = self.tasks_details[curriculum_idx]
        save_string_to_file(self.log_path + f"{task['Name']}/sample_{version_number}/" + "reward_code.md", reward_code)
        save_string_to_file(
            self.log_path + f"{task['Name']}/sample_{version_number}/" + "command_code.md", command_code
        )

        with open(reward_code_path, "r") as file:
            original_code = file.read()

        reward_code = "\n".join(line for line in reward_code.splitlines())
        new_code = original_code + "\n\n" + reward_code

        # Save the updated reward code
        new_code_path = "./orbit/berkeley_humanoid/tasks/locomotion/llm_curriculum/mdp/rewards.py"
        with open(new_code_path, "w") as file:
            file.write(new_code)

        print(f"Updated reward code saved to {new_code_path}")

        # Save the command code
        def insert_line_in_file(file_path, string_to_insert, line_number):
            source_file_path = "./orbit/berkeley_humanoid/tasks/locomotion/llm_curriculum/curriculum_env_cfg_source.py"
            # Read the current content of the file
            with open(source_file_path, 'r') as file:
                lines = file.readlines()
            
            # Insert the string at the specified line number
            lines.insert(line_number - 1, '        ' + string_to_insert + '\n')
            
            # Write the updated content back to the file
            with open(file_path, 'w') as file:
                file.writelines(lines)

        insert_line_in_file(command_code_path, command_code, 382)
        print(f"Updated command code saved to {command_code_path}")

        return reward_code

    def feedback(self, task, curriculum_idx, statistics):
        feedback_system = file_to_string(self.prompt_path + "/feedback_system.txt")
        feedback_user = file_to_string(self.prompt_path + "/feedback_user.txt")

        # Concatenate the task details into the user strings
        feedback_user = feedback_user.replace("<<Task_Name>>", task["Name"])
        feedback_user = feedback_user.replace("<<Task_Description>>", task["Description"])
        feedback_user = feedback_user.replace("<<Task_Reason>>", task["Reason"])

        # Add previous task information
        if curriculum_idx > 0:
            for i in range(curriculum_idx):
                task_history_details = self.tasks_details[i]
                feedback_history = file_to_string(self.prompt_path + "/feedback_history.txt")
                feedback_history = feedback_history.replace("<<Task_Name>>", task_history_details["Name"])
                feedback_history = feedback_history.replace("<<Task_Description>>", task_history_details["Description"])
                feedback_history = feedback_history.replace("<<Task_Reason>>", task_history_details["Reason"])

                feedback_user = feedback_user + "\n" + feedback_history

        # Statistics to string
        feedback_statistics = ""
        for agent in range(len(statistics)):
            feedback_statistics += f"Agent {agent}:\n"
            for key, value in statistics[agent].items():
                feedback_statistics += f"{key}: {value}\n"
            feedback_statistics += "\n"

        feedback_user = feedback_user + "\n" + feedback_statistics

        gpt_answer = gpt_interaction(self.client, GPT_MODEL, feedback_system, feedback_user)

        # Ensure the directory exists and write the curriculum to a file
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path + task["Name"] + "_statistics.md", "w") as file:
            file.write(feedback_user)
        with open(self.log_path + task["Name"] + ".md", "w") as file:
            file.write(gpt_answer)

        decision = gpt_answer.split("\n")[0]
        numbers = re.findall(r"\d+", decision)
        if numbers:
            return int(numbers[0])
        else:
            print("No number found in the decision.")
            return None
