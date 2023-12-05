from openai import OpenAI
import yaml
import os
from gpt.utils import *
import numpy as np
import pandas as pd

with open('./gpt/key.yaml', 'r') as stream:
    config = yaml.safe_load(stream)

client = OpenAI(api_key=config['OPENAI_API_KEY'])

def generate_curriculum(env):
  current_file_path = os.path.abspath(__file__)
  current_directory = os.path.dirname(current_file_path)

  initial_system = file_to_string(f'{current_directory}/' + env + '/curriculum_system.txt')
  initial_user = file_to_string(f'{current_directory}/' + env + '/curriculum_user.txt')

  completion = client.chat.completions.create(
     model="gpt-4-1106-preview", # gpt-4-1106-preview, gpt-4-0613, gpt-4-32k, gpt-3.5-turbo-1106
     messages=[
        {"role": "system", "content": initial_system},
        {"role": "user", "content": initial_user}
      ]
  )
  
  print(completion.choices[0].message.content)

  with open(f'{current_directory}/' + env + '/curriculum_assistant.md', 'w') as file:
    file.write(completion.choices[0].message.content)


def feedback(env, reward_task, reward_df, task, obs_traj, action_traj):
  current_file_path = os.path.abspath(__file__)
  current_directory = os.path.dirname(current_file_path)

  curriculum_system = file_to_string(f'{current_directory}/' + env + '/curriculum_system.txt')
  curriculum_user = file_to_string(f'{current_directory}/' + env + '/curriculum_user.txt')

  # load the curriculum and reward function generated in curriculum generation step
  curriculum_assistant = file_to_string(f'{current_directory}/' + env + '/curriculum_assistant.md')

  feedback_user = file_to_string(f'{current_directory}/' + env + '/feedback_user.txt')
  feedback_user =  feedback_user.format(TASK=task, REWARD_REFLECTION_HERE=reward_reflection(reward_task, reward_df))

  completion = client.chat.completions.create(
     model="gpt-4-1106-preview", # gpt-4-1106-preview, gpt-4-0613, gpt-4-32k, gpt-3.5-turbo-1106
     messages=[
        {"role": "system", "content": curriculum_system},
        {"role": "user", "content": curriculum_user},
        {"role": "assistant", "content": curriculum_assistant},
        {"role": "user", "content": feedback_user}
      ]
  )
  
  print(completion.choices[0].message)

def reward_reflection(reward_task, reward_df: pd.DataFrame):
  reward_task_string = "Reward for task: " + np.array2string(reward_task, precision=3, separator=',', suppress_small=True)
  reward_string = [reward_task_string]
  for column in reward_df.columns():
     reward_string.append(column + ": " + np.array2string(reward_df[column].to_numpy(), precision=3, separator=',', suppress_small=True))

  feedback_string = "\n".join(reward_string)

  return feedback_string