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

  initial_system = file_to_string(f'{current_directory}/' + env + '/env_specific_curriculum_system.txt')
  initial_user = file_to_string(f'{current_directory}/' + env + '/env_specific_curriculum_user.txt')

  completion = client.chat.completions.create(
     model="gpt-4-1106-preview", # gpt-4-1106-preview, gpt-4-0613, gpt-4-32k, gpt-3.5-turbo-1106
     messages=[
        {"role": "system", "content": initial_system},
        {"role": "user", "content": initial_user}
      ]
  )
  
  print(completion.choices[0].message.content)

  with open(f'{current_directory}/' + env + '/env_specific_curriculum.md', 'w') as file:
    file.write(completion.choices[0].message.content)

def generate_reward(env, itr):
  current_file_path = os.path.abspath(__file__)
  current_directory = os.path.dirname(current_file_path)

  initial_system = file_to_string(f'{current_directory}/' + env + '/env_specific_reward_system.txt')
  initial_user = file_to_string(f'{current_directory}/' + env + '/env_specific_reward_user.txt')

  curriculum_user = file_to_string(f'{current_directory}/' + env + '/env_specific_curriculum.md')

  user = initial_user + "\n" + curriculum_user

  completion = client.chat.completions.create(
     model="gpt-4-1106-preview", # gpt-4-1106-preview, gpt-4-0613, gpt-4-32k, gpt-3.5-turbo-1106
     messages=[
        {"role": "system", "content": initial_system},
        {"role": "user", "content": user}
      ]
  )
  
  print(completion.choices[0].message.content)

  with open(f'{current_directory}/' + env + '/env_specific_reward_' + str(itr) + '.md', 'w') as file:
    file.write(completion.choices[0].message.content)


def feedback(env, task, statistics):
  current_file_path = os.path.abspath(__file__)
  current_directory = os.path.dirname(current_file_path)

  feedback_system = file_to_string(f'{current_directory}/prompt/{env}/feedback_system.txt')
  feedback_task = file_to_string(f'{current_directory}/prompt/{env}/feedback_{task}.txt')

  # Statistics to string
  feedback_statistics = ""
  for agent in range(len(statistics)):
    feedback_statistics += f"Agent {agent}:\n"
    for key in statistics[agent]:
      feedback_statistics += f"{key}: {statistics[agent][key]}\n"

  user = feedback_task + "\n" + feedback_statistics


  completion = client.chat.completions.create(
     model="gpt-4-1106-preview", # gpt-4-1106-preview, gpt-4-0613, gpt-4-32k, gpt-3.5-turbo-1106
     messages=[
        {"role": "system", "content": feedback_system},
        {"role": "user", "content": user},
      ]
  )
  
  print(completion.choices[0].message.content)
