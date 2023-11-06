import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./logs/Curriculum/HalfCheetah-v4/progress.csv')

reward_main = data['eval/mean_reward_main']
reward_task = data['eval/mean_reward_task']
task = data['eval/current_task']