from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from utils.envs_utils import *

import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Create the vectorized environment
    env_id = "Curriculum/Hopper-v5"
    logger_path = "./logs/" + env_id #+ "_4"
    task = "hopping_forward"
    num_cpu = 4
    eval_env = SubprocVecEnv([make_env(env_id, i, task=task) for i in range(num_cpu)])

    model = PPO.load(logger_path + "/ppo_" + "move_forward")

    # get trajectory
    obs = eval_env.reset()
    obs_trajectory = [obs[0]]
    action_trajectory = []
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        action_trajectory.append(action[0])
        obs, rewards, dones, info = eval_env.step(action)
        obs_trajectory.append(obs[0])

    # (length, dimension) array
    obs_trajectory = np.array(obs_trajectory)
    action_trajectory = np.array(action_trajectory)

    np.savetxt( "move_forward_observation.txt", obs_trajectory, delimiter=",", fmt='%.2f')
    np.savetxt( "move_forward_action.txt", action_trajectory, delimiter=",", fmt='%.2f') 