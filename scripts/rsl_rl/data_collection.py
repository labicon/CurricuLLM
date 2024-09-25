"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--traj_log_path", type=str, default=None, help="Path to save the trajectory log.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch
import numpy as np
import pickle

from rsl_rl.runners import OnPolicyRunner

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_onnx,
)

# Import extensions to set up environment tasks
import orbit.berkeley_humanoid.tasks


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

    # reset environment
    obs, _ = env.get_observations()

    # base_lin_vel_traj = []
    # base_ang_vel_traj = []
    # projected_gravity_traj = []
    # velocity_command_traj = []
    # hip_pos_traj = []
    # kfe_pos_traj = []
    # ffe_pos_traj = []
    # faa_pos_traj = []
    # joint_vel_traj = []
    # last_actions_traj = []
    episode_length_traj = []
    velocity_error_traj = []
    fail_count = 0
    success_count = 0

    step = 0
    traj_length = np.zeros(args_cli.num_envs)

    # simulate environment
    while step < 5000:
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)

        # obs: shape (n_envs, obs_dim=48) action: shape (n_envs, action_dim=12)
        numpy_obs = obs.cpu().numpy()
        velocity_error = np.linalg.norm((env.get_observations()[1]['observations']['base_lin_vel'].cpu().numpy()[:,:2] - numpy_obs[:, 6:8]), axis=1)

        velocity_error_traj.append(velocity_error)
        # detect env id of done environments
        done_env_ids = np.where(dones.cpu().numpy())[0]
        # increment the trajectory length
        traj_length += 1
        # for each done environment, record the episode length
        for env_id in done_env_ids:
            print(f"Episode length: {traj_length[env_id]}")
            print("Env ID: ", env_id)
            episode_length_traj.append(traj_length[env_id])

            # If the trajectory length is smaller than 100, report as failure
            if traj_length[env_id] < 100:
                fail_count += 1
            else:
                success_count += 1

            traj_length[env_id] = 0


        step += 1

    # close the simulator
    env.close()

    state_log = {}
    # state_log["base_lin_vel"] = np.round(np.mean(np.concatenate(base_lin_vel_traj, axis=0), axis=0), 3)
    # state_log["base_ang_vel"] = np.round(np.mean(np.concatenate(base_ang_vel_traj, axis=0), axis=0), 3)
    # state_log["velocity_command"] = np.round(np.mean(np.concatenate(velocity_command_traj, axis=0), axis=0), 3)
    state_log["episode_length_mean"] = np.round(np.mean(np.array(episode_length_traj), axis=0), 3)
    state_log["episode_length_std"] = np.round(np.std(np.array(episode_length_traj), axis=0), 3)
    if len(velocity_error_traj) == 0:
        velocity_error_traj = np.zeros(1)
    velocity_error_traj = np.concatenate(velocity_error_traj, axis=0)
    state_log["velocity_error_mean"] = np.round(np.mean(velocity_error_traj, axis=0), 3)
    state_log["velocity_error_std"] = np.round(np.std(velocity_error_traj, axis=0), 3)
    state_log["velocity_error_max"] = np.round(np.max(velocity_error_traj, axis=0), 3)
    state_log["velocity_error_min"] = np.round(np.min(velocity_error_traj, axis=0), 3)

    print("Failed episodes: ", fail_count)
    print("Successful episodes: ", success_count)

    for key, value in state_log.items():
        print(key, value)

    print(f"Saving state log to: ", args_cli.traj_log_path)
    with open(args_cli.traj_log_path + '/results.pkl', "wb+") as file:
        pickle.dump(state_log, file)


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
