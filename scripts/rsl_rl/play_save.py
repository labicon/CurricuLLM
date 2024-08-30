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

    base_lin_vel_traj = []
    base_ang_vel_traj = []
    projected_gravity_traj = []
    velocity_command_traj = []
    hip_pos_traj = []
    kfe_pos_traj = []
    ffe_pos_traj = []
    faa_pos_traj = []
    joint_vel_traj = []
    last_actions_traj = []
    episode_length = []

    step = 0

    # simulate environment
    while step < 1000:
        traj_length = 0
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)

        traj_length += 1

        # obs: shape (n_envs, obs_dim=48) action: shape (n_envs, action_dim=12)
        # print(f"obs: {obs.shape}")
        # print(f"actions: {actions.shape}")
        numpy_obs = obs.cpu().numpy()
        base_lin_vel = numpy_obs[0, :3]
        base_ang_vel = numpy_obs[0, 3:6]
        projected_gravity = numpy_obs[0, 6:9]
        velocity_command = numpy_obs[0, 9:12]
        hip_pos = numpy_obs[0, 12:18]
        kfe_pos = numpy_obs[0, 18:20]
        ffe_pos = numpy_obs[0, 20:22]
        faa_pos = numpy_obs[0, 22:24]
        joint_vel = numpy_obs[0, 24:36]
        last_actions = numpy_obs[0, 36:48]
        

        base_lin_vel_traj.append(base_lin_vel)
        base_ang_vel_traj.append(base_ang_vel)
        projected_gravity_traj.append(projected_gravity)
        velocity_command_traj.append(velocity_command)
        hip_pos_traj.append(hip_pos)
        kfe_pos_traj.append(kfe_pos)
        ffe_pos_traj.append(ffe_pos)
        faa_pos_traj.append(faa_pos)
        joint_vel_traj.append(joint_vel)
        last_actions_traj.append(last_actions)

        if dones[0] == 1:
            episode_length.append(traj_length)

        step += 1

    # close the simulator
    env.close()

    state_log = {}
    state_log["base_lin_vel"] = np.round(np.mean(np.array(base_lin_vel_traj), axis=0), 3)
    state_log["base_ang_vel"] = np.round(np.mean(np.array(base_ang_vel_traj), axis=0), 3)
    state_log["projected_gravity"] = np.round(np.mean(np.array(projected_gravity_traj), axis=0), 3)
    state_log["velocity_command"] = np.round(np.mean(np.array(velocity_command_traj), axis=0), 3)
    state_log["hip_pos"] = np.round(np.mean(np.array(hip_pos_traj), axis=0), 3)
    state_log["kfe_pos"] = np.round(np.mean(np.array(kfe_pos_traj), axis=0), 3)
    state_log["ffe_pos"] = np.round(np.mean(np.array(ffe_pos_traj), axis=0), 3)
    state_log["faa_pos"] = np.round(np.mean(np.array(faa_pos_traj), axis=0), 3)
    state_log["joint_vel"] = np.round(np.mean(np.array(joint_vel_traj), axis=0), 3)
    state_log["last_actions"] = np.round(np.mean(np.array(last_actions_traj), axis=0), 3)
    state_log["episode_length"] = np.round(np.mean(np.array(episode_length), axis=0), 3)

    for key, value in state_log.items():
        print(key, value)

    print(f"Saving state log to: ", args_cli.traj_log_path)
    with open(args_cli.traj_log_path + 'states.pkl', "wb+") as file:
        pickle.dump(state_log, file)


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
