from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from orbit.berkeley_humanoid.sensors.contact_foot_height_sensor import ContactFootHeightSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import RLTaskEnv, ManagerBasedRLEnv


def feet_air_time(
    env: RLTaskEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold_min: float, threshold_max: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # negative reward for small steps
    air_time = (last_air_time - threshold_min) * first_contact
    # no reward for large steps
    air_time = torch.clamp(air_time, max=threshold_max - threshold_min)
    reward = torch.sum(air_time, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env, command_name: str, threshold_min: float, threshold_max: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold_max)
    # no reward for small steps
    reward *= reward > threshold_min
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def foot_clearance_reward(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    contact_sensor: ContactFootHeightSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    current_relative_height = contact_sensor.data.current_relative_height[:, sensor_cfg.body_ids]
    foot_z_target_error = torch.square(current_relative_height - target_height)
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def normalized_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    command = env.command_manager.get_command("base_velocity")
    command[2] = command[2] * 0.3
    asset: Articulation = env.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_b
    base_ang_vel = asset.data.root_ang_vel_b
    base_ang_vel[:,2] = base_ang_vel[:,2] * 0.3
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids] * 3.0e-3
    hip_asset_cfg = SceneEntityCfg("robot", joint_names=[".*HR", ".*HAA"])
    hip_asset: Articulation = env.scene[hip_asset_cfg.name]
    hip_pos_deviation = (
        hip_asset.data.joint_pos[:, hip_asset_cfg.joint_ids] - hip_asset.data.default_joint_pos[:, hip_asset_cfg.joint_ids]
    ) * 0.3
    knee_asset_cfg = SceneEntityCfg("robot", joint_names=[".*KFE"])
    knee_asset: Articulation = env.scene[knee_asset_cfg.name]
    knee_pos_deviation = (
        knee_asset.data.joint_pos[:,knee_asset_cfg.joint_ids] - knee_asset.data.default_joint_pos[:,knee_asset_cfg.joint_ids]
    ) * 0.1
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids] * 0.1
    joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids] * 0.01
    nonflat_base_orientation = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) * 5
    base_height_diff = asset.data.root_pos_w[:, 2] - asset.data.default_root_state[:, 2] 

    return command, base_lin_vel, base_ang_vel, hip_pos_deviation, knee_pos_deviation, joint_vel, joint_acc, torque, nonflat_base_orientation, base_height_diff

# def reward_curriculum(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Compute the reward for the current curriculum step."""
#     command, base_lin_vel, base_ang_vel, hip_pos_deviation, knee_pos_deviation, \
#     joint_vel, joint_acc, torque, \
#     nonflat_base_orientation, base_height_diff = normalized_obs(env, asset_cfg)
    # terminated = env.termination_manager.terminated.float()

#     n_envs = env.scene.num_envs

#     # Implement your reward function here
#     reward = torch.zeros(n_envs, device=env.device)
#     return reward
