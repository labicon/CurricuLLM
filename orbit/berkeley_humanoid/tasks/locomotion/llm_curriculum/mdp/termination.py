# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.managers.command_manager import CommandTerm


def bad_velocity_tracking(
        env: ManagerBasedRLEnv, 
        limit_error: float = 10.0, 
        starting_step: float = 0.0, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's travled position is too far from integration of velocity.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # Extract the command velocity
    command: CommandTerm = env.command_manager.get_term("base_velocity")
    # compute the distance the robot walked
    distance = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]

    error_threshold = torch.tensor(limit_error) * env.episode_length_buf * env.step_dt
    
    # distance that the robot should have walked
    heading_command = command.heading_target
    # The robot should have walked following the command velocity and heading
    # Rotation matrix to convert the command velocity to the world frame
    rotation_matrix = torch.stack([torch.cos(heading_command), torch.sin(heading_command), -torch.sin(heading_command), torch.cos(heading_command)], dim=1).reshape(-1, 2, 2)
    # Matrix multiplication with (num_envs, 2, 2) and (num_envs, 2) to get the world frame velocity
    command_location = torch.matmul(rotation_matrix, command.command[:, :2].unsqueeze(-1)).squeeze(-1) * env.episode_length_buf.unsqueeze(1) * env.step_dt

    # if ((torch.norm(distance, dim=1) - torch.norm(command_location, dim=1)) > error_threshold)[0] * (env.episode_length_buf > starting_step)[0]:
    #     print("Terminating episode: ", env.episode_length_buf[0])
    #     print("Env step dt: ", env.step_dt)
    #     print("Error threshold: ", error_threshold[0])
    #     print("command: ", command.command[0])
    #     print("base lin vel: ", asset.data.root_lin_vel_b[0])
    #     print("heading: ", asset.data.heading_w[0])
    #     print("heading command: ", heading_command[0])
    #     print("Robot distance: ", distance[0])
    #     print("Robot origin: ", env.scene.env_origins[0])
    #     print("Origin for robot 2: ", env.scene.env_origins[1])
    #     print("Command location: ", command_location[0])
    #     print("Distance: ", torch.norm(distance - command_location, dim=1)[0])
    
    # If robot travel distance is less than the command distance, then it is not tracking the velocity
    # So terminate the episode
    return (torch.abs((torch.norm(distance, dim=1) - torch.norm(command_location, dim=1))) > error_threshold) * (env.episode_length_buf > starting_step)

def bad_heading_tracking(
        env:ManagerBasedRLEnv,
        limit_error: float = 45.0 * math.pi/180.0,
        starting_step: float = 0.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's heading angle is too far from the command heading angle"""
    asset: RigidObject = env.scene[asset_cfg.name]
    heading = asset.data.heading_w
    heading_command: CommandTerm = env.command_manager.get_term("base_velocity").heading_target

    # if env.episode_length_buf < starting_step:
    #     return torch.zeros_like(heading, dtype=torch.bool)
    
    return (torch.abs(heading_command - heading) > limit_error) * (env.episode_length_buf > starting_step)
    

