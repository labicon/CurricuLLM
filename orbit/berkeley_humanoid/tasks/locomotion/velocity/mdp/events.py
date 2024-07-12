from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.envs.mdp.events import _randomize_prop_by_op

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        pos_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos



def randomize_joint_parameters(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float] | None = None,
    armature_distribution_params: tuple[float, float] | None = None,
    lower_limit_distribution_params: tuple[float, float] | None = None,
    upper_limit_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the joint parameters of an articulation by adding, scaling, or setting random values.

    This function allows randomizing the joint parameters of the asset.
    These correspond to the physics engine joint properties that affect the joint behavior.

    The function samples random values from the given distribution parameters and applies the operation to the joint properties.
    It then sets the values into the physics simulation. If the distribution parameters are not provided for a
    particular property, the function does not modify the property.

    .. tip::
        This function uses CPU tensors to assign the joint properties. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    # sample joint properties from the given ranges and set into the physics simulation
    # -- friction
    if friction_distribution_params is not None:
        for actuator in asset.actuators.values():
            actuator_joint_ids = [joint_id in joint_ids for joint_id in actuator.joint_indices]
            if sum(actuator_joint_ids) > 0:
                friction = actuator.friction_torque.to(asset.device).clone()
                friction = _randomize_prop_by_op(
                    friction, friction_distribution_params, env_ids, torch.arange(friction.shape[1]), operation=operation, distribution=distribution
                )[env_ids][:, actuator_joint_ids]
                actuator.friction_torque[env_ids[:, None], actuator_joint_ids] = friction

                friction = actuator.friction_vel.to(asset.device).clone()
                friction = _randomize_prop_by_op(
                    friction, friction_distribution_params, env_ids, torch.arange(friction.shape[1]), operation=operation, distribution=distribution
                )[env_ids][:, actuator_joint_ids]
                actuator.friction_vel[env_ids[:, None], actuator_joint_ids] = friction
    # -- armature
    if armature_distribution_params is not None:
        armature = asset.data.default_joint_armature.to(asset.device).clone()
        armature = _randomize_prop_by_op(
            armature, armature_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]
        asset.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=env_ids)
    # -- dof limits
    if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
        dof_limits = asset.data.default_joint_limits.to(asset.device).clone()
        if lower_limit_distribution_params is not None:
            lower_limits = dof_limits[..., 0]
            lower_limits = _randomize_prop_by_op(
                lower_limits,
                lower_limit_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )[env_ids][:, joint_ids]
            dof_limits[env_ids[:, None], joint_ids, 0] = lower_limits
        if upper_limit_distribution_params is not None:
            upper_limits = dof_limits[..., 1]
            upper_limits = _randomize_prop_by_op(
                upper_limits,
                upper_limit_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )[env_ids][:, joint_ids]
            dof_limits[env_ids[:, None], joint_ids, 1] = upper_limits
        if (dof_limits[env_ids[:, None], joint_ids, 0] > dof_limits[env_ids[:, None], joint_ids, 1]).any():
            raise ValueError(
                "Randomization term 'randomize_joint_parameters' is setting lower joint limits that are greater than"
                " upper joint limits."
            )

        asset.write_joint_limits_to_sim(dof_limits[env_ids][:, joint_ids], joint_ids=joint_ids, env_ids=env_ids)