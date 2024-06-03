from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.envs.mdp.events import _randomize_prop_by_op

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv


# Fix randomization of specific joint parameters
def randomize_joint_parameters(
        env: BaseEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        friction_range: tuple[float, float] | None = None,
        armature_range: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform"] = "uniform",
):
    """Randomize the joint parameters of an articulation by adding, scaling, or setting random values.

    This function allows randomizing the joint parameters (friction and armature) of the asset. These correspond
    to the physics engine joint properties that affect the joint behavior.

    The function samples random values from the given ranges and applies the operation to the joint properties.
    It then sets the values into the physics simulation. If the ranges are not provided for a
    particular property, the function does not modify the property.

    .. tip::
        This function uses CPU tensors to assign the joint properties. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    # sample joint properties from the given ranges and set into the physics simulation
    # -- friction
    if friction_range is not None:
        friction = asset.root_physx_view.get_dof_friction_coefficients().to(asset.device)
        friction = _randomize_prop_by_op(
            friction, friction_range, env_ids, joint_ids, operation=operation, distribution=distribution
        )[:, joint_ids]
        asset.write_joint_friction_to_sim(friction, joint_ids=joint_ids, env_ids=env_ids)
    # -- armature
    if armature_range is not None:
        armature = asset.root_physx_view.get_dof_armatures().to(asset.device)
        armature = _randomize_prop_by_op(
            armature, armature_range, env_ids, joint_ids, operation=operation, distribution=distribution
        )[:, joint_ids]
        asset.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=env_ids)
