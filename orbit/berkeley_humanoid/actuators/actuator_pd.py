# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.core.utils.types import ArticulationActions

from omni.isaac.lab.utils import DelayBuffer, LinearInterpolation

from omni.isaac.lab.actuators import IdealPDActuator

if TYPE_CHECKING:
    from .actuator_cfg import (
        IdentificatedActuatorCfg,
    )


class IdentifiedActuator(IdealPDActuator):
    cfg: IdentificatedActuatorCfg

    def __init__(self, cfg: IdentificatedActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # parse configuration
        if self.cfg.friction_torque is not None:
            self._friction_torque = self.cfg.friction_torque
        else:
            self._friction_torque = torch.inf
        if self.cfg.friction_vel is not None:
            self._friction_vel = self.cfg.friction_vel
        else:
            self._friction_vel = torch.inf

    def compute(
            self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # call the base method
        control_action = super().compute(control_action, joint_pos, joint_vel)
        control_action.joint_efforts = control_action.joint_efforts - self._friction_torque * torch.tanh(
            joint_vel / self._friction_vel)

        self.applied_effort = control_action.joint_efforts
        return control_action
