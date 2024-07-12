# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING


from dataclasses import MISSING

from omni.isaac.core.utils.types import ArticulationActions

from omni.isaac.lab.utils import DelayBuffer, LinearInterpolation

from omni.isaac.lab.actuators import IdealPDActuator

if TYPE_CHECKING:
    from .actuator_cfg import IdentifiedActuatorCfg


class IdentifiedActuator(IdealPDActuator):
    cfg: IdentifiedActuatorCfg

    def __init__(self, cfg: IdentifiedActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.friction_torque = self._parse_joint_parameter(self.cfg.friction_torque, 0.)
        self.activation_vel = self._parse_joint_parameter(self.cfg.activation_vel, torch.inf)
        self.friction_vel = self._parse_joint_parameter(self.cfg.friction_vel, 0.)

    def compute(
            self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # call the base method
        control_action = super().compute(control_action, joint_pos, joint_vel)
        control_action.joint_efforts = control_action.joint_efforts - (self.friction_torque * torch.tanh(
            joint_vel / self.activation_vel) + self.friction_vel * joint_vel)

        self.applied_effort = control_action.joint_efforts
        control_action.joint_positions = None
        control_action.joint_velocities = None

        print(self.friction_torque)

        return control_action
