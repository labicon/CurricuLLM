
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.physics.tensors.impl.api as physx
from pxr import PhysxSchema

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.utils.math import convert_quat

from omni.isaac.orbit.sensors.contact_sensor.contact_sensor import ContactSensor
from .contact_foot_height_sensor_data import ContactFootHeightSensorData

if TYPE_CHECKING:
    from .contact_foot_height_sensor_cfg import ContactFootHeightSensorCfg

class ContactFootHeightSensor(ContactSensor):
    def __init__(self, cfg: ContactFootHeightSensorCfg):
        """Initializes the contact sensor object.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data: ContactFootHeightSensorData = ContactFootHeightSensorData()
        # initialize self._body_physx_view for running in extension mode
        self._body_physx_view = None

    def _initialize_impl(self):
        super()._initialize_impl()
        self._data.detach_body_height = torch.zeros(self._num_envs, self._num_bodies, device=self._device)
        self._data.current_relative_height = torch.zeros(self._num_envs, self._num_bodies, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        
        super()._update_buffers_impl(env_ids)
        is_contact = torch.norm(self._data.net_forces_w[env_ids, :, :], dim=-1) > self.cfg.force_threshold
        is_first_detached = (self._data.current_contact_time[env_ids] > 0) * ~is_contact
        self._data.detach_body_height[env_ids] = torch.where(
            is_first_detached,
            self._data.pos_w[env_ids, :, 2],
            self._data.detach_body_height[env_ids],
        )
        self._data.current_relative_height = self._data.pos_w[env_ids, :, 2] - self._data.detach_body_height[env_ids]

