
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
        """Fills the buffers of the sensor data."""
        # default to all sensors
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)

        # obtain the contact forces
        # TODO: We are handling the indexing ourself because of the shape; (N, B) vs expected (N * B).
        #   This isn't the most efficient way to do this, but it's the easiest to implement.
        net_forces_w = self.contact_physx_view.get_net_contact_forces(dt=self._sim_physics_dt)
        self._data.net_forces_w[env_ids, :, :] = net_forces_w.view(-1, self._num_bodies, 3)[env_ids]
        # update contact force history
        if self.cfg.history_length > 0:
            self._data.net_forces_w_history[env_ids, 1:] = self._data.net_forces_w_history[env_ids, :-1].clone()
            self._data.net_forces_w_history[env_ids, 0] = self._data.net_forces_w[env_ids]

        # obtain the contact force matrix
        if len(self.cfg.filter_prim_paths_expr) != 0:
            # shape of the filtering matrix: (num_envs, num_bodies, num_filter_shapes, 3)
            num_filters = self.contact_physx_view.filter_count
            # acquire and shape the force matrix
            force_matrix_w = self.contact_physx_view.get_contact_force_matrix(dt=self._sim_physics_dt)
            force_matrix_w = force_matrix_w.view(-1, self._num_bodies, num_filters, 3)
            self._data.force_matrix_w[env_ids] = force_matrix_w[env_ids]
        # obtain the pose of the sensor origin
        if self.cfg.track_pose:
            pose = self.body_physx_view.get_transforms().view(-1, self._num_bodies, 7)[env_ids]
            pose[..., 3:] = convert_quat(pose[..., 3:], to="wxyz")
            self._data.pos_w[env_ids], self._data.quat_w[env_ids] = pose.split([3, 4], dim=-1)
        # obtain the air time
        if self.cfg.track_air_time:
            # -- time elapsed since last update
            # since this function is called every frame, we can use the difference to get the elapsed time
            elapsed_time = self._timestamp[env_ids] - self._timestamp_last_update[env_ids]
            # -- check contact state of bodies
            is_contact = torch.norm(self._data.net_forces_w[env_ids, :, :], dim=-1) > self.cfg.force_threshold
            is_first_contact = (self._data.current_air_time[env_ids] > 0) * is_contact
            is_first_detached = (self._data.current_contact_time[env_ids] > 0) * ~is_contact
            # -- update the last contact time if body has just become in contact
            self._data.last_air_time[env_ids] = torch.where(
                is_first_contact,
                self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),
                self._data.last_air_time[env_ids],
            )
            # -- increment time for bodies that are not in contact
            self._data.current_air_time[env_ids] = torch.where(
                ~is_contact, self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1), 0.0
            )

            # NEW: -- update the height of the body when it has just detached
            self._data.detach_body_height[env_ids] = torch.where(
                is_first_detached,
                self._data.pos_w[env_ids, :, 2],
                self._data.detach_body_height[env_ids],
            )
            # -- update the last contact time if body has just detached
            self._data.last_contact_time[env_ids] = torch.where(
                is_first_detached,
                self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),
                self._data.last_contact_time[env_ids],
            )
            # -- increment time for bodies that are in contact
            self._data.current_contact_time[env_ids] = torch.where(
                is_contact, self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1), 0.0
            )

        # NEW: -- update the relative height of the body
        self._data.current_relative_height = self._data.pos_w[env_ids, :, 2] - self._data.detach_body_height[env_ids]

