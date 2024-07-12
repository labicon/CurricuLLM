import torch
from collections.abc import Iterable
from dataclasses import MISSING
from typing import Literal

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuators import IdealPDActuatorCfg


@configclass
class IdentifiedActuatorCfg(IdealPDActuatorCfg):
    """Configuration for direct control (DC) motor actuator model."""

    class_type: type = actuator_pd.IdentifiedActuator

    friction_torque: float = MISSING
    """ (in N-m)."""
    friction_vel: float = MISSING
    """ (in Rad/s)."""
