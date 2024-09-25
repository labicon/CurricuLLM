import math
from dataclasses import MISSING

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils import configclass

from .velocity_command import ConstVelocityCommand


@configclass
class ConstVelocityCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = ConstVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    heading_command: bool = MISSING
    """Whether to use heading command or angular velocity command.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """
    heading_control_stiffness: float = MISSING
    """Scale factor to convert the heading error to angular velocity command."""
    rel_standing_envs: float = MISSING
    """Probability threshold for environments where the robots that are standing still."""
    rel_heading_envs: float = MISSING
    """Probability threshold for environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command)."""

    @configclass
    class Value:
        """Constant Value for the velocity commands."""

        lin_vel_x: float = MISSING  # [m/s]
        lin_vel_y: float = MISSING  # [m/s]
        ang_vel_z: float = MISSING  # [rad/s]
        heading: float = MISSING  # [rad]

    value: Value = MISSING
    """Distribution ranges for the velocity commands."""
