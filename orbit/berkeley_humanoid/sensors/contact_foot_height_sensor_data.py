import torch

from omni.isaac.orbit.sensors.contact_sensor.contact_sensor_data import ContactSensorData

class ContactFootHeightSensorData(ContactSensorData):
    """Data class for the foot height sensor."""

    detach_body_height: torch.Tensor | None = None
    """height of body (in m) when just detached since the last contact.

    Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
    """

    
    current_relative_height: torch.Tensor | None = None
    """height of body (in m) relative to the ground.

    Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
    """