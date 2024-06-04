from omni.isaac.orbit.utils import configclass


from omni.isaac.orbit.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from .contact_foot_height_sensor import ContactFootHeightSensor


@configclass
class ContactFootHeightSensorCfg(ContactSensorCfg):
    """Configuration for the foot height sensor."""

    class_type: type = ContactFootHeightSensor

    track_pose: bool = True

    track_air_time: bool = True