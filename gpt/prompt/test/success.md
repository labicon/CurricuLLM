Based on the provided environment code and the description of Task 2, the success of the "Learn to Squat and Rise" task can be evaluated using the z height of the torso, the angles of the joints, and the overall health of the state as described in the `is_healthy` method. We need to ensure that the Hopper stays within healthy ranges without hopping or moving horizontally. 

The success function will check that:
1. The next z height is within the specified healthy range.
2. The hopper should not be hopping, which implies there should be minimal change in the x component of position between observations.
3. The angles of the thigh, leg, and foot joints remain within healthy boundaries.
4. The hopper stays 'healthy' according to the `is_healthy` function.

```python
def compute_success(observation, action, next_observation) -> bool:
    # Extract the relevant parts of the observation
    prev_z, next_z = observation[1], next_observation[1]
    prev_x, next_x = observation[0], next_observation[0]
    angles = next_observation[3:6]  # thigh, leg, foot joint angles

    # Define the healthy bounds as provided in the environment code
    min_z, max_z = (0.7, np.inf)
    min_angle, max_angle = (-0.2, 0.2)

    # Check if the hopper's z height is within the healthy range
    success_z = min_z < next_z < max_z
    
    # Check if the hopper has not moved horizontally significantly which implies no hopping
    success_x = (abs(next_x - prev_x) < 0.01)  # Allow for a small tolerance in horizontal movement
    
    # Check if the joint angles are within the healthy range
    success_angles = all(min_angle < angle < max_angle for angle in angles)
    
    # Check if the overall state is healthy
    success_health = next_observation[7]  # 7th index in next_observation contains the health status
    
    # Success is when all conditions are true
    success = success_z and success_x and success_angles and bool(success_health)
    
    return success
```

This function assumes that `next_observation` already contains the result of calling `is_healthy` method as the 7th index, which is a Boolean value indicating the healthy state of the agent. If the `next_observation` does not include the result of `is_healthy`, that part of the code should be adjusted accordingly.