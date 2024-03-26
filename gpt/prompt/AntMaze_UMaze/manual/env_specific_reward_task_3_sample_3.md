Task 4 Name
Navigation and Turning

Task 4 Description
The reward function for "Navigation and Turning" should encourage the ant to navigate through the environment towards various goals without expecting it to actually reach the goals. The ant should adjust its torso_orientation to face the direction of the next goal position.

```python
def navigation_and_turning_reward(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Define the weights for different components of the reward
    coordinate_weight = 0.5  # Weight for the torso coordinate movement
    orientation_weight = 0.3  # Weight for the torso orientation
    angular_velocity_weight = 0.2  # Weight to minimize angular velocity
    
    # Extract necessary observation components
    torso_coordinate_xz = self.torso_coordinate(ant_obs)[:2]  # We care mainly about x and z axis
    torso_orientation = self.torso_orientation(ant_obs)
    torso_angular_velocity = self.torso_angular_velocity(ant_obs)
    goal_pos = self.goal_pos()[:2]  # Consider only the x and z components of the goal position

    # Compute the direction vector from the ant to the goal
    direction_to_goal = goal_pos - torso_coordinate_xz
    
    # Normalize the direction vector
    if np.linalg.norm(direction_to_goal) > 0:
        direction_to_goal /= np.linalg.norm(direction_to_goal)

    # Compute the reward for moving closer to the goal using xz coordinates
    coordinate_reward = np.tanh(np.linalg.norm(torso_coordinate_xz - goal_pos))

    # Compute the reward for facing towards the goal (using the dot product between orientation and direction vector)
    orientation_reward = np.dot(torso_orientation[:2], direction_to_goal)

    # Minimize angular velocity to encourage smooth turning
    angular_velocity_penalty = -np.tanh(np.linalg.norm(torso_angular_velocity))

    # Calculate the total weighted reward
    reward = (
        coordinate_weight * coordinate_reward +
        orientation_weight * orientation_reward +
        angular_velocity_weight * angular_velocity_penalty
    )

    # Store the individual reward components for analysis and debugging
    reward_components = {
        'coordinate_reward': coordinate_reward,
        'orientation_reward': orientation_reward,
        'angular_velocity_penalty': angular_velocity_penalty
    }

    return reward, reward_components
```