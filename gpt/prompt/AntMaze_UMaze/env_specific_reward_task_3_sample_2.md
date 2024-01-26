Task 4 Name
Navigation and Turning

Task 4 Description
Maximize torso_coordinate while changing torso_orientation to face multiple random goal_pos without actually reaching them. This task focuses on the ant robot's ability to orient and navigate towards multiple random goals in its environment. It requires the ant to be able to turn towards a given goal without necessarily reaching it.

```python
def navigation_and_turning_reward(self, ant_obs: np.ndarray) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Constants/Weights for reward components
    coordinate_weight = 1.0
    orientation_weight = 10.0
    
    # Getting components from the environment
    xyz_coordinate = self.torso_coordinate(ant_obs)          # Torso position (x, y, z coordinates)
    orientation = self.torso_orientation(ant_obs)            # Torso orientation (quaternions)
    goal_pos = self.goal_pos()                               # Position of the goal
    flat_goal_pos = goal_pos[:2]                             # Flattening to 2D for x and y coordinates

    # Calculating distance to the goal
    distance_to_goal = self.goal_distance(ant_obs)
    
    # Component 1: Maximizing x and y coordinates (Navigation)
    navigation_reward = np.tanh(np.linalg.norm(xyz_coordinate[:2]))     # Taking only x and y for navigation
    navigation_reward *= coordinate_weight                             # Apply weighting
    
    # Component 2: Reward for changing orientation to face the goal (Turning)
    # This uses the difference in yaw between the ant's current orientation and the target goal.
    # Note: This is a simplification for the example. Proper orientation difference in 3D space would involve quaternion math.
    desired_yaw_to_goal = np.arctan2(flat_goal_pos[1] - xyz_coordinate[1], flat_goal_pos[0] - xyz_coordinate[0])
    current_yaw = np.arctan2(orientation[2], orientation[3])  # Assuming orientation[3] represents the forward direction (cosine component)
    yaw_diff = np.linalg.norm(desired_yaw_to_goal - current_yaw) 
    
    # Apply tangens hyperbolicus to reward signal for orientation difference
    orientation_reward = -np.tanh(yaw_diff)     # We use negative because we want to minimize the difference
    orientation_reward *= orientation_weight    # Apply weighting
    
    # Reward is a combination of both navigation and orientation adjustments
    reward = navigation_reward + orientation_reward
    
    # Breakdown of reward components for analysis
    reward_components = {
        'navigation_reward': navigation_reward,
        'orientation_reward': orientation_reward
    }
    
    return reward, reward_components
```