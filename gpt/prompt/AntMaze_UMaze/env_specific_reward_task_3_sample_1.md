Task 4 Name
Navigation and Turning

Task 4 Description
The goal of the Navigation and Turning task is to maximize the movement of the ant's torso (forward progression) while also changing the torso orientation to face towards various goal positions randomly placed in the environment. The ant does not need to reach the goals but must turn to face them, promoting the behavior of navigation and turning without focusing on goal completion.

```python
def navigation_and_turning_reward(self, ant_obs: np.ndarray) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Extract relevant information from observations
    torso_pos = self.torso_coordinate(ant_obs)
    torso_orient = self.torso_orientation(ant_obs)
    goal_position = self.goal_pos()
    
    # Components of the reward signal
    progress_weight = 1.0  # Weighting for forward progression
    orientation_weight = 0.5  # Weighting for torso orientation adjustment
    
    # Reward for moving forward, penalizing movements that are not in the x-direction
    progress_reward = np.tanh(torso_pos[0]) - 0.5 * np.linalg.norm(torso_pos[1:])
    
    # Compute the vector pointing toward the goal from the current position
    direction_to_goal = goal_position[:2] - torso_pos[:2]
    # Normalize vectors for computing the dot product
    forward_vector = np.array([1.0, 0.0])  # Assuming that the ant's forward direction aligns with the x-axis
    direction_to_goal /= (np.linalg.norm(direction_to_goal) + 1e-8)
    
    # Reward for torso orientation aligns with vector to the goal position
    orientation_reward = np.dot(direction_to_goal, forward_vector)
    # Use an arccos to evaluate how well the torso is oriented toward the goal (0 means facing)
    orientation_reward = np.arccos(orientation_reward) / np.pi   # Normalized between [0, 1]
    orientation_reward = np.tanh(1 - orientation_reward)          # Encourage to face the goal
    
    # Compute the total reward with the associated weights for each component
    reward = progress_weight * progress_reward + orientation_weight * orientation_reward
    
    # Create a dictionary to store the individual reward components
    reward_components = {
        'progress_reward': progress_weight * progress_reward,
        'orientation_reward': orientation_weight * orientation_reward,
    }
    
    return reward, reward_components
```

Please note that the forward direction of the ant is assumed to align with the x-axis in the world coordinate system for this calculation, and any deviation from the forward movement reduces the progress reward. The orientation reward encourages the ant to turn such that its x-axis is aligned with the vector pointing to the goal.