Task 2 Name
Orientation Control

Task 2 Description
Maintain `torso_orientation` as a value of `[1.0, 0.0, 0.0, 0.0]` to achieve a stable and forward-facing orientation as the ant navigates through the maze.

```python
def orientation_control_reward(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Ideal orientation of the torso
    ideal_orientation = np.array([1.0, 0.0, 0.0, 0.0])
    
    # Get the current orientation from the observation
    current_orientation = self.torso_orientation(ant_obs)
    
    # Compute the L2 norm to measure how close the current orientation is to the ideal orientation
    orientation_error = np.linalg.norm(current_orientation - ideal_orientation)
    
    # Use negative exponential of the orientation error as the reward to encourage lower error values
    orientation_reward = np.exp(-orientation_error)
    
    # Assign a weight to the orientation reward component
    orientation_weight = 1.0
    
    # Calculate total reward for orientation control
    total_orientation_reward = orientation_weight * orientation_reward
    
    # Return the total reward and a dictionary of individual reward components
    reward_components = {'orientation_error': orientation_error, 'orientation_reward': total_orientation_reward}
    return total_orientation_reward, reward_components
```

Please note that the exponential function is used here to encourage the ant to maintain its orientation close to the ideal orientation, by penalizing deviations from it more as they get larger. This reward mechanism provides a smooth gradient which is often beneficial for learning.