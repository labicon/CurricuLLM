Task 2 Name
Orientation Control

Task 2 Description
Maintain `torso_orientation` as a value of `[1.0, 0.0, 0.0, 0.0]` (i.e., the ant should keep its orientation steady and aligned to a default pose).

```python
def orientation_control_reward(self, ant_obs: np.ndarray) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Ideal orientation of the ant's torso
    ideal_orientation = np.array([1.0, 0.0, 0.0, 0.0])
    # Get the current orientation from the observation
    current_orientation = self.torso_orientation(ant_obs)
    # Compute the L2 norm of the difference (deviation from the ideal orientation)
    orientation_error = np.linalg.norm(current_orientation - ideal_orientation)
    
    # Define a weight for how much we care about the orientation control
    orientation_control_weight = 0.3
    
    # Use the negative tanh of the error as the reward to maintain the orientation.
    # Negative because we want to minimize the error.
    # The weight scales the importance of this component of the total reward.
    orientation_reward = -orientation_control_weight * np.tanh(orientation_error)
    
    # Creating a dictionary to store the individual reward components
    reward_components = {
        'orientation_reward': orientation_reward
    }

    # Total reward is the sum of components, but here we only have one component
    total_reward = orientation_reward
    
    return total_reward, reward_components
```
Ensure that the variable names and function calls match with those of your environment for the code to work correctly. The `orientation_control_weight` is adjustable to tune the importance of orientation control relative to other potential reward components.