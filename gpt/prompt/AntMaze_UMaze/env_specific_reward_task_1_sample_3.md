Task 2 Name
Orientation Control

Task 2 Description
Maintain `torso_orientation` as a value of [1.0, 0.0, 0.0, 0.0]

```python
def orientation_control_reward(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:

    # Define the ideal orientation the torso should maintain
    ideal_orientation = np.array([1.0, 0.0, 0.0, 0.0])
    # Extract the actual orientation from ant_obs
    actual_orientation = self.torso_orientation(ant_obs)

    # Compute the L2 norm (Euclidean distance) between the actual orientation and the ideal orientation
    orientation_error = np.linalg.norm(ideal_orientation - actual_orientation)

    # Transform the orientation error into a reward, negative because we want to minimize this error
    orientation_reward = -orientation_error

    # Define a weighting parameter for the orientation reward component
    orientation_weight = 1.0

    # Calculate the total reward by applying the weight
    total_reward = orientation_weight * orientation_reward

    # Store the individual reward component in a dictionary
    reward_components = {
        'orientation_reward': orientation_reward
    }

    return total_reward, reward_components
```