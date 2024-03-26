Task 2 Name
Orientation Control

Task 2 Description
Maintain torso_orientation as a value of [1.0, 0.0, 0.0, 0.0]

```python
def orientation_control_reward(self, ant_obs: np.ndarray) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Define desired orientation as a 4D vector (for yaw, pitch, roll)
    desired_orientation = np.array([1.0, 0.0, 0.0, 0.0])

    # Retrieve the orientation of the torso
    current_orientation = self.torso_orientation(ant_obs)

    # Calculate the L2 norm difference between current orientation and the desired orientation
    orientation_error = np.linalg.norm(current_orientation - desired_orientation)

    # Transform the orientation error into a reward using negative exponential to penalize deviation 
    orientation_reward_comp = np.exp(-orientation_error)

    # Reward weighting factor for orientation control (this should be tuned appropriately)
    orientation_weight = 0.3

    # Compute the weighted reward for orientation control
    orientation_reward = orientation_weight * orientation_reward_comp

    reward_components = {
        'orientation_control': orientation_reward
    }

    # Total reward is just the orientation control component in this case
    total_reward = np.sum(list(reward_components.values()))

    return total_reward, reward_components
```