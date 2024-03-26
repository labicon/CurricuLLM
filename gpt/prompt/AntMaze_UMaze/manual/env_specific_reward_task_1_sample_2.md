Task 2 Name: Orientation Control
Task 2 Description: The objective is to maintain the torso_orientation close to a value of [1.0, 0.0, 0.0, 0.0], which represents a neutral orientation in a 3D environment (e.g. the ant facing upright).

```python
def orientation_control_reward(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Orientational goal for upright stance
    desired_orientation = np.array([1.0, 0.0, 0.0, 0.0])

    # Retrieve the current torso orientation from observations
    current_orientation = self.torso_orientation(ant_obs)

    # Compute the L2 norm of the orientation error, which represents how far the ant is from being upright
    orientation_error = np.linalg.norm(current_orientation - desired_orientation)

    # Use negative L2 norm to penalize deviation from the desired orientation
    # A lower error yields a higher (less negative) reward
    orientation_reward = -orientation_error

    # A weighting parameter to scale the contribution of the orientation reward component
    orientation_weight = 1.0  # This value can be adjusted to balance different reward components

    # Calculate the total weighted reward
    weighted_orientation_reward = orientation_weight * orientation_reward

    # Construct a dictionary containing individual reward component scores
    reward_components = {
        "orientation_reward": orientation_reward,  # Raw orientation reward without weighting
        "weighted_orientation_reward": weighted_orientation_reward  # Orientation reward with weighting
    }

    # Total reward is the weighted sum of all individual components
    reward = weighted_orientation_reward

    return reward, reward_components
```